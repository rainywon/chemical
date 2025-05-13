import hashlib
import sys
import torch
import pandas as pd  # 导入pandas，用于操作Excel文件
import re  # 导入正则表达式模块，用于处理文本
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入文档分割工具
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 导入HuggingFace嵌入模型
from langchain_community.vectorstores import FAISS  # 导入FAISS用于构建向量数据库
from langchain_community.document_loaders import UnstructuredPDFLoader  # 新增导入
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os  # 导入os模块，用于获取工作目录
import json
from pathlib import Path  # 导入Path，用于路径处理
from datetime import datetime  # 导入datetime，用于记录时间戳
from typing import List, Dict, Optional, Set, Tuple  # 导入类型提示
import logging  # 导入日志模块，用于记录运行日志
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池模块，支持并行加载PDF文件
from tqdm import tqdm  # 导入进度条模块，用于显示加载进度
from config import Config  # 导入配置类，用于加载配置参数
import shutil  # 用于文件操作

from pdf_cor_extractor.pdf_ocr_extractor import PDFProcessor


# 配置日志格式
# 配置日志格式，指定输出到stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],  # 明确输出到stdout
    force=True  # 关键：强制覆盖现有配置
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    def __init__(self, config: Config):
        """
        初始化向量数据库构建器
        Args:
            config (Config): 配置类，包含必要的配置
        """
        self.config = config
        
        # 设置缓存目录路径（保留用于存储分块分析结果）
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置向量数据库路径
        self.vector_dir = Path(config.vector_db_path)
        self.vector_backup_dir = self.vector_dir / "backups"
        
        # 将源文件目录定义放在初始化方法中
        self.subfolders = ['标准']  # '标准性文件','法律', '规范性文件'
        
        # 检查文件匹配模式
        if not hasattr(config, 'files') or not config.files:
            # 如果config中没有files参数，使用默认值
            self.config.files = ["data/**/*.pdf", "data/**/*.txt", "data/**/*.md", "data/**/*.docx"]
        
        # 添加GPU使用配置
        self.use_gpu_for_ocr = "cuda" in self.config.device
        
        # 已处理文件状态
        self.failed_files_count = 0
        
        # 是否输出详细的分块内容
        self.print_detailed_chunks = getattr(config, 'print_detailed_chunks', False)
        # 详细输出时每个文本块显示的最大字符数
        self.max_chunk_preview_length = getattr(config, 'max_chunk_preview_length', 200)
        
        logger.info("初始化向量数据库构建器...")

    def _is_non_content_page(self, page_content: str, page_num: int) -> bool:
        """
        检测页面是否为非内容页面，如封面、目录、目次、前言等，这些页面在分块时应当被过滤掉
        
        Args:
            page_content: 页面文本内容
            page_num: 页面编号
            
        Returns:
            bool: 如果是非内容页面返回True，否则返回False
        """
        # 如果是第一页，很可能是封面
        if page_num == 0 or page_num == 1:
            # 封面页通常很短，或者只包含标题、作者等信息
            if len(page_content.strip()) < 200:
                return True
            
            # 封面页通常包含这些关键词
            cover_keywords = ['封面', '版权', '版权所有', '发布', '编写', '编著', 
                             '著作权', '保留所有权利', '版权声明', '修订版']
            for keyword in cover_keywords:
                if keyword in page_content:
                    return True
        
        # 检测目录、目次页面
        toc_keywords = ['目录', '目 录', '目次', '目 次', '章节', '第一章', '第二章', '第三章', '附录']
        
        # 如果页面中包含多个目录关键词，可能是目录页
        keyword_count = sum(1 for keyword in toc_keywords if keyword in page_content)
        if keyword_count >= 1:
            return True
        
        # 检测前言页面
        preface_keywords = ['前言', '前 言', '序言', '序 言', '引言', '引 言', '绪论']
        for keyword in preface_keywords:
            # 如果前言关键词出现在页面开头部分，很可能是前言页
            if keyword in page_content[:200] or f"\n{keyword}\n" in page_content:
                logger.info(f"检测到前言页，关键词: {keyword}")
                return True
        
        # 检查页面是否有典型的目录结构（行首是章节标题，行尾是页码）
        lines = page_content.split('\n')
        pattern_count = 0
        for line in lines:
            line = line.strip()
            # 匹配类似 "第X章 内容..........10" 的模式
            if line and (line[0] == '第' or line.startswith('附录')) and line.strip()[-1].isdigit():
                pattern_count += 1
                
        # 如果有多行符合目录特征，可能是目录页
        if pattern_count >= 3:
            return True
        
        return False

    def _load_single_document(self, file_path: Path) -> Optional[List[Document]]:
        """多线程加载单个文档文件（支持 PDF、DOCX、DOC）"""
        try:
            file_extension = file_path.suffix.lower()
            docs = []

            if file_extension == ".pdf":
                try:
                    # 检查PDF页数
                    import fitz
                    with fitz.open(str(file_path)) as doc:
                        page_count = doc.page_count
                        logger.info(f"[文档加载] PDF文件 '{file_path.name}' 共有 {page_count} 页")
                        
                    # 使用配置中的参数初始化处理器
                    processor = PDFProcessor(
                        file_path=str(file_path), 
                        lang='ch', 
                        use_gpu=self.use_gpu_for_ocr
                    )
                    
                    # 根据页数选择合适的GPU参数配置
                    if page_count > 30:
                        logger.info(f"[文档加载] PDF页数较多({page_count}页)，应用大文档优化配置")
                        processor.configure_gpu(**self.config.pdf_ocr_large_doc_params)
                    else:
                        # 使用标准参数配置
                        processor.configure_gpu(**self.config.pdf_ocr_params)
                    
                    # 处理PDF
                    docs = processor.process()
                    
                    # 过滤掉非内容页面（封面、目录、前言等）
                    if docs:
                        filtered_docs = []
                        filtered_count = 0
                        filtered_types = []
                        
                        for i, doc in enumerate(docs):
                            if not self._is_non_content_page(doc.page_content, i):
                                filtered_docs.append(doc)
                            else:
                                filtered_count += 1
                                # 尝试判断页面类型
                                page_type = "非内容页面"
                                if i == 0:
                                    page_type = "封面"
                                elif "目录" in doc.page_content or "目次" in doc.page_content:
                                    page_type = "目录/目次"
                                elif "前言" in doc.page_content:
                                    page_type = "前言"
                                    
                                filtered_types.append(page_type)
                                logger.info(f"[文档加载] 过滤掉 '{file_path.name}' 的第 {i+1} 页（疑似{page_type}）")
                                
                        if filtered_count > 0:
                            # 汇总过滤情况
                            type_summary = {}
                            for t in filtered_types:
                                if t not in type_summary:
                                    type_summary[t] = 0
                                type_summary[t] += 1
                                
                            type_str = ", ".join([f"{k}页{v}页" for k, v in type_summary.items()])
                            logger.info(f"[文档加载] 从 '{file_path.name}' 中过滤掉 {filtered_count} 页非内容页面（{type_str}）")
                            docs = filtered_docs
                    
                    # 检查处理结果
                    if docs and len(docs) < page_count * 0.5:
                        logger.warning(f"[文档加载] 警告: 只识别出 {len(docs)}/{page_count} 页，低于50%，可能有问题")
                    elif docs:
                        logger.info(f"[文档加载] 成功识别 {len(docs)}/{page_count} 页")
                    
                    # 处理后清理内存
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"[文档加载] 处理PDF文件 '{file_path.name}' 失败: {str(e)}")
                    self.failed_files_count += 1
                    return None
                    
            elif file_extension in [".docx", ".doc"]:
                try:
                    # 首先尝试导入依赖模块
                    try:
                        import docx2txt
                    except ImportError:
                        logger.error(f"缺少处理Word文档所需的依赖包，请运行: pip install docx2txt")
                        # 记录错误但继续执行，以便处理其他文件类型
                        self.failed_files_count += 1
                        return None
                        
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                    
                    # 尝试过滤Word文档的非内容页面
                    if docs and len(docs) > 1:  # 如果Word文档被分成了多个页面
                        filtered_docs = []
                        filtered_count = 0
                        filtered_types = []
                        
                        for i, doc in enumerate(docs):
                            if not self._is_non_content_page(doc.page_content, i):
                                filtered_docs.append(doc)
                            else:
                                filtered_count += 1
                                # 尝试判断页面类型
                                page_type = "非内容页面"
                                if i == 0:
                                    page_type = "封面"
                                elif "目录" in doc.page_content or "目次" in doc.page_content:
                                    page_type = "目录/目次"
                                elif "前言" in doc.page_content:
                                    page_type = "前言"
                                    
                                filtered_types.append(page_type)
                                logger.info(f"[文档加载] 过滤掉 '{file_path.name}' 的第 {i+1} 部分（疑似{page_type}）")
                                
                        if filtered_count > 0:
                            # 汇总过滤情况
                            type_summary = {}
                            for t in filtered_types:
                                if t not in type_summary:
                                    type_summary[t] = 0
                                type_summary[t] += 1
                                
                            type_str = ", ".join([f"{k}{v}页" for k, v in type_summary.items()])
                            logger.info(f"[文档加载] 从 '{file_path.name}' 中过滤掉 {filtered_count} 部分非内容页面（{type_str}）")
                            docs = filtered_docs
                except Exception as e:
                    logger.error(f"[文档加载] 处理DOCX文件 '{file_path.name}' 失败: {str(e)}")
                    
                    # 尝试使用替代方法
                    try:
                        logger.info(f"[文档加载] 尝试使用替代方法加载Word文档...")
                        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(str(file_path))
                        docs = loader.load()
                        logger.info(f"[文档加载] 成功使用替代方法加载Word文档: {file_path.name}")
                        
                        # 也尝试过滤非内容页面
                        if docs and len(docs) > 1:
                            filtered_docs = []
                            filtered_count = 0
                            filtered_types = []
                            
                            for i, doc in enumerate(docs):
                                if not self._is_non_content_page(doc.page_content, i):
                                    filtered_docs.append(doc)
                                else:
                                    filtered_count += 1
                                    # 尝试判断页面类型
                                    page_type = "非内容页面"
                                    if i == 0 or i == 1:
                                        page_type = "封面"
                                    elif "目录" in doc.page_content or "目次" in doc.page_content:
                                        page_type = "目录/目次"
                                    elif "前言" in doc.page_content:
                                        page_type = "前言"
                                        
                                    filtered_types.append(page_type)
                                    logger.info(f"[文档加载] 过滤掉 '{file_path.name}' 的第 {i+1} 部分（疑似{page_type}）")
                                    
                            if filtered_count > 0:
                                # 汇总过滤情况
                                type_summary = {}
                                for t in filtered_types:
                                    if t not in type_summary:
                                        type_summary[t] = 0
                                    type_summary[t] += 1
                                    
                                type_str = ", ".join([f"{k}{v}页" for k, v in type_summary.items()])
                                logger.info(f"[文档加载] 从 '{file_path.name}' 中过滤掉 {filtered_count} 部分非内容页面（{type_str}）")
                                docs = filtered_docs
                    except Exception as e2:
                        logger.error(f"[文档加载] 替代方法也失败: {str(e2)}")
                        self.failed_files_count += 1
                        return None
            else:
                logger.warning(f"[文档加载] 不支持的文件格式: {file_path.name}")
                return None

            if docs:
                # 统一添加元数据
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["file_name"] = file_path.name
                return docs
            return None

        except Exception as e:
            logger.error(f"[文档加载] 加载 {file_path} 失败: {str(e)}")
            self.failed_files_count += 1
            return None

    def load_documents(self) -> List:
        """加载所有文档"""
        logger.info("⌛ 开始加载文档...")

        # 获取所有文档文件
        document_files = []
        for subfolder in self.subfolders:
            folder_path = self.config.data_dir / subfolder
            if folder_path.exists() and folder_path.is_dir():
                document_files.extend([f for f in folder_path.rglob("*") 
                                    if f.suffix.lower() in ['.pdf', '.docx', '.doc']])
            else:
                logger.warning(f"子文件夹 {subfolder} 不存在或不是目录: {folder_path}")
                
        # 过滤并排序文件（先处理较小的文件，避免大文件占用显存）
        document_files = sorted(document_files, key=lambda x: x.stat().st_size)
        logger.info(f"发现 {len(document_files)} 个待处理文件")
        
        results = []
        # 限制线程池大小以避免资源争用
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._load_single_document, file) for file in document_files]
            with tqdm(total=len(futures), desc="加载文档", unit="files") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.extend(res)
                        pbar.update(1)
                        pbar.set_postfix_str(f"已加载 {len(res)} 页")
                    else:
                        pbar.update(1)
        
        # 在处理完成后清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        logger.info(f"✅ 成功加载 {len(results)} 页文档")
        logger.info(f"❌ 未成功加载 {self.failed_files_count} 个文件")
        
        return results

    def process_files(self) -> List:
        """优化的文件处理流程，使用章节分块方法"""
        logger.info("开始文件处理流程")
        
        # 加载所有文档
        all_docs = self.load_documents()

        if not all_docs:
            logger.warning("没有可处理的文件内容")
            return []

        # 首先按文件合并页面内容，避免跨页分块断裂
        logger.info("合并文件页面内容，准备进行整体分块...")
        
        # 按文件分组整理文档
        file_docs = {}
        for doc in all_docs:
            source = doc.metadata.get("source", "")
            if source not in file_docs:
                file_docs[source] = []
            file_docs[source].append(doc)
        
        # 对每个文件的页面进行排序和合并
        whole_docs = []
        for source, docs in file_docs.items():
            # 按页码排序
            sorted_docs = sorted(docs, key=lambda x: x.metadata.get("page", 0))
            
            # 合并文件所有页面的内容
            full_content = "\n".join([doc.page_content for doc in sorted_docs])
            
            # 创建完整文档对象
            file_doc = Document(
                page_content=full_content,
                metadata={
                    "source": source,
                    "file_name": sorted_docs[0].metadata.get("file_name", ""),
                    "page_count": len(sorted_docs),
                    "is_merged_doc": True  # 标记为合并后的完整文档
                }
            )
            whole_docs.append(file_doc)
            
        logger.info(f"已将 {len(all_docs)} 页内容合并为 {len(whole_docs)} 个完整文档")
        
        # 使用按章节分块方法
        chunks = []
        
        with tqdm(total=len(whole_docs), desc="处理文档章节分块") as pbar:
            for doc in whole_docs:
                metadata = doc.metadata.copy()
                # 移除分块后不再适用的元数据
                if "is_merged_doc" in metadata:
                    del metadata["is_merged_doc"]
                
                # 按章节分块
                sections = self._split_by_section(doc.page_content)
                logger.info(f"正在处理{doc.metadata.get('file_name', '未知文件')}")
                logger.info(f"sections: {len(sections)}")
                # 如果找到章节，则使用章节分块
                if sections:
                    logger.info(f"找到 {len(sections)} 个章节，使用章节结构进行分块")
                    for i, (title, content, section_meta) in enumerate(sections):
                        if not content.strip():  # 跳过空章节
                            continue
                            
                        # 生成内容哈希
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        
                        # 合并元数据
                        enhanced_metadata = metadata.copy()
                        enhanced_metadata.update(section_meta)  # 添加章节元数据
                        enhanced_metadata["content_hash"] = content_hash
                        enhanced_metadata["chunk_index"] = i
                        enhanced_metadata["total_chunks"] = len(sections)
                        enhanced_metadata["chunk_type"] = "section"
                        
                        chunks.append(Document(
                            page_content=content,
                            metadata=enhanced_metadata
                        ))
                else:
                    # 如果未找到章节结构，则直接使用递归分块
                    logger.warning(f"未检测到章节结构，直接使用递归分块方法")
                    
                    # 递归文本分割配置
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap,
                        separators=[
                            "\n\n", "\n", "。", "；", "！", "？", "，", " ", ""
                        ],
                        length_function=len,
                        add_start_index=True,
                        is_separator_regex=False
                    )
                    
                    # 对完整文档进行分块
                    split_texts = text_splitter.split_text(doc.page_content)
                    
                    # 处理每个文本块
                    for i, text in enumerate(split_texts):
                        if not text.strip():  # 跳过空文本块
                            continue
                            
                        # 生成内容哈希
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        enhanced_metadata = metadata.copy()
                        enhanced_metadata["content_hash"] = content_hash
                        enhanced_metadata["chunk_index"] = i
                        enhanced_metadata["total_chunks"] = len(split_texts)
                        enhanced_metadata["chunk_type"] = "fixed_size"
                        
                        chunks.append(Document(
                            page_content=text,
                            metadata=enhanced_metadata
                        ))
                
                pbar.update(1)
        
        # 直接使用生成的文本块，不进行后处理
        logger.info(f"生成 {len(chunks)} 个文本块")
        
        # 打印分块结果概览
        self._print_chunks_summary(chunks)
        
        # 保存分块到文件，方便用户查看
        self.save_chunks_to_file(chunks)

        return chunks

    def _split_by_section(self, text: str) -> List[Tuple[str, str, Dict]]:
        """
        根据章节标题（如1、1.1、1.1.1格式）将文本分割成段落，
        同时将一级标题与其直接下属内容合并为一个块
        
        Args:
            text: 完整的文档文本
            
        Returns:
            List[Tuple[str, str, Dict]]: 返回章节标题、章节内容和元数据的元组列表
        """
        logger.info("开始按章节结构进行文档分块...")
        import re
        
        # 合并所有模式
        patterns = {
            # 标准格式
            'standard': [
                # 第一级到第四级标题
                r'^\s*(\d+)\.?\s+([^\n]+)$',
                r'^\s*(\d+\.\d+)\.?\s+([^\n]+)$',
                r'^\s*(\d+\.\d+\.\d+)\.?\s+([^\n]+)$',
                r'^\s*(\d+\.\d+\.\d+\.\d+)\.?\s+([^\n]+)$',
                # 中文序号和括号序号标题
                r'^\s*([一二三四五六七八九十]+)[、.．]\s+([^\n]+)$',
                r'^\s*[（(]([一二三四五六七八九十]+)[)）]\s+([^\n]+)$',
                # 附录格式
                r'^\s*(附录\s*[A-Za-z])[.．、]?\s*([^\n]+)?$'
            ],
            # 无空格格式
            'no_space': [
                r'^\s*(\d+)\.([\S].*?)$',
                r'^\s*(\d+\.\d+)([\S].*?)$',
                r'^\s*(\d+\.\d+\.\d+)([\S].*?)$',
                r'^\s*(\d+\.\d+\.\d+\.\d+)([\S].*?)$'
            ],
            # 独立章节编号
            'standalone': [
                r'^\s*(\d+)\s*$',
                r'^\s*(\d+)\.?\s*$',
                r'^\s*(\d+\.\d+)\.?\s*$',
                r'^\s*(\d+\.\d+\.\d+)\.?\s*$',
                r'^\s*(\d+\.\d+\.\d+\.\d+)\.?\s*$'
            ],
            # 事故报告特有格式
            'accident_report': [
                # 事故报告中的中文编号（一、二、三、四、五等）
                r'^\s*(一|二|三|四|五|六|七|八|九|十|[一二三四五六七八九十]{2})\s*[、,:：.．]\s*(.+)$',
                # 事故报告中的"1."、"2."格式
                r'^\s*(\d+)\s*[、,:：.．]\s*(.+)$',
                # 日期格式标题（如"2014年1月9日事故情况"）
                r'^(\d{4}年\d{1,2}月\d{1,2}日.*?)[:：]?\s*(.*)$',
                # "（一）、（二）"格式
                r'^\s*[（(]\s*(一|二|三|四|五|六|七|八|九|十|[一二三四五六七八九十]{2})\s*[)）]\s*[、,:：.．]?\s*(.+)$'
            ]
        }
        
        # 分割文本为行
        lines = text.split('\n')
        
        # 初始化变量
        sections = []
        current = {
            "num": "",
            "title": "",
            "content": [],
            "level": 0,
            "children": []  # 存储子章节
        }
        found_first_section = False
        
        # 用于日期时间检测的正则表达式
        date_time_pattern = re.compile(r'(\d{4}年\d{1,2}月\d{1,2}日(?:\s*[上下]午)?\s*\d{1,2}[时:](?:\d{1,2}分?)?)')
        
        # 事故报告特征的检测计数
        accident_report_features = 0
        
        # 临时存储区，用于合并一级标题与其子项
        pending_main_sections = []
        current_main_section = None
        last_section_level = 0
        
        line_num = 0
        while line_num < len(lines):
            line = lines[line_num].strip()
            line_num += 1
            
            # 检测是否包含日期时间（事故报告特征）
            if date_time_pattern.search(line):
                accident_report_features += 1
            
            # 处理空行
            if not line:
                if current["content"]:
                    current["content"].append("")
                continue
            
            # 识别标题变量
            match_info = {
                "is_header": False,
                "level": 0,
                "num": "",
                "title": "",
                "next_line_used": False,
                "pattern_type": ""
            }
            
            # 检查各种标题模式
            for pattern_type, pattern_list in patterns.items():
                if match_info["is_header"]:
                    break
                    
                for i, pattern in enumerate(pattern_list):
                    match = re.match(pattern, line)
                    if not match:
                        continue
                        
                    match_info["is_header"] = True
                    match_info["level"] = i + 1
                    match_info["num"] = match.group(1)
                    match_info["pattern_type"] = pattern_type
                    
                    # 处理标题文本
                    if len(match.groups()) > 1 and match.group(2):
                        match_info["title"] = match.group(2).strip()
                    else:
                        match_info["title"] = match_info["num"]
                    
                    # 事故报告特有的处理
                    if pattern_type == 'accident_report':
                        # 增加事故报告特征计数
                        accident_report_features += 1
                        # 为中文数字设置合适的级别
                        if re.match(r'^[一二三四五六七八九十]{1,2}$', match_info["num"]):
                            match_info["level"] = 1  # 一级标题（一、二、三...）
                        elif re.match(r'^\d+$', match_info["num"]):
                            match_info["level"] = 2  # 二级标题（1、2、3...）
                    
                    # 对于独立编号格式，检查下一行是否为标题内容
                    if pattern_type == 'standalone':
                        if line_num < len(lines):
                            # 只向前查看最多3行
                            for look_ahead in range(line_num, min(line_num + 3, len(lines))):
                                next_line = lines[look_ahead].strip()
                                if not next_line:
                                    continue
                                
                                # 确认下一行不是标题
                                is_next_header = False
                                for ptype in patterns.values():
                                    for p in ptype:
                                        if re.match(p, next_line):
                                            is_next_header = True
                                            break
                                    if is_next_header:
                                        break
                                
                                if not is_next_header and len(next_line) < 50:
                                    match_info["title"] = next_line
                                    match_info["next_line_used"] = True
                                    line_num = look_ahead + 1
                                    break
                    
                    break
            
            # 检查是否找到第一个一级标题
            if match_info["is_header"] and not found_first_section and match_info["level"] == 1:
                found_first_section = True
            
            # 处理章节标题行
            if match_info["is_header"] and (found_first_section or "附录" in match_info["num"] or accident_report_features > 2):
                # 当识别到标题时，保存当前的章节
                if current["num"] and current["content"]:
                    content = "\n".join(current["content"])
                    metadata = {
                        "section_num": current["num"],
                        "section_title": current["title"],
                        "section_level": current["level"],
                        "section_type": "accident_report" if accident_report_features > 2 else "standard"
                    }
                    
                    # 判断是否是一级标题
                    is_main_heading = current["level"] == 1
                    
                    # 如果是一级标题，存储到主章节对象中
                    if is_main_heading:
                        if current_main_section:
                            # 将当前主章节添加到输出列表
                            pending_main_sections.append(current_main_section)
                        
                        # 创建新的主章节对象
                        current_main_section = {
                            "title": current["num"] + " " + current["title"],
                            "content": content,
                            "metadata": metadata,
                            "children": []  # 子章节列表
                        }
                    # 如果是子标题且存在当前主章节，则添加到子章节列表中
                    elif current_main_section:
                        current_main_section["children"].append({
                            "title": current["num"] + " " + current["title"],
                            "content": content,
                            "metadata": metadata
                        })
                    # 如果是子标题但没有主章节（例如文档开始就是子章节），则直接添加到输出列表
                    else:
                        sections.append((current["num"] + " " + current["title"], content, metadata))
                
                # 创建新章节
                current = {
                    "num": match_info["num"],
                    "title": match_info["title"],
                    "level": match_info["level"],
                    "content": [f"{match_info['num']} {match_info['title']}"] if not match_info["next_line_used"] else [f"{match_info['num']} {match_info['title']}"]
                }
                
                # 记录上一个章节的级别
                last_section_level = match_info["level"]
            else:
                # 常规内容行
                if current["content"] or not found_first_section:
                    current["content"].append(line)
                # 如果是首段内容且还没有章节，则创建一个"概述"章节
                elif not current["content"] and not sections and len(line) > 20:
                    current = {
                        "num": "概述",
                        "title": "",
                        "level": 0,
                        "content": [line]
                    }
        
        # 处理最后一个章节
        if current["num"] and current["content"]:
            content = "\n".join(current["content"])
            metadata = {
                "section_num": current["num"],
                "section_title": current["title"],
                "section_level": current["level"],
                "section_type": "accident_report" if accident_report_features > 2 else "standard"
            }
            
            # 判断是否是一级标题
            is_main_heading = current["level"] == 1
            
            if is_main_heading:
                if current_main_section:
                    pending_main_sections.append(current_main_section)
                
                current_main_section = {
                    "title": current["num"] + " " + current["title"],
                    "content": content,
                    "metadata": metadata,
                    "children": []
                }
            elif current_main_section:
                current_main_section["children"].append({
                    "title": current["num"] + " " + current["title"],
                    "content": content,
                    "metadata": metadata
                })
            else:
                sections.append((current["num"] + " " + current["title"], content, metadata))
        
        # 添加最后一个主章节
        if current_main_section:
            pending_main_sections.append(current_main_section)
        
        # 处理所有待处理的主章节及其子章节，合并内容
        for main_section in pending_main_sections:
            if main_section["children"]:
                # 合并主章节内容和所有子章节内容
                combined_content = main_section["content"] + "\n\n"
                
                for child in main_section["children"]:
                    combined_content += child["content"] + "\n\n"
                
                # 保留主章节的元数据，但添加包含子章节信息
                main_section["metadata"]["contains_subsections"] = True
                main_section["metadata"]["subsection_count"] = len(main_section["children"])
                
                # 将合并后的章节添加到输出列表
                sections.append((main_section["title"], combined_content.strip(), main_section["metadata"]))
            else:
                # 如果没有子章节，则直接添加主章节
                sections.append((main_section["title"], main_section["content"], main_section["metadata"]))
        
        # 如果全文没有识别到章节，但有内容且包含事故报告特征
        if not sections and accident_report_features > 1:
            # 使用简单段落分割（通过连续空行）
            paragraphs = []
            current_para = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_para.append(line)
                elif current_para:  # 空行且当前段落有内容
                    paragraphs.append("\n".join(current_para))
                    current_para = []
            
            # 添加最后一个段落
            if current_para:
                paragraphs.append("\n".join(current_para))
            
            # 将段落转换为章节
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 30:  # 只处理较长的段落
                    # 尝试从段落中提取标题
                    first_line = paragraph.split("\n")[0] if "\n" in paragraph else ""
                    title = first_line[:50] if len(first_line) > 10 else f"段落{i+1}"
                    
                    metadata = {
                        "section_num": f"P{i+1}",
                        "section_title": title,
                        "section_level": 1,
                        "section_type": "accident_report_paragraph"
                    }
                    sections.append((f"P{i+1} {title}", paragraph, metadata))
        
        # 输出结果摘要
        logger.info(f"按章节结构分块完成，共找到 {len(sections)} 个章节")
        if sections:
            for i, (title, _, meta) in enumerate(sections[:min(5, len(sections))]):
                section_type = meta.get("section_type", "standard")
                contains_subsections = meta.get("contains_subsections", False)
                subsection_str = f"，包含 {meta.get('subsection_count', 0)} 个子章节" if contains_subsections else ""
                logger.info(f"  • 章节 {i+1}: {title} (级别: {meta['section_level']}, 类型: {section_type}{subsection_str})")
            
            if len(sections) > 5:
                logger.info(f"  ... 以及 {len(sections)-5} 个其他章节")
        
        return sections

    def _ensure_complete_sentences(self, text: str) -> str:
        """确保文本块以完整句子开始和结束
        
        Args:
            text: 原始文本块
            
        Returns:
            处理后的文本块，确保以完整句子开始和结束
        """
        if not text or len(text) < 10:  # 文本过短则直接返回
            return text
            
        # 中文句子结束标记
        sentence_end_marks = ['。', '！', '？', '；', '\n']
        # 句子开始的可能标记（中文段落开头、章节标题等）
        sentence_start_patterns = ['\n', '第.{1,3}章', '第.{1,3}节']
        
        # 处理文本块开头
        text_stripped = text.lstrip()
        # 如果不是以句末标点开头，也不是以大写字母或数字开头（可能是新段落），则可能是不完整句子
        is_incomplete_start = True
        
        # 检查是否以完整句子或段落开始的标记
        for pattern in sentence_start_patterns:
            if text.startswith(pattern) or text_stripped[0].isupper() or text_stripped[0].isdigit():
                is_incomplete_start = False
                break
        
        if is_incomplete_start:
            # 查找第一个完整句子的开始
            for mark in sentence_end_marks:
                pos = text.find(mark)
                if pos > 0:
                    # 找到句末标记后的内容作为起点
                    try:
                        # 确保句末标记后还有内容
                        if pos + 1 < len(text):
                            text = text[pos+1:].lstrip()
                            break
                    except:
                        # 出错则保持原样
                        pass
        
        # 处理文本块结尾
        is_incomplete_end = True
        # 检查是否以完整句子结束
        for mark in sentence_end_marks:
            if text.endswith(mark):
                is_incomplete_end = False
                break
        
        if is_incomplete_end:
            # 找最后一个完整句子的结束位置
            last_pos = -1
            for mark in sentence_end_marks:
                pos = text.rfind(mark)
                if pos > last_pos:
                    last_pos = pos
                    
            if last_pos > 0:
                # 截取到最后一个完整句子结束
                text = text[:last_pos+1]
        
        return text.strip()
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """对分块后的文本进行后处理，优化块的质量
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            处理后的分块列表
        """
        if not chunks:
            return []
            
        logger.info("对文本块进行后处理优化...")
        processed_chunks = []
        
        # 按文档源分组处理
        doc_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(chunk)
        
        total_merged = 0
        
        # 处理每个文档的块
        for source, source_chunks in doc_chunks.items():
            # 按块索引排序
            sorted_chunks = sorted(source_chunks, 
                                   key=lambda x: x.metadata.get("chunk_index", 0))
            
            # 检查和处理相邻块
            for i, chunk in enumerate(sorted_chunks):
                # 只对章节类型的块应用完整句子处理
                if chunk.metadata.get("chunk_type") == "section":
                    chunk.page_content = self._ensure_complete_sentences(chunk.page_content)
                
                # 跳过空块
                if not chunk.page_content.strip():
                    continue
            
                processed_chunks.append(chunk)
        
        logger.info(f"后处理完成，优化后的块数: {len(processed_chunks)}")
        return processed_chunks
        

    def _print_chunks_summary(self, chunks: List[Document]):
        """打印文本分块结果概览"""
        if not chunks:
            logger.info("没有文本块可供显示")
            return
            
        # 统计信息
        total_chunks = len(chunks)
        avg_chunk_length = sum(len(chunk.page_content) for chunk in chunks) / total_chunks
        files_count = len(set(chunk.metadata.get("source", "") for chunk in chunks))
        
        logger.info("\n" + "="*50)
        logger.info("📊 文本分块处理概览")
        logger.info("="*50)
        logger.info(f"📄 总块数: {total_chunks}")
        logger.info(f"📊 平均块长度: {avg_chunk_length:.1f} 字符")
        logger.info(f"📂 涉及文件数: {files_count}")
        
        # 文件级统计
        file_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "未知来源")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(chunk)
        
        logger.info("\n📂 文件级分块统计:")
        for file_path, file_chunks_list in sorted(file_chunks.items(), key=lambda x: len(x[1]), reverse=True):
            file_name = Path(file_path).name if isinstance(file_path, str) else "未知文件"
            logger.info(f"  • {file_name}: {len(file_chunks_list)} 块")
        
        
        # 输出详细分块内容 (如果开启)
        if self.print_detailed_chunks:
            self._print_detailed_chunks(chunks)
            
        logger.info("="*50)

    def _print_detailed_chunks(self, chunks: List[Document]):
        """输出详细的分块内容"""
        logger.info("\n" + "="*50)
        logger.info("📑 详细文本块内容")
        logger.info("="*50)
        
        # 将分块按文件分组
        file_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "未知来源")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(chunk)
        
        # 为了更有组织地输出，先按文件输出
        for file_path, file_chunks_list in sorted(file_chunks.items(), key=lambda x: len(x[1]), reverse=True):
            file_name = Path(file_path).name if isinstance(file_path, str) else "未知文件"
            logger.info(f"\n📄 文件: {file_name} (共{len(file_chunks_list)}块)")
            
            # 输出该文件的前3个块
            for i, chunk in enumerate(file_chunks_list[:3]):
                page_num = chunk.metadata.get("page", "未知页码")
                chunk_size = len(chunk.page_content)
                
                # 获取预览内容
                content_preview = chunk.page_content
                if len(content_preview) > self.max_chunk_preview_length:
                    content_preview = content_preview[:self.max_chunk_preview_length] + "..."
                
                # 替换换行符以便于控制台显示
                content_preview = content_preview.replace("\n", "\\n")
                
                logger.info(f"\n  块 {i+1}/{len(file_chunks_list[:3])} [第{page_num}页, {chunk_size}字符]:")
                logger.info(f"  {content_preview}")
            
            # 如果文件中的块数超过3个，显示省略信息
            if len(file_chunks_list) > 3:
                logger.info(f"  ... 还有 {len(file_chunks_list) - 3} 个块未显示 ...")
                
        # 输出保存完整分块内容的提示
        chunks_detail_file = self.cache_dir / "chunks_detail.txt"
        try:
            with open(chunks_detail_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    source = chunk.metadata.get("source", "未知来源")
                    file_name = Path(source).name if isinstance(source, str) else "未知文件"
                    page_num = chunk.metadata.get("page", "未知页码")
                    
                    f.write(f"=== 块 {i+1}/{len(chunks)} [{file_name} - 第{page_num}页] ===\n")
                    f.write(chunk.page_content)
                    f.write("\n\n")
            
            logger.info(f"\n✅ 所有文本块的详细内容已保存至: {chunks_detail_file}")
        except Exception as e:
            logger.error(f"保存详细块内容失败: {str(e)}")
        
        logger.info("="*50)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """创建嵌入模型实例"""
        logger.info("初始化嵌入模型...")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_path,  # 嵌入模型的路径
            model_kwargs={"device": self.config.device},  # 设置设备为CPU或GPU
            encode_kwargs={
                "batch_size": self.config.batch_size,  # 批处理大小
                "normalize_embeddings": self.config.normalize_embeddings  # 是否归一化嵌入
            },
        )

    def backup_vector_db(self):
        """备份现有向量数据库"""
        vector_db_path = Path(self.config.vector_db_path)
        if not vector_db_path.exists():
            return False
            
        try:
            # 创建备份目录
            backup_dir = vector_db_path.parent / f"{vector_db_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有文件到备份目录
            for item in vector_db_path.glob('*'):
                if item.is_file():
                    shutil.copy2(item, backup_dir)
                elif item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                    
            logger.info(f"✅ 向量数据库已备份至 {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"备份向量数据库失败: {str(e)}")
            return False

    def build_vector_store(self):
        """构建向量数据库
        
        Returns:
            List[Document]: 处理后的文档块列表
        """
        logger.info("开始构建向量数据库")

        # 创建必要目录
        Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)

        # 处理文档
        chunks = self.process_files()  # 处理文档并分块
        
        if not chunks:
            logger.warning("没有文档块可以处理，跳过向量存储构建")
            return []

        # 备份现有向量数据库
        if Path(self.config.vector_db_path).exists() and any(Path(self.config.vector_db_path).glob('*')):
            self.backup_vector_db()

        # 生成嵌入模型
        embeddings = self.create_embeddings()

        # 构建向量存储
        logger.info("生成向量...")
        # 构建向量存储时显式指定
        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE  # 明确指定余弦相似度
        )

        # 保存向量数据库
        vector_store.save_local(str(self.config.vector_db_path))  # 保存向量存储到指定路径
        logger.info(f"向量数据库已保存至 {self.config.vector_db_path}")  # 输出保存路径
        
        # 返回处理后的文档块
        return chunks

    def save_chunks_to_file(self, chunks: List[Document]):
        """将文档分块保存到文件中，支持多种格式，但不作为缓存存储
        
        Args:
            chunks: 文档分块列表
        """
        if not chunks:
            logger.info("没有文本块可供保存")
            return
        
        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为纯文本格式，方便直接查看
        text_file = self.cache_dir / "chunks_text.txt"
        try:
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(f"文档分块总览\n")
                f.write(f"=============\n")
                f.write(f"总块数: {len(chunks)}\n")
                f.write(f"涉及文件数: {len(set(chunk.metadata.get('source', '') for chunk in chunks))}\n\n")
                
                # 按文件分组输出
                file_chunks = {}
                for chunk in chunks:
                    source = chunk.metadata.get("source", "未知来源")
                    if source not in file_chunks:
                        file_chunks[source] = []
                    file_chunks[source].append(chunk)
                
                for file_path, file_chunks_list in sorted(file_chunks.items(), key=lambda x: len(x[1]), reverse=True):
                    file_name = Path(file_path).name if isinstance(file_path, str) else "未知文件"
                    f.write(f"\n{'='*80}\n")
                    f.write(f"文件: {file_name} (共{len(file_chunks_list)}块)\n")
                    f.write(f"{'='*80}\n\n")
                    
                    for i, chunk in enumerate(file_chunks_list):
                        # 获取章节信息
                        section_num = chunk.metadata.get("section_num", "")
                        section_title = chunk.metadata.get("section_title", "")
                        chunk_index = chunk.metadata.get("chunk_index", i)
                        total_chunks = chunk.metadata.get("total_chunks", len(file_chunks_list))
                        position = chunk.metadata.get("position", "")
                        chunk_type = chunk.metadata.get("chunk_type", "")
                        
                        # 构建块标题
                        header = f"----- 块 {chunk_index+1}/{total_chunks} "
                        if section_num and section_title:
                            header += f"[章节: {section_num} {section_title}, "
                        header += f"位置:{position}, {len(chunk.page_content)}字符"
                        if chunk_type:
                            header += f", 类型:{chunk_type}"
                        header += "] -----\n"
                        
                        # 写入块信息
                        f.write(header)
                        f.write(chunk.page_content)
                        f.write("\n\n")
            
            logger.info(f"✅ 文本格式的分块内容已保存至: {text_file}")
        except Exception as e:
            logger.error(f"保存文本格式的分块内容失败: {str(e)}")
        
        # 保存为JSON格式，包含完整的元数据
        json_file = self.cache_dir / "chunks_detail.json"
        try:
            chunks_data = []
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "index": i,
                    "content": chunk.page_content,
                    "length": len(chunk.page_content),
                    "metadata": chunk.metadata
                }
                chunks_data.append(chunk_data)
                
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat(),
                    "chunks": chunks_data
                }, f, ensure_ascii=False, indent=2)
                
            logger.info(f"✅ JSON格式的分块详细信息已保存至: {json_file}")
        except Exception as e:
            logger.error(f"保存JSON格式的分块详细信息失败: {str(e)}")
        
        # 保存CSV格式的摘要信息，方便导入电子表格查看
        csv_file = self.cache_dir / "chunks_summary.csv"
        try:
            with open(csv_file, "w", encoding="utf-8") as f:
                # 写入CSV头
                f.write("索引,文件名,章节编号,章节标题,块索引,总块数,位置,字符数,内容预览\n")
                
                for i, chunk in enumerate(chunks):
                    source = chunk.metadata.get("source", "未知来源")
                    file_name = Path(source).name if isinstance(source, str) else "未知文件"
                    section_num = chunk.metadata.get("section_num", "")
                    section_title = chunk.metadata.get("section_title", "")
                    chunk_index = chunk.metadata.get("chunk_index", i)
                    total_chunks = chunk.metadata.get("total_chunks", 0)
                    position = chunk.metadata.get("position", "")
                    length = len(chunk.page_content)
                    
                    # 内容预览，去除换行符
                    preview = chunk.page_content[:100].replace("\n", " ").replace("\r", " ")
                    if len(chunk.page_content) > 100:
                        preview += "..."
                    preview = f'"{preview}"'  # 用引号包围，避免CSV解析错误
                    
                    f.write(f"{i},{file_name},{section_num},{section_title},{chunk_index},{total_chunks},{position},{length},{preview}\n")
                
            logger.info(f"✅ CSV格式的分块摘要已保存至: {csv_file}")
        except Exception as e:
            logger.error(f"保存CSV格式的分块摘要失败: {str(e)}")

    def save_chunks_to_excel(self, chunks: List[Document], output_dir: str = None):
        """将文档分块保存到Excel文件中，按照源文件分组
        
        Args:
            chunks: 文档分块列表
            output_dir: 输出目录，默认为C:/Users/Administrator/Desktop/chunks
        """
        if not chunks:
            logger.info("没有文本块可供保存")
            return
        
        # 设置输出目录，如果未指定则使用桌面路径
        if output_dir is None:
            output_dir = r"C:\Users\Administrator\Desktop\chunks"
            
        # 确保输出目录存在
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始将文档分块保存到Excel文件，输出目录: {output_dir}")
        
        # 按文件分组整理文档块
        file_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(chunk)
        
        # 术语判断正则表达式
        term_patterns = [
            r'^(?:\d+\.?)+\s*术语(?:和定义)?$',  # 匹配"3.术语"、"3.1 术语和定义"等
            r'^第\s*\d+\s*章\s*术语(?:和定义)?$',  # 匹配"第3章 术语和定义"
            r'^[一二三四五六七八九十]+、\s*术语(?:和定义)?$'  # 匹配"三、术语和定义"
        ]
        
        # 处理每个文件的块
        total_files = len(file_chunks)
        processed_files = 0
        
        with tqdm(total=total_files, desc="保存Excel文件") as pbar:
            for source, source_chunks in file_chunks.items():
                try:
                    # 获取文件名，用作Excel文件名
                    file_path = Path(source)
                    file_name = file_path.stem  # 不带扩展名的文件名
                    excel_filename = f"{file_name}.xlsx"
                    excel_path = output_path / excel_filename
                    
                    # 创建数据结构以存储Excel数据
                    excel_data = []
                    
                    # 按块索引排序
                    sorted_chunks = sorted(source_chunks, 
                                           key=lambda x: x.metadata.get("chunk_index", 0))
                    
                    # 处理每个文本块
                    for chunk in sorted_chunks:
                        content = chunk.page_content
                        
                        # 获取章节编号和标题
                        section_num = chunk.metadata.get("section_num", "")
                        section_title = chunk.metadata.get("section_title", "")
                        
                        # 原文内容：删除章节编号
                        # 如果内容以章节编号开头，则删除
                        original_content = content
                        if section_num:
                            # 尝试删除开头的章节编号和标题
                            header_pattern = f"^{re.escape(section_num)}\\s*{re.escape(section_title)}\\s*"
                            original_content = re.sub(header_pattern, "", original_content, count=1)
                        
                        # 入库内容：直接使用原始content
                        db_content = content
                        
                        # 判断是否是术语块
                        is_term = False
                        
                        # 检查块的标题是否符合术语章节的特征
                        if section_title and section_num:
                            full_title = f"{section_num} {section_title}"
                            for pattern in term_patterns:
                                if re.search(pattern, full_title, re.IGNORECASE):
                                    is_term = True
                                    break
                        
                        # 检查块内容是否包含术语定义的特征
                        if not is_term:
                            # 术语定义通常采用"术语名 定义"的格式
                            definition_patterns = [
                                r'\d+\.\d+\s+[\u4e00-\u9fa5a-zA-Z]+\s+[\u4e00-\u9fa5]',  # 3.1 术语 定义
                                r'[\u4e00-\u9fa5a-zA-Z]+\s+[\u4e00-\u9fa5]'  # 术语 定义
                            ]
                            
                            definition_count = 0
                            content_lines = content.split('\n')
                            for line in content_lines:
                                for pattern in definition_patterns:
                                    if re.search(pattern, line):
                                        definition_count += 1
                                        break
                            
                            # 如果有多行符合术语定义模式，则认为是术语块
                            if definition_count >= 3:  # 至少有3个术语定义
                                is_term = True
                        
                        # 将文本块添加到Excel数据中
                        excel_data.append({
                            "原文内容": original_content.strip(),
                            "入库内容": db_content.strip(),
                            "术语": "是" if is_term else "否"
                        })
                    
                    # 创建DataFrame并保存为Excel
                    if excel_data:
                        df = pd.DataFrame(excel_data)
                        df.to_excel(excel_path, index=False, engine='openpyxl')
                        logger.info(f"✅ 已将 '{file_name}' 的 {len(excel_data)} 个文本块保存至Excel文件: {excel_path}")
                    else:
                        logger.warning(f"⚠️ '{file_name}' 没有可保存的文本块")
                
                except Exception as e:
                    logger.error(f"❌ 保存 '{file_name}' 的Excel文件时出错: {str(e)}")
                
                processed_files += 1
                pbar.update(1)
                pbar.set_postfix_str(f"已处理 {processed_files}/{total_files} 个文件")
        
        logger.info(f"✅ 所有文档分块已保存到Excel文件，共处理 {processed_files} 个文件")


if __name__ == "__main__":
    try:
        # 初始化配置
        config = Config()
        
        # 打印工作目录信息
        logger.info(f"工作目录: {os.getcwd()}")
        logger.info(f"数据目录: {config.data_dir}")
        logger.info(f"缓存目录: {config.cache_dir}")
        logger.info(f"向量数据库目录: {config.vector_db_path}")
        
        # Excel输出目录
        excel_output_dir = r"C:\Users\Administrator\Desktop\chunks"
        logger.info(f"Excel文件将保存至: {excel_output_dir}")
        
        # 构建向量数据库
        builder = VectorDBBuilder(config)
        
        # 处理文档并获取分块
        logger.info("开始处理文档并导出Excel...")
        chunks = builder.process_files()
        
        # 将文档分块保存到Excel文件
        builder.save_chunks_to_excel(chunks, excel_output_dir)
        
        logger.info("所有处理完成")

    except Exception as e:
        logger.exception("程序运行出错")  # 记录程序异常
    finally:
        logger.info("程序运行结束")  # 程序结束日志
