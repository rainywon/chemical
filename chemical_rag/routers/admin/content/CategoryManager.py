# 引入 FastAPI 中的 APIRouter 和 HTTPException 模块，用于创建路由和处理异常
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
# 引入 Pydantic 中的 BaseModel 类，用于定义请求体的数据结构和验证
from pydantic import BaseModel
# 从数据库模块导入 execute_query 和 execute_update 函数，用于执行查询和更新操作
from database import execute_query, execute_update
# 引入时间模块处理日期范围
from datetime import datetime, timedelta
# 引入 typing 模块中的 Optional 和 List 类型
from typing import Optional, List, Dict, Any
# 引入操作系统相关模块
import os
import shutil
from pathlib import Path
import glob
import logging
import traceback
# 导入配置
from config import Config

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 导入向量数据库构建器
try:
    from chemical_rag.build_vector_store import VectorDBBuilder
except ImportError:
    # 尝试直接导入
    from build_vector_store import VectorDBBuilder

# 导入向量存储
from langchain_community.vectorstores import FAISS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 APIRouter 实例，用于定义路由
router = APIRouter()

# 获取配置实例
config = Config()

# 设置文件存储路径
KNOWLEDGE_BASE_PATH = config.knowledge_base_path
# 确保路径存在
os.makedirs(KNOWLEDGE_BASE_PATH, exist_ok=True)

# 获取当前管理员ID
async def get_current_admin(request: Request):
    try:
        # 从Authorization头获取token
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            raise HTTPException(status_code=401, detail="无效的认证信息")
        
        token = auth_header.split(' ')[1]
        
        # 查询admin_tokens表
        admin_result = execute_query(
            """SELECT admin_id FROM admin_tokens WHERE token = %s AND is_valid = 1 AND expire_at > NOW()""",
            (token,)
        )
        
        if not admin_result:
            raise HTTPException(status_code=401, detail="无效的token或token已过期")
        
        admin_id = admin_result[0]['admin_id']
        
        # 验证管理员是否存在
        admin_info = execute_query(
            """SELECT admin_id, full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        if not admin_info:
            raise HTTPException(status_code=401, detail="管理员不存在")
        
        return admin_id
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"管理员认证失败: {str(e)}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise HTTPException(status_code=401, detail=f"管理员认证失败: {str(e)}")

# 安全记录管理员操作
def log_admin_operation(admin_id: int, operation_type: str, description: str):
    if not admin_id:
        return
    
    try:
        # 获取管理员姓名
        admin_result = execute_query(
            """SELECT full_name FROM admins WHERE admin_id = %s""",
            (admin_id,)
        )
        
        admin_name = admin_result[0]['full_name'] if admin_result else f"管理员{admin_id}"
        # 在描述前添加管理员姓名
        full_description = f"{admin_name}{description}"
        execute_update(
            """INSERT INTO operation_logs (admin_id, operation_type, operation_desc, created_at) 
               VALUES (%s, %s, %s, NOW())""",
            (admin_id, operation_type, full_description)
        )
    except Exception as e:
        # 记录错误但不中断主要流程
        logger.error(f"记录操作日志失败: {str(e)}")

# 定义文件信息模型
class FileInfo(BaseModel):
    id: str
    fileName: str
    fileType: str
    fileSize: int
    createdTime: str
    lastModified: str
    path: str

# 定义批量删除请求模型
class BatchDeleteRequest(BaseModel):
    file_ids: List[str]

# 获取文件列表
@router.get("/admin/content/knowledge-files", tags=["知识库管理"])
async def get_file_list(
    request: Request,
    search_query: Optional[str] = Query(None, description="搜索关键词"),
    file_type: Optional[str] = Query(None, description="文件类型"),
    start_date: Optional[str] = Query(None, description="开始日期"),
    end_date: Optional[str] = Query(None, description="结束日期"),
    sort_by: Optional[str] = Query("name-asc", description="排序方式"),
    page: int = Query(1, description="页码"),
    page_size: int = Query(20, description="每页数量")
):
    """
    获取知识库文件列表，支持搜索、筛选和排序
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "查询", "查询知识库文件列表")
        
        # 获取所有文件
        file_list = []
        file_paths = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.*"))
        
        # 记录查找到的总文件数
        
        for i, file_path in enumerate(file_paths):
            file_stats = os.stat(file_path)
            file_path_obj = Path(file_path)
            file_name = file_path_obj.name
            file_ext = file_path_obj.suffix
            
            file_info = {
                "id": str(i + 1),
                "fileName": file_name,
                "fileType": file_ext,
                "fileSize": file_stats.st_size,
                "createdTime": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "lastModified": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "path": file_path
            }
            
            # 应用搜索过滤
            if search_query and search_query.lower() not in file_name.lower():
                continue
                
            # 应用文件类型过滤
            if file_type and file_ext != file_type:
                continue
                
            # 应用日期范围过滤
            if start_date or end_date:
                file_date = datetime.fromtimestamp(file_stats.st_mtime)
                
                if start_date:
                    start_datetime = datetime.fromisoformat(start_date)
                    if file_date < start_datetime:
                        continue
                        
                if end_date:
                    end_datetime = datetime.fromisoformat(end_date)
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                    if file_date > end_datetime:
                        continue
            
            file_list.append(file_info)
        
        # 记录过滤后的文件数
        
        # 应用排序
        if sort_by:
            field, direction = sort_by.split('-')
            reverse = direction == 'desc'
            
            if field == 'name':
                file_list.sort(key=lambda x: x["fileName"], reverse=reverse)
            elif field == 'size':
                file_list.sort(key=lambda x: x["fileSize"], reverse=reverse)
            elif field == 'date':
                file_list.sort(key=lambda x: x["lastModified"], reverse=reverse)
        
        # 计算总数
        total_count = len(file_list)
        
        # 确保页码合法
        if page < 1:
            page = 1
        
        # 确保每页数量合法
        if page_size < 1:
            page_size = 20
        elif page_size > 100:
            page_size = 100  # 限制最大每页数量为100
        
        # 分页
        start_idx = (page - 1) * page_size
        
        # 确保起始索引不超出范围
        if start_idx >= total_count:
            if total_count > 0:
                page = (total_count - 1) // page_size + 1
                start_idx = (page - 1) * page_size
            else:
                start_idx = 0
        
        end_idx = min(start_idx + page_size, total_count)
        paginated_files = file_list[start_idx:end_idx]
        
        # 记录分页信息
        
        return {
            "success": True,
            "data": {
                "files": paginated_files,
                "total": total_count,
                "page": page,
                "pageSize": page_size
            }
        }
    except Exception as e:
        logger.error(f"获取文件列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取文件列表失败: {str(e)}")

# 上传文件
@router.post("/admin/content/knowledge-files/upload", tags=["知识库管理"])
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...)
):
    """
    上传文件到知识库
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        uploaded_files = []
        vector_db_updated = False
        vector_db_errors = []
        
        # 初始化向量数据库构建器
        vector_builder = VectorDBBuilder(config)
        
        for file in files:
            # 检查文件类型
            supported_extensions = ['.xlsx', '.xls', '.pdf', '.docx', '.doc']
            file_ext = os.path.splitext(file.filename)[1].lower()
            
            if file_ext not in supported_extensions:
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"不支持的文件类型: {file.filename}, 支持的格式有: Excel文件(.xlsx, .xls), PDF文件(.pdf), Word文档(.docx, .doc)"}
                )
            
            # 检查文件大小限制
            content = await file.read()
            # PDF文件可以更大一些，最大允许20MB
            size_limit = 20 * 1024 * 1024 if file_ext == '.pdf' else 10 * 1024 * 1024
            
            if len(content) > size_limit:
                max_size = "20MB" if file_ext == '.pdf' else "10MB"
                return JSONResponse(
                    status_code=400,
                    content={"success": False, "message": f"文件过大: {file.filename}, 最大允许{max_size}"}
                )
            
            # 保存文件 - 直接使用已读取的内容，不需要重置文件指针
            file_path = os.path.join(KNOWLEDGE_BASE_PATH, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            print(f"文件 {file.filename} 已保存到 {file_path}")
            uploaded_files.append(file.filename)
            
            # 为上传的文件增量更新向量数据库
            try:
                logger.info(f"开始为文件 {file.filename} 更新向量数据库")
                success = vector_builder.process_single_file(file_path)
                if success:
                    vector_db_updated = True
                    logger.info(f"文件 {file.filename} 已成功添加到向量数据库")
                    # 判断文件类型，如果是PDF、Word文件则在生成Excel文件后删除原始文件
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    if file_ext in ['.pdf', '.docx', '.doc']:
                        try:
                            # 删除原始文件
                            os.remove(file_path)
                            logger.info(f"已删除原始{file_ext}文件: {file_path}，仅保留文本分块后的Excel文件")
                        except Exception as del_err:
                            logger.error(f"删除原始{file_ext}文件失败: {str(del_err)}")
                else:
                    vector_db_errors.append(f"无法将文件 {file.filename} 添加到向量数据库")
                    logger.warning(f"无法将文件 {file.filename} 添加到向量数据库")
            except Exception as e:
                error_msg = f"向量数据库更新失败 ({file.filename}): {str(e)}"
                vector_db_errors.append(error_msg)
                logger.error(error_msg)
                # 继续处理其他文件，不让单个文件的失败影响整体上传
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "上传文件", f"上传了{len(uploaded_files)}个文件")
        
        # 根据向量数据库更新状态返回不同的消息
        if vector_db_updated:
            if vector_db_errors:
                # 部分文件更新成功
                return {"success": True, 
                        "message": f"成功上传 {len(uploaded_files)} 个文件，但部分文件未能添加到向量数据库", 
                        "files": uploaded_files,
                        "vector_db_errors": vector_db_errors}
            else:
                # 全部更新成功
                return {"success": True, 
                        "message": f"成功上传 {len(uploaded_files)} 个文件并更新了知识库向量数据库", 
                        "files": uploaded_files}
        else:
            # 上传成功但向量数据库更新失败
            return {"success": True, 
                    "message": f"成功上传 {len(uploaded_files)} 个文件，但未能更新向量数据库", 
                    "files": uploaded_files,
                    "vector_db_errors": vector_db_errors}
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

# 下载文件
@router.get("/admin/content/knowledge-files/download/{file_name}", tags=["知识库管理"])
async def download_file(
    request: Request,
    file_name: str
):
    """
    下载知识库文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "下载文件", f"下载了文件{file_name}")
        
        return FileResponse(
            path=file_path, 
            filename=file_name,
            media_type='application/octet-stream'
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件下载失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件下载失败: {str(e)}")

# 删除单个文件
@router.delete("/admin/content/knowledge-files/{file_name}", tags=["知识库管理"])
async def delete_file(
    request: Request,
    file_name: str
):
    """
    删除知识库文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 记录文件路径，用于向量数据库清理
        file_path_str = str(Path(file_path).resolve())
        
        # 删除文件
        os.remove(file_path)
        
        # 尝试从向量数据库中删除相关数据
        vector_db_updated = False
        vector_db_error = None
        
        try:
            # 创建向量数据库构建器
            vector_builder = VectorDBBuilder(config)
            
            # 检查向量数据库是否存在
            vector_db_path = Path(config.vector_db_path)
            if vector_db_path.exists() and any(vector_db_path.glob("*")):
                # 创建嵌入模型
                embeddings = vector_builder.create_embeddings()
                
                # 加载向量数据库
                vector_store = FAISS.load_local(
                    str(vector_db_path),
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # 查询向量数据库中的所有向量
                all_docs = vector_store.docstore._dict
                
                # 计算要删除的向量ID
                docs_to_delete = []
                deletion_count = 0
                
                for doc_id, doc in all_docs.items():
                    # 检查文档的来源是否匹配要删除的文件
                    doc_source = doc.metadata.get('source', '')
                    if doc_source == file_path_str or file_name in doc_source:
                        docs_to_delete.append(doc_id)
                        deletion_count += 1
                
                # 如果有要删除的向量
                if docs_to_delete:
                    # 删除向量
                    for doc_id in docs_to_delete:
                        del vector_store.docstore._dict[doc_id]
                    
                    # 保存修改后的向量数据库
                    vector_store.save_local(str(vector_db_path))
                    
                    logger.info(f"从向量数据库中删除了 {deletion_count} 个与文件 {file_name} 相关的向量")
                    vector_db_updated = True
                else:
                    logger.info(f"向量数据库中未找到与文件 {file_name} 相关的向量")
            else:
                logger.info("向量数据库不存在，无需清理")
        except Exception as e:
            vector_db_error = str(e)
            logger.error(f"从向量数据库中删除数据失败: {str(e)}")
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "删除", f"删除了文件{file_name}")
        
        # 根据向量数据库更新状态返回不同的消息
        if vector_db_updated:
            return {"success": True, "message": f"成功删除文件 {file_name} 并从向量数据库中移除相关数据"}
        elif vector_db_error:
            return {"success": True, "message": f"成功删除文件 {file_name}，但从向量数据库中移除数据时出错: {vector_db_error}"}
        else:
            return {"success": True, "message": f"成功删除文件 {file_name}，向量数据库无需更新"}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件删除失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件删除失败: {str(e)}")

# 批量删除文件
@router.post("/admin/content/knowledge-files/batch-delete", tags=["知识库管理"])
async def batch_delete_files(
    request: Request,
    batch_request: BatchDeleteRequest
):
    """
    批量删除知识库文件
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        deleted_files = []
        failed_files = []
        file_paths = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.*"))
        deleted_file_paths = []
        
        # 获取所有文件的ID和名称映射
        file_map = {}
        for i, file_path in enumerate(file_paths):
            file_name = os.path.basename(file_path)
            file_map[str(i + 1)] = {"name": file_name, "path": file_path}
        
        # 删除指定ID的文件
        for file_id in batch_request.file_ids:
            if file_id in file_map:
                file_info = file_map[file_id]
                try:
                    # 记录文件路径
                    deleted_file_paths.append(str(Path(file_info["path"]).resolve()))
                    
                    # 删除文件
                    os.remove(file_info["path"])
                    deleted_files.append(file_info["name"])
                except Exception as e:
                    failed_files.append({"name": file_info["name"], "error": str(e)})
        
        # 尝试从向量数据库中删除相关数据
        vector_db_updated = False
        vector_db_error = None
        vector_db_deleted_count = 0
        
        try:
            if deleted_file_paths:
                # 创建向量数据库构建器
                vector_builder = VectorDBBuilder(config)
                
                # 检查向量数据库是否存在
                vector_db_path = Path(config.vector_db_path)
                if vector_db_path.exists() and any(vector_db_path.glob("*")):
                    # 创建嵌入模型
                    embeddings = vector_builder.create_embeddings()
                    
                    # 加载向量数据库
                    vector_store = FAISS.load_local(
                        str(vector_db_path),
                        embeddings,
                        allow_dangerous_deserialization=True
                    )
                    
                    # 查询向量数据库中的所有向量
                    all_docs = vector_store.docstore._dict
                    
                    # 计算要删除的向量ID
                    docs_to_delete = []
                    
                    for doc_id, doc in all_docs.items():
                        # 检查文档的来源是否匹配要删除的文件
                        doc_source = doc.metadata.get('source', '')
                        
                        # 检查文档源是否在已删除文件列表中
                        for deleted_path in deleted_file_paths:
                            if doc_source == deleted_path or any(Path(deleted_path).name in doc_source for deleted_path in deleted_file_paths):
                                docs_to_delete.append(doc_id)
                                vector_db_deleted_count += 1
                                break
                    
                    # 如果有要删除的向量
                    if docs_to_delete:
                        # 删除向量
                        for doc_id in docs_to_delete:
                            del vector_store.docstore._dict[doc_id]
                        
                        # 保存修改后的向量数据库
                        vector_store.save_local(str(vector_db_path))
                        
                        logger.info(f"从向量数据库中删除了 {vector_db_deleted_count} 个向量")
                        vector_db_updated = True
                    else:
                        logger.info("没有找到需要从向量数据库中删除的向量")
                else:
                    logger.info("向量数据库不存在，无需清理")
        except Exception as e:
            vector_db_error = str(e)
            logger.error(f"从向量数据库中删除数据失败: {str(e)}")
        
        # 记录操作日志（不阻止主要功能）
        log_admin_operation(admin_id, "删除", f"批量删除了{len(deleted_files)}个文件")
        
        # 根据向量数据库更新状态返回不同的消息
        base_response = {
            "success": True,
            "deleted_files": deleted_files,
            "failed_files": failed_files
        }
        
        if vector_db_updated:
            base_response["message"] = f"成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个，并从向量数据库中移除了相关数据"
        elif vector_db_error:
            base_response["message"] = f"成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个，但从向量数据库中移除数据时出错: {vector_db_error}"
        else:
            base_response["message"] = f"成功删除 {len(deleted_files)} 个文件，失败 {len(failed_files)} 个，向量数据库无需更新"
            
        return base_response
    except Exception as e:
        logger.error(f"批量删除文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量删除文件失败: {str(e)}")

# 获取文件预览
@router.get("/admin/content/knowledge-files/preview/{file_name}", tags=["知识库管理"])
async def preview_file(
    request: Request,
    file_name: str,
    max_rows: int = Query(5, description="最大预览行数")
):
    """
    获取文件的预览内容
    """
    try:
        # 获取管理员ID
        admin_id = await get_current_admin(request)
        
        file_path = os.path.join(KNOWLEDGE_BASE_PATH, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 检查文件类型
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # 根据文件类型提供不同的预览
        if file_ext in ['.xlsx', '.xls']:
            # Excel文件预览
            try:
                import pandas as pd
            except ImportError:
                return {"success": False, "message": "服务器缺少pandas库，无法预览Excel文件"}
            
            # 读取Excel文件
            try:
                df = pd.read_excel(file_path, nrows=max_rows)
                columns = [{"prop": f"col{i+1}", "label": col} for i, col in enumerate(df.columns)]
                
                # 将DataFrame转换为适合前端显示的格式
                data = []
                for _, row in df.iterrows():
                    item = {}
                    for i, col in enumerate(df.columns):
                        item[f"col{i+1}"] = str(row[col])
                    data.append(item)
                
                # 记录操作日志（不阻止主要功能）
                log_admin_operation(admin_id, "查询", f"预览了文件{file_name}")
                
                return {"success": True, "columns": columns, "data": data}
            except Exception as e:
                return {"success": False, "message": f"无法读取Excel文件: {str(e)}"}
        
        elif file_ext == '.pdf':
            # PDF文件预览（提取前几页文本）
            try:
                # 尝试使用PyPDF2提取文本
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return {"success": False, "message": "服务器缺少PyPDF2库，无法预览PDF文件"}
                
                reader = PdfReader(file_path)
                pages = min(3, len(reader.pages))  # 最多显示前3页
                
                # 提取文本
                text_content = []
                for i in range(pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    if text:
                        # 限制每页的文本长度
                        preview_text = text[:1000] + "..." if len(text) > 1000 else text
                        text_content.append({"page": i+1, "content": preview_text})
                
                # 记录操作日志
                log_admin_operation(admin_id, "查询", f"预览了PDF文件{file_name}")
                
                # 返回PDF预览数据
                return {
                    "success": True, 
                    "file_type": "pdf",
                    "total_pages": len(reader.pages),
                    "preview_pages": pages,
                    "content": text_content
                }
            except Exception as e:
                logger.error(f"PDF预览失败: {str(e)}")
                return {"success": False, "message": f"无法预览PDF文件: {str(e)}"}
                
        elif file_ext in ['.docx', '.doc']:
            # Word文档预览
            try:
                try:
                    import docx2txt
                except ImportError:
                    return {"success": False, "message": "服务器缺少docx2txt库，无法预览Word文档"}
                
                # 提取文本
                text = docx2txt.process(file_path)
                
                # 限制预览长度
                preview_text = text[:2000] + "..." if len(text) > 2000 else text
                
                # 记录操作日志
                log_admin_operation(admin_id, "查询", f"预览了Word文档{file_name}")
                
                # 返回预览数据
                return {
                    "success": True, 
                    "file_type": "docx",
                    "content": preview_text
                }
            except Exception as e:
                logger.error(f"Word文档预览失败: {str(e)}")
                return {"success": False, "message": f"无法预览Word文档: {str(e)}"}
        else:
            # 不支持的文件类型
            return {"success": False, "message": f"不支持预览的文件类型: {file_ext}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文件预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件预览失败: {str(e)}")
