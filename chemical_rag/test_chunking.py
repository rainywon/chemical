import os
from pathlib import Path
from build_vector_store import VectorDBBuilder
from config import Config
from langchain_core.documents import Document

# 示例文本 - 模拟事故报告结构，更清晰地区分一级和二级标题
sample_text = """
事故调查报告

一、概述
这是一个概述性的文本。描述了事故的基本情况。

二、事故经过
事故发生于2023年5月10日。时间：上午10点30分。
地点：化工厂北区储罐区。
当事人：操作员张三、李四。

三、事故原因分析
1. 直接原因
操作失误导致系统崩溃。

2. 间接原因
管理疏忽，培训不足。

3. 根本原因
安全制度不完善，责任落实不到位。

四、事故责任认定及处理建议
1. 操作员张三负有直接责任，建议处罚。
2. 主管李四负有管理责任，警告处分。
3. 安全部门负有监督不力责任，进行整改。

五、事故防范和整改措施
1. 完善安全操作规程，增加以下内容：
   a) 明确操作步骤
   b) 增加安全检查点
   c) 完善应急处置流程

2. 加强员工培训
   a) 每月组织安全培训
   b) 开展应急演练

3. 落实安全责任制
   a) 部门负责人签订责任状
   b) 加强过程监督
"""

# 初始化配置和构建器
config = Config()
builder = VectorDBBuilder(config)

print("开始测试章节分块功能...\n")

# 直接调用_split_by_section方法测试分块
sections = builder._split_by_section(sample_text)

print(f"识别出 {len(sections)} 个章节级别的内容")

# 创建Document对象，与正常处理流程相似
chunks = []
for i, (title, content, section_meta) in enumerate(sections):
    if not content.strip():  # 跳过空章节
        continue
        
    chunks.append(Document(
        page_content=content,
        metadata={
            "section_num": section_meta["section_num"],
            "section_title": section_meta["section_title"],
            "section_level": section_meta["section_level"],
            "chunk_index": i,
            "total_chunks": len(sections),
            "chunk_type": "section",
            **section_meta  # 包含其他元数据
        }
    ))

# 打印分块结果统计
print("\n" + "="*80)
print("文本分块分析结果:")
print("="*80)

# 按级别统计
level_counts = {}
for chunk in chunks:
    level = chunk.metadata.get("section_level", 0)
    contains_subs = chunk.metadata.get("contains_subsections", False)
    
    if level not in level_counts:
        level_counts[level] = {"total": 0, "with_subs": 0}
    
    level_counts[level]["total"] += 1
    if contains_subs:
        level_counts[level]["with_subs"] += 1

# 输出统计结果
print(f"总文本块数: {len(chunks)}")
for level, stats in sorted(level_counts.items()):
    print(f"级别 {level} 的章节: {stats['total']} 块, 其中 {stats['with_subs']} 块包含子章节")

# 详细输出每个块的内容
print("\n" + "="*80)
print("章节分块详情:")
print("="*80)

for i, chunk in enumerate(chunks):
    level = chunk.metadata.get("section_level", 0)
    level_indicator = "【一级】" if level == 1 else "【二级】" if level == 2 else f"【级别{level}】"
    
    print(f"\n----- 块 {i+1}/{len(chunks)} {level_indicator} [章节: {chunk.metadata['section_num']} {chunk.metadata['section_title']}, " + 
          f"字符数: {len(chunk.page_content)}] -----")
    
    # 打印子章节信息
    if chunk.metadata.get("contains_subsections", False):
        print(f"✓ 此块包含 {chunk.metadata.get('subsection_count')} 个子章节")
        # 检查内容中是否包含子章节的编号
        has_sub_content = False
        for j in range(1, 10):  # 检查数字1-9开头的子章节
            if f"\n{j}. " in chunk.page_content or f"\n{j}、" in chunk.page_content:
                has_sub_content = True
                break
        if has_sub_content:
            print("✓ 确认内容中包含子章节内容")
        else:
            print("✗ 警告：未在内容中找到明确的子章节内容")
    
    # 打印内容预览（如果太长则只显示前200字符）
    if len(chunk.page_content) > 500:
        print(f"{chunk.page_content[:500]}...\n(内容过长，省略剩余部分)")
    else:
        print(chunk.page_content)
    print()

# 保存分块到文件以便查看
result_file = Path("chunking_result.txt")
try:
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("章节分块测试结果\n")
        f.write("===============\n\n")
        
        for i, chunk in enumerate(chunks):
            level = chunk.metadata.get("section_level", 0)
            level_indicator = "【一级】" if level == 1 else "【二级】" if level == 2 else f"【级别{level}】"
            
            f.write(f"\n----- 块 {i+1}/{len(chunks)} {level_indicator} [章节: {chunk.metadata['section_num']} {chunk.metadata['section_title']}, " + 
                  f"字符数: {len(chunk.page_content)}] -----\n")
            
            # 记录子章节信息
            if chunk.metadata.get("contains_subsections", False):
                f.write(f"✓ 此块包含 {chunk.metadata.get('subsection_count')} 个子章节\n")
            
            f.write(chunk.page_content)
            f.write("\n\n")
            
    print(f"\n分块结果已保存到 {result_file}")
except Exception as e:
    print(f"保存结果失败: {str(e)}")

print("\n" + "="*80)
print("测试完成")
print("="*80) 