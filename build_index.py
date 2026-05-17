# -*- coding: utf-8 -*-
"""
初会RAG助手 - 知识库构建（索引入库）

功能：解析文档/图片/视频，生成embedding，保存到Chroma向量数据库
"""
import os
import re
from config import KNOWLEDGE_BASE_DIR, BOOK_TYPES, KNOWLEDGE_TYPES, EXAM_LEVELS, SOURCE_TYPES, DIFFICULTY_LEVELS


def infer_metadata(file_path: str, content: str = "") -> dict:
    """根据文件名、路径和内容推断元数据"""
    metadata = {
        'book_type': "初级会计实务",
        'chapter': "",
        'section': "",
        'sub_topic': "",
        'knowledge_type': "概念定义",
        'exam_level': "了解",
        'difficulty': "中等",
        'source': "其他",
        'file_name': os.path.basename(file_path)
    }
    
    file_name = os.path.basename(file_path).lower()
    dir_name = os.path.basename(os.path.dirname(file_path)).lower()
    
    for book in BOOK_TYPES:
        if book.lower() in file_name or book.lower() in dir_name:
            metadata['book_type'] = book
            break
    
    chapter_match = re.search(r'第[一二三四五六七八九十\d]+章[^第]*', file_name)
    if chapter_match:
        chapter_str = chapter_match.group()
        section_match = re.search(r'第[一二三四五六七八九十\d]+节', chapter_str)
        if section_match:
            metadata['chapter'] = chapter_str.replace(section_match.group(), "").strip()
            metadata['section'] = section_match.group()
        else:
            metadata['chapter'] = chapter_str.strip()
    
    if not metadata['chapter']:
        chapter_match = re.search(r'第一章|第二章|第三章|第四章|第五章|第六章|第七章|第八章|第九章|第十章', file_name)
        if chapter_match:
            metadata['chapter'] = chapter_match.group()
    
    if content:
        content_chapter = re.search(r'第[一二三四五六七八九十\d]+章\s+[\u4e00-\u9fa5]+', content[:500])
        if content_chapter and not metadata['chapter']:
            metadata['chapter'] = content_chapter.group()
    
    for source in SOURCE_TYPES:
        if source.lower() in file_name or source.lower() in dir_name:
            metadata['source'] = source
            break
    
    if "2025" in file_name:
        metadata['source'] = "2025官方教材"
    elif "2026" in file_name:
        metadata['source'] = "2026教材"
    elif "三色" in file_name:
        metadata['source'] = "三色笔记"
    elif "真题" in file_name:
        metadata['source'] = "历年真题"
    
    # 基于文件名的判断（降低优先级，主要基于内容判断）
    file_knowledge_type = None
    if any(keyword in file_name for keyword in ["分录大全", "会计分录"]):
        file_knowledge_type = "会计分录"
    elif any(keyword in file_name for keyword in ["税法", "经济法"]):
        file_knowledge_type = "税法法条"
    elif any(keyword in file_name for keyword in ["真题", "习题", "练习", "母题", "550题", "600题"]):
        file_knowledge_type = "真题习题"
    elif any(keyword in file_name for keyword in ["辨析", "对比"]):
        file_knowledge_type = "易错辨析"
    
    if file_knowledge_type:
        metadata['knowledge_type'] = file_knowledge_type
    else:
        # 默认都是概念定义
        metadata['knowledge_type'] = "概念定义"
        metadata['difficulty'] = "简单"
    
    if content:
        # 更严格的判断逻辑，避免误判
        content_sample = content[:2000]
        
        # 会计分录 - 需要有明确的借贷格式
        if any(keyword in content_sample for keyword in ["借：", "贷："]) and ("会计分录" in content_sample or len(re.findall(r"借：", content_sample)) >= 2):
            metadata['knowledge_type'] = "会计分录"
        # 税法法条 - 需要有明确的税法关键词
        elif any(keyword in content_sample for keyword in ["税率", "纳税义务", "增值税", "所得税", "消费税", "印花税", "关税"]):
            metadata['knowledge_type'] = "税法法条"
        # 计算公式 - 需要有明确的公式标记
        elif any(keyword in content_sample for keyword in ["计算公式", "公式：", "计算如下", "×", "÷"]) and "=" in content_sample:
            metadata['knowledge_type'] = "计算公式"
        # 真题习题 - 需要有明确的题目/答案结构
        elif any(keyword in content_sample for keyword in ["答案", "解析", "正确选项", "单选题", "多选题", "判断题"]):
            metadata['knowledge_type'] = "真题习题"
        # 易错辨析 - 需要有明确的对比关键词
        elif any(keyword in content_sample for keyword in ["区别", "对比", "辨析", "不同", "vs", "VS"]):
            metadata['knowledge_type'] = "易错辨析"
    
    if any(keyword in file_name for keyword in ["必考", "重点", "核心"]):
        metadata['exam_level'] = "必考"
    elif any(keyword in file_name for keyword in ["高频", "常考", "常见"]):
        metadata['exam_level'] = "高频"
    elif any(keyword in file_name for keyword in ["了解", "了解即可"]):
        metadata['exam_level'] = "了解"
    
    if content:
        if "必考" in content[:500]:
            metadata['exam_level'] = "必考"
        elif "高频考点" in content[:500]:
            metadata['exam_level'] = "高频"
    
    return metadata


def extract_sub_topic(content: str) -> str:
    """从内容中提取子知识点"""
    patterns = [
        r'【(.+?)】',
        r'(\d+\.\d+\s*.+)',
        r'([一二三四五六七八九十]+\、.+)',
        r'^(.+?[:：])'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content[:300])
        if match:
            result = match.group(1).strip()
            if len(result) > 2 and len(result) < 50:
                return result
    return ""


def process_document(file_path: str, chroma_manager, splitter):
    """处理单个文档文件"""
    from utils.document_parser import DocumentParser
    
    print(f"\n处理文档: {file_path}")
    
    try:
        text, extra_info = DocumentParser.parse_file(file_path)
        metadata = infer_metadata(file_path, text)
        
        source_type = metadata.get('knowledge_type', '')
        chunks = splitter.split_by_type(text, source_type)
        
        filtered_chunks = []
        for chunk in chunks:
            if len(chunk) > 0:
                # 确保chunk不超过最大长度
                if len(chunk) > 8192:
                    # 对超长的chunk进行二次切分
                    sub_chunks = splitter._split_large_chunk(chunk)
                    filtered_chunks.extend(sub_chunks)
                else:
                    filtered_chunks.append(chunk)
        
        print(f"  文档长度: {len(text)} 字符, 切分为 {len(chunks)} 个chunk, 过滤后 {len(filtered_chunks)} 个")
        print(f"  推断类型: {metadata['knowledge_type']}, 章节: {metadata['chapter']}")
        
        metadatas = []
        for i, chunk in enumerate(filtered_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['type'] = 'text'
            chunk_metadata['page_number'] = extra_info.get('page_numbers', [1])[0] if extra_info else 1
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(filtered_chunks)
            chunk_metadata['sub_topic'] = extract_sub_topic(chunk)
            chunk_metadata['content_length'] = len(chunk)
            metadatas.append(chunk_metadata)
        
        if filtered_chunks:
            ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(filtered_chunks))]
            chroma_manager.add_documents(filtered_chunks, metadatas, ids)
        
        return len(filtered_chunks)
    except Exception as e:
        print(f"  处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


def process_images(chroma_manager):
    """处理图片文件"""
    image_dir = os.path.join(KNOWLEDGE_BASE_DIR, "images")
    if not os.path.exists(image_dir):
        return 0
    
    count = 0
    print("\n处理图片...")
    for img_filename in os.listdir(image_dir):
        if img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(image_dir, img_filename)
            print(f"  - {img_filename}")
            
            metadata = {
                'type': 'image',
                'book_type': "初级会计实务",
                'knowledge_type': "概念定义",
                'exam_level': "了解",
                'difficulty': "简单",
                'source': "图片资源",
                'file_name': img_filename
            }
            
            chroma_manager.add_image(img_path, metadata)
            count += 1
    
    return count


def build_and_save():
    """构建知识库并保存"""
    print("=== 构建初会RAG知识库 ===")
    
    from rag.chroma_manager import ChromaManager
    from utils.text_splitter import ChuHuiTextSplitter
    
    chroma_manager = ChromaManager()
    splitter = ChuHuiTextSplitter()
    
    text_count = 0
    image_count = 0
    
    for subdir in ["books", "notes", "exercises"]:
        dir_path = os.path.join(KNOWLEDGE_BASE_DIR, subdir)
        if not os.path.exists(dir_path):
            continue
        
        print(f"\n遍历目录: {subdir}")
        for filename in os.listdir(dir_path):
            if filename.startswith('.'):
                continue
            
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in ['.pdf', '.docx', '.txt', '.md']:
                    text_count += process_document(file_path, chroma_manager, splitter)
    
    image_count = process_images(chroma_manager)
    
    total = chroma_manager.get_collection_stats()
    print(f"\n=== 构建完成 ===")
    print(f"文本块: {text_count}")
    print(f"图片: {image_count}")
    print(f"知识库总条目: {total}")


if __name__ == "__main__":
    build_and_save()