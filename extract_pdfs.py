# -*- coding: utf-8 -*-
"""
PDF文本提取脚本 - 从PDF中提取文本并进行基础清洗
"""
import os
import re
from PyPDF2 import PdfReader


def clean_text(text):
    """清洗文本"""
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\r+', '\n', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 1:
            cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


def extract_pdf_text(pdf_path):
    """从PDF中提取文本"""
    try:
        reader = PdfReader(pdf_path)
        all_text = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                page_text = f"=== 第{page_num}页 ===\n{text}"
                all_text.append(page_text)

        full_text = '\n'.join(all_text)
        cleaned_text = clean_text(full_text)

        return cleaned_text
    except Exception as e:
        print(f"  提取失败: {str(e)}")
        return None


def get_pdf_info(pdf_path):
    """获取PDF基本信息"""
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        return num_pages
    except:
        return 0


def process_pdfs_in_directory(source_dir, output_base_dir):
    """处理目录中的所有PDF文件"""
    files_info = []

    if not os.path.exists(source_dir):
        print(f"目录不存在: {source_dir}")
        return files_info

    for filename in os.listdir(source_dir):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(source_dir, filename)
            print(f"\n处理: {filename}")

            num_pages = get_pdf_info(pdf_path)
            print(f"  页数: {num_pages}")

            text = extract_pdf_text(pdf_path)
            if text:
                text_length = len(text)
                print(f"  提取文本长度: {text_length} 字符")

                file_info = {
                    'original_filename': filename,
                    'pdf_path': pdf_path,
                    'text': text,
                    'num_pages': num_pages,
                    'text_length': text_length
                }
                files_info.append(file_info)
            else:
                print(f"  未能提取文本")

    return files_info


def save_extracted_text(file_info, output_dir):
    """保存提取的文本到文件"""
    original_name = file_info['original_filename']
    base_name = os.path.splitext(original_name)[0]

    safe_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
    txt_filename = f"{safe_name}.txt"
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(file_info['text'])

    print(f"  已保存: {txt_filename} ({len(file_info['text'])} 字符)")
    return txt_path


def infer_category(filename):
    """根据文件名推断分类"""
    filename_lower = filename.lower()

    if '经济法' in filename_lower:
        return '经济法'
    elif '实务' in filename_lower:
        return '实务'
    elif '分录' in filename_lower:
        return '分录'
    elif '真题' in filename_lower or '母题' in filename_lower:
        return '真题'
    elif '三色' in filename_lower or '笔记' in filename_lower:
        return '笔记'
    elif '大纲' in filename_lower:
        return '大纲'
    elif '备考' in filename_lower or '指南' in filename_lower:
        return '备考'
    else:
        return '其他'


def get_output_dir(category, base_output_dir):
    """获取输出目录"""
    category_dirs = {
        '实务': os.path.join(base_output_dir, 'books'),
        '经济法': os.path.join(base_output_dir, 'books'),
        '分录': os.path.join(base_output_dir, 'notes'),
        '真题': os.path.join(base_output_dir, 'exercises'),
        '笔记': os.path.join(base_output_dir, 'notes'),
        '大纲': os.path.join(base_output_dir, 'books'),
        '备考': os.path.join(base_output_dir, 'notes'),
        '其他': os.path.join(base_output_dir, 'notes')
    }

    output_dir = category_dirs.get(category, os.path.join(base_output_dir, 'notes'))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    """主函数"""
    source_dir = r"c:\Users\初号机驾驶员\Desktop\AI大模型应用第22期\初会资料"
    base_output_dir = r"c:\Users\初号机驾驶员\Desktop\AI大模型应用第22期\初会RAG助手\knowledge_base"

    print("="*60)
    print("PDF文本提取工具")
    print("="*60)
    print(f"\n源目录: {source_dir}")
    print(f"输出目录: {base_output_dir}")

    files_info = process_pdfs_in_directory(source_dir, base_output_dir)

    if not files_info:
        print("\n未找到PDF文件或提取失败")
        return

    print(f"\n\n成功处理 {len(files_info)} 个PDF文件")
    print("="*60)

    saved_count = 0
    for file_info in files_info:
        filename = file_info['original_filename']
        category = infer_category(filename)
        output_dir = get_output_dir(category, base_output_dir)

        print(f"\n[{category}] {filename}")
        txt_path = save_extracted_text(file_info, output_dir)
        saved_count += 1

    print(f"\n\n处理完成! 共保存 {saved_count} 个文本文件")


if __name__ == "__main__":
    main()