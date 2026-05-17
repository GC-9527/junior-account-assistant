import os
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from typing import Tuple, List, Dict


class DocumentParser:
    @staticmethod
    def extract_text_with_page_numbers(pdf_path: str) -> Tuple[str, List[int]]:
        """从PDF中提取文本并记录每行文本对应的页码"""
        text = ""
        page_numbers = []
        pdf = PdfReader(pdf_path)

        for page_number, page in enumerate(pdf.pages, start=1):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
                page_numbers.extend([page_number] * len(extracted_text.split("\n")))

        return text, page_numbers

    @staticmethod
    def parse_pdf(pdf_path: str) -> Tuple[str, Dict]:
        """解析PDF文件"""
        text, page_numbers = DocumentParser.extract_text_with_page_numbers(pdf_path)
        return text, {"page_numbers": page_numbers}

    @staticmethod
    def parse_docx(docx_path: str) -> str:
        """解析DOCX文件，提取全部文本"""
        doc = DocxDocument(docx_path)
        all_text = []

        for element in doc.element.body:
            if element.tag.endswith('p'):
                paragraph_text = ""
                for run in element.findall('.//w:t', {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}):
                    paragraph_text += run.text if run.text else ""
                if paragraph_text.strip():
                    all_text.append(paragraph_text.strip())

            elif element.tag.endswith('tbl'):
                table = [t for t in doc.tables if t._element is element][0]
                if table.rows:
                    md_table = []
                    header = [cell.text.strip() for cell in table.rows[0].cells]
                    md_table.append("| " + " | ".join(header) + " |")
                    md_table.append("|" + "---|" * len(header))
                    for row in table.rows[1:]:
                        row_data = [cell.text.strip() for cell in row.cells]
                        md_table.append("| " + " | ".join(row_data) + " |")
                    all_text.append("\n".join(md_table))

        return "\n".join(all_text)

    @staticmethod
    def parse_txt(txt_path: str) -> str:
        """解析TXT文件"""
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    @staticmethod
    def parse_file(file_path: str) -> Tuple[str, Dict]:
        """根据文件类型选择相应的解析方法"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.pdf':
            return DocumentParser.parse_pdf(file_path)
        elif ext == '.docx':
            return DocumentParser.parse_docx(file_path), {}
        elif ext == '.txt' or ext == '.md':  # 支持 Markdown
            return DocumentParser.parse_txt(file_path), {}
        else:
            raise ValueError(f"不支持的文件类型: {ext}")