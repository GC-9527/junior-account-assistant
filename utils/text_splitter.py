import re
from typing import List, Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_RULES, KNOWLEDGE_TYPES


class ChuHuiTextSplitter:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.default_chunk_size = chunk_size
        self.default_chunk_overlap = chunk_overlap
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunk_length = 8000

    def set_knowledge_type(self, knowledge_type: str):
        """根据知识类型设置切分参数"""
        rule = CHUNK_RULES.get(knowledge_type, CHUNK_RULES["default"])
        self.chunk_size = rule["chunk_size"]
        self.chunk_overlap = rule["chunk_overlap"]

    def _find_boundary(self, text: str, end_pos: int, direction: str = 'backward') -> int:
        """寻找合适的切分边界"""
        boundaries = ['\n\n', '\n\n\n', '\n', '。', '；', '！', '？', '】', '）', '》', '、', '。\n', '；\n']
        
        if direction == 'backward':
            search_range = range(max(0, end_pos - 60), end_pos)
            for pos in reversed(search_range):
                for boundary in boundaries:
                    if text[pos:pos+len(boundary)] == boundary:
                        return pos + len(boundary)
        else:
            search_range = range(end_pos, min(len(text), end_pos + 60))
            for pos in search_range:
                for boundary in boundaries:
                    if text[pos:pos+len(boundary)] == boundary:
                        return pos + len(boundary)
        
        return end_pos

    def _is_complete_knowledge(self, text: str) -> bool:
        """判断是否为完整知识点"""
        complete_patterns = [
            r'^【.*】.*$',
            r'^\d+\..*$',
            r'^[一二三四五六七八九十]+、.*$',
            r'^[（(].*[)）].*$',
            r'^[甲乙丙丁].*$',
            r'^[①②③④⑤⑥⑦⑧⑨⑩].*$',
            r'^分录[一二三四五六七八九十]+：.*$',
            r'^考点.*：.*$',
            r'^例题.*：.*$'
        ]
        for pattern in complete_patterns:
            if re.match(pattern, text.strip()):
                return True
        return False

    def _detect_knowledge_type(self, text: str) -> str:
        """自动检测知识类型"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["分录", "做账", "记账", "借：", "贷："]):
            return "会计分录"
        elif any(keyword in text_lower for keyword in ["税法", "税率", "纳税", "增值税", "所得税"]):
            return "税法法条"
        elif any(keyword in text_lower for keyword in ["公式", "计算", "计算方法"]):
            return "计算公式"
        elif any(keyword in text_lower for keyword in ["真题", "习题", "答案", "解析"]):
            return "真题习题"
        elif any(keyword in text_lower for keyword in ["辨析", "区别", "不同", "对比"]):
            return "易错辨析"
        else:
            return "概念定义"

    def split_text(self, text: str, knowledge_type: str = None) -> List[str]:
        """按初会专属规则切分文本"""
        if knowledge_type:
            self.set_knowledge_type(knowledge_type)
        
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            max_end = start + min(self.chunk_size, self.max_chunk_length)
            
            if max_end >= text_length:
                chunk = text[start:].strip()
                if chunk and len(chunk) <= self.max_chunk_length:
                    chunks.append(chunk)
                elif chunk:
                    sub_chunks = self._split_large_chunk(chunk)
                    chunks.extend(sub_chunks)
                break

            end = self._find_boundary(text, max_end, 'backward')
            
            if end - start < self.chunk_size // 2:
                end = self._find_boundary(text, max_end, 'forward')

            if end - start > self.max_chunk_length:
                end = start + self.max_chunk_length

            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return self._merge_small_chunks(chunks)

    def _split_large_chunk(self, text: str) -> List[str]:
        """分割超过最大长度限制的chunk"""
        chunks = []
        start = 0
        text_length = len(text)
        chunk_size = self.max_chunk_length - 100
        
        while start < text_length:
            end = start + chunk_size
            
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            end = self._find_boundary(text, end, 'backward')
            
            if end - start < chunk_size // 2:
                end = start + chunk_size
            
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的chunk"""
        merged = []
        current = ""
        
        min_chunk_size = self.chunk_size * 0.4
        
        for chunk in chunks:
            if len(current) + len(chunk) < self.chunk_size * 0.9:
                current += chunk if not current else "\n" + chunk
            else:
                if current and len(current) >= min_chunk_size:
                    merged.append(current)
                elif current:
                    if merged:
                        merged[-1] += "\n" + current
                    else:
                        merged.append(current)
                current = chunk
        
        if current:
            if merged and len(current) < min_chunk_size:
                merged[-1] += "\n" + current
            else:
                merged.append(current)
        
        return merged

    def split_exercise(self, text: str) -> List[str]:
        """切分真题/习题，确保题干+选项+答案+解析完整"""
        exercises = []
        current_exercise = ""
        
        lines = text.split('\n')
        exercise_pattern = r'^\s*\d+[.．、)）]\s*'
        
        for line in lines:
            line = line.strip()
            
            if re.match(exercise_pattern, line) and current_exercise:
                if len(current_exercise.strip()) > 50:
                    exercises.append(current_exercise.strip())
                current_exercise = line
            else:
                current_exercise += ("\n" if current_exercise else "") + line
        
        if current_exercise.strip() and len(current_exercise.strip()) > 50:
            exercises.append(current_exercise.strip())
        
        return exercises

    def split_entry(self, text: str) -> List[str]:
        """切分会计分录，确保一组分录完整"""
        entries = []
        current_entry = ""
        
        lines = text.split('\n')
        entry_start_pattern = r'^(借|贷)\s+[\u4e00-\u9fa5]+'
        
        for line in lines:
            line = line.strip()
            
            if re.match(entry_start_pattern, line):
                if current_entry:
                    entries.append(current_entry.strip())
                    current_entry = line
                else:
                    current_entry = line
            elif line:
                current_entry += ("\n" if current_entry else "") + line
        
        if current_entry.strip():
            entries.append(current_entry.strip())
        
        return entries

    def split_table(self, text: str) -> List[str]:
        """切分表格内容，保持表格完整性"""
        tables = []
        current_table = []
        in_table = False
        
        lines = text.split('\n')
        
        for line in lines:
            if '|' in line and len(line.split('|')) >= 3:
                in_table = True
                current_table.append(line)
            elif in_table and current_table:
                tables.append('\n'.join(current_table))
                current_table = []
                in_table = False
        
        if current_table:
            tables.append('\n'.join(current_table))
        
        if tables:
            return tables
        return [text]

    def split_by_type(self, text: str, knowledge_type: str = "概念定义") -> List[str]:
        """根据知识类型选择切分策略"""
        if knowledge_type in ["真题习题", "练习题", "历年真题", "习题"]:
            return self.split_exercise(text)
        elif knowledge_type in ["会计分录", "分录大全", "分录"]:
            return self.split_entry(text)
        elif '|' in text and text.count('|') >= 3:
            return self.split_table(text)
        else:
            return self.split_text(text, knowledge_type)

    def intelligent_split(self, text: str) -> List[str]:
        """智能切分，自动检测内容类型"""
        detected_type = self._detect_knowledge_type(text)
        return self.split_by_type(text, detected_type)