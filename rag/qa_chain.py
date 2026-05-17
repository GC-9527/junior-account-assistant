# -*- coding: utf-8 -*-
"""
初会RAG助手 - 优化版问答链
"""
import time
import dashscope
from config import DASHSCOPE_API_KEY, LLM_MODEL
from rag.retriever import Retriever
from utils.query_rewriter import SmartQueryRewriter

dashscope.api_key = DASHSCOPE_API_KEY


class QAChain:
    def __init__(self, max_retries: int = 3, retry_delay: int = 2):
        self.retriever = Retriever()
        self.query_rewriter = SmartQueryRewriter()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _build_prompt(self, query: str, context_docs: list) -> str:
        """构建QA提示词"""
        context_str = ""
        for i, doc in enumerate(context_docs):
            source_info = doc['metadata'].get('source', '未知来源')
            knowledge_type = doc['metadata'].get('knowledge_type', '')
            chapter = doc['metadata'].get('chapter', '')
            
            context_str += f"【知识块 {i+1}】"
            if chapter:
                context_str += f" 章节: {chapter}"
            if knowledge_type:
                context_str += f" 类型: {knowledge_type}"
            context_str += f" 来源: {source_info}\n"
            context_str += f"{doc['content']}\n\n"

        prompt = f"""你是一个初级会计考试智能辅导助手。请根据以下背景知识回答用户问题。

【背景知识】
{context_str}

【用户问题】
{query}

【回答要求】
1. 必须基于提供的背景知识进行回答，不要编造信息
2. 如果背景知识中没有相关内容，请明确说明"根据当前知识库，无法回答该问题"
3. 回答要准确、简洁，符合初级会计考试的要求
4. 涉及会计分录的问题，请使用标准的借贷记账法格式
5. 涉及计算的问题，请列出详细的计算步骤
6. 在回答末尾注明参考来源

【回答】
"""
        return prompt

    def _call_llm_with_retry(self, prompt: str) -> str:
        """调用LLM生成回答（带重试机制）"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                response = dashscope.Generation.call(
                    model=LLM_MODEL,
                    prompt=prompt,
                    result_format='text',
                    temperature=0.1,
                    max_tokens=2048
                )
                
                if response.status_code == 200:
                    return response.output.text
                else:
                    last_error = f"API错误: {response.message}"
                    
            except Exception as e:
                last_error = str(e)
                print(f"  尝试 {attempt + 1}/{self.max_retries} 失败: {last_error}")
            
            if attempt < self.max_retries - 1:
                print(f"  等待 {self.retry_delay} 秒后重试...")
                time.sleep(self.retry_delay)
        
        raise Exception(f"LLM调用失败，已重试 {self.max_retries} 次: {last_error}")

    def ask(self, query: str, k: int = 5, use_rewrite: bool = True, use_expansion: bool = False, **filters) -> dict:
        """执行RAG问答 - 优化版"""
        rewritten_query = query
        rewrite_info = None
        
        if use_rewrite:
            try:
                rewrite_result = self.query_rewriter.auto_rewrite(query, use_expansion=use_expansion)
                rewritten_query = rewrite_result['rewritten_query']
                rewrite_info = {
                    'original_query': query,
                    'rewritten_query': rewritten_query,
                    'query_type': rewrite_result.get('query_type', ''),
                    'confidence': rewrite_result.get('confidence', 0)
                }
                print(f"原始查询: {query}")
                print(f"改写后查询: {rewritten_query}")
                if rewrite_info['query_type']:
                    print(f"查询类型: {rewrite_info['query_type']}")
                    print(f"置信度: {rewrite_info['confidence']}")
            except Exception as e:
                print(f"查询改写失败，使用原始查询: {e}")
                rewritten_query = query
        
        return self._single_ask(rewritten_query, k, rewrite_info=rewrite_info, **filters)

    def _single_ask(self, query: str, k: int = 5, rewrite_info: dict = None, **filters) -> dict:
        """执行单次问答（带重试机制）"""
        
        docs = None
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if filters:
                    docs = self.retriever.search_with_filter(query, k=k, **filters)
                else:
                    docs = self.retriever.search(query, k=k, use_strict_filter=False)
                break
                
            except Exception as e:
                last_error = str(e)
                print(f"  检索尝试 {attempt + 1}/{self.max_retries} 失败: {last_error}")
                
                if attempt < self.max_retries - 1:
                    print(f"  等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
        
        if docs is None:
            return {
                'query': query,
                'answer': f"检索失败，已重试 {self.max_retries} 次: {last_error}",
                'sources': [],
                'context': [],
                'rewrite_info': rewrite_info,
                'error': last_error
            }
        
        if not docs:
            return {
                'query': query,
                'answer': "根据当前知识库，无法回答该问题",
                'sources': [],
                'context': [],
                'rewrite_info': rewrite_info
            }
        
        prompt = self._build_prompt(query, docs)
        
        try:
            answer = self._call_llm_with_retry(prompt)
        except Exception as e:
            return {
                'query': query,
                'answer': f"生成回答失败: {str(e)}",
                'sources': [],
                'context': docs,
                'rewrite_info': rewrite_info,
                'error': str(e)
            }
        
        sources = []
        for doc in docs:
            metadata = doc.get('metadata', {})
            source = metadata.get('source', '未知来源')
            
            file_name = metadata.get('file_name', '')
            if file_name and file_name != source:
                source_display = f"{source} ({file_name})"
            else:
                source_display = source
            
            source_info = {
                'source': source_display,
                'file_name': file_name,
                'chapter': metadata.get('chapter', ''),
                'knowledge_type': metadata.get('knowledge_type', ''),
                'sub_topic': metadata.get('sub_topic', ''),
                'similarity': f"{(1 - doc['distance']) * 100:.2f}%",
                'score': round((1 - doc['distance']) * 100, 2)
            }
            sources.append(source_info)
        
        result = {
            'query': query,
            'answer': answer,
            'sources': sources,
            'context': docs,
            'rewrite_info': rewrite_info
        }
        
        if rewrite_info:
            result['original_query'] = rewrite_info['original_query']
            result['query_type'] = rewrite_info.get('query_type', '')
            result['confidence'] = rewrite_info.get('confidence', 0)
        
        return result

    def batch_ask(self, queries: list, k: int = 5) -> list:
        """批量问答"""
        results = []
        for query in queries:
            result = self.ask(query, k=k)
            results.append(result)
        return results
