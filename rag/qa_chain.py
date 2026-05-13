
import dashscope
from config import DASHSCOPE_API_KEY, LLM_MODEL
from rag.retriever import Retriever
from utils.query_rewriter_v2 import SmartQueryRewriter

dashscope.api_key = DASHSCOPE_API_KEY


class QAChain:
    def __init__(self):
        self.retriever = Retriever()
        self.query_rewriter = SmartQueryRewriter()

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

    def _call_llm(self, prompt: str) -> str:
        """调用LLM生成回答"""
        response = dashscope.Generation.call(
            model=LLM_MODEL,
            prompt=prompt,
            result_format='text',
            temperature=0.1,
            max_tokens=2048
        )
        return response.output.text

    def ask(self, query: str, k: int = 5, use_rewrite: bool = True, use_expansion: bool = False, **filters) -> dict:
        """执行RAG问答 - 简化版"""
        rewritten_query = query
        
        if use_rewrite:
            rewrite_result = self.query_rewriter.auto_rewrite(query, use_expansion=use_expansion)
            rewritten_query = rewrite_result['rewritten_query']
            print(f"原始查询: {query}")
            print(f"改写后查询: {rewritten_query}")
        
        return self._single_ask(rewritten_query, k, **filters)

    def _single_ask(self, query: str, k: int = 5, **filters) -> dict:
        """执行单次问答"""
        if filters:
            docs = self.retriever.search_with_filter(query, k=k, **filters)
        else:
            docs = self.retriever.search(query, k=k, use_strict_filter=False)
        
        if not docs:
            return {
                'query': query,
                'answer': "根据当前知识库，无法回答该问题",
                'sources': [],
                'context': []
            }
        
        prompt = self._build_prompt(query, docs)
        answer = self._call_llm(prompt)
        
        sources = []
        for doc in docs:
            source_info = {
                'source': doc['metadata'].get('source', '未知来源'),
                'chapter': doc['metadata'].get('chapter', ''),
                'knowledge_type': doc['metadata'].get('knowledge_type', ''),
                'similarity': f"{(1 - doc['distance']) * 100:.2f}%"
            }
            sources.append(source_info)
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'context': docs
        }

    def batch_ask(self, queries: list, k: int = 5) -> list:
        """批量问答"""
        results = []
        for query in queries:
            result = self.ask(query, k=k)
            results.append(result)
        return results

