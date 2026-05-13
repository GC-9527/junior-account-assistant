
# -*- coding: utf-8 -*-
"""
初会RAG助手 - 简化版Query改写模块
专注于真正能提高检索匹配度的功能
"""
import dashscope
from config import DASHSCOPE_API_KEY

dashscope.api_key = DASHSCOPE_API_KEY


class SimpleQueryRewriter:
    """简化版查询改写器 - 只做必要的优化"""
    
    def __init__(self):
        # 初会领域术语映射表（用户表达方式 → 知识库常用表达）
        self.term_mapping = {
            "怎么做账": "账务处理",
            "做账": "账务处理",
            "记账": "账务处理",
            "进项税": "进项税额",
            "销项税": "销项税额",
            "不能抵扣": "不得抵扣",
            "折旧": "固定资产折旧",
            "摊销": "无形资产摊销",
            "税法": "增值税法律制度 消费税法律制度 企业所得税法律制度 个人所得税法律制度",
            "经济法": "会计法律制度 支付结算法律制度 增值税法律制度 消费税法律制度",
            "税率是多少": "税率",
            "怎么算": "计算公式 计算方法",
            "有哪些": "包括哪些 情形有哪些",
            "区别": "区别 不同之处 对比",
            "不一样": "区别 不同之处",
        }
    
    def rewrite(self, query):
        """
        简化版查询改写
        1. 替换术语映射
        2. 轻微调整，不过度复杂化
        """
        rewritten_query = query
        
        # 应用术语映射
        for user_term, knowledge_term in self.term_mapping.items():
            if user_term in rewritten_query:
                rewritten_query = rewritten_query.replace(user_term, knowledge_term)
        
        return rewritten_query


def get_completion(prompt, model="qwen-turbo-latest"):
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0.3,
    )
    return response.output.choices[0].message.content


class SmartQueryRewriter:
    """智能版查询改写器 - 基于知识库感知"""
    
    def __init__(self):
        self.simple_rewriter = SimpleQueryRewriter()
    
    def rewrite_for_retrieval(self, query):
        """
        专门为检索优化的改写
        目标：提高与知识库内容的匹配度
        """
        instruction = """
你是一个初级会计考试知识库的查询优化助手。
你的任务是将用户的问题改写成更有可能在知识库中找到匹配内容的表达方式。

注意事项：
1. 保持问题的核心意图不变
2. 使用初级会计考试的标准专业术语
3. 不要过度复杂化，不要拆分成多个问题
4. 可以适当扩展，但不要改变原意
5. 如果是简单问题，保持原样即可

常见术语对照：
- "怎么做账" → "账务处理"
- "进项税" → "进项税额"
- "不能抵扣" → "不得抵扣"
- "有什么不同" → "区别"
- "怎么算" → "计算方法"或"计算公式"

请直接返回改写后的查询，不要添加其他内容。
"""
        
        prompt = f"""
{instruction}

用户原始问题：
{query}

改写后的查询：
"""
        
        try:
            llm_rewritten = get_completion(prompt)
            # 同时应用简单术语映射
            final_rewritten = self.simple_rewriter.rewrite(llm_rewritten)
            return final_rewritten
        except Exception as e:
            print(f"智能改写失败，使用简单改写: {e}")
            return self.simple_rewriter.rewrite(query)
    
    def expand_query(self, query):
        """
        查询扩展 - 添加相关关键词
        增加命中概率
        """
        instruction = """
你是一个初级会计考试知识库的查询扩展助手。
请为用户的查询添加2-5个相关的关键词或短语，用空格分隔。
这些关键词应该是初级会计考试中常见的、可能在知识库中出现的术语。

例如：
- 用户查询："固定资产折旧"
- 扩展后："固定资产折旧 折旧方法 折旧计算 累计折旧"

请直接返回扩展后的查询，不要添加其他内容。
"""
        
        prompt = f"""
{instruction}

用户原始查询：
{query}

扩展后的查询：
"""
        
        try:
            expanded = get_completion(prompt)
            return f"{query} {expanded}"
        except Exception as e:
            print(f"查询扩展失败: {e}")
            return query
    
    def auto_rewrite(self, query, use_expansion=False):
        """
        自动改写入口
        """
        # 先做基础改写
        rewritten = self.rewrite_for_retrieval(query)
        
        # 可选：查询扩展
        if use_expansion:
            rewritten = self.expand_query(rewritten)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten
        }

