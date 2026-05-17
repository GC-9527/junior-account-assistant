# -*- coding: utf-8 -*-
"""
增强版查询改写模块
集成Tavily Search和优化的意图识别
"""
import json
import re
import os
import dashscope
from typing import List, Dict, Optional, Tuple
from http import HTTPStatus
from config import DASHSCOPE_API_KEY

dashscope.api_key = DASHSCOPE_API_KEY

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    print("警告: Tavily SDK未安装，Web搜索功能不可用")


class IntentClassifier:
    """增强的意图分类器"""
    
    INTENT_PATTERNS = {
        'definition': {
            'keywords': ['什么是', '什么叫做', '定义是', '概念是', '含义是', '指的是', '什么叫'],
            'weight': 0.9
        },
        'calculation': {
            'keywords': ['计算', '怎么算', '如何算', '多少', '公式是', '怎么计算', '如何计算'],
            'weight': 0.85
        },
        'comparison': {
            'keywords': ['区别', '不同', '差异', '比较', '哪个更好', '哪个是', '相比', '对比', '有什么不同', '有什么差异'],
            'weight': 0.8
        },
        'procedure': {
            'keywords': ['如何做', '怎么做', '步骤是', '流程是', '方法', '如何处理', '怎么处理', '怎么写', '如何填写'],
            'weight': 0.85
        },
        'explanation': {
            'keywords': ['为什么', '原因是', '解释', '说明', '原因是什么', '为什么是', '为什么要'],
            'weight': 0.75
        },
        'example': {
            'keywords': ['例子', '举例', '例如', '案例', '示例', '举个例子'],
            'weight': 0.7
        },
        'tax_law': {
            'keywords': ['税法', '税率', '抵扣', '免税', '纳税', '增值税', '所得税', '营业税'],
            'weight': 0.9
        },
        'accounting_entry': {
            'keywords': ['分录', '借贷', '记账', '科目', '借方', '贷方', '账户'],
            'weight': 0.9
        },
        'exam_related': {
            'keywords': ['考试', '及格', '分数线', '题型', '真题', '考点', '考试重点'],
            'weight': 0.85
        },
        'web_search': {
            'keywords': ['最新', '2025', '2026', '今年', '现在', '最新政策', '最近'],
            'weight': 0.95
        }
    }
    
    @classmethod
    def classify(cls, query: str) -> Tuple[str, float, List[str]]:
        """
        分类查询类型
        
        Returns:
            (主类型, 置信度, 所有匹配的类型列表)
        """
        query_lower = query.lower()
        matched_types = []
        scores = {}
        
        for intent_type, config in cls.INTENT_PATTERNS.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in query_lower:
                    score += config['weight']
                    matched_types.append(intent_type)
            
            if score > 0:
                scores[intent_type] = score
        
        if not scores:
            return 'general', 0.5, ['general']
        
        # 返回得分最高的类型
        main_type = max(scores, key=scores.get)
        max_score = scores[main_type]
        confidence = min(max_score / 1.0, 1.0)
        
        return main_type, confidence, matched_types


class TavilySearcher:
    """Tavily搜索引擎集成"""
    
    def __init__(self, api_key: str = None):
        """
        初始化Tavily搜索客户端
        
        Args:
            api_key: Tavily API密钥
        """
        self.api_key = api_key or TAVILY_API_KEY
        self.client = None
        
        if TAVILY_AVAILABLE and self.api_key:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception as e:
                print(f"Tavily客户端初始化失败: {e}")
    
    def search(self, query: str, max_results: int = 5, include_answer: bool = True) -> Optional[Dict]:
        """
        执行Tavily搜索
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            include_answer: 是否包含AI生成的答案
            
        Returns:
            搜索结果字典
        """
        if not self.client:
            return None
        
        try:
            results = self.client.search(
                query=query,
                max_results=max_results,
                include_answer=include_answer,
                include_raw_content=False,
                search_depth="advanced"
            )
            return results
        except Exception as e:
            print(f"Tavily搜索失败: {e}")
            return None
    
    def is_available(self) -> bool:
        """检查Tavily是否可用"""
        return self.client is not None


class QueryExpander:
    """查询扩展器 - 使用大模型生成相关查询"""
    
    def __init__(self, model: str = "qwen-turbo-latest"):
        self.model = model
    
    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """
        扩展查询，生成多个变体
        
        Args:
            query: 原始查询
            num_variations: 生成的变体数量
            
        Returns:
            查询变体列表
        """
        prompt = f"""
你是一个查询优化专家。请为以下查询生成{num_variations}个不同的变体，这些变体应该：
1. 使用不同的表达方式
2. 涵盖不同的角度或方面
3. 保持查询的原始意图

原始查询: {query}

请以JSON数组格式输出，例如：["变体1", "变体2", "变体3"]
"""
        
        try:
            response = self._call_llm(prompt)
            expansions = json.loads(response)
            if isinstance(expansions, list):
                return expansions[:num_variations]
        except Exception as e:
            print(f"查询扩展失败: {e}")
        
        return [query]
    
    def generate_related_queries(self, query: str) -> List[str]:
        """
        生成相关查询
        
        Args:
            query: 原始查询
            
        Returns:
            相关查询列表
        """
        prompt = f"""
基于以下查询，生成5个相关的子查询或补充查询：

原始查询: {query}

这些查询应该：
1. 涵盖查询的不同方面
2. 包括相关的概念和术语
3. 提出具体的子问题

请以JSON数组格式输出。
"""
        
        try:
            response = self._call_llm(prompt)
            queries = json.loads(response)
            if isinstance(queries, list):
                return queries[:5]
        except Exception as e:
            print(f"相关查询生成失败: {e}")
        
        return [query]
    
    def _call_llm(self, prompt: str) -> str:
        """调用大模型"""
        messages = [{"role": "user", "content": prompt}]
        response = dashscope.Generation.call(
            model=self.model,
            messages=messages,
            result_format='message',
            temperature=0.7,
        )
        
        if response.status_code == HTTPStatus.OK:
            return response.output.choices[0].message.content
        else:
            raise Exception(f"LLM调用失败: {response.message}")


class EnhancedQueryRewriter:
    """增强版查询改写器"""
    
    def __init__(self, use_tavily: bool = True, use_expansion: bool = True):
        """
        初始化增强版查询改写器
        
        Args:
            use_tavily: 是否使用Tavily搜索
            use_expansion: 是否使用查询扩展
        """
        self.classifier = IntentClassifier()
        self.expander = QueryExpander()
        self.use_tavily = use_tavily and TAVILY_AVAILABLE
        self.use_expansion = use_expansion
        
        if self.use_tavily:
            self.tavily = TavilySearcher()
        else:
            self.tavily = None
    
    def rewrite(self, query: str, context: str = "") -> Dict:
        """
        改写查询
        
        Args:
            query: 原始查询
            context: 对话上下文
            
        Returns:
            包含改写结果和元数据的字典
        """
        # 1. 意图分类
        intent, confidence, matched_intents = self.classifier.classify(query)
        
        # 2. 使用大模型进行智能改写
        rewritten_query = self._llm_rewrite(query, intent, context)
        
        # 3. 生成查询变体（如果启用）
        variations = []
        if self.use_expansion:
            variations = self.expander.expand(query, num_variations=2)
        
        # 4. 准备结果
        result = {
            'original_query': query,
            'rewritten_query': rewritten_query,
            'intent': intent,
            'confidence': confidence,
            'matched_intents': matched_intents,
            'variations': variations,
            'use_tavily': False,
            'tavily_results': None
        }
        
        # 5. 如果需要web搜索
        if 'web_search' in matched_intents or confidence > 0.9:
            if self.use_tavily:
                tavily_results = self._tavily_search(rewritten_query)
                result['use_tavily'] = True
                result['tavily_results'] = tavily_results
        
        return result
    
    def _llm_rewrite(self, query: str, intent: str, context: str) -> str:
        """使用大模型改写查询"""
        
        intent_prompts = {
            'definition': "将查询改写成一个明确的定义性问题，例如：'请解释X的概念'",
            'calculation': "将查询改写成一个具体的计算问题，例如：'X的计算方法是...'",
            'comparison': "将查询改写成一个清晰的对比性问题，例如：'请比较X和Y的区别'",
            'procedure': "将查询改写成一个步骤性问题，例如：'X的处理步骤是...'",
            'explanation': "将查询改写成一个解释性问题，例如：'请解释为什么X...'",
            'example': "将查询改写成一个示例请求，例如：'请举例说明X...'",
            'tax_law': "将查询改写成一个税法相关问题，确保包含具体的法规条款",
            'accounting_entry': "将查询改写成一个会计分录问题，例如：'X业务的会计分录是...'",
            'exam_related': "将查询改写成一个考试相关问题，确保包含考试要点",
            'web_search': "将查询改写成一个适合网络搜索的问题",
            'general': "将查询改写成一个清晰、直接的问题"
        }
        
        instruction = intent_prompts.get(intent, intent_prompts['general'])
        
        prompt = f"""
你是一个查询优化专家。请根据以下指令改写用户的查询。

### 改写要求 ###
{instruction}

### 原始查询 ###
{query}

### 对话上下文 ###
{context if context else "无"}

### 改写后的查询 ###
请只输出改写后的查询，不要其他内容。
"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = dashscope.Generation.call(
                model="qwen-turbo-latest",
                messages=messages,
                result_format='message',
                temperature=0,
            )
            
            if response.status_code == HTTPStatus.OK:
                rewritten = response.output.choices[0].message.content.strip()
                return rewritten if rewritten else query
        except Exception as e:
            print(f"LLM改写失败: {e}")
        
        return query
    
    def _tavily_search(self, query: str) -> Optional[Dict]:
        """执行Tavily搜索"""
        if not self.tavily or not self.tavily.is_available():
            return None
        
        try:
            results = self.tavily.search(query, max_results=5, include_answer=True)
            return results
        except Exception as e:
            print(f"Tavily搜索失败: {e}")
            return None
    
    def batch_rewrite(self, queries: List[str]) -> List[Dict]:
        """
        批量改写查询
        
        Args:
            queries: 查询列表
            
        Returns:
            改写结果列表
        """
        return [self.rewrite(q) for q in queries]


class SmartQueryRewriter(EnhancedQueryRewriter):
    """智能查询改写器 - EnhancedQueryRewriter的别名"""
    
    def auto_rewrite(self, query: str, use_expansion: bool = False) -> Dict:
        """
        自动查询改写
        
        Args:
            query: 原始查询
            use_expansion: 是否使用查询扩展
            
        Returns:
            包含改写结果的字典
        """
        result = self.rewrite(query, "")
        
        if use_expansion and result.get('variations'):
            return {
                'rewritten_query': result['rewritten_query'],
                'query_type': result['intent'],
                'confidence': result['confidence'],
                'variations': result['variations']
            }
        
        return {
            'rewritten_query': result['rewritten_query'],
            'query_type': result['intent'],
            'confidence': result['confidence']
        }


def get_completion(prompt, model="qwen-turbo-latest"):
    """兼容旧版本的函数"""
    messages = [{"role": "user", "content": prompt}]
    response = dashscope.Generation.call(
        model=model,
        messages=messages,
        result_format='message',
        temperature=0,
    )
    return response.output.choices[0].message.content


class QueryRewriter:
    """兼容旧版本的查询改写器"""
    
    def __init__(self, model="qwen-turbo-latest"):
        self.model = model
        self.enhanced_rewriter = EnhancedQueryRewriter()
    
    def rewrite_context_dependent_query(self, current_query, conversation_history):
        return self.enhanced_rewriter._llm_rewrite(current_query, 'general', conversation_history)
    
    def rewrite_comparative_query(self, query, context_info):
        return self.enhanced_rewriter._llm_rewrite(query, 'comparison', context_info)
    
    def rewrite_ambiguous_reference_query(self, current_query, conversation_history):
        return self.enhanced_rewriter._llm_rewrite(current_query, 'general', conversation_history)
    
    def rewrite_multi_intent_query(self, query):
        variations = self.enhanced_rewriter.expander.expand(query, num_variations=3)
        return variations
    
    def rewrite_rhetorical_query(self, current_query, conversation_history):
        return self.enhanced_rewriter._llm_rewrite(current_query, 'explanation', conversation_history)
    
    def auto_rewrite_query(self, query, conversation_history="", context_info=""):
        result = self.enhanced_rewriter.rewrite(query, context_info)
        return {
            "query_type": result['intent'],
            "rewritten_query": result['rewritten_query'],
            "confidence": result['confidence']
        }
    
    def auto_rewrite_and_execute(self, query, conversation_history="", context_info=""):
        result = self.enhanced_rewriter.rewrite(query, context_info)
        return {
            "original_query": result['original_query'],
            "detected_type": result['intent'],
            "confidence": result['confidence'],
            "rewritten_query": result['rewritten_query'],
            "auto_rewrite_result": result
        }
