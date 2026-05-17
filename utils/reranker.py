# -*- coding: utf-8 -*-
"""
Rerank重排序模块
使用通义千问qwen3-rerank模型进行智能重排序
"""
import re
from typing import List, Dict, Tuple
import jieba
import jieba.posseg as pseg
import dashscope
from http import HTTPStatus
from config import DASHSCOPE_API_KEY


dashscope.api_key = DASHSCOPE_API_KEY


class TongyiReranker:
    """使用通义千问qwen3-rerank模型的排序器"""
    
    def __init__(self, model_name: str = "qwen3-rerank", top_n: int = 5):
        """
        初始化通义千问排序器
        
        Args:
            model_name: 模型名称，默认为 qwen3-rerank
            top_n: 返回的top n个结果
        """
        self.model_name = model_name
        self.top_n = top_n
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        使用通义千问rerank模型对结果进行重排序
        
        Args:
            query: 用户查询
            results: 原始检索结果列表
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果
        """
        if not results:
            return []
        
        try:
            # 提取文档内容
            documents = []
            for result in results:
                content = result.get('content', '')
                if content:
                    documents.append(content)
            
            if not documents:
                return results[:top_k]
            
            # 调用通义千问rerank API
            response = dashscope.TextReRank.call(
                model=self.model_name,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=True,
                instruct="Given a web search query, retrieve relevant passages that answer the query."
            )
            
            if response.status_code != HTTPStatus.OK:
                print(f"通义Rerank API调用失败: {response.message}")
                # 如果API调用失败，使用原始结果
                return results[:top_k]
            
            # 解析响应
            reranked_results = []
            for item in response.output.results:
                idx = item.index
                score = item.relevance_score
                
                # 从原始结果中找到对应索引的文档
                if idx < len(results):
                    result = results[idx].copy()
                    result['rerank_score'] = score
                    result['similarity'] = f"{score * 100:.2f}%"
                    result['distance'] = 1 - score
                    reranked_results.append(result)
            
            return reranked_results
            
        except Exception as e:
            print(f"通义Rerank重排序失败: {str(e)}")
            # 出错时使用原始结果
            return results[:top_k]


class HybridReranker:
    """混合排序器 - 结合规则排序和通义Rerank"""
    
    def __init__(self, use_tongyi: bool = True):
        """
        初始化混合排序器
        
        Args:
            use_tongyi: 是否优先使用通义rerank
        """
        self.tongyi_reranker = TongyiReranker() if use_tongyi else None
        self.rule_reranker = RuleBasedReranker()
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        混合重排序
        
        优先使用通义千问rerank，失败时使用规则排序
        """
        if not results:
            return []
        
        # 优先尝试使用通义rerank
        if self.tongyi_reranker:
            tongyi_results = self.tongyi_reranker.rerank(query, results, top_k)
            if tongyi_results:
                return tongyi_results
        
        # fallback到规则排序
        return self.rule_reranker.rerank(query, results, top_k)


class RuleBasedReranker:
    """基于规则的重排序器"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        初始化规则排序器
        
        Args:
            weights: 各策略的权重配置
        """
        self.weights = weights or {
            'semantic': 0.4,
            'keyword': 0.3,
            'position': 0.2,
            'quality': 0.1
        }
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        基于规则对检索结果进行重排序
        
        Args:
            query: 用户查询
            results: 原始检索结果
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果
        """
        if not results:
            return []
        
        query_keywords = self._extract_keywords(query)
        
        scored_results = []
        for result in results:
            scores = self._calculate_scores(query, query_keywords, result)
            total_score = sum(
                scores[key] * self.weights[key] 
                for key in scores.keys()
            )
            
            result['rerank_score'] = total_score
            result['score_details'] = scores
            scored_results.append(result)
        
        scored_results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        for result in scored_results[:top_k]:
            result['similarity'] = f"{result['rerank_score'] * 100:.2f}%"
            result['distance'] = 1 - result['rerank_score']
        
        return scored_results[:top_k]
    
    def _extract_keywords(self, text: str) -> set:
        """提取关键词"""
        words = pseg.cut(text)
        keywords = set()
        for word, flag in words:
            if flag in ['n', 'v', 'a', 'nz', 'ns', 'nt', 'nr', 'nz']:
                if len(word) > 1:
                    keywords.add(word)
        return keywords
    
    def _calculate_scores(self, query: str, query_keywords: set, result: Dict) -> Dict[str, float]:
        """计算各项分数"""
        return {
            'semantic': self._calculate_semantic_score(query, result),
            'keyword': self._calculate_keyword_score(query_keywords, result),
            'position': self._calculate_position_score(query, result),
            'quality': self._calculate_quality_score(result)
        }
    
    def _calculate_semantic_score(self, query: str, result: Dict) -> float:
        """计算语义相似度分数"""
        hybrid_score = result.get('hybrid_score', 0)
        vector_score = result.get('vector_score', 0)
        bm25_score = result.get('bm25_score', 0)
        
        max_bm25 = 100
        normalized_bm25 = min(bm25_score / max_bm25, 1.0) if bm25_score > 0 else 0
        
        semantic_score = (hybrid_score * 0.6 + vector_score * 0.3 + normalized_bm25 * 0.1)
        
        return min(semantic_score, 1.0)
    
    def _calculate_keyword_score(self, query_keywords: set, result: Dict) -> float:
        """计算关键词匹配分数"""
        content = result.get('content', '').lower()
        metadata = result.get('metadata', {})
        
        if not query_keywords:
            return 0.5
        
        matched_keywords = 0
        total_keywords = len(query_keywords)
        
        for keyword in query_keywords:
            if keyword.lower() in content:
                matched_keywords += 1
                count = content.count(keyword.lower())
                if count > 1:
                    matched_keywords += min(count - 1, 2) * 0.5
        
        knowledge_type = metadata.get('knowledge_type', '').lower()
        for keyword in query_keywords:
            if keyword.lower() in knowledge_type:
                matched_keywords += 0.5
        
        return min(matched_keywords / total_keywords, 1.0)
    
    def _calculate_position_score(self, query: str, result: Dict) -> float:
        """计算位置相关分数"""
        content = result.get('content', '')
        query_lower = query.lower()
        content_lower = content.lower()
        
        first_pos = content_lower.find(query_lower)
        
        if first_pos == -1:
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                pos = content_lower.find(keyword.lower())
                if pos != -1:
                    first_pos = pos
                    break
        
        if first_pos == -1:
            return 0.5
        
        position_ratio = first_pos / max(len(content), 1)
        position_score = 1.0 - position_ratio
        
        return position_score
    
    def _calculate_quality_score(self, result: Dict) -> float:
        """计算文档质量分数"""
        content = result.get('content', '')
        metadata = result.get('metadata', {})
        
        quality_score = 0.0
        
        content_length = len(content)
        if 50 <= content_length <= 1000:
            quality_score += 0.3
        elif content_length > 0:
            quality_score += 0.2
        
        knowledge_type = metadata.get('knowledge_type', '')
        high_value_types = ['概念定义', '会计分录', '税法法条', '计算公式']
        if knowledge_type in high_value_types:
            quality_score += 0.2
        
        exam_level = metadata.get('exam_level', '')
        if exam_level == '必考':
            quality_score += 0.2
        elif exam_level == '高频':
            quality_score += 0.15
        
        source = metadata.get('source', '')
        if '官方' in source or '教材' in source:
            quality_score += 0.1
        
        if '...' not in content and 'fôY' not in content:
            quality_score += 0.2
        
        return min(quality_score, 1.0)


class AdvancedReranker(HybridReranker):
    """高级重排序器 - 增强版混合排序"""
    
    def __init__(self):
        super().__init__(use_tongyi=True)
        self.rule_reranker = AdvancedRuleReranker()


class AdvancedRuleReranker(RuleBasedReranker):
    """高级规则排序器"""
    
    def __init__(self):
        super().__init__(weights={
            'semantic': 0.35,
            'keyword': 0.35,
            'position': 0.15,
            'quality': 0.15
        })
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """增强版规则排序"""
        if not results:
            return []
        
        query_keywords = self._extract_keywords(query)
        
        for result in results:
            scores = self._calculate_scores(query, query_keywords, result)
            base_score = sum(
                scores[key] * self.weights[key] 
                for key in scores.keys()
            )
            
            query_type = self._classify_query(query)
            special_score = self._apply_query_type_rules(query_type, result, query_keywords)
            
            result['rerank_score'] = base_score * 0.7 + special_score * 0.3
            result['score_details'] = scores
            result['query_type'] = query_type
        
        results.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        for result in results[:top_k]:
            result['similarity'] = f"{result['rerank_score'] * 100:.2f}%"
            result['distance'] = 1 - result['rerank_score']
        
        return results[:top_k]
    
    def _classify_query(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['区别', '不同', '差异', '比较', '对比']):
            return 'comparison'
        if any(word in query_lower for word in ['计算', '算', '公式', '多少']):
            return 'calculation'
        if any(word in query_lower for word in ['什么', '定义', '概念', '是']):
            return 'definition'
        if any(word in query_lower for word in ['如何', '怎么', '步骤', '方法']):
            return 'procedure'
        
        return 'general'
    
    def _apply_query_type_rules(self, query_type: str, result: Dict, query_keywords: set) -> float:
        """应用查询类型特殊规则"""
        content = result.get('content', '').lower()
        metadata = result.get('metadata', {})
        
        bonus = 0.0
        
        if query_type == 'comparison':
            concept_count = sum(1 for kw in query_keywords if kw.lower() in content)
            bonus = min(concept_count * 0.1, 0.5)
            if metadata.get('knowledge_type') == '易错辨析':
                bonus += 0.3
        
        elif query_type == 'calculation':
            if any(char in content for char in ['=', '×', '÷', '+', '-', '%']):
                bonus += 0.3
            if metadata.get('knowledge_type') == '计算公式':
                bonus += 0.3
        
        elif query_type == 'definition':
            if metadata.get('knowledge_type') == '概念定义':
                bonus += 0.3
            if len(content) < 500:
                bonus += 0.2
        
        elif query_type == 'procedure':
            if metadata.get('knowledge_type') == '会计分录':
                bonus += 0.3
            if any(marker in content for marker in ['1.', '2.', '3.', '（1）', '（2）']):
                bonus += 0.2
        
        return min(bonus, 1.0)
