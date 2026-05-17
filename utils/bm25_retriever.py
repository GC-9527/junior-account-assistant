# -*- coding: utf-8 -*-
"""
BM25 检索器模块
使用 jieba 分词和 BM25 算法进行关键词检索
"""
import jieba
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple
import numpy as np


class BM25Retriever:
    """BM25关键词检索器"""
    
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.metadata_list = []
        self.doc_ids = []
    
    def _tokenize(self, text: str) -> List[str]:
        """使用 jieba 分词"""
        words = pseg.cut(text)
        tokens = []
        for word, flag in words:
            if flag in ['n', 'v', 'a', 'd', 'nz', 'ns', 'nt']:
                tokens.append(word)
        return tokens
    
    def build_index(self, documents: List[Dict]):
        """构建 BM25 索引"""
        self.corpus = []
        self.metadata_list = []
        self.doc_ids = []
        
        for doc in documents:
            content = doc.get('content', '')
            self.corpus.append(self._tokenize(content))
            self.metadata_list.append(doc.get('metadata', {}))
            self.doc_ids.append(doc.get('id', ''))
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """执行 BM25 检索"""
        if not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                results.append({
                    'content': ''.join(self.corpus[idx]),
                    'metadata': self.metadata_list[idx],
                    'bm25_score': score,
                    'id': self.doc_ids[idx] if idx < len(self.doc_ids) else f"bm25_{idx}"
                })
        
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        return results[:top_k]
    
    def get_all_documents(self) -> List[Dict]:
        """获取所有文档"""
        return [
            {
                'content': ''.join(tokens),
                'metadata': meta,
                'id': doc_id
            }
            for tokens, meta, doc_id in zip(self.corpus, self.metadata_list, self.doc_ids)
        ]


class HybridRetriever:
    """混合检索器 - BM25关键词检索 + 向量检索"""
    
    def __init__(self, bm25_weight: float = 0.15, vector_weight: float = 0.85):
        """
        初始化混合检索器
        
        Args:
            bm25_weight: BM25检索权重
            vector_weight: 向量检索权重
        """
        self.bm25_retriever = BM25Retriever()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
    
    def build_index(self, documents: List[Dict]):
        """构建混合索引"""
        self.bm25_retriever.build_index(documents)
    
    def _normalize_scores(self, scores: List[float], method: str = 'min_max') -> List[float]:
        """
        归一化分数
        
        Args:
            scores: 原始分数列表
            method: 归一化方法 ('min_max' 或 'softmax')
        """
        if not scores:
            return []
        
        scores = np.array(scores)
        
        if method == 'softmax':
            exp_scores = np.exp(scores - np.max(scores))
            return (exp_scores / exp_scores.sum()).tolist()
        else:  # min_max
            min_s = np.min(scores)
            max_s = np.max(scores)
            if max_s == min_s:
                return [1.0 / len(scores)] * len(scores)
            return ((scores - min_s) / (max_s - min_s)).tolist()
    
    def _calculate_rrf(self, bm25_ranks: List[int], vector_ranks: List[int], k: int = 60) -> List[float]:
        """
        计算倒数排名融合分数 (Reciprocal Rank Fusion)
        
        Args:
            bm25_ranks: BM25检索结果排名列表
            vector_ranks: 向量检索结果排名列表
            k: RRF参数，通常设为60
        """
        rrf_scores = {}
        
        for rank, doc_id in bm25_ranks:
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        for rank, doc_id in vector_ranks:
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
        
        return rrf_scores
    
    def hybrid_search(self, query: str, vector_results: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        执行混合检索
        
        检索策略：
        1. 分别进行BM25和向量检索
        2. 使用RRF融合两种检索结果
        3. 返回融合后的top_k个结果
        
        Args:
            query: 查询文本
            vector_results: 向量检索结果
            top_k: 返回结果数量
            
        Returns:
            混合检索结果列表
        """
        # 1. 执行BM25检索（获取更多候选结果）
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 3)
        
        # 2. 如果BM25没有结果，直接返回向量检索结果
        if not bm25_results:
            for result in vector_results[:top_k]:
                result['hybrid_score'] = 1 - result.get('distance', 1)
                result['bm25_score'] = 0
                result['vector_score'] = 1 - result.get('distance', 1)
            return vector_results[:top_k]
        
        # 3. 准备BM25排名
        bm25_ranks = [(i, r.get('id', f"bm25_{i}")) for i, r in enumerate(bm25_results)]
        
        # 4. 准备向量检索排名
        vector_ranks = []
        for i, result in enumerate(vector_results):
            doc_id = result.get('id', f"vector_{i}")
            vector_ranks.append((i, doc_id))
        
        # 5. 使用RRF融合
        rrf_scores = self._calculate_rrf(bm25_ranks, vector_ranks, k=60)
        
        # 6. 构建最终结果
        merged_results = {}
        
        # 添加BM25结果
        for i, result in enumerate(bm25_results):
            doc_id = result.get('id', f"bm25_{i}")
            
            # 获取RRF分数
            rrf_score = rrf_scores.get(doc_id, 0)
            
            # 归一化BM25分数
            bm25_scores = [r['bm25_score'] for r in bm25_results]
            normalized_bm25 = self._normalize_scores(bm25_scores)[i]
            
            # 计算混合分数
            hybrid_score = (
                normalized_bm25 * self.bm25_weight * 0.5 +  # BM25贡献50%权重
                rrf_score * 0.5  # RRF贡献50%权重
            )
            
            merged_results[doc_id] = {
                'content': result['content'],
                'metadata': result['metadata'],
                'id': doc_id,
                'bm25_score': result['bm25_score'],
                'vector_score': 0,
                'hybrid_score': hybrid_score,
                'rrf_score': rrf_score
            }
        
        # 添加向量结果
        for i, result in enumerate(vector_results):
            doc_id = result.get('id', f"vector_{i}")
            
            if doc_id in merged_results:
                # 文档已在BM25结果中，更新向量分数
                merged_results[doc_id]['vector_score'] = 1 - result.get('distance', 1)
                
                # 重新计算混合分数
                vector_scores = [1 - r.get('distance', 1) for r in vector_results]
                normalized_vector = self._normalize_scores(vector_scores)[i]
                
                rrf_score = rrf_scores.get(doc_id, 0)
                
                merged_results[doc_id]['hybrid_score'] = (
                    normalized_vector * self.vector_weight * 0.5 +
                    rrf_score * 0.5
                )
            else:
                # 文档不在BM25结果中，添加为新结果
                vector_scores = [1 - r.get('distance', 1) for r in vector_results]
                normalized_vector = self._normalize_scores(vector_scores)[i]
                
                rrf_score = rrf_scores.get(doc_id, 0)
                
                merged_results[doc_id] = {
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'id': doc_id,
                    'bm25_score': 0,
                    'vector_score': 1 - result.get('distance', 1),
                    'hybrid_score': normalized_vector * self.vector_weight,
                    'rrf_score': rrf_score
                }
        
        # 7. 按混合分数排序
        final_results = sorted(
            merged_results.values(),
            key=lambda x: x['hybrid_score'],
            reverse=True
        )
        
        # 8. 格式化结果
        for result in final_results[:top_k]:
            result['similarity'] = f"{result['hybrid_score'] * 100:.2f}%"
            result['distance'] = 1 - result['hybrid_score']
        
        return final_results[:top_k]
    
    def search_bm25_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """仅使用BM25检索"""
        results = self.bm25_retriever.search(query, top_k)
        for result in results:
            result['similarity'] = f"{result['bm25_score'] * 100:.2f}%"
            result['distance'] = 1 / (1 + result['bm25_score'])
        return results
