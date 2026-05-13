# -*- coding: utf-8 -*-
"""
BM25 检索器模块
使用 jieba 分词和 BM25 算法进行关键词检索
"""
import jieba
import jieba.posseg as pseg
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.metadata_list = []
        
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
        
        for doc in documents:
            content = doc.get('content', '')
            self.corpus.append(self._tokenize(content))
            self.metadata_list.append(doc.get('metadata', {}))
        
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
                    'id': f"bm25_{idx}"
                })
        
        results.sort(key=lambda x: x['bm25_score'], reverse=True)
        return results[:top_k]


class HybridRetriever:
    def __init__(self, bm25_weight: float = 0.4, vector_weight: float = 0.6):
        self.bm25_retriever = BM25Retriever()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
    
    def build_index(self, documents: List[Dict]):
        """构建混合索引"""
        self.bm25_retriever.build_index(documents)
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数"""
        if not scores:
            return []
        max_score = max(scores)
        min_score = min(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def hybrid_search(self, query: str, vector_results: List[Dict], top_k: int = 5) -> List[Dict]:
        """执行混合检索"""
        bm25_results = self.bm25_retriever.search(query, top_k=top_k)
        
        bm25_scores = [r['bm25_score'] for r in bm25_results]
        normalized_bm25 = self._normalize_scores(bm25_scores)
        
        vector_scores = [(1 - r.get('distance', 1)) for r in vector_results]
        normalized_vector = self._normalize_scores(vector_scores)
        
        merged_results = {}
        
        for i, result in enumerate(bm25_results):
            doc_id = result.get('metadata', {}).get('id', result.get('id', f"bm25_{i}"))
            merged_results[doc_id] = {
                'content': result['content'],
                'metadata': result['metadata'],
                'bm25_score': result['bm25_score'],
                'vector_score': 0,
                'hybrid_score': normalized_bm25[i] * self.bm25_weight
            }
        
        for i, result in enumerate(vector_results):
            doc_id = result.get('metadata', {}).get('id', result.get('id', f"vector_{i}"))
            if doc_id in merged_results:
                merged_results[doc_id]['vector_score'] = 1 - result.get('distance', 1)
                merged_results[doc_id]['hybrid_score'] += normalized_vector[i] * self.vector_weight
            else:
                merged_results[doc_id] = {
                    'content': result['content'],
                    'metadata': result['metadata'],
                    'bm25_score': 0,
                    'vector_score': 1 - result.get('distance', 1),
                    'hybrid_score': normalized_vector[i] * self.vector_weight
                }
        
        final_results = sorted(merged_results.values(), key=lambda x: x['hybrid_score'], reverse=True)
        
        for result in final_results[:top_k]:
            result['similarity'] = f"{result['hybrid_score'] * 100:.2f}%"
            result['distance'] = 1 - result['hybrid_score']
        
        return final_results[:top_k]