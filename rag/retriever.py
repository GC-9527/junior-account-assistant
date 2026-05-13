from rag.chroma_manager import ChromaManager
from utils.bm25_retriever import HybridRetriever
from config import RETRIEVAL_K, BM25_WEIGHT, VECTOR_WEIGHT


class Retriever:
    def __init__(self, collection_name: str = "chuhui_rag"):
        self.chroma_manager = ChromaManager(collection_name)
        self.hybrid_retriever = HybridRetriever(
            bm25_weight=BM25_WEIGHT,
            vector_weight=VECTOR_WEIGHT
        )
        self._init_bm25_index()

    def _init_bm25_index(self):
        """初始化 BM25 索引"""
        try:
            all_docs = self.chroma_manager.get_all_documents()
            if all_docs:
                self.hybrid_retriever.build_index(all_docs)
        except Exception as e:
            print(f"初始化 BM25 索引失败: {e}")

    def _detect_knowledge_type(self, query: str) -> str:
        """根据查询内容检测知识点类型"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["分录", "做账", "记账", "借贷"]):
            return "会计分录"
        elif any(keyword in query_lower for keyword in ["税法", "税率", "期限", "纳税"]):
            return "税法法条"
        elif any(keyword in query_lower for keyword in ["公式", "计算", "算"]):
            return "计算公式"
        elif any(keyword in query_lower for keyword in ["真题", "习题", "错题", "练习"]):
            return "真题习题"
        elif any(keyword in query_lower for keyword in ["辨析", "区别", "不同", "对比"]):
            return "易错辨析"
        else:
            return "概念定义"

    def _detect_book_type(self, query: str) -> str:
        """根据查询内容检测教材类型"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ["实务", "资产", "负债", "收入", "费用", "利润", "报表"]):
            return "初级会计实务"
        elif any(keyword in query_lower for keyword in ["经济法", "法律", "法规", "合同", "票据"]):
            return "经济法基础"
        else:
            return None

    def _build_where_clause(self, conditions: dict) -> dict:
        """构建Chroma的where子句"""
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions
        
        return {"$and": [{k: v} for k, v in conditions.items()]}

    def search(self, query: str, k: int = RETRIEVAL_K, filter_type: str = None, use_hybrid: bool = False, use_strict_filter: bool = False) -> list:
        """检索相关文档"""
        where_conditions = {}
        
        if use_strict_filter:
            if filter_type:
                where_conditions["knowledge_type"] = filter_type
            else:
                detected_type = self._detect_knowledge_type(query)
                where_conditions["knowledge_type"] = detected_type
            
            detected_book = self._detect_book_type(query)
            if detected_book:
                where_conditions["book_type"] = detected_book
        
        where_clause = self._build_where_clause(where_conditions) if where_conditions else None
        vector_results = self.chroma_manager.query(query, n_results=k, where=where_clause)
        vector_results = self._format_results(vector_results)
        
        if use_hybrid:
            return self.hybrid_retriever.hybrid_search(query, vector_results, top_k=k)
        
        return vector_results

    def search_with_filter(self, query: str, k: int = RETRIEVAL_K, use_hybrid: bool = True, **filters) -> list:
        """带自定义过滤条件的检索"""
        where_conditions = {}
        
        if 'book_type' in filters:
            where_conditions['book_type'] = filters['book_type']
        if 'chapter' in filters:
            where_conditions['chapter'] = filters['chapter']
        if 'knowledge_type' in filters:
            where_conditions['knowledge_type'] = filters['knowledge_type']
        if 'exam_level' in filters:
            where_conditions['exam_level'] = filters['exam_level']
        if 'source' in filters:
            where_conditions['source'] = filters['source']
        
        where_clause = self._build_where_clause(where_conditions)
        vector_results = self.chroma_manager.query(query, n_results=k, where=where_clause)
        vector_results = self._format_results(vector_results)
        
        if use_hybrid:
            return self.hybrid_retriever.hybrid_search(query, vector_results, top_k=k)
        
        return vector_results

    def _format_results(self, results: dict) -> list:
        """格式化检索结果"""
        formatted = []
        if results and 'documents' in results and results['documents']:
            for i in range(len(results['documents'][0])):
                doc = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0,
                    'id': results['ids'][0][i] if results['ids'] else f"doc_{i}"
                }
                formatted.append(doc)
        return formatted

    def search_all(self, query: str, k: int = RETRIEVAL_K, use_hybrid: bool = True) -> list:
        """不带过滤条件的检索"""
        vector_results = self.chroma_manager.query(query, n_results=k)
        vector_results = self._format_results(vector_results)
        
        if use_hybrid:
            return self.hybrid_retriever.hybrid_search(query, vector_results, top_k=k)
        
        return vector_results

    def update_bm25_index(self):
        """更新 BM25 索引"""
        self._init_bm25_index()