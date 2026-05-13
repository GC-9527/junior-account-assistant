
# -*- coding: utf-8 -*-
"""
测试纯向量检索效果
"""
from rag.chroma_manager import ChromaManager


def test_query(query, k=5):
    print("\n" + "=" * 80)
    print(f"查询: {query}")
    print("=" * 80)
    
    chroma_manager = ChromaManager()
    results = chroma_manager.query(query, n_results=k)
    
    if not results or not results.get('documents'):
        print("未找到相关文档")
        return
    
    print(f"\n找到 {len(results['documents'][0])} 个相关文档:")
    for i in range(len(results['documents'][0])):
        print(f"\n--- 文档 {i+1} ---")
        print(f"距离: {results['distances'][0][i]:.4f}")
        print(f"类型: {results['metadatas'][0][i].get('knowledge_type', '未知')}")
        print(f"来源: {results['metadatas'][0][i].get('source', '未知')}")
        print(f"内容: {results['documents'][0][i][:300]}...")


def main():
    test_queries = [
        "固定资产折旧的方法有哪些",
        "什么是会计分录",
        "应收账款和应付账款的区别",
        "交易性金融资产如何做账",
        "个人所得税税率是多少"
    ]
    
    for query in test_queries:
        test_query(query, k=3)


if __name__ == "__main__":
    main()
