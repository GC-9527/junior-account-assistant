
# -*- coding: utf-8 -*-
"""
检查知识库中的文档元数据和内容
"""
from rag.chroma_manager import ChromaManager


def main():
    print("=" * 80)
    print("检查知识库")
    print("=" * 80)
    
    chroma_manager = ChromaManager()
    all_docs = chroma_manager.get_all_documents()
    
    print(f"\n知识库总文档数: {len(all_docs)}")
    
    # 统计知识类型分布
    type_count = {}
    for doc in all_docs:
        k_type = doc.get('metadata', {}).get('knowledge_type', '未知')
        type_count[k_type] = type_count.get(k_type, 0) + 1
    
    print("\n知识类型分布:")
    for k_type, count in sorted(type_count.items()):
        print(f"  {k_type}: {count}")
    
    # 查看前10个文档的详细信息
    print("\n" + "=" * 80)
    print("前10个文档详情:")
    print("=" * 80)
    for i, doc in enumerate(all_docs[:10]):
        print(f"\n文档 {i+1}:")
        print(f"  ID: {doc.get('id')}")
        print(f"  类型: {doc.get('metadata', {}).get('knowledge_type')}")
        print(f"  来源: {doc.get('metadata', {}).get('source')}")
        print(f"  章节: {doc.get('metadata', {}).get('chapter')}")
        print(f"  内容预览: {doc.get('content', '')[:150]}...")


if __name__ == "__main__":
    main()
