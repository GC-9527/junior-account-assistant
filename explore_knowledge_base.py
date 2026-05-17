
# -*- coding: utf-8 -*-
"""
探索知识库内容，找出我们可以确定有答案的问题
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.chroma_manager import ChromaManager


def main():
    print("=" * 80)
    print("初会RAG知识库内容探索")
    print("=" * 80)
    
    chroma_manager = ChromaManager()
    all_docs = chroma_manager.get_all_documents()
    
    print(f"\n知识库总文档数: {len(all_docs)}")
    
    # 只显示前30个文档的前200个字符
    print("\n" + "=" * 80)
    print("前30个文档内容预览（前200字符）:")
    print("=" * 80)
    
    for i, doc in enumerate(all_docs[:30]):
        print(f"\n--- 文档 {i+1} ---")
        print(f"ID: {doc.get('id')}")
        print(f"类型: {doc.get('metadata', {}).get('knowledge_type', '未知')}")
        print(f"来源: {doc.get('metadata', {}).get('source', '未知')}")
        content = doc.get('content', '')
        # 打印前200个字符，避免编码问题
        try:
            print(f"内容: {content[:200]}")
        except Exception as e:
            print(f"内容: [编码问题，无法显示] 错误: {e}")
    
    print("\n" + "=" * 80)
    print("知识库内容探索完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

