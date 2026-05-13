# -*- coding: utf-8 -*-
"""
初会RAG助手 - 查询处理

功能：加载Chroma索引，处理用户query，支持Query改写，生成回答
"""
import os
from rag.qa_chain import QAChain


def print_result(result: dict):
    """打印查询结果"""
    print(f"\n{'='*60}")
    print(f"原始问题: {result.get('original_query', result.get('query', ''))}")
    
    if 'rewritten_query' in result and result['rewritten_query'] != result.get('original_query'):
        print(f"改写后: {result['rewritten_query']}")
        print(f"查询类型: {result.get('query_type', '')}")
        print(f"置信度: {result.get('confidence', '')}%")
    
    if 'answers' in result:
        for i, sub_result in enumerate(result['answers']):
            print(f"\n--- 子问题 {i+1} ---")
            print(f"问题: {sub_result['query']}")
            print(f"回答:\n{sub_result['answer']}")
            if sub_result['sources']:
                print("\n参考来源:")
                for source in sub_result['sources']:
                    print(f"  - {source.get('source', '')}")
    else:
        print(f"\n回答:\n{result['answer']}")
        
        if result.get('sources'):
            print("\n参考来源:")
            for source in result['sources']:
                info = f"  - {source.get('source', '')}"
                if source.get('chapter'):
                    info += f" | 章节: {source['chapter']}"
                if source.get('knowledge_type'):
                    info += f" | 类型: {source['knowledge_type']}"
                if source.get('similarity'):
                    info += f" | 相似度: {source['similarity']}"
                print(info)
    
    print('='*60)


def main():
    """主函数"""
    print("=== 初会RAG助手 ===")
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出\n")
    
    qa_chain = QAChain()
    
    while True:
        query = input("请输入问题: ").strip()
        
        if query.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if not query:
            print("请输入有效问题")
            continue
        
        try:
            result = qa_chain.ask(query, k=3)
            print_result(result)
        except Exception as e:
            print(f"查询失败: {str(e)}")


if __name__ == "__main__":
    main()