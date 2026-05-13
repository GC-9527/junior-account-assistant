
# -*- coding: utf-8 -*-
"""
测试另一个问题
"""
from rag.qa_chain import QAChain


def main():
    print("=" * 80)
    print("测试Query改写 - 进项税问题")
    print("=" * 80)
    
    qa = QAChain()
    
    # 测试问题
    question = "进项税不能抵扣的情形有哪些？"
    
    print(f"\n问题: {question}")
    print("-" * 80)
    
    try:
        result = qa.ask(question, k=3, use_rewrite=True, use_expansion=False)
        
        print("\n检索到的内容:")
        for i, doc in enumerate(result.get('context', [])):
            distance = doc.get('distance', 1)
            similarity = max(0, min(100, (1 - distance / 2) * 100))
            print(f"\n{i+1}. 相似度: {similarity:.2f}%")
            print(f"   内容: {doc.get('content', '')[:180]}...")
        
        print("\n" + "=" * 80)
        print("最终答案:")
        print(result.get('answer', '无答案'))
        print("=" * 80)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

