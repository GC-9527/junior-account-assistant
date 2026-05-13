
# -*- coding: utf-8 -*-
"""
测试新的Query改写效果
"""
from rag.qa_chain import QAChain


def test_question(question):
    """测试单个问题"""
    print("\n" + "=" * 80)
    print(f"问题: {question}")
    print("=" * 80)
    
    qa = QAChain()
    try:
        # 测试新的改写（不使用扩展）
        print("\n--- 测试1: 使用新Query改写（不扩展）---")
        result = qa.ask(question, k=3, use_rewrite=True, use_expansion=False)
        
        # 测试新的改写（使用扩展）
        print("\n--- 测试2: 使用新Query改写（带扩展）---")
        result_with_expansion = qa.ask(question, k=3, use_rewrite=True, use_expansion=True)
        
        # 测试不使用改写
        print("\n--- 测试3: 不使用Query改写 ---")
        result_no_rewrite = qa.ask(question, k=3, use_rewrite=False)
        
        print("\n" + "=" * 80)
        print("测试完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 80)
    print("初会RAG助手 - Query改写效果对比测试")
    print("=" * 80)
    
    # 测试一些典型的问题
    test_cases = [
        "交易性金融资产怎么做账？",
        "进项税不能抵扣的情形有哪些？",
        "固定资产折旧方法有哪些？",
        "消费税的纳税环节？",
        "增值税视同销售？"
    ]
    
    for question in test_cases:
        test_question(question)
        print("\n\n")


if __name__ == "__main__":
    main()

