
# -*- coding: utf-8 -*-
"""
初会RAG助手 - 测试脚本（基于知识库中确定存在的知识点）
"""
from rag.qa_chain import QAChain
from utils.query_rewriter import QueryRewriter


def print_divider():
    """打印分隔线"""
    print("=" * 80)


def test_question(test_num, question):
    """测试单个问题"""
    print(f"\n测试{test_num}: {question}")
    print_divider()
    print(f"Query: {question}")
    
    # Query改写检测
    rewriter = QueryRewriter()
    rewrite_result = rewriter.auto_rewrite_query(question)
    query_type = rewrite_result.get('query_type', '未知类型')
    confidence = rewrite_result.get('confidence', 0)
    
    print(f"查询类型检测: {query_type}")
    print(f"置信度: {confidence}%")
    
    if rewrite_result.get('rewritten_query') != question:
        print(f"改写后查询: {rewrite_result.get('rewritten_query')}")
    
    print_divider()
    
    # RAG问答
    qa = QAChain()
    try:
        result = qa.ask(question, k=3)
        
        # 处理多意图查询
        if 'answers' in result:
            print(f"【检测到多意图查询，共 {len(result['answers'])} 个子问题】")
            for i, sub_result in enumerate(result['answers']):
                print(f"\n--- 子问题 {i+1} ---")
                print(f"问题: {sub_result.get('query', '')}")
                print(f"回答:\n{sub_result.get('answer', '')}")
                if sub_result.get('sources'):
                    print("\n参考来源:")
                    for source in sub_result['sources']:
                        print(f"  - {source.get('source', '')} | 相似度: {source.get('similarity', '')}")
        else:
            # 检索结果
            print("【检索结果】")
            if result.get('context'):
                print(f"命中 {len(result['context'])} 条相关知识")
                for i, doc in enumerate(result['context']):
                    # 计算相似度：distance越小越相似，范围一般是0-2（取决于embedding模型）
                    distance = doc.get('distance', 1)
                    # 将distance转换为0-100的相似度分数
                    sim = max(0, min(100, (1 - distance / 2) * 100))
                    print(f"\n{i+1}. 相似度: {sim:.2f}%")
                    print(f"   来源: {doc['metadata'].get('source', '')}")
                    print(f"   类型: {doc['metadata'].get('knowledge_type', '')}")
                    print(f"   内容: {doc['content'][:150]}...")
            
            print("\n【最终答案】")
            print(result.get('answer', '无回答'))
            
            if result.get('sources'):
                print("\n【参考来源】")
                for source in result['sources']:
                    info = f"  - {source.get('source', '')}"
                    if source.get('chapter'):
                        info += f" | 章节: {source['chapter']}"
                    if source.get('knowledge_type'):
                        info += f" | 类型: {source['knowledge_type']}"
                    if source.get('similarity'):
                        info += f" | 相似度: {source['similarity']}"
                    print(info)
    
    except Exception as e:
        print(f"查询失败: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print_divider()


def main():
    """主函数"""
    print("=" * 80)
    print("  初会RAG助手 - 验证知识库中的问题（确定有答案）")
    print("=" * 80)
    
    # 基于我们在知识库中实际看到的内容设计的测试问题
    test_cases = [
        "自然人民事行为能力是如何划分的？",
        "法律责任有哪些类型？",
        "会计档案的保管期限是多久？",
        "支付结算的基本要求有哪些？",
        "票据权利时效是怎么规定的？",
        "商业汇票的付款期限是多久？",
        "增值税视同销售的情形有哪些？",
        "增值税混合销售和兼营有什么区别？",
        "增值税中不得抵扣的进项税额有哪些？",
        "增值税纳税义务发生时间是什么时候？",
        "消费税的纳税环节有哪些？",
        "消费税的计税方法有哪几种？"
    ]
    
    for i, question in enumerate(test_cases, 1):
        test_question(i, question)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

