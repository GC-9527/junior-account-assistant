# -*- coding: utf-8 -*-
"""
初会RAG助手 - 测试脚本
测试几个常见的初级会计问题
"""
from rag.qa_chain import QAChain
from utils.query_rewriter import QueryRewriter


def print_divider():
    """打印分隔线"""
    print("=" * 80)


def test_question(test_num, question, expected_type="概念定义"):
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
                    print(f"   内容: {doc['content'][:120]}...")
            
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
    
    print_divider()


def main():
    """主函数"""
    print("=" * 80)
    print("          初会RAG助手 - 测试脚本")
    print("=" * 80)
    
    test_cases = [
        ("固定资产折旧的方法有哪些", "概念定义"),
        ("什么是会计分录", "概念定义"),
        ("应收账款和应付账款的区别", "易错辨析"),
        ("交易性金融资产如何做账", "会计分录"),
        ("个人所得税税率是多少", "税法法条"),
    ]
    
    for i, (question, q_type) in enumerate(test_cases, 1):
        test_question(i, question, q_type)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()