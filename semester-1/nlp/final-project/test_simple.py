#!/usr/bin/env python3
"""
简化测试版本，避免复杂的初始化问题
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.llm_client import llm_client
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def test_llm_connection():
    """测试LLM连接"""
    print("测试LLM连接...")

    system_prompt = "你是一个有用的助手，请用简洁的语言回答。"
    user_prompt = "请简单介绍一下你自己。"

    result = llm_client.chat(system_prompt, user_prompt)

    if "content" in result and result["content"]:
        print("✓ LLM连接成功！")
        print(f"回复: {result['content']}")
        return True
    else:
        print("✗ LLM连接失败")
        print(f"错误: {result.get('error', '未知错误')}")
        return False


def simple_chat():
    """简单聊天模式"""
    print("\n" + "=" * 50)
    print("智能搜索引擎 - 简化测试版")
    print("输入 'quit' 或 '退出' 结束程序")
    print("=" * 50)

    while True:
        try:
            query = input("\n请输入问题: ").strip()

            if query.lower() in ['quit', '退出', 'exit']:
                print("再见！")
                break

            if not query:
                continue

            print("思考中...")

            # 直接使用LLM回答
            system_prompt = """你是一个智能搜索引擎助手。请准确、简洁地回答用户的问题。
            如果问题涉及实时信息（如天气、股票、新闻等），请说明这是基于你的训练数据，
            可能不是最新信息。"""

            result = llm_client.chat(system_prompt, query, max_tokens=1000)

            if "content" in result and result["content"]:
                print(f"\n回答: {result['content']}")
            else:
                print(f"\n抱歉，回答时出现错误: {result.get('error', '未知错误')}")

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n处理问题时出现错误: {e}")


if __name__ == "__main__":
    print("启动智能搜索引擎测试版...")

    # 测试连接
    if test_llm_connection():
        # 进入聊天模式
        simple_chat()
    else:
        print("初始化失败，请检查网络连接和API配置")