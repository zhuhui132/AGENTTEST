"""
Agent使用示例
"""
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agent import Agent
from memory import MemorySystem
from rag import RAGSystem
from tools import ToolSystem, calculator, weather_query

def basic_agent_example():
    """基础Agent使用示例"""
    print("=== 基础Agent使用示例 ===")

    # 创建Agent
    agent = Agent("示例助手")

    # 基本对话
    messages = [
        "你好",
        "我想了解一下天气",
        "谢谢你的帮助"
    ]

    for message in messages:
        print(f"用户: {message}")
        result = agent.process_message(message)
        print(f"助手: {result['response']}")
        print(f"使用的记忆: {len(result['memories_used'])} 条")
        print(f"参考的文档: {len(result['docs_used'])} 篇")
        print("-" * 50)

    # 查看Agent状态
    state = agent.get_state()
    print(f"Agent状态: {state}")

def memory_system_example():
    """记忆系统使用示例"""
    print("\n=== 记忆系统使用示例 ===")

    agent = Agent("记忆测试助手")

    # 手动添加记忆
    memory_id1 = agent.memory.add_memory(
        "用户名叫张三，住在上海",
        weight=3.0,
        metadata={"source": "user_profile", "importance": "high"}
    )

    memory_id2 = agent.memory.add_memory(
        "用户喜欢吃辣的食物",
        weight=2.0,
        metadata={"source": "preference", "category": "food"}
    )

    memory_id3 = agent.memory.add_memory(
        "用户是一名程序员",
        weight=2.5,
        metadata={"source": "profession", "category": "work"}
    )

    print(f"添加了3条记忆，ID分别为: {memory_id1}, {memory_id2}, {memory_id3}")

    # 检索相关记忆
    memories = agent.memory.retrieve("用户信息", limit=5)
    print(f"检索到 {len(memories)} 条相关记忆:")
    for memory in memories:
        print(f"  - {memory['content']} (分数: {memory['score']:.3f})")

    # 基于记忆的对话
    personal_messages = [
        "我是谁？",
        "我住在哪里？",
        "我喜欢什么食物？",
        "我的职业是什么？"
    ]

    print("\n基于记忆的对话:")
    for message in personal_messages:
        result = agent.process_message(message)
        print(f"用户: {message}")
        print(f"助手: {result['response']}")
        print(f"使用的记忆: {[mem['content'] for mem in result['memories_used']]}")
        print("-" * 50)

def rag_system_example():
    """RAG系统使用示例"""
    print("\n=== RAG系统使用示例 ===")

    agent = Agent("RAG测试助手")

    # 添加文档到知识库
    docs = [
        ("北京是中国的首都，有着悠久的历史", {"type": "geography", "importance": "high"}),
        ("上海是中国最大的城市，经济中心", {"type": "geography", "importance": "high"}),
        ("Python是一种流行的编程语言", {"type": "technology", "importance": "medium"}),
        ("机器学习是人工智能的一个重要分支", {"type": "technology", "importance": "medium"}),
        ("健康饮食应该包含蔬菜、水果和蛋白质", {"type": "health", "importance": "medium"})
    ]

    doc_ids = []
    for content, metadata in docs:
        doc_id = agent.rag.add_document(content, metadata)
        doc_ids.append(doc_id)

    print(f"添加了 {len(doc_ids)} 篇文档到知识库")

    # 基于知识的对话
    knowledge_questions = [
        "中国的首都是哪里？",
        "上海是什么样的城市？",
        "什么是Python？",
        "什么是机器学习？",
        "健康饮食应该包括什么？"
    ]

    for question in knowledge_questions:
        result = agent.process_message(question)
        print(f"用户: {question}")
        print(f"助手: {result['response']}")
        print(f"参考的文档: {[doc['content'] for doc in result['docs_used']]}")
        print("-" * 50)

def tools_system_example():
    """工具系统使用示例"""
    print("\n=== 工具系统使用示例 ===")

    agent = Agent("工具测试助手")

    # 注册工具
    calculator_id = agent.tools.register_tool(
        "calculator",
        calculator,
        "数学计算工具，支持加减乘除"
    )

    weather_id = agent.tools.register_tool(
        "weather",
        weather_query,
        "天气查询工具，支持主要城市天气查询"
    )

    print(f"注册了工具: calculator (ID: {calculator_id}), weather (ID: {weather_id})")

    # 使用工具的对话
    tool_messages = [
        "请帮我计算 100 除以 4 的结果",
        "北京今天的天气怎么样？",
        "计算 25 乘以 8 再减去 50",
        "上海的天气如何？"
    ]

    for message in tool_messages:
        result = agent.process_message(message)
        print(f"用户: {message}")
        print(f"助手: {result['response']}")
        print("-" * 50)

def complex_integration_example():
    """复杂集成示例"""
    print("\n=== 复杂集成示例 ===")

    agent = Agent("集成测试助手")

    # 设置丰富的数据
    # 记忆
    agent.memory.add_memory("用户叫李明，是北京人", weight=3.0)
    agent.memory.add_memory("用户是一名数据科学家", weight=2.5)
    agent.memory.add_memory("用户计划下个月去上海出差", weight=2.0)

    # 文档
    agent.rag.add_document("北京到上海的高铁约需要4-6小时", {"type": "travel"})
    agent.rag.add_document("上海是中国的经济金融中心", {"type": "geography"})
    agent.rag.add_document("数据科学需要统计学和编程技能", {"type": "profession"})

    # 工具
    agent.tools.register_tool("calculator", calculator)
    agent.tools.register_tool("weather", weather_query)

    # 复杂对话场景
    complex_dialogue = [
        "请帮我规划一下去上海的行程",
        "我想知道北京和上海的天气对比",
        "作为数据科学家，我需要准备什么？",
        "如果我在上海待5天，预算3000元够吗？",
        "请帮我总结一下我的情况和建议"
    ]

    print("开始复杂对话场景...")
    conversation_history = []

    for message in complex_dialogue:
        print(f"\n用户: {message}")
        result = agent.process_message(message)
        print(f"助手: {result['response']}")

        print("详细信息:")
        print(f"  - 使用记忆: {len(result['memories_used'])} 条")
        print(f"  - 参考文档: {len(result['docs_used'])} 篇")

        conversation_history.append({
            "user": message,
            "assistant": result["response"],
            "context": result["context"]
        })

        print("-" * 80)

    # 总结对话
    print(f"\n对话总结:")
    print(f"总对话轮次: {len(conversation_history)}")
    print(f"Agent状态: {agent.get_state()}")

    # 记忆统计
    memory_stats = agent.memory.get_memory_stats()
    print(f"记忆系统统计: {memory_stats}")

    # 文档统计
    doc_stats = agent.rag.get_document_stats()
    print(f"文档系统统计: {doc_stats}")

    # 工具统计
    tool_stats = agent.tools.get_tool_usage_stats()
    print(f"工具使用统计: {tool_stats}")

def performance_monitoring_example():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")

    import time

    agent = Agent("性能监控助手")

    # 准备一些数据
    for i in range(10):
        agent.memory.add_memory(f"测试记忆{i}", weight=float(i+1))
        agent.rag.add_document(f"测试文档{i}的内容")

    # 性能测试消息
    test_messages = [
        f"性能测试消息 {i}" for i in range(20)
    ]

    response_times = []

    print("开始性能测试...")
    for message in test_messages:
        start_time = time.time()
        result = agent.process_message(message)
        end_time = time.time()

        response_time = end_time - start_time
        response_times.append(response_time)

        print(f"消息处理时间: {response_time:.3f}s")

    # 统计分析
    import statistics

    avg_time = statistics.mean(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    median_time = statistics.median(response_times)

    print(f"\n性能统计:")
    print(f"平均响应时间: {avg_time:.3f}s")
    print(f"最小响应时间: {min_time:.3f}s")
    print(f"最大响应时间: {max_time:.3f}s")
    print(f"中位数响应时间: {median_time:.3f}s")
    print(f"总处理时间: {sum(response_times):.3f}s")
    print(f"处理消息数量: {len(response_times)}")

def error_handling_example():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")

    agent = Agent("错误处理测试助手")

    # 测试各种错误情况
    error_cases = [
        ("", "空消息测试"),
        ("   ", "空白消息测试"),
    ]

    for message, description in error_cases:
        print(f"\n{description}:")
        try:
            result = agent.process_message(message)
            print(f"意外成功: {result['response']}")
        except ValueError as e:
            print(f"预期的ValueError: {e}")
        except Exception as e:
            print(f"意外错误: {type(e).__name__}: {e}")

    # 正常消息测试
    normal_message = "这是一个正常的消息"
    print(f"\n正常消息测试:")
    try:
        result = agent.process_message(normal_message)
        print(f"成功处理: {result['response']}")
    except Exception as e:
        print(f"处理失败: {type(e).__name__}: {e}")

if __name__ == "__main__":
    # 运行所有示例
    try:
        basic_agent_example()
        memory_system_example()
        rag_system_example()
        tools_system_example()
        complex_integration_example()
        performance_monitoring_example()
        error_handling_example()

        print("\n=== 所有示例运行完成 ===")

    except Exception as e:
        print(f"运行示例时出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
