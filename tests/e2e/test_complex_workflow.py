"""
复杂工作流端到端测试
测试Agent在复杂场景下的完整工作流
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.agents.agent import IntelligentAgent
from src.core.types import AgentConfig


class TestComplexWorkflow:
    """复杂工作流测试类"""

    @pytest.fixture
    def mock_llm(self):
        """模拟LLM"""
        llm = Mock()
        llm.generate = AsyncMock(side_effect=self._mock_llm_responses())
        return llm

    def _mock_llm_responses(self):
        """模拟LLM响应序列"""
        responses = [
            Mock(content="我了解了您的需求。让我帮您分析这个问题。", finish_reason="stop"),
            Mock(content="首先，我需要收集更多信息。您能提供更多细节吗？", finish_reason="stop"),
            Mock(content="基于您提供的信息，我建议采用以下步骤来解决：1. 数据收集 2. 模型训练 3. 结果评估", finish_reason="stop"),
            Mock(content="让我为您执行第一步：数据收集", finish_reason="tool_calls"),
            Mock(content="数据收集完成，现在开始模型训练", finish_reason="tool_calls"),
            Mock(content="模型训练完成，正在进行结果评估", finish_reason="tool_calls"),
            Mock(content="整个流程已完成！根据评估结果，您的项目取得了很好的成果。", finish_reason="stop")
        ]
        return iter(responses)

    @pytest.fixture
    def agent(self, mock_llm):
        """Agent fixture"""
        config = AgentConfig(
            model_name="test-model",
            max_tokens=2000,
            temperature=0.7,
            enable_memory=True,
            enable_rag=True,
            enable_tools=True
        )
        return IntelligentAgent(config=config, llm=mock_llm)

    @pytest.mark.asyncio
    async def test_complete_problem_solving_workflow(self, agent):
        """测试完整问题解决工作流"""
        # 模拟复杂问题
        problem = "我需要开发一个机器学习模型来预测客户流失，请帮我制定完整的解决方案"

        # 第一轮：问题理解
        response1 = await agent.process_message(problem)

        assert "需求" in response1.content or "问题" in response1.content
        assert response1.finish_reason == "stop"

        # 第二轮：信息收集
        info_request = "我们有6个月的历史数据，包含客户行为、交易记录等，主要使用Python和scikit-learn"
        response2 = await agent.process_message(info_request)

        assert "步骤" in response2.content or "计划" in response2.content
        assert response2.finish_reason == "stop"

        # 第三轮：执行确认
        confirm_message = "请按照您提出的步骤执行"
        response3 = await agent.process_message(confirm_message)

        assert response3.finish_reason == "tool_calls" or "数据收集" in response3.content

        # 模拟工具调用后的响应
        for i in range(4):  # 模拟后续的交互
            if response3.finish_reason == "tool_calls":
                # 模拟工具完成
                continue_message = "请继续下一步"
            else:
                continue_message = "继续"

            response = await agent.process_message(continue_message)

            assert response.content is not None
            assert response.finish_reason in ["stop", "tool_calls"]

    @pytest.mark.asyncio
    async def test_multi_step_code_generation_workflow(self, agent):
        """测试多步代码生成工作流"""
        # 第一步：需求理解
        request = "我需要一个Python函数来处理用户数据清洗，包括缺失值处理、异常值检测和数据标准化"

        response1 = await agent.process_message(request)

        assert "功能" in response1.content or "步骤" in response1.content

        # 第二步：详细设计
        design_message = "请先设计整体架构"
        response2 = await agent.process_message(design_message)

        assert "架构" in response2.content or "设计" in response2.content

        # 第三步：代码生成
        code_message = "现在请实现这个函数"
        response3 = await agent.process_message(code_message)

        assert response3.finish_reason == "tool_calls" or "def" in response3.content or "import" in response3.content

        # 第四步：测试和优化
        test_message = "请添加测试用例和优化建议"
        response4 = await agent.process_message(test_message)

        assert "测试" in response4.content or "优化" in response4.content

    @pytest.mark.asyncio
    async def test_research_and_analysis_workflow(self, agent):
        """测试研究分析工作流"""
        # 研究主题
        topic = "请帮我研究最新的深度学习在自然语言处理中的应用进展"

        # 信息收集阶段
        research_response = await agent.process_message(topic)

        assert len(research_response.content) > 100  # 应该有详细内容

        # 深度分析阶段
        analysis_request = "请分析这些进展的技术原理和实际应用场景"
        analysis_response = await agent.process_message(analysis_request)

        assert "技术" in analysis_response.content or "原理" in analysis_response.content
        assert "应用" in analysis_response.content or "场景" in analysis_response.content

        # 实践建议阶段
        practice_request = "基于这些信息，请提供实践建议和实施步骤"
        practice_response = await agent.process_message(practice_request)

        assert "建议" in practice_response.content or "步骤" in practice_response.content

    @pytest.mark.asyncio
    async def test_concurrent_task_management(self, agent):
        """测试并发任务管理"""
        tasks = [
            "分析这个数据集的特征重要性",
            "设计一个推荐系统架构",
            "优化现有算法的性能",
            "编写技术文档"
        ]

        # 并发启动多个任务
        import asyncio
        task_prompts = [f"请帮我处理任务{i+1}: {task}" for i, task in enumerate(tasks)]

        responses = await asyncio.gather(*[
            agent.process_message(prompt) for prompt in task_prompts
        ])

        assert len(responses) == 4

        # 验证每个任务都有合理的响应
        for i, response in enumerate(responses):
            assert response.content is not None
            assert len(response.content) > 50  # 应该有详细内容

    @pytest.mark.asyncio
    async def test_memory_integration_workflow(self, agent):
        """测试记忆集成工作流"""
        # 第一轮：建立上下文
        context1 = "我是一名数据科学家，工作在金融科技公司，主要处理客户风险预测"
        response1 = await agent.process_message(context1)

        # 第二轮：引用之前的信息
        reference1 = "基于我刚才提到的背景，请推荐合适的机器学习算法"
        response2 = await agent.process_message(reference1)

        assert "金融" in response2.content or "风险" in response2.content or "预测" in response2.content

        # 第三轮：继续引用
        reference2 = "我还提到了数据科学，请推荐相关的工具和框架"
        response3 = await agent.process_message(reference2)

        assert "工具" in response3.content or "框架" in response3.content
        assert "数据科学" in response3.content

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, agent):
        """测试错误恢复工作流"""
        # 模拟出现错误的任务
        failing_task = "请处理这个不存在的文件：/path/to/missing/file.csv"

        response1 = await agent.process_message(failing_task)

        # 应该处理错误并提供替代方案
        assert "文件" in response1.content or "错误" in response1.content or "不存在" in response1.content
        assert "建议" in response1.content or "替代" in response1.content

        # 恢复流程
        recovery_message = "好的，请提供一个替代方案"
        response2 = await agent.process_message(recovery_message)

        assert response2.content is not None
        assert len(response2.content) > 100  # 应该有详细的替代方案

    @pytest.mark.asyncio
    async def test_progressive_refinement_workflow(self, agent):
        """测试渐进改进工作流"""
        # 初始需求
        initial_request = "请帮我写一个简单的Python函数"
        response1 = await agent.process_message(initial_request)

        # 第一次改进
        refinement1 = "请添加错误处理"
        response2 = await agent.process_message(refinement1)

        assert "错误" in response2.content or "try" in response2.content or "except" in response2.content

        # 第二次改进
        refinement2 = "请添加文档字符串"
        response3 = await agent.process_message(refinement2)

        assert "文档" in response3.content or """" in response3.content or "# " in response3.content

        # 第三次改进
        refinement3 = "请添加类型注解"
        response4 = await agent.process_message(refinement3)

        assert "类型" in response4.content or ":" in response4.content

    @pytest.mark.asyncio
    async def test_long_conversation_maintenance(self, agent):
        """测试长对话维护"""
        messages = [
            "我想学习机器学习",
            "请从基础概念开始",
            "什么是监督学习？",
            "能给我一个具体的例子吗？",
            "非监督学习又是什么？",
            "这两种方法有什么区别？",
            "在实际项目中如何选择？",
            "请推荐一些学习资源",
            "总结一下今天学到的内容"
        ]

        conversation_history = []

        for i, message in enumerate(messages):
            response = await agent.process_message(message)
            conversation_history.append((message, response.content))

            # 验证对话连贯性
            if i > 0:
                # 检查是否能引用之前的内容
                prev_context = " ".join([conv[1] for conv in conversation_history[:-2]])
                if i >= 3:  # 在对话进行一段后
                    assert len(response.content) > 50  # 应该有详细回应
                    assert any(word in response.content for word in ["学习", "监督", "非监督", "方法"])

        # 验证总结能力
        summary_response = conversation_history[-1][1]
        assert "总结" in summary_response or "回顾" in summary_response or "今天" in summary_response

    @pytest.mark.asyncio
    async def test_domain_specific_workflow(self, agent):
        """测试特定领域工作流 - 医疗AI"""
        # 医疗领域的问题
        medical_query = "我需要开发一个医疗图像诊断系统，用于检测肺部CT图像中的异常"

        # 第一轮：需求分析
        response1 = await agent.process_message(medical_query)

        assert "医疗" in response1.content or "图像" in response1.content or "诊断" in response1.content

        # 第二轮：技术方案
        tech_request = "请提供具体的技术架构和实现方案"
        response2 = await agent.process_message(tech_request)

        assert "架构" in response2.content or "技术" in response2.content
        assert "深度学习" in response2.content or "CNN" in response2.content

        # 第三轮：合规性考虑
        compliance_request = "还需要考虑医疗行业的合规性要求"
        response3 = await agent.process_message(compliance_request)

        assert "合规" in response3.content or "FDA" in response3.content or "标准" in response3.content
        assert "安全" in response3.content or "隐私" in response3.content

    @pytest.mark.asyncio
    async def test_performance_stress_workflow(self, agent):
        """测试性能压力工作流"""
        # 快速连续的消息
        rapid_messages = [f"快速消息{i}" for i in range(20)]

        start_time = asyncio.get_event_loop().time()

        responses = []
        for message in rapid_messages:
            response = await agent.process_message(message)
            responses.append(response)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # 验证所有消息都得到响应
        assert len(responses) == 20

        # 验证平均响应时间
        avg_time = total_time / 20
        assert avg_time < 5.0  # 平均每条消息响应时间小于5秒

        # 验证没有响应为空
        for response in responses:
            assert response.content is not None
            assert len(response.content) > 0


if __name__ == "__main__":
    pytest.main([__file__])
