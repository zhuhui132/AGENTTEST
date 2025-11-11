# 🤖 Agent架构设计

## 📚 概述

智能Agent是人工智能系统的核心组件，能够感知环境、进行推理、制定决策并执行动作。本文档详细介绍Agent的架构设计原理和实现方法。

## 🏗️ Agent基础架构

### 核心组件模型
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import time

class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    ERROR = "error"

@dataclass
class AgentPerception:
    """Agent感知信息"""
    timestamp: float
    observations: Dict[str, Any]
    confidence: float
    source: str
    metadata: Dict[str, Any] = None

@dataclass
class AgentAction:
    """Agent动作定义"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float
    expected_outcome: str
    timeout: float = 30.0

class BaseAgent(ABC):
    """Agent基础抽象类"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.state = AgentState.IDLE
        self.perception_buffer = []
        self.action_history = []
        self.goal_stack = []
        self.current_goal = None

        # 初始化组件
        self.perception_system = self._init_perception()
        self.reasoning_engine = self._init_reasoning()
        self.planning_system = self._init_planning()
        self.execution_system = self._init_execution()
        self.memory_system = self._init_memory()

    @abstractmethod
    def _init_perception(self):
        """初始化感知系统"""
        pass

    @abstractmethod
    def _init_reasoning(self):
        """初始化推理引擎"""
        pass

    @abstractmethod
    def _init_planning(self):
        """初始化规划系统"""
        pass

    @abstractmethod
    def _init_execution(self):
        """初始化执行系统"""
        pass

    @abstractmethod
    def _init_memory(self):
        """初始化记忆系统"""
        pass

    async def perceive(self, environment_data: Dict[str, Any]) -> List[AgentPerception]:
        """感知环境"""
        self.state = AgentState.OBSERVING

        perceptions = await self.perception_system.process(environment_data)
        self.perception_buffer.extend(perceptions)

        # 保持缓冲区大小
        if len(self.perception_buffer) > 100:
            self.perception_buffer = self.perception_buffer[-100:]

        self.state = AgentState.IDLE
        return perceptions

    async def reason(self, perceptions: List[AgentPerception]) -> Dict[str, Any]:
        """推理和决策"""
        self.state = AgentState.THINKING

        reasoning_result = await self.reasoning_engine.reason(
            perceptions,
            self.current_goal,
            self.memory_system
        )

        self.state = AgentState.IDLE
        return reasoning_result

    async def plan(self, reasoning_result: Dict[str, Any]) -> List[AgentAction]:
        """制定计划"""
        self.state = AgentState.PLANNING

        plan = await self.planning_system.create_plan(
            reasoning_result,
            self.current_goal,
            self.action_history
        )

        self.state = AgentState.IDLE
        return plan

    async def execute(self, actions: List[AgentAction]) -> List[Dict[str, Any]]:
        """执行动作"""
        self.state = AgentState.ACTING

        results = []
        for action in actions:
            try:
                result = await self.execution_system.execute(action)
                self.action_history.append(action)
                results.append(result)
            except Exception as e:
                results.append({
                    'action': action,
                    'success': False,
                    'error': str(e)
                })

        self.state = AgentState.IDLE
        return results

    async def perceive_reason_plan_act(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """完整的感知-推理-规划-执行循环"""
        start_time = time.time()

        try:
            # 1. 感知
            perceptions = await self.perceive(environment_data)

            # 2. 推理
            reasoning_result = await self.reason(perceptions)

            # 3. 规划
            actions = await self.plan(reasoning_result)

            # 4. 执行
            results = await self.execute(actions)

            execution_time = time.time() - start_time

            return {
                'success': True,
                'perceptions': perceptions,
                'reasoning_result': reasoning_result,
                'actions': actions,
                'results': results,
                'execution_time': execution_time
            }

        except Exception as e:
            self.state = AgentState.ERROR
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

    def set_goal(self, goal: Dict[str, Any]):
        """设置目标"""
        self.goal_stack.append(goal)
        self.current_goal = goal

    def complete_goal(self):
        """完成当前目标"""
        if self.goal_stack:
            self.goal_stack.pop()
            self.current_goal = self.goal_stack[-1] if self.goal_stack else None
```

## 🧠 感知系统

### 多模态感知系统
```python
class MultiModalPerceptionSystem:
    """多模态感知系统"""

    def __init__(self):
        self.sensors = {}
        self.perception_fusion = PerceptionFusion()

    def register_sensor(self, sensor_name: str, sensor):
        """注册传感器"""
        self.sensors[sensor_name] = sensor

    async def process(self, environment_data: Dict[str, Any]) -> List[AgentPerception]:
        """处理环境数据"""
        perceptions = []

        # 处理各模态数据
        for sensor_name, sensor in self.sensors.items():
            if sensor_name in environment_data:
                sensor_data = environment_data[sensor_name]
                modal_perceptions = await sensor.process(sensor_data)
                perceptions.extend(modal_perceptions)

        # 融合多模态感知
        if len(perceptions) > 1:
            fused_perception = await self.perception_fusion.fuse(perceptions)
            perceptions.append(fused_perception)

        return perceptions

class TextPerception:
    """文本感知"""

    async def process(self, text_data: str) -> List[AgentPerception]:
        perceptions = []

        # 文本分析
        from transformers import pipeline

        # 情感分析
        sentiment_analyzer = pipeline("sentiment-analysis")
        sentiment = sentiment_analyzer(text_data)[0]

        # 实体识别
        ner_analyzer = pipeline("ner", aggregation_strategy="simple")
        entities = ner_analyzer(text_data)

        # 意图识别
        intent = await self._identify_intent(text_data)

        perception = AgentPerception(
            timestamp=time.time(),
            observations={
                'text': text_data,
                'sentiment': sentiment,
                'entities': entities,
                'intent': intent
            },
            confidence=0.8,
            source='text'
        )

        perceptions.append(perception)
        return perceptions

    async def _identify_intent(self, text: str) -> str:
        """识别用户意图"""
        # 简化的意图识别
        text_lower = text.lower()

        intents = {
            'question': ['什么', '如何', '为什么', '?'],
            'request': ['请', '帮', '需要', '想要'],
            'greeting': ['你好', 'hello', 'hi'],
            'goodbye': ['再见', 'bye', '拜拜']
        }

        for intent, keywords in intents.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent

        return 'general'

class ImagePerception:
    """图像感知"""

    def __init__(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    async def process(self, image_data) -> List[AgentPerception]:
        perceptions = []

        # 图像描述生成
        inputs = self.processor(image_data, return_tensors="pt")
        out = self.model.generate(**inputs, max_length=50)
        description = self.processor.decode(out[0], skip_special_tokens=True)

        # 物体检测
        objects = await self._detect_objects(image_data)

        # 场景分析
        scene = await self._analyze_scene(image_data)

        perception = AgentPerception(
            timestamp=time.time(),
            observations={
                'description': description,
                'objects': objects,
                'scene': scene
            },
            confidence=0.7,
            source='image'
        )

        perceptions.append(perception)
        return perceptions

    async def _detect_objects(self, image_data) -> List[str]:
        """检测物体"""
        # 简化实现，实际应用中应该使用专门的物体检测模型
        return ['person', 'car', 'building']

    async def _analyze_scene(self, image_data) -> str:
        """分析场景"""
        # 简化实现
        return 'outdoor urban scene'

class PerceptionFusion:
    """感知融合"""

    async def fuse(self, perceptions: List[AgentPerception]) -> AgentPerception:
        """融合多模态感知信息"""
        # 合并观察数据
        merged_observations = {}
        total_confidence = 0
        sources = []

        for perception in perceptions:
            merged_observations.update(perception.observations)
            total_confidence += perception.confidence
            sources.append(perception.source)

        # 计算融合后的置信度
        avg_confidence = total_confidence / len(perceptions)

        # 添加融合信息
        merged_observations['fusion_info'] = {
            'fused_sources': sources,
            'fusion_method': 'weighted_average',
            'fusion_timestamp': time.time()
        }

        fused_perception = AgentPerception(
            timestamp=time.time(),
            observations=merged_observations,
            confidence=avg_confidence,
            source='fusion'
        )

        return fused_perception
```

## 🧭 推理引擎

### 基于规则的推理系统
```python
class RuleBasedReasoningEngine:
    """基于规则的推理引擎"""

    def __init__(self):
        self.rules = []
        self.inference_engine = SimpleInferenceEngine()

    def add_rule(self, condition, conclusion, confidence=1.0):
        """添加推理规则"""
        rule = {
            'condition': condition,
            'conclusion': conclusion,
            'confidence': confidence
        }
        self.rules.append(rule)

    async def reason(self, perceptions: List[AgentPerception],
                    current_goal: Optional[Dict],
                    memory_system) -> Dict[str, Any]:
        """执行推理"""
        facts = self._extract_facts(perceptions)

        # 应用规则进行推理
        conclusions = []
        for rule in self.rules:
            if self._match_condition(rule['condition'], facts):
                conclusions.append({
                    'conclusion': rule['conclusion'],
                    'confidence': rule['confidence'],
                    'rule_applied': rule
                })

        # 选择最高置信度的结论
        if conclusions:
            best_conclusion = max(conclusions, key=lambda x: x['confidence'])

            return {
                'reasoning_type': 'rule_based',
                'conclusion': best_conclusion['conclusion'],
                'confidence': best_conclusion['confidence'],
                'applied_rules': [best_conclusion['rule_applied']],
                'facts_used': facts
            }
        else:
            return {
                'reasoning_type': 'rule_based',
                'conclusion': None,
                'confidence': 0.0,
                'applied_rules': [],
                'facts_used': facts
            }

    def _extract_facts(self, perceptions: List[AgentPerception]) -> Dict[str, Any]:
        """从感知中提取事实"""
        facts = {}

        for perception in perceptions:
            facts.update(perception.observations)

        return facts

    def _match_condition(self, condition: str, facts: Dict[str, Any]) -> bool:
        """匹配规则条件"""
        # 简化的条件匹配，实际应该使用更复杂的逻辑
        try:
            # 安全的字典访问
            for key, value in condition.items():
                if key not in facts or facts[key] != value:
                    return False
            return True
        except:
            return False

class NeuralReasoningEngine:
    """基于神经网络的推理引擎"""

    def __init__(self):
        self.model = self._build_reasoning_model()
        self.reasoning_history = []

    def _build_reasoning_model(self):
        """构建推理模型"""
        import torch
        import torch.nn as nn

        class ReasoningNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = torch.softmax(self.fc3(x), dim=-1)
                return x

        return ReasoningNetwork(512, 256, 128)

    async def reason(self, perceptions: List[AgentPerception],
                    current_goal: Optional[Dict],
                    memory_system) -> Dict[str, Any]:
        """神经网络推理"""
        # 特征提取
        features = self._extract_features(perceptions, current_goal, memory_system)

        # 神经网络推理
        import torch
        with torch.no_grad():
            reasoning_output = self.model(features)

        # 解析推理结果
        reasoning_result = self._parse_neural_output(reasoning_output)

        self.reasoning_history.append({
            'timestamp': time.time(),
            'input_features': features,
            'output': reasoning_result
        })

        return reasoning_result

    def _extract_features(self, perceptions, current_goal, memory_system):
        """提取推理特征"""
        # 简化的特征提取
        import numpy as np

        # 感知特征
        perception_features = []
        for perception in perceptions:
            perception_features.extend([
                perception.confidence,
                len(str(perception.observations))
            ])

        # 目标特征
        goal_features = [0] * 50  # 简化处理
        if current_goal:
            goal_features = [
                len(str(current_goal)),
                current_goal.get('priority', 0) / 10.0,
                current_goal.get('urgency', 0) / 10.0
            ] + [0] * 47

        # 记忆特征
        memory_features = [0] * 256  # 简化处理

        # 合并特征
        features = perception_features[:10] + goal_features[:50] + memory_features[:200]

        # 填充到固定长度
        while len(features) < 512:
            features.append(0)

        import torch
        return torch.FloatTensor(features)

    def _parse_neural_output(self, output):
        """解析神经网络输出"""
        import torch

        # 获取最可能的推理结果
        probabilities = torch.softmax(output, dim=-1)
        max_prob, max_index = torch.max(probabilities, dim=-1)

        reasoning_types = [
            'action_selection', 'goal_planning', 'problem_solving',
            'decision_making', 'learning', 'communication'
        ]

        return {
            'reasoning_type': 'neural',
            'conclusion': reasoning_types[max_index] if max_index < len(reasoning_types) else 'unknown',
            'confidence': max_prob.item(),
            'probabilities': probabilities.tolist(),
            'all_probabilities': {
                reasoning_types[i]: prob for i, prob in enumerate(probabilities.tolist())
                if i < len(reasoning_types)
            }
        }
```

## 📋 规划系统

### 层次化任务规划
```python
class HierarchicalPlanner:
    """层次化任务规划器"""

    def __init__(self):
        self.task_library = TaskLibrary()
        self.plan_optimizer = PlanOptimizer()

    async def create_plan(self, reasoning_result: Dict[str, Any],
                         current_goal: Optional[Dict],
                         action_history: List) -> List[AgentAction]:
        """创建执行计划"""
        if not current_goal:
            return []

        # 分解目标为子任务
        subtasks = await self._decompose_goal(current_goal)

        # 为每个子任务选择动作
        actions = []
        for subtask in subtasks:
            task_actions = await self._plan_subtask(subtask)
            actions.extend(task_actions)

        # 优化计划
        optimized_actions = await self.plan_optimizer.optimize(actions)

        return optimized_actions

    async def _decompose_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分解目标为子任务"""
        goal_type = goal.get('type', 'general')

        if goal_type == 'conversation':
            return await self._decompose_conversation_goal(goal)
        elif goal_type == 'information_retrieval':
            return await self._decompose_retrieval_goal(goal)
        elif goal_type == 'problem_solving':
            return await self._decompose_problem_goal(goal)
        else:
            return [goal]  # 不分解

    async def _decompose_conversation_goal(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分解对话目标"""
        subtasks = []

        # 理解用户意图
        subtasks.append({
            'type': 'intent_understanding',
            'input': goal.get('user_input', ''),
            'priority': 1
        })

        # 检索相关信息
        subtasks.append({
            'type': 'information_retrieval',
            'query': goal.get('user_input', ''),
            'priority': 2
        })

        # 生成响应
        subtasks.append({
            'type': 'response_generation',
            'context': goal.get('context', {}),
            'priority': 3
        })

        return subtasks

    async def _plan_subtask(self, subtask: Dict[str, Any]) -> List[AgentAction]:
        """为子任务规划动作"""
        task_type = subtask['type']

        if task_type == 'intent_understanding':
            return [
                AgentAction(
                    action_type='analyze_intent',
                    parameters={'text': subtask['input']},
                    confidence=0.9,
                    expected_outcome='intent_classified'
                )
            ]
        elif task_type == 'information_retrieval':
            return [
                AgentAction(
                    action_type='search_knowledge',
                    parameters={'query': subtask['query']},
                    confidence=0.8,
                    expected_outcome='relevant_documents_found'
                )
            ]
        elif task_type == 'response_generation':
            return [
                AgentAction(
                    action_type='generate_response',
                    parameters={'context': subtask['context']},
                    confidence=0.8,
                    expected_outcome='response_generated'
                )
            ]

        return []

    async def optimize_plan(self, actions: List[AgentAction]) -> List[AgentAction]:
        """优化执行计划"""
        # 按优先级排序
        sorted_actions = sorted(actions, key=lambda x: self._get_action_priority(x))

        # 移除冗余动作
        optimized_actions = self._remove_redundant_actions(sorted_actions)

        # 合并相似动作
        merged_actions = self._merge_similar_actions(optimized_actions)

        return merged_actions

    def _get_action_priority(self, action: AgentAction) -> int:
        """获取动作优先级"""
        priorities = {
            'analyze_intent': 1,
            'search_knowledge': 2,
            'generate_response': 3,
            'execute_tool': 2,
            'update_memory': 4
        }

        return priorities.get(action.action_type, 5)

    def _remove_redundant_actions(self, actions: List[AgentAction]) -> List[AgentAction]:
        """移除冗余动作"""
        unique_actions = []
        seen_action_types = set()

        for action in actions:
            if action.action_type not in seen_action_types:
                unique_actions.append(action)
                seen_action_types.add(action.action_type)

        return unique_actions

    def _merge_similar_actions(self, actions: List[AgentAction]) -> List[AgentAction]:
        """合并相似动作"""
        # 简化实现：按动作类型分组
        action_groups = {}
        for action in actions:
            action_type = action.action_type
            if action_type not in action_groups:
                action_groups[action_type] = []
            action_groups[action_type].append(action)

        # 合并同类型动作
        merged_actions = []
        for action_type, group_actions in action_groups.items():
            if len(group_actions) == 1:
                merged_actions.extend(group_actions)
            else:
                # 合并参数
                merged_params = {}
                for action in group_actions:
                    merged_params.update(action.parameters)

                merged_action = AgentAction(
                    action_type=action_type,
                    parameters=merged_params,
                    confidence=sum(a.confidence for a in group_actions) / len(group_actions),
                    expected_outcome=f"merged_{action_type}_result"
                )
                merged_actions.append(merged_action)

        return merged_actions

class TaskLibrary:
    """任务库"""

    def __init__(self):
        self.tasks = {
            'conversation': {
                'subtasks': ['understand_intent', 'retrieve_context', 'generate_response'],
                'success_criteria': ['intent_understood', 'context_retrieved', 'response_relevant']
            },
            'search': {
                'subtasks': ['parse_query', 'execute_search', 'rank_results'],
                'success_criteria': ['query_parsed', 'search_executed', 'results_ranked']
            }
        }

    def get_task_template(self, task_type: str) -> Dict[str, Any]:
        """获取任务模板"""
        return self.tasks.get(task_type, {})

class PlanOptimizer:
    """计划优化器"""

    async def optimize(self, actions: List[AgentAction]) -> List[AgentAction]:
        """优化动作序列"""
        # 时间复杂度优化
        optimized = self._optimize_time_complexity(actions)

        # 资源使用优化
        optimized = self._optimize_resource_usage(optimized)

        # 并行化优化
        optimized = self._enable_parallel_execution(optimized)

        return optimized

    def _optimize_time_complexity(self, actions: List[AgentAction]) -> List[AgentAction]:
        """优化时间复杂度"""
        # 将独立动作移到前面执行
        independent_actions = []
        dependent_actions = []

        for action in actions:
            if self._is_independent(action, actions):
                independent_actions.append(action)
            else:
                dependent_actions.append(action)

        return independent_actions + dependent_actions

    def _is_independent(self, action: AgentAction, all_actions: List[AgentAction]) -> bool:
        """检查动作是否独立"""
        # 简化实现
        independent_actions = ['update_memory', 'log_activity']
        return action.action_type in independent_actions

    def _optimize_resource_usage(self, actions: List[AgentAction]) -> List[AgentAction]:
        """优化资源使用"""
        # 将资源密集型动作分散执行
        cpu_intensive = []
        memory_intensive = []
        normal = []

        for action in actions:
            if self._is_cpu_intensive(action):
                cpu_intensive.append(action)
            elif self._is_memory_intensive(action):
                memory_intensive.append(action)
            else:
                normal.append(action)

        # 交错执行
        optimized = []
        max_length = max(len(cpu_intensive), len(memory_intensive), len(normal))

        for i in range(max_length):
            if i < len(normal):
                optimized.append(normal[i])
            if i < len(memory_intensive):
                optimized.append(memory_intensive[i])
            if i < len(cpu_intensive):
                optimized.append(cpu_intensive[i])

        return optimized

    def _enable_parallel_execution(self, actions: List[AgentAction]) -> List[AgentAction]:
        """启用并行执行"""
        # 标记可并行执行的动作
        for action in actions:
            if self._can_run_parallel(action):
                action.parameters['parallel'] = True

        return actions

    def _is_cpu_intensive(self, action: AgentAction) -> bool:
        """检查是否CPU密集"""
        cpu_intensive_actions = ['generate_response', 'search_knowledge']
        return action.action_type in cpu_intensive_actions

    def _is_memory_intensive(self, action: AgentAction) -> bool:
        """检查是否内存密集"""
        memory_intensive_actions = ['load_model', 'process_large_data']
        return action.action_type in memory_intensive_actions

    def _can_run_parallel(self, action: AgentAction) -> bool:
        """检查是否可并行执行"""
        parallel_actions = ['search_knowledge', 'update_memory', 'log_activity']
        return action.action_type in parallel_actions
```

## ⚙️ 执行系统

### 多线程执行引擎
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

class ExecutionEngine:
    """执行引擎"""

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.tool_registry = ToolRegistry()
        self.execution_history = []

    async def execute(self, actions: List[AgentAction]) -> List[Dict[str, Any]]:
        """执行动作列表"""
        results = []

        # 分析动作依赖关系
        dependency_graph = self._build_dependency_graph(actions)

        # 执行动作
        if self._has_dependencies(dependency_graph):
            # 有依赖关系，按顺序执行
            results = await self._execute_with_dependencies(actions, dependency_graph)
        else:
            # 无依赖关系，并行执行
            results = await self._execute_parallel(actions)

        # 记录执行历史
        self.execution_history.append({
            'timestamp': time.time(),
            'actions': actions,
            'results': results
        })

        return results

    async def _execute_parallel(self, actions: List[AgentAction]) -> List[Dict[str, Any]]:
        """并行执行动作"""
        results = []

        # 创建异步任务
        tasks = []
        for action in actions:
            if action.parameters.get('parallel', False):
                task = self._execute_single_action(action)
                tasks.append(task)
            else:
                # 串行执行
                result = await self._execute_single_action(action)
                results.append(result)

        # 并行执行
        if tasks:
            parallel_results = await asyncio.gather(*tasks)
            results.extend(parallel_results)

        return results

    async def _execute_with_dependencies(self, actions: List[AgentAction],
                                       dependency_graph: Dict) -> List[Dict[str, Any]]:
        """按依赖关系执行"""
        results = []
        executed = set()

        while len(executed) < len(actions):
            # 找到可以执行的动作
            ready_actions = []
            for i, action in enumerate(actions):
                if i not in executed:
                    dependencies = dependency_graph.get(i, [])
                    if all(dep in executed for dep in dependencies):
                        ready_actions.append((i, action))

            # 执行就绪的动作
            for idx, action in ready_actions:
                result = await self._execute_single_action(action)
                results.append(result)
                executed.add(idx)

        return results

    async def _execute_single_action(self, action: AgentAction) -> Dict[str, Any]:
        """执行单个动作"""
        start_time = time.time()

        try:
            # 获取工具
            tool = self.tool_registry.get_tool(action.action_type)

            if tool is None:
                raise ValueError(f"Unknown action type: {action.action_type}")

            # 执行工具
            result = await tool.execute(action.parameters)

            execution_time = time.time() - start_time

            return {
                'action': action,
                'success': True,
                'result': result,
                'execution_time': execution_time
            }

        except Exception as e:
            execution_time = time.time() - start_time

            return {
                'action': action,
                'success': False,
                'error': str(e),
                'execution_time': execution_time
            }

    def _build_dependency_graph(self, actions: List[AgentAction]) -> Dict[int, List[int]]:
        """构建依赖关系图"""
        # 简化的依赖关系构建
        dependencies = {}

        for i, action in enumerate(actions):
            deps = []

            # 搜索动作依赖于理解意图
            if action.action_type == 'search_knowledge':
                for j, prev_action in enumerate(actions[:i]):
                    if prev_action.action_type == 'analyze_intent':
                        deps.append(j)

            # 响应生成依赖于搜索和意图理解
            elif action.action_type == 'generate_response':
                for j, prev_action in enumerate(actions[:i]):
                    if prev_action.action_type in ['analyze_intent', 'search_knowledge']:
                        deps.append(j)

            dependencies[i] = deps

        return dependencies

    def _has_dependencies(self, dependency_graph: Dict[int, List[int]]) -> bool:
        """检查是否有依赖关系"""
        return any(deps for deps in dependency_graph.values())

class ToolRegistry:
    """工具注册器"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, tool):
        """注册工具"""
        self.tools[name] = tool

    def get_tool(self, name: str):
        """获取工具"""
        return self.tools.get(name)

# 示例工具实现
class SearchTool:
    """搜索工具"""

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """执行搜索"""
        query = parameters.get('query', '')

        # 模拟搜索
        await asyncio.sleep(0.5)

        results = [
            {'title': f"Search result for: {query}",
             'content': f"Content about {query}",
             'relevance': 0.9}
        ]

        return {
            'results': results,
            'query': query,
            'count': len(results)
        }

class GenerateResponseTool:
    """响应生成工具"""

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """生成响应"""
        context = parameters.get('context', {})

        # 模拟生成
        await asyncio.sleep(0.3)

        response = f"Generated response based on context: {str(context)[:50]}..."

        return {
            'response': response,
            'tokens': len(response.split()),
            'confidence': 0.8
        }
```

## 💾 记忆系统

### 分层记忆架构
```python
class LayeredMemorySystem:
    """分层记忆系统"""

    def __init__(self):
        self.working_memory = WorkingMemory(capacity=7)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()

    async def store(self, information: Dict[str, Any], memory_type: str = 'auto') -> str:
        """存储信息到记忆系统"""
        if memory_type == 'auto':
            memory_type = self._determine_memory_type(information)

        if memory_type == 'working':
            return await self.working_memory.store(information)
        elif memory_type == 'episodic':
            return await self.episodic_memory.store(information)
        elif memory_type == 'semantic':
            return await self.semantic_memory.store(information)
        elif memory_type == 'procedural':
            return await self.procedural_memory.store(information)

        return None

    async def retrieve(self, query: Dict[str, Any], memory_type: str = 'all') -> List[Dict[str, Any]]:
        """从记忆系统检索信息"""
        results = []

        if memory_type in ['all', 'working']:
            results.extend(await self.working_memory.retrieve(query))
        if memory_type in ['all', 'episodic']:
            results.extend(await self.episodic_memory.retrieve(query))
        if memory_type in ['all', 'semantic']:
            results.extend(await self.semantic_memory.retrieve(query))
        if memory_type in ['all', 'procedural']:
            results.extend(await self.procedural_memory.retrieve(query))

        return results

    def _determine_memory_type(self, information: Dict[str, Any]) -> str:
        """自动确定记忆类型"""
        # 简化的记忆类型判断
        if 'action' in information:
            return 'procedural'
        elif 'conversation' in information:
            return 'episodic'
        elif 'fact' in information:
            return 'semantic'
        else:
            return 'working'

class WorkingMemory:
    """工作记忆"""

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items = []
        self.access_count = {}

    async def store(self, information: Dict[str, Any]) -> str:
        """存储到工作记忆"""
        import uuid

        item_id = str(uuid.uuid4())
        item = {
            'id': item_id,
            'information': information,
            'timestamp': time.time()
        }

        self.items.append(item)
        self.access_count[item_id] = 1

        # 维护容量限制
        if len(self.items) > self.capacity:
            # 移除访问次数最少的项
            oldest_item = min(self.items,
                           key=lambda x: self.access_count.get(x['id'], 0))
            self.items.remove(oldest_item)
            del self.access_count[oldest_item['id']]

        return item_id

    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从工作记忆检索"""
        results = []

        for item in self.items:
            if self._match_query(item['information'], query):
                self.access_count[item['id']] = self.access_count.get(item['id'], 0) + 1
                results.append(item)

        return results

    def _match_query(self, information: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """匹配查询"""
        # 简化的匹配逻辑
        for key, value in query.items():
            if key in information and information[key] != value:
                return False
        return True

class EpisodicMemory:
    """情景记忆"""

    def __init__(self):
        self.episodes = []
        self.episode_index = {}

    async def store(self, information: Dict[str, Any]) -> str:
        """存储情景记忆"""
        import uuid

        episode_id = str(uuid.uuid4())
        episode = {
            'id': episode_id,
            'information': information,
            'timestamp': time.time(),
            'emotional_weight': self._calculate_emotional_weight(information)
        }

        self.episodes.append(episode)
        self._update_index(episode)

        return episode_id

    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检索情景记忆"""
        results = []

        # 简化的检索逻辑
        for episode in self.episodes:
            if self._match_query(episode['information'], query):
                results.append(episode)

        # 按时间倒序排列
        results.sort(key=lambda x: x['timestamp'], reverse=True)

        return results[:10]  # 返回最近10个相关情景

    def _calculate_emotional_weight(self, information: Dict[str, Any]) -> float:
        """计算情感权重"""
        # 简化实现
        return information.get('importance', 0.5)

    def _update_index(self, episode: Dict[str, Any]):
        """更新索引"""
        # 简化实现：按关键词索引
        keywords = self._extract_keywords(episode['information'])

        for keyword in keywords:
            if keyword not in self.episode_index:
                self.episode_index[keyword] = []
            self.episode_index[keyword].append(episode['id'])

    def _extract_keywords(self, information: Dict[str, Any]) -> List[str]:
        """提取关键词"""
        # 简化实现
        text = str(information)
        return text.lower().split()[:5]  # 取前5个词作为关键词

    def _match_query(self, information: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """匹配查询"""
        # 简化的匹配逻辑
        query_text = str(query).lower()
        info_text = str(information).lower()
        return query_text in info_text

class SemanticMemory:
    """语义记忆"""

    def __init__(self):
        self.facts = {}
        self.concepts = {}

    async def store(self, information: Dict[str, Any]) -> str:
        """存储语义记忆"""
        import uuid

        fact_id = str(uuid.uuid4())

        fact = {
            'id': fact_id,
            'fact': information,
            'timestamp': time.time(),
            'confidence': information.get('confidence', 1.0)
        }

        self.facts[fact_id] = fact

        # 更新概念网络
        self._update_concept_network(fact)

        return fact_id

    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检索语义记忆"""
        results = []

        for fact in self.facts.values():
            if self._semantic_match(fact['fact'], query):
                results.append(fact)

        # 按置信度排序
        results.sort(key=lambda x: x['confidence'], reverse=True)

        return results

    def _semantic_match(self, fact: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """语义匹配"""
        # 简化实现
        fact_text = str(fact).lower()
        query_text = str(query).lower()

        # 检查关键词重叠
        fact_words = set(fact_text.split())
        query_words = set(query_text.split())

        overlap = fact_words & query_words
        return len(overlap) > 0

    def _update_concept_network(self, fact: Dict[str, Any]):
        """更新概念网络"""
        # 简化实现
        words = str(fact['fact']).split()

        for word in words:
            if word not in self.concepts:
                self.concepts[word] = []
            self.concepts[word].append(fact['id'])

class ProceduralMemory:
    """程序性记忆"""

    def __init__(self):
        self.procedures = {}
        self.skills = {}

    async def store(self, information: Dict[str, Any]) -> str:
        """存储程序性记忆"""
        import uuid

        procedure_id = str(uuid.uuid4())

        procedure = {
            'id': procedure_id,
            'procedure': information,
            'timestamp': time.time(),
            'success_rate': 1.0,
            'usage_count': 0
        }

        self.procedures[procedure_id] = procedure
        self._update_skills(procedure)

        return procedure_id

    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检索程序性记忆"""
        results = []

        for procedure in self.procedures.values():
            if self._procedure_match(procedure['procedure'], query):
                results.append(procedure)

        # 按成功率排序
        results.sort(key=lambda x: x['success_rate'], reverse=True)

        return results

    def _procedure_match(self, procedure: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """匹配程序"""
        # 简化实现
        procedure_type = procedure.get('type', '')
        query_type = query.get('type', '')

        return procedure_type == query_type

    def _update_skills(self, procedure: Dict[str, Any]):
        """更新技能库"""
        skill_type = procedure['procedure'].get('type', 'general')

        if skill_type not in self.skills:
            self.skills[skill_type] = []

        self.skills[skill_type].append(procedure['id'])
```

## 📊 Agent架构示例

### 完整的对话Agent
```python
class ConversationalAgent(BaseAgent):
    """对话Agent"""

    def _init_perception(self):
        """初始化感知系统"""
        perception_system = MultiModalPerceptionSystem()

        # 注册文本感知
        perception_system.register_sensor('text', TextPerception())

        # 注册图像感知
        perception_system.register_sensor('image', ImagePerception())

        return perception_system

    def _init_reasoning(self):
        """初始化推理引擎"""
        # 混合推理引擎
        self.rule_engine = RuleBasedReasoningEngine()
        self.neural_engine = NeuralReasoningEngine()

        # 添加基础规则
        self.rule_engine.add_rule(
            condition={'conversation': True},
            conclusion={'type': 'conversation'},
            confidence=0.9
        )

        return self.rule_engine

    def _init_planning(self):
        """初始化规划系统"""
        return HierarchicalPlanner()

    def _init_execution(self):
        """初始化执行系统"""
        execution_engine = ExecutionEngine(max_workers=4)

        # 注册工具
        execution_engine.tool_registry.register_tool('analyze_intent', IntentAnalysisTool())
        execution_engine.tool_registry.register_tool('search_knowledge', SearchTool())
        execution_engine.tool_registry.register_tool('generate_response', GenerateResponseTool())
        execution_engine.tool_registry.register_tool('update_memory', MemoryUpdateTool())

        return execution_engine

    def _init_memory(self):
        """初始化记忆系统"""
        return LayeredMemorySystem()

    async def handle_conversation(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """处理对话"""
        # 设置目标
        goal = {
            'type': 'conversation',
            'user_input': user_input,
            'context': context or {},
            'priority': 1.0
        }
        self.set_goal(goal)

        # 执行感知-推理-规划-执行循环
        environment_data = {
            'text': user_input,
            'context': context or {}
        }

        result = await self.perceive_reason_plan_act(environment_data)

        if result['success']:
            # 提取响应
            for action_result in result['results']:
                if action_result['success']:
                    if action_result['action'].action_type == 'generate_response':
                        response = action_result['result']['response']

                        # 存储到记忆
                        await self.memory_system.store({
                            'type': 'conversation',
                            'user_input': user_input,
                            'agent_response': response,
                            'timestamp': time.time()
                        }, 'episodic')

                        return response

        return "抱歉，我无法处理您的请求。"

# 使用示例
async def test_conversational_agent():
    """测试对话Agent"""
    agent = ConversationalAgent("chat_agent")

    # 处理对话
    response1 = await agent.handle_conversation("你好，请介绍一下你自己")
    print(f"用户: 你好，请介绍一下你自己")
    print(f"Agent: {response1}")

    response2 = await agent.handle_conversation("什么是人工智能？")
    print(f"用户: 什么是人工智能？")
    print(f"Agent: {response2}")

    # 处理带上下文的对话
    response3 = await agent.handle_conversation(
        "那深度学习呢？",
        context={'previous_topic': 'artificial_intelligence'}
    )
    print(f"用户: 那深度学习呢？")
    print(f"Agent: {response3}")

# 运行示例
# asyncio.run(test_conversational_agent())
```

## 📝 总结

Agent架构设计是构建智能系统的核心，本文档介绍了从基础架构到完整实现的各个方面。

### 🎯 关键要点
- **分层架构**: 感知-推理-规划-执行的完整循环
- **多模态感知**: 支持文本、图像等多种输入
- **混合推理**: 结合规则和神经网络的优势
- **记忆系统**: 分层记忆架构支持长期学习
- **规划优化**: 智能的任务规划和执行优化

### 🚀 下一步
- 学习[记忆系统设计](02-记忆系统.md)
- 了解[工具调用系统](03-工具调用.md)
- 掌握[RAG集成Agent](04-RAG系统.md)
- 探索[上下文管理](05-上下文管理.md)
