# 多模态Agent API文档

## 概览

多模态Agent API提供了处理文本、图像、音频、视频等多种模态数据的完整接口。支持单模态和多模态融合处理。

## 核心类

### MultiModalAgent

多模态智能Agent类，继承自IntelligentAgent，扩展了多模态处理能力。

#### 构造函数

```python
MultiModalAgent(config: AgentConfig) -> MultiModalAgent
```

**参数：**
- `config` (AgentConfig): Agent配置对象

**返回：**
- `MultiModalAgent`: 多模态Agent实例

**示例：**
```python
from src.agents.multi_modal_agent import MultiModalAgent
from src.core.types import AgentConfig, LLMConfig

# 创建配置
config = AgentConfig(
    name="multimodal_agent",
    llm_config=LLMConfig(
        model_name="gpt-4-vision-preview",
        api_key="your-api-key"
    ),
    vision_enabled=True,
    audio_enabled=True,
    fusion_enabled=True
)

# 创建Agent
agent = MultiModalAgent(config)
await agent.initialize()
```

#### 主要方法

##### process_multimodal_message

处理多模态消息的核心方法。

```python
async process_multimodal_message(
    message: Union[MultimodalMessage, Dict[str, Any]],
    conversation_id: Optional[str] = None,
    processing_mode: ProcessingMode = ProcessingMode.MULTI_MODAL,
    **kwargs
) -> AgentResponse
```

**参数：**
- `message` (MultimodalMessage | Dict): 多模态消息
- `conversation_id` (str, 可选): 对话ID
- `processing_mode` (ProcessingMode): 处理模式
- `**kwargs`: 额外参数

**返回：**
- `AgentResponse`: Agent响应结果

**处理模式：**
- `ProcessingMode.TEXT_ONLY`: 仅处理文本
- `ProcessingMode.VISION_ONLY`: 仅处理图像
- `ProcessingMode.AUDIO_ONLY`: 仅处理音频
- `ProcessingMode.MULTI_MODAL`: 多模态融合处理
- `ProcessingMode.SEQUENTIAL`: 顺序处理

**示例：**
```python
from src.agents.multi_modal_agent import MultimodalMessage, MultimodalContent, ModalityType

# 创建多模态消息
message = MultimodalMessage(
    content=[
        MultimodalContent(
            type=ModalityType.TEXT,
            data="请分析这张图片"
        ),
        MultimodalContent(
            type=ModalityType.IMAGE,
            data=image_bytes  # 图像字节数据
        )
    ],
    role="user",
    timestamp=time.time()
)

# 处理消息
response = await agent.process_multimodal_message(
    message,
    conversation_id="conversation_123",
    processing_mode=ProcessingMode.MULTI_MODAL
)

print(f"响应: {response.content}")
print(f"主要模态: {response.metadata['primary_modality']}")
```

##### analyze_image

分析图像内容。

```python
async analyze_image(image_data: Any, **kwargs) -> Dict[str, Any]
```

**参数：**
- `image_data` (Any): 图像数据（bytes、PIL Image、numpy数组等）
- `**kwargs`: 分析参数

**返回：**
- `Dict[str, Any]`: 图像分析结果

**示例：**
```python
# 分析本地图像
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

result = await agent.analyze_image(
    image_bytes,
    detail="high",  # 高精度分析
    language="zh"   # 中文描述
)

print(f"图像描述: {result['description']}")
print(f"置信度: {result['confidence']}")
```

##### transcribe_audio

转录音频为文本。

```python
async transcribe_audio(audio_data: Any, **kwargs) -> str
```

**参数：**
- `audio_data` (Any): 音频数据
- `**kwargs`: 转录参数

**返回：**
- `str`: 转录文本

**示例：**
```python
# 转录音频文件
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()

transcript = await agent.transcribe_audio(
    audio_bytes,
    language="zh-CN",  # 中文
    model="whisper-1"
)

print(f"转录结果: {transcript}")
```

##### generate_speech

合成语音。

```python
async generate_speech(text: str, **kwargs) -> bytes
```

**参数：**
- `text` (str): 要合成的文本
- `**kwargs`: 语音合成参数

**返回：**
- `bytes`: 语音音频数据

**示例：**
```python
# 合成语音
audio_data = await agent.generate_speech(
    "你好，欢迎使用多模态Agent系统",
    voice="nova",      # 语音类型
    speed=1.0,         # 语速
    format="mp3"       # 音频格式
)

# 保存音频文件
with open("output.mp3", "wb") as f:
    f.write(audio_data)
```

## 数据类型

### MultimodalMessage

多模态消息类。

```python
@dataclass
class MultimodalMessage(Message):
    content: List[MultimodalContent] = field(default_factory=list)
    primary_modality: ModalityType = ModalityType.TEXT
```

**属性：**
- `content` (List[MultimodalContent]): 模态内容列表
- `primary_modality` (ModalityType): 主要模态类型

**方法：**
- `get_text_content()`: 获取文本内容
- `get_image_content()`: 获取图像内容
- `get_audio_content()`: 获取音频内容

**示例：**
```python
# 创建多模态消息
message = MultimodalMessage(
    content=[
        MultimodalContent(type=ModalityType.TEXT, data="文本内容"),
        MultimodalContent(type=ModalityType.IMAGE, data=image_data),
        MultimodalContent(type=ModalityType.AUDIO, data=audio_data)
    ]
)

# 获取特定模态内容
text = message.get_text_content()
image = message.get_image_content()
audio = message.get_audio_content()
```

### MultimodalContent

多模态内容类。

```python
@dataclass
class MultimodalContent:
    type: ModalityType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**属性：**
- `type` (ModalityType): 模态类型
- `data` (Any): 内容数据
- `metadata` (Dict[str, Any]): 元数据

**模态类型：**
```python
class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
```

**示例：**
```python
# 创建文本内容
text_content = MultimodalContent(
    type=ModalityType.TEXT,
    data="这是一段文本",
    metadata={"language": "zh", "encoding": "utf-8"}
)

# 创建图像内容
image_content = MultimodalContent(
    type=ModalityType.IMAGE,
    data=image_bytes,
    metadata={
        "format": "jpeg",
        "size": (1920, 1080),
        "color_space": "RGB"
    }
)

# 创建音频内容
audio_content = MultimodalContent(
    type=ModalityType.AUDIO,
    data=audio_bytes,
    metadata={
        "format": "mp3",
        "sample_rate": 44100,
        "duration": 30.5
    }
)
```

### ProcessingMode

处理模式枚举。

```python
class ProcessingMode(Enum):
    TEXT_ONLY = "text_only"
    VISION_ONLY = "vision_only"
    AUDIO_ONLY = "audio_only"
    MULTI_MODAL = "multi_modal"
    SEQUENTIAL = "sequential"
```

## 使用场景

### 场景1：图像分析

```python
# 用户上传图片并询问
async def analyze_user_image(image_path: str, question: str):
    with open(image_path, "rb") as f:
        image_data = f.read()

    message = MultimodalMessage(
        content=[
            MultimodalContent(type=ModalityType.TEXT, data=question),
            MultimodalContent(type=ModalityType.IMAGE, data=image_data)
        ]
    )

    response = await agent.process_multimodal_message(message)
    return response.content

# 使用示例
result = await analyze_user_image(
    "sunset.jpg",
    "这张图片展示了什么？描述一下色彩和构图。"
)
print(result)
```

### 场景2：语音对话

```python
# 语音转文本然后处理
async def voice_conversation(audio_path: str):
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # 转录语音
    transcript = await agent.transcribe_audio(audio_data)

    # 处理文本
    response = await agent.process_message(transcript)

    # 合成回复语音
    audio_response = await agent.generate_speech(response.content)

    return {
        "transcript": transcript,
        "text_response": response.content,
        "audio_response": audio_response
    }

# 使用示例
result = await voice_conversation("user_question.mp3")
print(f"用户说: {result['transcript']}")
print(f"AI回复: {result['text_response']}")
```

### 场景3：多模态学习助手

```python
# 处理包含图像、文本的教材
async def study_assistant(textbook_text: str, image_path: str):
    with open(image_path, "rb") as f:
        image_data = f.read()

    message = MultimodalMessage(
        content=[
            MultimodalContent(
                type=ModalityType.TEXT,
                data=f"教材内容: {textbook_text}\n\n请结合图片进行讲解。"
            ),
            MultimodalContent(type=ModalityType.IMAGE, data=image_data)
        ]
    )

    response = await agent.process_multimodal_message(
        message,
        processing_mode=ProcessingMode.MULTI_MODAL
    )

    return response.content

# 使用示例
explanation = await study_assistant(
    "光合作用是植物将光能转化为化学能的过程...",
    "photosynthesis_diagram.jpg"
)
print(explanation)
```

### 场景4：视频内容分析

```python
async def analyze_video_content(video_path: str):
    # 提取关键帧和音频（简化示例）
    key_frames = extract_key_frames(video_path)  # 自定义函数
    audio_track = extract_audio(video_path)     # 自定义函数

    multimodal_contents = [
        MultimodalContent(
            type=ModalityType.TEXT,
            data="请分析这个视频的内容和主要观点"
        )
    ]

    # 添加关键帧
    for i, frame in enumerate(key_frames):
        multimodal_contents.append(
            MultimodalContent(
                type=ModalityType.IMAGE,
                data=frame,
                metadata={"frame_number": i, "timestamp": i * 2.0}
            )
        )

    # 添加音频
    multimodal_contents.append(
        MultimodalContent(
            type=ModalityType.AUDIO,
            data=audio_track
        )
    )

    message = MultimodalMessage(content=multimodal_contents)

    response = await agent.process_multimodal_message(
        message,
        processing_mode=ProcessingMode.SEQUENTIAL
    )

    return response.content

# 使用示例
analysis = await analyze_video_content("lecture_video.mp4")
print(analysis)
```

## 配置选项

### 视觉配置

```python
vision_config = {
    "provider": "openai",           # 视觉模型提供商
    "model": "gpt-4-vision-preview", # 视觉模型
    "max_tokens": 1000,              # 最大生成token数
    "detail": "high",                # 图像分析精度: "low" | "high"
    "temperature": 0.7,              # 生成温度
    "timeout": 30.0                  # 超时时间
}
```

### 音频配置

```python
audio_config = {
    "provider": "openai",            # 音频模型提供商
    "transcription_model": "whisper-1",  # 转录模型
    "tts_model": "tts-1",            # 语音合成模型
    "voice": "nova",                 # 默认语音
    "speed": 1.0,                    # 语速
    "format": "mp3",                 # 音频格式
    "language": "zh-CN"              # 默认语言
}
```

### 多模态融合配置

```python
fusion_config = {
    "strategy": "attention",          # 融合策略
    "weights": {                     # 模态权重
        "text": 0.4,
        "image": 0.4,
        "audio": 0.2
    },
    "fusion_layer": "late",          # 融合层: "early" | "late" | "intermediate"
    "cross_modal_attention": True,    # 启用跨模态注意力
    "modality_dropout": 0.1          # 模态dropout率
}
```

## 错误处理

### 常见错误类型

```python
from src.core.exceptions import ModalityError, AgentError

try:
    response = await agent.process_multimodal_message(message)
except ModalityError as e:
    print(f"模态处理错误: {e}")
except AgentError as e:
    print(f"Agent错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 错误响应处理

```python
response = await agent.process_multimodal_message(message)

if response.error:
    print(f"处理出错: {response.error}")
    # 根据错误类型采取不同处理
    if "vision" in response.error.lower():
        # 图像处理错误
        fallback_response = await agent.process_message(
            "仅基于文本处理您的问题"
        )
    else:
        # 其他错误
        fallback_response = response
```

## 性能优化

### 并发处理

```python
import asyncio

async def batch_process_multimodal(messages: List[MultimodalMessage]):
    tasks = [
        agent.process_multimodal_message(msg)
        for msg in messages
    ]

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理结果和异常
    successful_responses = []
    errors = []

    for i, result in enumerate(responses):
        if isinstance(result, Exception):
            errors.append((i, result))
        else:
            successful_responses.append((i, result))

    return successful_responses, errors
```

### 缓存优化

```python
from functools import lru_cache
import hashlib

def get_content_hash(content: bytes) -> str:
    return hashlib.md5(content).hexdigest()

@lru_cache(maxsize=100)
async def cached_image_analysis(image_hash: str, image_data: bytes):
    return await agent.analyze_image(image_data)

async def optimized_image_processing(image_data: bytes):
    image_hash = get_content_hash(image_data)
    return await cached_image_analysis(image_hash, image_data)
```

## 最佳实践

### 1. 模态选择策略

```python
def determine_processing_mode(message: MultimodalMessage) -> ProcessingMode:
    content_types = [content.type for content in message.content]

    if len(content_types) == 1:
        if content_types[0] == ModalityType.TEXT:
            return ProcessingMode.TEXT_ONLY
        elif content_types[0] == ModalityType.IMAGE:
            return ProcessingMode.VISION_ONLY
        elif content_types[0] == ModalityType.AUDIO:
            return ProcessingMode.AUDIO_ONLY
    else:
        # 多模态内容，根据复杂度选择策略
        if len(content_types) <= 2:
            return ProcessingMode.MULTI_MODAL
        else:
            return ProcessingMode.SEQUENTIAL

    return ProcessingMode.MULTI_MODAL
```

### 2. 内容预处理

```python
async def preprocess_multimodal_content(
    content: MultimodalContent
) -> MultimodalContent:
    if content.type == ModalityType.IMAGE:
        # 图像预处理
        processed_data = preprocess_image(content.data)
        return MultimodalContent(
            type=content.type,
            data=processed_data,
            metadata=content.metadata
        )
    elif content.type == ModalityType.AUDIO:
        # 音频预处理
        processed_data = preprocess_audio(content.data)
        return MultimodalContent(
            type=content.type,
            data=processed_data,
            metadata=content.metadata
        )
    else:
        return content
```

### 3. 响应质量评估

```python
def evaluate_response_quality(
    response: AgentResponse,
    original_message: MultimodalMessage
) -> Dict[str, Any]:
    quality_metrics = {
        "confidence": response.confidence,
        "modality_coverage": len(set([
            content.type for content in original_message.content
        ])),
        "processing_time": response.metadata.get("processing_time", 0),
        "has_error": bool(response.error)
    }

    # 综合质量评分
    quality_score = (
        quality_metrics["confidence"] * 0.4 +
        (quality_metrics["modality_coverage"] / 4) * 0.3 +
        min(1.0, 5.0 / quality_metrics["processing_time"]) * 0.2 +
        (0 if quality_metrics["has_error"] else 1) * 0.1
    )

    quality_metrics["overall_score"] = quality_score
    return quality_metrics
```

### 4. 资源管理

```python
import psutil

class MultimodalResourceManager:
    def __init__(self):
        self.max_memory_usage = 0.8  # 最大内存使用率
        self.max_cpu_usage = 0.9     # 最大CPU使用率

    async def check_resources(self) -> bool:
        memory_percent = psutil.virtual_memory().percent / 100
        cpu_percent = psutil.cpu_percent(interval=1) / 100

        return (
            memory_percent < self.max_memory_usage and
            cpu_percent < self.max_cpu_usage
        )

    async def optimize_processing(self, message: MultimodalMessage) -> ProcessingMode:
        if await self.check_resources():
            return ProcessingMode.MULTI_MODAL
        else:
            # 资源不足时使用顺序处理
            return ProcessingMode.SEQUENTIAL
```

## 监控和日志

### 多模态处理监控

```python
import logging
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MultimodalMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    modality_counts: Dict[str, int] = None
    average_processing_time: float = 0.0

    def __post_init__(self):
        if self.modality_counts is None:
            self.modality_counts = {}

class MultimodalMonitor:
    def __init__(self):
        self.metrics = MultimodalMetrics()
        self.logger = logging.getLogger("multimodal_monitor")

    async def log_request(self, message: MultimodalMessage):
        self.metrics.total_requests += 1

        for content in message.content:
            modality = content.type.value
            self.metrics.modality_counts[modality] = \
                self.metrics.modality_counts.get(modality, 0) + 1

    async def log_response(self, response: AgentResponse):
        if response.error:
            self.metrics.failed_requests += 1
            self.logger.error(f"多模态处理失败: {response.error}")
        else:
            self.metrics.successful_requests += 1
            processing_time = response.metadata.get("processing_time", 0)
            self._update_average_time(processing_time)

    def _update_average_time(self, new_time: float):
        total = self.metrics.successful_requests + self.metrics.failed_requests
        current_avg = self.metrics.average_processing_time
        self.metrics.average_processing_time = (
            (current_avg * (total - 1) + new_time) / total
        )

    def get_metrics(self) -> Dict[str, Any]:
        success_rate = (
            self.metrics.successful_requests / max(1, self.metrics.total_requests)
        )

        return {
            "total_requests": self.metrics.total_requests,
            "success_rate": success_rate,
            "average_processing_time": self.metrics.average_processing_time,
            "modality_distribution": self.metrics.modality_counts
        }
```

## 部署建议

### Docker配置

```dockerfile
# 多模态Agent Dockerfile
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libsndfile1

WORKDIR /app

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "src.multimodal_server"]
```

### 资源需求

| 功能模块 | 最小配置 | 推荐配置 |
|----------|----------|----------|
| 基础Agent | 2GB RAM, 2 CPU | 4GB RAM, 4 CPU |
| 图像处理 | 4GB RAM, 4 CPU | 8GB RAM, 8 CPU |
| 音频处理 | 2GB RAM, 2 CPU | 4GB RAM, 4 CPU |
| 多模态融合 | 8GB RAM, 8 CPU | 16GB RAM, 16 CPU |
| GPU支持 | NVIDIA T4 | NVIDIA A100 |

这个多模态Agent API文档提供了完整的接口说明、使用示例和最佳实践指南，帮助开发者充分利用多模态AI能力。
