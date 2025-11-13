"""
多模态Agent实现

支持文本、图像、音频、视频等多种模态的智能Agent。
"""

import asyncio
import base64
import io
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .intelligent_agent import IntelligentAgent
from ..core.interfaces import BaseAgent, BaseLLM, BaseVisionModel, BaseAudioModel
from ..core.types import (
    AgentResponse, Message, ModalityType,
    MultimodalContent, AgentConfig
)
from ..core.exceptions import AgentError, ModalityError
from ..utils.logger import get_logger


class ProcessingMode(Enum):
    """处理模式"""
    TEXT_ONLY = "text_only"
    VISION_ONLY = "vision_only"
    AUDIO_ONLY = "audio_only"
    MULTI_MODAL = "multi_modal"
    SEQUENTIAL = "sequential"


@dataclass
class MultimodalMessage(Message):
    """多模态消息"""
    content: List[MultimodalContent] = field(default_factory=list)
    primary_modality: ModalityType = ModalityType.TEXT

    def __post_init__(self):
        if isinstance(self.content, str):
            # 兼容单模态文本消息
            self.content = [MultimodalContent(
                type=ModalityType.TEXT,
                data=self.content
            )]
            self.primary_modality = ModalityType.TEXT

    def get_text_content(self) -> Optional[str]:
        """获取文本内容"""
        for content_item in self.content:
            if content_item.type == ModalityType.TEXT:
                return content_item.data
        return None

    def get_image_content(self) -> Optional[Any]:
        """获取图像内容"""
        for content_item in self.content:
            if content_item.type == ModalityType.IMAGE:
                return content_item.data
        return None

    def get_audio_content(self) -> Optional[Any]:
        """获取音频内容"""
        for content_item in self.content:
            if content_item.type == ModalityType.AUDIO:
                return content_item.data
        return None


class MultiModalAgent(IntelligentAgent):
    """
    多模态智能Agent

    扩展了基础Agent，支持：
    - 图像理解和分析
    - 语音识别和合成
    - 视频处理
    - 多模态融合推理
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.logger = get_logger(f"multimodal_agent.{config.name}")

        # 多模态组件
        self._vision_model: Optional[BaseVisionModel] = None
        self._audio_model: Optional[BaseAudioModel] = None

        # 配置
        self._vision_enabled = config.get("vision_enabled", False)
        self._audio_enabled = config.get("audio_enabled", False)
        self._fusion_enabled = config.get("fusion_enabled", True)

        # 多模态处理统计
        self._multimodal_stats = {
            "image_processed": 0,
            "audio_processed": 0,
            "video_processed": 0,
            "fusion_inferences": 0,
            "modality_errors": 0
        }

    async def initialize(self) -> None:
        """初始化多模态Agent"""
        try:
            # 先初始化基础组件
            await super().initialize()

            # 初始化视觉模型
            if self._vision_enabled:
                await self._initialize_vision_model()

            # 初始化音频模型
            if self._audio_enabled:
                await self._initialize_audio_model()

            self.logger.info("多模态Agent初始化完成")

        except Exception as e:
            self.logger.error(f"多模态Agent初始化失败: {e}")
            raise

    async def _initialize_vision_model(self) -> None:
        """初始化视觉模型"""
        from ..llm.vision import VisionModelFactory

        vision_config = self.config.get("vision_config", {})
        self._vision_model = await VisionModelFactory.create_model(vision_config)
        await self._vision_model.initialize()
        self.logger.info("视觉模型初始化完成")

    async def _initialize_audio_model(self) -> None:
        """初始化音频模型"""
        from ..llm.audio import AudioModelFactory

        audio_config = self.config.get("audio_config", {})
        self._audio_model = await AudioModelFactory.create_model(audio_config)
        await self._audio_model.initialize()
        self.logger.info("音频模型初始化完成")

    async def process_multimodal_message(
        self,
        message: Union[MultimodalMessage, Dict[str, Any]],
        conversation_id: Optional[str] = None,
        processing_mode: ProcessingMode = ProcessingMode.MULTI_MODAL,
        **kwargs
    ) -> AgentResponse:
        """
        处理多模态消息

        Args:
            message: 多模态消息
            conversation_id: 对话ID
            processing_mode: 处理模式
            **kwargs: 额外参数

        Returns:
            AgentResponse: Agent响应
        """
        start_time = time.time()

        try:
            # 消息标准化
            if isinstance(message, dict):
                multimodal_message = MultimodalMessage(
                    content=message.get("content", []),
                    role=message.get("role", "user"),
                    timestamp=message.get("timestamp", time.time())
                )
            else:
                multimodal_message = message

            # 确定主要模态
            if not multimodal_message.content:
                raise ModalityError("消息内容为空")

            primary_modality = self._determine_primary_modality(multimodal_message)

            # 处理不同模态内容
            processed_content = await self._process_modalities(
                multimodal_message, processing_mode
            )

            # 构建增强提示
            enhanced_prompt = await self._build_multimodal_prompt(
                processed_content, primary_modality
            )

            # LLM推理
            llm_response = await self._llm.generate(enhanced_prompt)

            # 处理工具调用（如果需要）
            if llm_response.tool_calls:
                tool_results = await self._execute_tool_calls(llm_response.tool_calls)
                final_response = await self._process_tool_results(
                    enhanced_prompt, tool_results
                )
            else:
                final_response = llm_response

            # 构建响应
            response = AgentResponse(
                content=final_response.content,
                conversation_id=conversation_id or self._generate_conversation_id(),
                message_id=self._generate_message_id(),
                reasoning=final_response.reasoning,
                confidence=final_response.confidence,
                metadata={
                    "processing_time": time.time() - start_time,
                    "primary_modality": primary_modality.value,
                    "processed_modalities": [content.type.value for content in multimodal_message.content],
                    "processing_mode": processing_mode.value,
                    "tools_used": final_response.tool_calls
                }
            )

            # 更新统计
            self._update_multimodal_stats(multimodal_message, processing_mode)

            return response

        except Exception as e:
            self._multimodal_stats["modality_errors"] += 1
            self.logger.error(f"多模态消息处理失败: {e}")
            raise AgentError(f"多模态消息处理失败: {e}")

    def _determine_primary_modality(self, message: MultimodalMessage) -> ModalityType:
        """确定主要模态"""
        modality_counts = {}
        for content in message.content:
            modality_counts[content.type] = modality_counts.get(content.type, 0) + 1

        # 优先级：图像 > 音频 > 视频 > 文本
        priority_order = [
            ModalityType.IMAGE,
            ModalityType.AUDIO,
            ModalityType.VIDEO,
            ModalityType.TEXT
        ]

        for modality in priority_order:
            if modality in modality_counts:
                return modality

        return ModalityType.TEXT

    async def _process_modalities(
        self,
        message: MultimodalMessage,
        processing_mode: ProcessingMode
    ) -> Dict[str, Any]:
        """处理各种模态内容"""
        processed = {
            "text": [],
            "image_descriptions": [],
            "audio_transcripts": [],
            "video_summaries": [],
            "metadata": {}
        }

        tasks = []

        for content in message.content:
            if content.type == ModalityType.TEXT:
                processed["text"].append(content.data)

            elif content.type == ModalityType.IMAGE and self._vision_model:
                tasks.append(self._process_image(content.data))

            elif content.type == ModalityType.AUDIO and self._audio_model:
                tasks.append(self._process_audio(content.data))

            elif content.type == ModalityType.VIDEO:
                tasks.append(self._process_video(content.data))

        # 并行处理非文本模态
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"模态处理失败: {result}")
                    continue

                if result["type"] == "image":
                    processed["image_descriptions"].append(result["description"])
                elif result["type"] == "audio":
                    processed["audio_transcripts"].append(result["transcript"])
                elif result["type"] == "video":
                    processed["video_summaries"].append(result["summary"])

        return processed

    async def _process_image(self, image_data: Any) -> Dict[str, Any]:
        """处理图像内容"""
        try:
            description = await self._vision_model.analyze_image(image_data)
            self._multimodal_stats["image_processed"] += 1

            return {
                "type": "image",
                "description": description,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"图像处理失败: {e}")
            return {
                "type": "image",
                "description": "图像处理失败",
                "error": str(e)
            }

    async def _process_audio(self, audio_data: Any) -> Dict[str, Any]:
        """处理音频内容"""
        try:
            transcript = await self._audio_model.transcribe(audio_data)
            self._multimodal_stats["audio_processed"] += 1

            return {
                "type": "audio",
                "transcript": transcript,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"音频处理失败: {e}")
            return {
                "type": "audio",
                "transcript": "音频转写失败",
                "error": str(e)
            }

    async def _process_video(self, video_data: Any) -> Dict[str, Any]:
        """处理视频内容"""
        try:
            # 视频处理通常包括：
            # 1. 提取关键帧
            # 2. 分析关键帧图像
            # 3. 提取音频并转写
            # 4. 生成视频摘要

            # 这里简化处理
            summary = await self._extract_video_summary(video_data)
            self._multimodal_stats["video_processed"] += 1

            return {
                "type": "video",
                "summary": summary,
                "timestamp": time.time()
            }

        except Exception as e:
            self.logger.error(f"视频处理失败: {e}")
            return {
                "type": "video",
                "summary": "视频处理失败",
                "error": str(e)
            }

    async def _extract_video_summary(self, video_data: Any) -> str:
        """提取视频摘要"""
        # 简化实现，实际应用中需要更复杂的视频分析
        if self._vision_model and self._audio_model:
            # 提取关键帧并分析
            # 提取音频并转写
            # 综合生成摘要
            pass

        return "视频内容摘要（需要进一步分析）"

    async def _build_multimodal_prompt(
        self,
        processed_content: Dict[str, Any],
        primary_modality: ModalityType
    ) -> str:
        """构建多模态提示"""
        prompt_parts = []

        # 添加文本内容
        if processed_content["text"]:
            text_content = "\n".join(processed_content["text"])
            prompt_parts.append(f"文本内容:\n{text_content}")

        # 添加图像描述
        if processed_content["image_descriptions"]:
            image_content = "\n".join([
                f"图像{i+1}: {desc}"
                for i, desc in enumerate(processed_content["image_descriptions"])
            ])
            prompt_parts.append(f"图像描述:\n{image_content}")

        # 添加音频转录
        if processed_content["audio_transcripts"]:
            audio_content = "\n".join([
                f"音频{i+1}: {transcript}"
                for i, transcript in enumerate(processed_content["audio_transcripts"])
            ])
            prompt_parts.append(f"音频转录:\n{audio_content}")

        # 添加视频摘要
        if processed_content["video_summaries"]:
            video_content = "\n".join([
                f"视频{i+1}: {summary}"
                for i, summary in enumerate(processed_content["video_summaries"])
            ])
            prompt_parts.append(f"视频摘要:\n{video_content}")

        # 添加模态融合指导
        if self._fusion_enabled and len([k for k, v in processed_content.items() if v]) > 1:
            fusion_guidance = f"""
请注意这是一个多模态输入，主要模态是{primary_modality.value}。
请综合考虑各种模态的信息，给出融合的理解和回答。
"""
            prompt_parts.append(fusion_guidance)

        return "\n\n".join(prompt_parts)

    def _update_multimodal_stats(
        self,
        message: MultimodalMessage,
        processing_mode: ProcessingMode
    ) -> None:
        """更新多模态统计"""
        if processing_mode == ProcessingMode.MULTI_MODAL:
            self._multimodal_stats["fusion_inferences"] += 1

    async def analyze_image(self, image_data: Any, **kwargs) -> Dict[str, Any]:
        """分析图像"""
        if not self._vision_model:
            raise ModalityError("视觉模型未启用")

        return await self._vision_model.analyze_image(image_data, **kwargs)

    async def transcribe_audio(self, audio_data: Any, **kwargs) -> str:
        """转写音频"""
        if not self._audio_model:
            raise ModalityError("音频模型未启用")

        return await self._audio_model.transcribe(audio_data, **kwargs)

    async def generate_speech(self, text: str, **kwargs) -> bytes:
        """合成语音"""
        if not self._audio_model or not hasattr(self._audio_model, 'synthesize'):
            raise ModalityError("语音合成功能未启用")

        return await self._audio_model.synthesize(text, **kwargs)

    def get_multimodal_stats(self) -> Dict[str, Any]:
        """获取多模态处理统计"""
        base_stats = self.get_stats()
        return {
            **base_stats,
            "multimodal": self._multimodal_stats,
            "capabilities": {
                "vision_enabled": self._vision_enabled,
                "audio_enabled": self._audio_enabled,
                "fusion_enabled": self._fusion_enabled
            }
        }

    async def cleanup(self) -> None:
        """清理多模态资源"""
        await super().cleanup()

        if self._vision_model:
            await self._vision_model.cleanup()

        if self._audio_model:
            await self._audio_model.cleanup()

        self.logger.info("多模态Agent资源清理完成")
