"""MemorySystem 单元测试"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timedelta

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.memory import MemorySystem  # noqa: E402
from src.core.types import MemoryConfig, MemoryType  # noqa: E402


@pytest.mark.asyncio
async def test_add_and_retrieve_with_metadata():
    memory = MemorySystem(MemoryConfig(max_memories=10, similarity_threshold=0.0))
    await memory.add_memory(
        "用户喜欢意大利面",
        metadata={"category": "preference", "role": "user"},
        memory_type=MemoryType.SEMANTIC,
    )
    await memory.add_memory(
        "助手推荐多喝水",
        metadata={"category": "advice", "role": "assistant"},
        memory_type=MemoryType.PROCEDURAL,
    )

    results = await memory.retrieve("意大利")
    assert len(results) == 1
    assert results[0].metadata["category"] == "preference"
    assert results[0].memory_type == MemoryType.SEMANTIC

    by_metadata = await memory.retrieve_by_metadata("role", "assistant")
    assert len(by_metadata) == 1
    assert by_metadata[0].metadata["category"] == "advice"


@pytest.mark.asyncio
async def test_bulk_add_and_stats():
    memory = MemorySystem(MemoryConfig(max_memories=5))
    entries = [
        ("记录一", 1.0, {"tag": "a"}, MemoryType.EPISODIC),
        ("记录二", 2.0, {"tag": "b"}, MemoryType.SEMANTIC),
        ("记录三", None, {"tag": "c"}, MemoryType.WORKING),
    ]
    ids = await memory.bulk_add(entries)
    assert len(ids) == 3

    stats = await memory.stats()
    assert stats["total_memories"] == 3
    assert stats["per_type"]["semantic"] == 1


@pytest.mark.asyncio
async def test_cleanup_expired_and_low_importance():
    config = MemoryConfig(max_memories=10, cleanup_interval=1)
    memory = MemorySystem(config)

    mem_id_recent = await memory.add_memory("近期访问", importance=1.0)
    mem_id_old = await memory.add_memory("过期数据", importance=0.05)

    # 手动调整访问时间以模拟过期
    async with memory._lock:  # type: ignore[attr-defined]
        memory._store[mem_id_old].last_accessed = datetime.now() - timedelta(hours=1)

    removed = await memory.cleanup()
    assert removed == 1

    remaining = await memory.retrieve("近期")
    assert remaining and remaining[0].id == mem_id_recent

