"""端到端测试：示例脚本"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from contextlib import contextmanager
from io import StringIO
from typing import Iterator

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@contextmanager
def captured_output() -> Iterator[StringIO]:
    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer
    try:
        yield buffer
    finally:
        sys.stdout = old_stdout


def _import_module(path: str):
    spec = importlib.util.spec_from_file_location("temp_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


@pytest.mark.e2e
def test_quickstart_script_runs_successfully():
    module = _import_module(os.path.join(PROJECT_ROOT, "examples", "quickstart.py"))
    assert hasattr(module, "main"), "quickstart.py 应定义 main()"
    with captured_output() as buf:
        asyncio.run(module.main())
    output = buf.getvalue()
    assert "IntelligentAgent Quickstart" in output
    assert "Agent 状态" in output


@pytest.mark.e2e
def test_agent_usage_example_outputs_statistics():
    module = _import_module(os.path.join(PROJECT_ROOT, "examples", "agent_usage_example.py"))
    assert hasattr(module, "main"), "agent_usage_example.py 应定义 main()"
    with captured_output() as buf:
        asyncio.run(module.main())
    output = buf.getvalue()
    assert "记忆系统演示" in output
    assert "工具系统演示" in output
    assert "Agent 状态" in output

