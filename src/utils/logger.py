"""
日志记录模块
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }

    def format(self, record):
        # 添加颜色
        level_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"

        # 格式化消息
        formatted = super().format(record)

        # 如果是JSON格式，不额外处理
        if hasattr(record, 'is_json') and record.is_json:
            return formatted

        return formatted


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # 添加额外字段
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)

        # 标记为JSON格式
        record.is_json = True

        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console_output: bool = True
) -> logging.Logger:
    """设置日志系统"""

    # 创建根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)

        if json_format:
            formatter = JSONFormatter()
        else:
            formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """获取日志器"""
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger


class ContextLogger:
    """上下文日志记录器"""

    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context

    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """带上下文记录日志"""
        if 'extra' not in kwargs:
            kwargs['extra'] = {}

        kwargs['extra'].update(self.context)
        self.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self._log_with_context(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


class PerformanceLogger:
    """性能日志记录器"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_function_call(self, func_name: str, execution_time: float,
                         success: bool = True, error: Optional[str] = None):
        """记录函数调用性能"""
        self.logger.info(
            f"Function call: {func_name}",
            extra={
                'function_name': func_name,
                'execution_time': execution_time,
                'success': success,
                'error': error,
                'type': 'performance'
            }
        )

    def log_memory_usage(self, process_name: str, memory_mb: float):
        """记录内存使用情况"""
        self.logger.info(
            f"Memory usage: {process_name} - {memory_mb:.2f} MB",
            extra={
                'process_name': process_name,
                'memory_mb': memory_mb,
                'type': 'memory'
            }
        )

    def log_request_metrics(self, endpoint: str, response_time: float,
                         status_code: int, request_size: int = 0,
                         response_size: int = 0):
        """记录请求指标"""
        self.logger.info(
            f"Request: {endpoint} - {response_time:.3f}s - {status_code}",
            extra={
                'endpoint': endpoint,
                'response_time': response_time,
                'status_code': status_code,
                'request_size': request_size,
                'response_size': response_size,
                'type': 'request'
            }
        )


# 默认日志器设置
if not logging.getLogger().handlers:
    setup_logging(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        log_file=os.getenv('LOG_FILE'),
        json_format=os.getenv('LOG_JSON', 'false').lower() == 'true'
    )
