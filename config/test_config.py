"""
测试配置文件
"""

# 基础测试配置
TEST_CONFIG = {
    # 超时配置（秒）
    "timeout": {
        "unit_test": 10,
        "integration_test": 30,
        "e2e_test": 120,
        "performance_test": 300
    },

    # 性能基准
    "performance": {
        "max_response_time": {
            "average": 2.0,
            "p95": 5.0,
            "p99": 10.0
        },
        "min_throughput": 5.0,  # 每秒请求数
        "max_memory_increase": 100 * 1024 * 1024,  # 100MB
        "max_cpu_usage": 80.0  # 百分比
    },

    # 负载测试配置
    "load_test": {
        "concurrent_users": 10,
        "requests_per_user": 5,
        "test_duration": 30,  # 秒
        "max_error_rate": 0.05  # 5%错误率
    },

    # 压力测试配置
    "stress_test": {
        "max_concurrent_threads": 50,
        "requests_per_thread": 10,
        "min_success_rate": 0.8,  # 80%成功率
        "peak_throughput_min": 20.0  # 最小峰值吞吐量
    },

    # 数据配置
    "data": {
        "max_memory_entries": 10000,
        "max_documents": 5000,
        "max_conversation_history": 1000,
        "context_window_size": 4000  # tokens
    },

    # 模拟配置
    "simulation": {
        "random_seed": 42,
        "delays": {
            "min": 0.1,
            "max": 0.5
        },
        "error_rate": 0.01  # 1%模拟错误率
    }
}

# 测试环境配置
ENVIRONMENT_CONFIG = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///:memory:",
        "cache_enabled": False
    },

    "testing": {
        "debug": False,
        "log_level": "INFO",
        "database_url": "sqlite:///test.db",
        "cache_enabled": True,
        "cache_ttl": 300  # 5分钟
    },

    "production": {
        "debug": False,
        "log_level": "WARNING",
        "database_url": "postgresql://user:pass@localhost/testdb",
        "cache_enabled": True,
        "cache_ttl": 3600  # 1小时
    }
}

# 测试数据配置
TEST_DATA = {
    "sample_conversations": [
        ["你好", "你好！有什么可以帮助您的吗？"],
        ["今天天气怎么样？", "抱歉，我无法实时获取天气信息。"],
        ["谢谢", "不客气！"]
    ],

    "test_memories": [
        "用户名叫张三",
        "用户住在北京",
        "用户是一名程序员",
        "用户喜欢咖啡"
    ],

    "test_documents": [
        ("Python是一种编程语言", {"type": "technology"}),
        ("北京是中国的首都", {"type": "geography"}),
        ("健康饮食很重要", {"type": "health"})
    ],

    "boundary_values": {
        "string_lengths": {
            "empty": "",
            "short": "a",
            "medium": "a" * 100,
            "long": "a" * 1000,
            "very_long": "a" * 10000
        },
        "numbers": {
            "zero": 0,
            "negative": -1,
            "positive": 1,
            "max_int": 2**31 - 1,
            "min_int": -2**31
        }
    }
}

# 监控配置
MONITORING_CONFIG = {
    "metrics": {
        "response_time": {
            "enabled": True,
            "buckets": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        },
        "throughput": {
            "enabled": True,
            "window": 60  # 秒
        },
        "error_rate": {
            "enabled": True,
            "window": 300  # 秒
        },
        "memory_usage": {
            "enabled": True,
            "interval": 10  # 秒
        }
    },

    "alerts": {
        "response_time_threshold": 5.0,  # 秒
        "error_rate_threshold": 0.05,   # 5%
        "memory_threshold": 1024 * 1024 * 1024,  # 1GB
        "cpu_threshold": 90.0  # 90%
    }
}

# 兼容性配置
COMPATIBILITY_CONFIG = {
    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
    "required_packages": {
        "pytest": ">=6.0",
        "psutil": ">=5.0",
        "requests": ">=2.25"
    },
    "optional_packages": {
        "matplotlib": ">=3.0",  # 用于性能图表
        "pandas": ">=1.0",      # 用于数据分析
        "numpy": ">=1.20"       # 用于数值计算
    }
}

# 报告配置
REPORT_CONFIG = {
    "output_directory": "./test_reports",
    "formats": ["html", "json", "xml"],
    "include_performance_charts": True,
    "include_coverage_report": True,
    "retention_days": 30,

    "email_notifications": {
        "enabled": False,
        "recipients": ["test@example.com"],
        "smtp_server": "smtp.example.com",
        "smtp_port": 587
    }
}
