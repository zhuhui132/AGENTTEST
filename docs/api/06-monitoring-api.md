# 监控系统API文档

## 概览

监控系统提供了全面的性能监控、健康检查、指标收集和告警功能，帮助开发者实时了解系统运行状态和性能指标。

## 核心组件

### MetricsCollector - 指标收集器

收集和管理各种性能指标的组件。

#### 构造函数

```python
from src.utils.metrics import MetricsCollector

MetricsCollector(name: str, config: Optional[MetricsConfig] = None) -> MetricsCollector
```

**参数：**
- `name` (str): 收集器名称
- `config` (MetricsConfig, 可选): 配置对象

#### 使用示例

```python
from src.utils.metrics import MetricsCollector

# 创建指标收集器
metrics = MetricsCollector("agent_performance")

# 记录计数指标
metrics.increment("total_requests")
metrics.increment("successful_requests")

# 记录时间指标
metrics.record("response_time", 1.23)
metrics.record("processing_latency", 0.8)

# 记录分布指标
metrics.histogram("token_usage", 1500)
metrics.histogram("memory_usage", 512)

# 设置状态指标
metrics.gauge("active_connections", 25)
metrics.gauge("cpu_usage", 75.5)

# 记录错误
metrics.error("processing_error", {"message": "API调用失败"})

# 获取指标数据
all_metrics = metrics.get_all_metrics()
print(f"总请求数: {all_metrics['total_requests']}")
print(f"平均响应时间: {all_metrics['response_time']['mean']}")
```

#### 指标类型

**计数器 (Counter):**
```python
# 递增计数器
metrics.increment("metric_name", value=1)

# 设置计数器值
metrics.set_counter("metric_name", 100)

# 获取计数器值
count = metrics.get_counter("metric_name")
```

**计时器 (Timer):**
```python
# 记录时间
metrics.record("response_time", 1.23)

# 使用上下文管理器
with metrics.timer("operation_duration"):
    # 执行操作
    result = some_operation()

# 使用装饰器
@metrics.time_decorator("function_execution")
def my_function():
    # 函数实现
    pass
```

**直方图 (Histogram):**
```python
# 记录分布值
metrics.histogram("request_size", 1024)
metrics.histogram("request_size", 2048)

# 获取统计信息
stats = metrics.get_histogram_stats("request_size")
print(f"平均值: {stats['mean']}")
print(f"百分位数: {stats['percentiles']}")
```

**状态值 (Gauge):**
```python
# 设置状态值
metrics.gauge("active_users", 150)
metrics.gauge("memory_usage", 8192)

# 增减状态值
metrics.increment_gauge("active_users", 5)
metrics.decrement_gauge("active_users", 2)
```

### HealthChecker - 健康检查器

监控系统和组件健康状态的组件。

#### 构造函数

```python
from src.utils.monitoring import HealthChecker

HealthChecker(name: str, config: Optional[HealthConfig] = None) -> HealthChecker
```

#### 使用示例

```python
from src.utils.monitoring import HealthChecker, HealthCheck, HealthStatus

# 创建健康检查器
health_checker = HealthChecker("agent_health")

# 注册健康检查
@health_checker.register("database_connection", timeout=5.0)
async def check_database():
    try:
        # 检查数据库连接
        await database.ping()
        return HealthCheck(
            status=HealthStatus.HEALTHY,
            message="数据库连接正常",
            details={"response_time": 0.1}
        )
    except Exception as e:
        return HealthCheck(
            status=HealthStatus.UNHEALTHY,
            message=f"数据库连接失败: {str(e)}",
            details={"error": str(e)}
        )

@health_checker.register("llm_service", timeout=10.0)
async def check_llm_service():
    try:
        # 检查LLM服务
        response = await llm_client.health_check()
        return HealthCheck(
            status=HealthStatus.HEALTHY,
            message="LLM服务正常",
            details=response
        )
    except Exception as e:
        return HealthCheck(
            status=HealthStatus.UNHEALTHY,
            message=f"LLM服务异常: {str(e)}"
        )

# 执行健康检查
health_status = await health_checker.check_all()
print(f"整体状态: {health_status.overall_status}")
for check_name, check in health_status.checks.items():
    print(f"{check_name}: {check.status.value} - {check.message}")

# 获取单个健康检查
db_health = await health_checker.check_single("database_connection")
print(f"数据库状态: {db_health.status}")

# 获取健康报告
report = health_checker.generate_health_report()
print(json.dumps(report, indent=2))
```

#### 健康状态类型

```python
class HealthStatus(Enum):
    HEALTHY = "healthy"        # 健康
    DEGRADED = "degraded"      # 降级
    UNHEALTHY = "unhealthy"    # 不健康
    UNKNOWN = "unknown"        # 未知
```

#### 健康检查配置

```python
from src.utils.monitoring import HealthCheckConfig

config = HealthCheckConfig(
    check_interval=30.0,        # 检查间隔（秒）
    timeout=10.0,              # 超时时间
    retry_count=3,             # 重试次数
    failure_threshold=2,       # 失败阈值
    recovery_threshold=2        # 恢复阈值
)

health_checker = HealthChecker("agent_health", config)
```

### AlertManager - 告警管理器

管理和发送告警通知的组件。

#### 构造函数

```python
from src.utils.monitoring import AlertManager

AlertManager(config: AlertConfig) -> AlertManager
```

#### 使用示例

```python
from src.utils.monitoring import (
    AlertManager, AlertConfig, Alert, AlertLevel, AlertChannel
)

# 配置告警管理器
config = AlertConfig(
    enabled=True,
    default_channel=AlertChannel.EMAIL,
    channels={
        AlertChannel.EMAIL: {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "username": "alerts@example.com",
            "password": "app_password",
            "recipients": ["admin@example.com"]
        },
        AlertChannel.SLACK: {
            "webhook_url": "https://hooks.slack.com/...",
            "channel": "#alerts"
        }
    },
    rules=[
        {
            "name": "high_error_rate",
            "condition": "error_rate > 0.1",
            "level": AlertLevel.CRITICAL,
            "message": "错误率过高"
        },
        {
            "name": "high_response_time",
            "condition": "avg_response_time > 5.0",
            "level": AlertLevel.WARNING,
            "message": "响应时间过长"
        }
    ]
)

# 创建告警管理器
alert_manager = AlertManager(config)

# 手动发送告警
await alert_manager.send_alert(
    level=AlertLevel.ERROR,
    message="数据库连接失败",
    details={
        "error": "Connection timeout",
        "timestamp": time.time(),
        "service": "database"
    }
)

# 触发告警规则检查
await alert_manager.check_rules({
    "error_rate": 0.15,
    "avg_response_time": 6.2
})

# 获取告警历史
alert_history = alert_manager.get_alert_history(
    start_time=time.time() - 3600,  # 最近1小时
    level=AlertLevel.ERROR
)

print(f"最近1小时错误告警数: {len(alert_history)}")
```

#### 告警级别

```python
class AlertLevel(Enum):
    INFO = "info"         # 信息
    WARNING = "warning"   # 警告
    ERROR = "error"       # 错误
    CRITICAL = "critical" # 严重
```

#### 告警渠道

```python
class AlertChannel(Enum):
    EMAIL = "email"       # 邮件
    SLACK = "slack"       # Slack
    WEBHOOK = "webhook"    # Webhook
    SMS = "sms"           # 短信
    CONSOLE = "console"   # 控制台
```

### PerformanceMonitor - 性能监控器

监控系统性能和资源使用情况的组件。

#### 构造函数

```python
from src.utils.monitoring import PerformanceMonitor

PerformanceMonitor(name: str, config: PerfMonitorConfig) -> PerformanceMonitor
```

#### 使用示例

```python
from src.utils.monitoring import PerformanceMonitor, PerfMonitorConfig
import psutil

# 配置性能监控
config = PerfMonitorConfig(
    interval=5.0,                    # 监控间隔（秒）
    metrics=[
        "cpu_percent",
        "memory_percent",
        "disk_usage",
        "network_io",
        "process_count"
    ],
    thresholds={
        "cpu_percent": 80.0,        # CPU使用率阈值
        "memory_percent": 85.0,     # 内存使用率阈值
        "disk_usage": 90.0         # 磁盘使用率阈值
    }
)

# 创建性能监控器
perf_monitor = PerformanceMonitor("system_performance", config)

# 启动监控
await perf_monitor.start()

# 获取当前性能数据
current_metrics = perf_monitor.get_current_metrics()
print(f"CPU使用率: {current_metrics['cpu_percent']}%")
print(f"内存使用率: {current_metrics['memory_percent']}%")
print(f"磁盘使用率: {current_metrics['disk_usage']}%")

# 获取历史性能数据
historical_data = perf_monitor.get_historical_data(
    start_time=time.time() - 3600,  # 最近1小时
    metrics=["cpu_percent", "memory_percent"]
)

# 检查性能阈值
threshold_violations = perf_monitor.check_thresholds()
if threshold_violations:
    for metric, value in threshold_violations.items():
        print(f"指标 {metric} 超过阈值: {value}")

# 停止监控
await perf_monitor.stop()
```

#### 性能指标类型

```python
# 系统指标
"cpu_percent"          # CPU使用率
"memory_percent"       # 内存使用率
"disk_usage"           # 磁盘使用率
"network_io"           # 网络IO
"process_count"        # 进程数

# 应用指标
"request_rate"         # 请求速率
"response_time"        # 响应时间
"error_rate"           # 错误率
"active_connections"   # 活跃连接数
"queue_length"         # 队列长度

# 自定义指标
"custom_metric"         # 自定义指标
```

## API接口

### 监控指标API

#### 获取所有指标

```python
GET /api/v1/metrics
```

**响应：**
```json
{
    "success": true,
    "data": {
        "counters": {
            "total_requests": 1000,
            "successful_requests": 950,
            "failed_requests": 50
        },
        "gauges": {
            "active_connections": 25,
            "cpu_usage": 75.5,
            "memory_usage": 8192
        },
        "histograms": {
            "response_time": {
                "count": 1000,
                "mean": 1.23,
                "sum": 1230.0,
                "min": 0.1,
                "max": 5.0,
                "percentiles": {
                    "50th": 1.0,
                    "95th": 2.5,
                    "99th": 4.0
                }
            }
        },
        "timers": {
            "operation_duration": {
                "mean": 0.8,
                "max": 2.5,
                "min": 0.1,
                "count": 500
            }
        }
    }
}
```

#### 获取特定指标

```python
GET /api/v1/metrics/{metric_name}
```

**参数：**
- `metric_name`: 指标名称

**响应：**
```json
{
    "success": true,
    "data": {
        "name": "response_time",
        "type": "histogram",
        "value": {
            "count": 1000,
            "mean": 1.23,
            "percentiles": {
                "50th": 1.0,
                "95th": 2.5,
                "99th": 4.0
            }
        }
    }
}
```

#### 创建指标

```python
POST /api/v1/metrics
```

**请求体：**
```json
{
    "name": "custom_metric",
    "type": "counter",
    "value": 100,
    "tags": {
        "service": "api",
        "environment": "production"
    }
}
```

#### 更新指标

```python
PUT /api/v1/metrics/{metric_name}
```

**请求体：**
```json
{
    "value": 150,
    "increment": true
}
```

### 健康检查API

#### 获取整体健康状态

```python
GET /api/v1/health
```

**响应：**
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "timestamp": "2023-12-25T15:30:00Z",
        "checks": {
            "database_connection": {
                "status": "healthy",
                "message": "数据库连接正常",
                "response_time": 0.1,
                "last_check": "2023-12-25T15:30:00Z"
            },
            "llm_service": {
                "status": "degraded",
                "message": "LLM服务响应较慢",
                "response_time": 2.5,
                "last_check": "2023-12-25T15:30:00Z"
            }
        },
        "summary": {
            "total_checks": 2,
            "healthy_checks": 1,
            "degraded_checks": 1,
            "unhealthy_checks": 0
        }
    }
}
```

#### 获取特定健康检查

```python
GET /api/v1/health/{check_name}
```

**响应：**
```json
{
    "success": true,
    "data": {
        "name": "database_connection",
        "status": "healthy",
        "message": "数据库连接正常",
        "details": {
            "response_time": 0.1,
            "connection_pool": {
                "active": 5,
                "idle": 15,
                "total": 20
            }
        },
        "last_check": "2023-12-25T15:30:00Z",
        "check_duration": 0.05
    }
}
```

#### 触发健康检查

```python
POST /api/v1/health/{check_name}/check
```

**响应：**
```json
{
    "success": true,
    "data": {
        "status": "healthy",
        "message": "健康检查完成",
        "execution_time": 0.05
    }
}
```

### 告警API

#### 获取告警历史

```python
GET /api/v1/alerts
```

**查询参数：**
- `level`: 告警级别过滤
- `start_time`: 开始时间
- `end_time`: 结束时间
- `limit`: 返回数量限制

**响应：**
```json
{
    "success": true,
    "data": {
        "alerts": [
            {
                "id": "alert_123",
                "level": "error",
                "message": "数据库连接失败",
                "details": {
                    "error": "Connection timeout"
                },
                "timestamp": "2023-12-25T15:30:00Z",
                "resolved": false,
                "resolved_at": null
            }
        ],
        "total": 50,
        "page": 1,
        "limit": 20
    }
}
```

#### 创建告警

```python
POST /api/v1/alerts
```

**请求体：**
```json
{
    "level": "error",
    "message": "自定义告警",
    "details": {
        "service": "api",
        "error_code": "DATABASE_ERROR"
    },
    "tags": {
        "environment": "production",
        "service": "api"
    }
}
```

#### 解析告警

```python
PUT /api/v1/alerts/{alert_id}/resolve
```

**请求体：**
```json
{
    "resolution": "问题已修复",
    "resolved_by": "admin"
}
```

### 性能监控API

#### 获取当前性能指标

```python
GET /api/v1/performance/current
```

**响应：**
```json
{
    "success": true,
    "data": {
        "timestamp": "2023-12-25T15:30:00Z",
        "metrics": {
            "cpu_percent": 75.5,
            "memory_percent": 68.2,
            "disk_usage": 45.8,
            "network_io": {
                "bytes_sent": 1048576,
                "bytes_recv": 2097152
            },
            "process_count": 156
        },
        "threshold_violations": {
            "cpu_percent": {
                "value": 75.5,
                "threshold": 80.0,
                "status": "warning"
            }
        }
    }
}
```

#### 获取历史性能数据

```python
GET /api/v1/performance/history
```

**查询参数：**
- `start_time`: 开始时间
- `end_time`: 结束时间
- `metrics`: 指标列表（逗号分隔）
- `interval`: 数据间隔（1m, 5m, 1h）

**响应：**
```json
{
    "success": true,
    "data": {
        "interval": "5m",
        "points": [
            {
                "timestamp": "2023-12-25T15:25:00Z",
                "cpu_percent": 70.2,
                "memory_percent": 65.8,
                "disk_usage": 45.5
            },
            {
                "timestamp": "2023-12-25T15:30:00Z",
                "cpu_percent": 75.5,
                "memory_percent": 68.2,
                "disk_usage": 45.8
            }
        ]
    }
}
```

## 配置管理

### 监控配置

```yaml
# config/monitoring.yaml
monitoring:
  enabled: true
  metrics:
    collection_interval: 10.0
    retention_days: 30
    export_formats: ["prometheus", "json"]

  health_checks:
    enabled: true
    check_interval: 30.0
    timeout: 10.0
    retry_count: 3

    checks:
      - name: "database"
        type: "http"
        url: "http://localhost:5432/health"
        timeout: 5.0
        expected_status: 200

      - name: "llm_service"
        type: "grpc"
        endpoint: "localhost:50051"
        service: "LLMService"
        method: "HealthCheck"
        timeout: 10.0

  alerts:
    enabled: true
    channels:
      email:
        enabled: true
        smtp_host: "smtp.gmail.com"
        smtp_port: 587
        username: "${ALERT_EMAIL_USER}"
        password: "${ALERT_EMAIL_PASS}"
        recipients: ["admin@example.com"]

      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK_URL}"
        channel: "#alerts"

      webhook:
        enabled: false
        url: "https://hooks.example.com/alerts"
        headers:
          Authorization: "Bearer ${WEBHOOK_TOKEN}"

    rules:
      - name: "high_error_rate"
        condition: "error_rate > 0.1"
        level: "critical"
        message: "错误率过高: {{error_rate}}"
        cooldown: 300

      - name: "high_response_time"
        condition: "avg_response_time > 5.0"
        level: "warning"
        message: "平均响应时间过长: {{avg_response_time}}s"
        cooldown: 600

      - name: "low_success_rate"
        condition: "success_rate < 0.95"
        level: "error"
        message: "成功率过低: {{success_rate}}"
        cooldown: 300

  performance:
    enabled: true
    interval: 5.0
    metrics:
      - "cpu_percent"
      - "memory_percent"
      - "disk_usage"
      - "network_io"
      - "process_count"

    thresholds:
      cpu_percent: 80.0
      memory_percent: 85.0
      disk_usage: 90.0

    alerts:
      enabled: true
      level: "warning"
      cooldown: 300
```

### 环境变量

```bash
# 监控配置
MONITORING_ENABLED=true
METRICS_COLLECTION_INTERVAL=10
METRICS_RETENTION_DAYS=30

# 告警配置
ALERT_EMAIL_USER=alerts@example.com
ALERT_EMAIL_PASS=app_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
WEBHOOK_TOKEN=secret_token

# 性能监控
PERF_MONITOR_INTERVAL=5
CPU_THRESHOLD=80.0
MEMORY_THRESHOLD=85.0
DISK_THRESHOLD=90.0
```

## 最佳实践

### 1. 指标设计

```python
# 良好的指标命名规范
metrics.increment("http_requests_total", tags={
    "method": "GET",
    "endpoint": "/api/v1/users",
    "status_code": "200"
})

metrics.record("http_request_duration", 0.123, tags={
    "method": "POST",
    "endpoint": "/api/v1/users"
})

metrics.gauge("database_connections_active", 25, tags={
    "database": "primary",
    "host": "db-primary"
})
```

### 2. 健康检查设计

```python
@health_checker.register("comprehensive_health", timeout=15.0)
async def comprehensive_health_check():
    checks = []

    # 数据库检查
    try:
        db_result = await database.ping()
        checks.append(("database", True, db_result))
    except Exception as e:
        checks.append(("database", False, str(e)))

    # 外部API检查
    try:
        api_result = await external_api.health_check()
        checks.append(("external_api", True, api_result))
    except Exception as e:
        checks.append(("external_api", False, str(e)))

    # 确定整体状态
    all_healthy = all(check[1] for check in checks)

    return HealthCheck(
        status=HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED,
        message=f"综合健康检查: {len([c for c in checks if c[1]])}/{len(checks)} 通过",
        details={
            "checks": [
                {
                    "name": check[0],
                    "status": "pass" if check[1] else "fail",
                    "message": check[2]
                }
                for check in checks
            ]
        }
    )
```

### 3. 告警规则配置

```yaml
# 分层告警规则
rules:
  # 信息级告警
  - name: "new_user_signup"
    condition: "new_user_count > 0"
    level: "info"
    message: "新用户注册: {{new_user_count}}"
    channels: ["slack"]

  # 警告级告警
  - name: "high_response_time"
    condition: "p95_response_time > 2.0"
    level: "warning"
    message: "95%响应时间超过2秒: {{p95_response_time}}s"
    channels: ["slack", "email"]
    cooldown: 600

  # 错误级告警
  - name: "service_down"
    condition: "service_up == 0"
    level: "error"
    message: "服务 {{service_name}} 不可用"
    channels: ["slack", "email", "webhook"]
    cooldown: 300
    escalation:
      - after: 600  # 10分钟后升级
        level: "critical"
        channels: ["sms", "phone"]

  # 严重级告警
  - name: "data_loss"
    condition: "data_corruption_detected"
    level: "critical"
    message: "检测到数据损坏，需要立即处理"
    channels: ["all"]
    immediate: true
```

### 4. 性能监控优化

```python
# 自定义性能监控指标
class CustomPerformanceMonitor(PerformanceMonitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics = {}

    async def collect_custom_metrics(self):
        """收集自定义业务指标"""
        # 业务指标
        active_users = await self.get_active_user_count()
        self.custom_metrics["active_users"] = active_users

        # 缓存性能
        cache_hit_rate = await self.get_cache_hit_rate()
        self.custom_metrics["cache_hit_rate"] = cache_hit_rate

        # 队列长度
        queue_lengths = await self.get_queue_lengths()
        for queue_name, length in queue_lengths.items():
            self.custom_metrics[f"queue_length_{queue_name}"] = length

    async def get_active_user_count(self) -> int:
        """获取活跃用户数"""
        # 实现获取逻辑
        pass

    async def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        # 实现获取逻辑
        pass

    async def get_queue_lengths(self) -> Dict[str, int]:
        """获取各队列长度"""
        # 实现获取逻辑
        pass
```

这个监控系统API文档提供了完整的监控功能说明、使用示例和最佳实践，帮助开发者构建可靠的系统监控体系。
