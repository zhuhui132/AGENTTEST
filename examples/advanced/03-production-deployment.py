"""
生产环境部署指南

详细演示AI Agent系统的生产环境部署方案。
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from src.agents.intelligent_agent import IntelligentAgent
from src.core.types import AgentConfig, LLMConfig, MemoryConfig, RAGConfig
from src.utils.logger import get_logger
from src.utils.monitoring import MetricsCollector, HealthChecker


# ============================================================================
# 1. 生产环境配置
# ============================================================================

@dataclass
class ProductionConfig:
    """生产环境配置"""

    # 环境设置
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"

    # 性能配置
    max_concurrent_requests: int = 100
    request_timeout: float = 30.0
    worker_processes: int = 4

    # 资源限制
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    max_disk_usage_percent: float = 85.0

    # 安全配置
    enable_authentication: bool = True
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60

    # 监控配置
    enable_monitoring: bool = True
    metrics_retention_days: int = 30
    health_check_interval: int = 30

    # 备份配置
    enable_backups: bool = True
    backup_interval_hours: int = 24
    backup_retention_days: int = 7


class ProductionEnvironmentSetup:
    """生产环境设置"""

    def __init__(self, config: ProductionConfig):
        self.config = config
        self.logger = get_logger("production_setup")

    def setup_environment(self):
        """设置环境变量和配置"""
        print("=== 生产环境设置 ===\n")

        # 环境变量设置
        env_vars = {
            "ENVIRONMENT": self.config.environment,
            "DEBUG": str(self.config.debug_mode),
            "LOG_LEVEL": self.config.log_level,

            "MAX_CONCURRENT_REQUESTS": str(self.config.max_concurrent_requests),
            "REQUEST_TIMEOUT": str(self.config.request_timeout),
            "WORKER_PROCESSES": str(self.config.worker_processes),

            "MAX_MEMORY_MB": str(self.config.max_memory_mb),
            "MAX_CPU_PERCENT": str(self.config.max_cpu_percent),
            "MAX_DISK_USAGE_PERCENT": str(self.config.max_disk_usage_percent),

            "ENABLE_AUTHENTICATION": str(self.config.enable_authentication),
            "ENABLE_RATE_LIMITING": str(self.config.enable_rate_limiting),
            "RATE_LIMIT_PER_MINUTE": str(self.config.rate_limit_per_minute),

            "ENABLE_MONITORING": str(self.config.enable_monitoring),
            "METRICS_RETENTION_DAYS": str(self.config.metrics_retention_days),
            "HEALTH_CHECK_INTERVAL": str(self.config.health_check_interval),

            "ENABLE_BACKUPS": str(self.config.enable_backups),
            "BACKUP_INTERVAL_HOURS": str(self.config.backup_interval_hours),
            "BACKUP_RETENTION_DAYS": str(self.config.backup_retention_days)
        }

        print("设置环境变量:")
        for key, value in env_vars.items():
            os.environ[key] = value
            print(f"  {key}={value}")

        print()
        return env_vars

    def create_directory_structure(self):
        """创建目录结构"""
        print("=== 创建目录结构 ===\n")

        directories = [
            "logs",
            "logs/application",
            "logs/access",
            "logs/error",
            "data",
            "data/memory",
            "data/rag",
            "data/cache",
            "backups",
            "backups/daily",
            "backups/memory",
            "backups/rag",
            "config",
            "config/environments",
            "config/secrets",
            "monitoring",
            "monitoring/metrics",
            "monitoring/alerts",
            "temp",
            "uploads",
            "models"
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✓ 创建目录: {directory}")

        print()

    def setup_logging(self):
        """设置日志配置"""
        print("=== 设置日志配置 ===\n")

        log_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S"
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "detailed",
                    "stream": "ext://sys.stdout"
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": "logs/application/app.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "detailed",
                    "filename": "logs/error/error.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                },
                "json_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "json",
                    "filename": "logs/application/app.json",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5
                }
            },
            "loggers": {
                "": {
                    "level": self.config.log_level,
                    "handlers": ["console", "file", "json_file"],
                    "propagate": False
                },
                "error": {
                    "level": "ERROR",
                    "handlers": ["console", "error_file"],
                    "propagate": False
                }
            }
        }

        # 保存日志配置
        with open("config/logging.json", "w") as f:
            json.dump(log_config, f, indent=2)

        print("✓ 日志配置已保存: config/logging.json")

        # 显示配置摘要
        print(f"日志级别: {self.config.log_level}")
        print(f"控制台输出: 启用")
        print(f"文件轮转: 10MB, 保留5个文件")
        print(f"JSON格式: 启用")
        print()


# ============================================================================
# 2. Docker部署配置
# ============================================================================

class DockerDeployment:
    """Docker部署配置"""

    def __init__(self):
        self.logger = get_logger("docker_deployment")

    def create_dockerfile(self):
        """创建Dockerfile"""
        print("=== 创建Dockerfile ===\n")

        dockerfile_content = """# 多阶段构建
FROM python:3.9-slim as builder

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY requirements-dev.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# 生产阶段
FROM python:3.9-slim

# 安装运行时依赖
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser

# 设置工作目录
WORKDIR /app

# 复制Python环境
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY src/ ./src/
COPY config/ ./config/
COPY examples/ ./examples/
COPY docs/ ./docs/

# 创建必要目录
RUN mkdir -p logs data backups temp uploads models
RUN chown -R appuser:appuser /app

# 切换到非root用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 环境变量
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production
ENV LOG_LEVEL=INFO

# 启动命令
CMD ["python", "-m", "src.server"]
"""

        with open("Dockerfile", "w") as f:
            f.write(dockerfile_content)

        print("✓ Dockerfile已创建")
        print()

    def create_docker_compose(self):
        """创建Docker Compose配置"""
        print("=== 创建Docker Compose配置 ===\n")

        docker_compose_content = """version: '3.8'

services:
  # 主应用
  agent-app:
    build: .
    container_name: agent-app
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/agent_db
      - REDIS_URL=redis://redis:6379/0
      - LLM_API_KEY=${LLM_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
      - ./backups:/app/backups
      - ./config:/app/config
    depends_on:
      - postgres
      - redis
      - elasticsearch
    restart: unless-stopped
    networks:
      - agent-network
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # PostgreSQL数据库
  postgres:
    image: postgres:13-alpine
    container_name: postgres
    environment:
      - POSTGRES_DB=agent_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - agent-network
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

  # Redis缓存
  redis:
    image: redis:7-alpine
    container_name: redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
      - ./config/redis/redis.conf:/usr/local/etc/redis/redis.conf
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - agent-network
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.25'

  # Elasticsearch搜索引擎
  elasticsearch:
    image: elasticsearch:8.5.0
    container_name: elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
      - ./config/elasticsearch/elasticsearch.yml:/usr/share/elasticsearch/config/elasticsearch.yml
    ports:
      - "9200:9200"
      - "9300:9300"
    restart: unless-stopped
    networks:
      - agent-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # Nginx反向代理
  nginx:
    image: nginx:alpine
    container_name: nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/nginx/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - agent-app
    restart: unless-stopped
    networks:
      - agent-network

  # Prometheus监控
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - agent-network

  # Grafana可视化
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - agent-network

volumes:
  postgres_data:
  redis_data:
  elasticsearch_data:
  prometheus_data:
  grafana_data:

networks:
  agent-network:
    driver: bridge
"""

        with open("docker-compose.yml", "w") as f:
            f.write(docker_compose_content)

        print("✓ Docker Compose配置已创建")
        print()

    def create_environment_file(self):
        """创建环境变量文件"""
        print("=== 创建环境变量文件 ===\n")

        env_file_content = """# 数据库配置
POSTGRES_PASSWORD=your_postgres_password
DATABASE_URL=postgresql://postgres:your_postgres_password@postgres:5432/agent_db

# Redis配置
REDIS_PASSWORD=your_redis_password
REDIS_URL=redis://:your_redis_password@redis:6379/0

# API密钥
LLM_API_KEY=your_llm_api_key
WEATHER_API_KEY=your_weather_api_key
SEARCH_API_KEY=your_search_api_key

# 监控配置
GRAFANA_PASSWORD=your_grafana_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/your_webhook
EMAIL_ALERTS=alerts@example.com

# 应用配置
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# 安全配置
SECRET_KEY=your_very_long_and_secure_secret_key
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key

# 性能配置
MAX_CONCURRENT_REQUESTS=100
WORKER_PROCESSES=4
REQUEST_TIMEOUT=30

# 外部服务
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_email_password
"""

        with open(".env", "w") as f:
            f.write(env_file_content)

        print("✓ 环境变量文件已创建: .env")
        print()

    def create_dockerignore(self):
        """创建.dockerignore文件"""
        print("=== 创建.dockerignore文件 ===\n")

        dockerignore_content = """# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
venv
.venv
pip-log.txt
pip-delete-this-directory.txt

# IDE
.vscode
.idea
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Temporary files
temp/
tmp/
*.tmp

# Data (for development)
data/
backups/

# Documentation
docs/
*.md

# Tests
tests/
.pytest_cache/
.coverage

# Development
.env.local
.env.development
docker-compose.override.yml
"""

        with open(".dockerignore", "w") as f:
            f.write(dockerignore_content)

        print("✓ .dockerignore文件已创建")
        print()


# ============================================================================
# 3. Kubernetes部署配置
# ============================================================================

class KubernetesDeployment:
    """Kubernetes部署配置"""

    def __init__(self):
        self.logger = get_logger("kubernetes_deployment")

    def create_namespace(self):
        """创建命名空间"""
        print("=== 创建Kubernetes命名空间 ===\n")

        namespace_content = """apiVersion: v1
kind: Namespace
metadata:
  name: agent-system
  labels:
    name: agent-system
    environment: production
"""

        Path("k8s").mkdir(exist_ok=True)
        with open("k8s/namespace.yaml", "w") as f:
            f.write(namespace_content)

        print("✓ 命名空间配置已创建: k8s/namespace.yaml")
        print()

    def create_configmap(self):
        """创建ConfigMap"""
        print("=== 创建ConfigMap ===\n")

        configmap_content = """apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: agent-system
data:
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  DEBUG: "false"
  MAX_CONCURRENT_REQUESTS: "100"
  WORKER_PROCESSES: "4"
  REQUEST_TIMEOUT: "30"
  HEALTH_CHECK_INTERVAL: "30"
  METRICS_RETENTION_DAYS: "30"

  # 数据库配置
  DB_HOST: "postgres-service"
  DB_PORT: "5432"
  DB_NAME: "agent_db"
  DB_USER: "postgres"

  # Redis配置
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"

  # Elasticsearch配置
  ES_HOST: "elasticsearch-service"
  ES_PORT: "9200"

  # 监控配置
  PROMETHEUS_HOST: "prometheus-service"
  PROMETHEUS_PORT: "9090"
"""

        with open("k8s/configmap.yaml", "w") as f:
            f.write(configmap_content)

        print("✓ ConfigMap配置已创建: k8s/configmap.yaml")
        print()

    def create_secret(self):
        """创建Secret"""
        print("=== 创建Secret ===\n")

        secret_content = """apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: agent-system
type: Opaque
data:
  # Base64编码的密钥
  postgres-password: eW91cl9wb3N0Z3Jlc19wYXNzd29yZA==  # your_postgres_password
  redis-password: eW91cl9yZWRpc19wYXNzd29yZA==             # your_redis_password
  llm-api-key: eW91cl9sbG1fYXBpX2tleQ==                # your_llm_api_key
  jwt-secret: eW91cl9qd3Rfc2VjcmV0X2tleQ==               # your_jwt_secret_key
  encryption-key: eW91cl9lbmNyeXB0aW9uX2tleQ==           # your_encryption_key

  # 邮件配置
  smtp-user: eW91cl9lbWFpbEBnbWFpbC5jb20=              # your_email@gmail.com
  smtp-pass: eW91cl9lbWFpbF9wYXNzd29yZA==               # your_email_password
"""

        with open("k8s/secret.yaml", "w") as f:
            f.write(secret_content)

        print("✓ Secret配置已创建: k8s/secret.yaml")
        print()

    def create_deployment(self):
        """创建Deployment"""
        print("=== 创建Deployment ===\n")

        deployment_content = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-app
  namespace: agent-system
  labels:
    app: agent-app
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-app
  template:
    metadata:
      labels:
        app: agent-app
        version: v1
    spec:
      containers:
      - name: agent-app
        image: agent-app:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 9090
          name: metrics

        # 环境变量
        envFrom:
        - configMapRef:
            name: agent-config
        - secretRef:
            name: agent-secrets

        # 资源限制
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3

        # 存储挂载
        volumeMounts:
        - name: logs-volume
          mountPath: /app/logs
        - name: data-volume
          mountPath: /app/data
        - name: config-volume
          mountPath: /app/config

      volumes:
      - name: logs-volume
        emptyDir: {}
      - name: data-volume
        persistentVolumeClaim:
          claimName: agent-data-pvc
      - name: config-volume
        configMap:
          name: agent-config

      # 节点选择
      nodeSelector:
        node-type: application

      # 容忍度
      tolerations:
      - key: "application"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

      # 亲和性
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - agent-app
              topologyKey: kubernetes.io/hostname
"""

        with open("k8s/deployment.yaml", "w") as f:
            f.write(deployment_content)

        print("✓ Deployment配置已创建: k8s/deployment.yaml")
        print()

    def create_service(self):
        """创建Service"""
        print("=== 创建Service ===\n")

        service_content = """apiVersion: v1
kind: Service
metadata:
  name: agent-app-service
  namespace: agent-system
  labels:
    app: agent-app
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  selector:
    app: agent-app
---
apiVersion: v1
kind: Service
metadata:
  name: agent-app-headless
  namespace: agent-system
  labels:
    app: agent-app
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  selector:
    app: agent-app
"""

        with open("k8s/service.yaml", "w") as f:
            f.write(service_content)

        print("✓ Service配置已创建: k8s/service.yaml")
        print()

    def create_ingress(self):
        """创建Ingress"""
        print("=== 创建Ingress ===\n")

        ingress_content = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-app-ingress
  namespace: agent-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: agent-app-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agent-app-service
            port:
              number: 80
"""

        with open("k8s/ingress.yaml", "w") as f:
            f.write(ingress_content)

        print("✓ Ingress配置已创建: k8s/ingress.yaml")
        print()

    def create_horizontal_pod_autoscaler(self):
        """创建水平Pod自动扩展"""
        print("=== 创建HPA ===\n")

        hpa_content = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-app-hpa
  namespace: agent-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
"""

        with open("k8s/hpa.yaml", "w") as f:
            f.write(hpa_content)

        print("✓ HPA配置已创建: k8s/hpa.yaml")
        print()


# ============================================================================
# 4. 监控和告警配置
# ============================================================================

class ProductionMonitoring:
    """生产环境监控配置"""

    def __init__(self):
        self.logger = get_logger("production_monitoring")

    def create_prometheus_config(self):
        """创建Prometheus配置"""
        print("=== 创建Prometheus配置 ===\n")

        prometheus_config = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Agent应用监控
  - job_name: 'agent-app'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - agent-system
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: agent-app
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\\d+)?;(\\d+)
        replacement: $1:$2
        target_label: __address__

  # Kubernetes系统监控
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - source_labels: [__meta_kubernetes_namespace, __meta_kubernetes_service_name, __meta_kubernetes_endpoint_port_name]
        action: keep
        regex: default;kubernetes;https

  - job_name: 'kubernetes-nodes'
    kubernetes_sd_configs:
      - role: node
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\\d+)?;(\\d+)
        replacement: $1:$2
        target_label: __address__
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name
"""

        Path("config/prometheus").mkdir(parents=True, exist_ok=True)
        with open("config/prometheus/prometheus.yml", "w") as f:
            f.write(prometheus_config)

        print("✓ Prometheus配置已创建: config/prometheus/prometheus.yml")
        print()

    def create_alert_rules(self):
        """创建告警规则"""
        print("=== 创建告警规则 ===\n")

        alert_rules = """groups:
  - name: agent-app-alerts
    rules:
      # 应用可用性告警
      - alert: AgentAppDown
        expr: up{job="agent-app"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Agent应用实例 {{ $labels.instance }} 停止响应"
          description: "Agent应用实例 {{ $labels.instance }} 已经停止响应超过1分钟"

      # 高错误率告警
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent应用错误率过高"
          description: "Agent应用5分钟内错误率超过10%，当前值: {{ $value }}"

      # 高响应时间告警
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent应用响应时间过长"
          description: "Agent应用95分位响应时间超过2秒，当前值: {{ $value }}秒"

      # 内存使用率告警
      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "容器内存使用率过高"
          description: "容器 {{ $labels.name }} 内存使用率超过85%，当前值: {{ $value }}%"

      # CPU使用率告警
      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "容器CPU使用率过高"
          description: "容器 {{ $labels.name }} CPU使用率超过80%，当前值: {{ $value }}%"

      # Pod重启告警
      - alert: PodRestartTooFrequently
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod重启过于频繁"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} 在15分钟内重启了 {{ $value }} 次"

      # 磁盘空间告警
      - alert: LowDiskSpace
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 15
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "磁盘空间不足"
          description: "节点 {{ $labels.instance }} 磁盘空间剩余不足15%，当前值: {{ $value }}%"

  - name: kubernetes-alerts
    rules:
      # Pod状态告警
      - alert: PodNotReady
        expr: kube_pod_status_ready{condition="true"} == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Pod未就绪"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} 已经未就绪超过5分钟"

      # 节点状态告警
      - alert: NodeNotReady
        expr: kube_node_status_condition{condition="Ready",status="true"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "节点未就绪"
          description: "节点 {{ $labels.node }} 已经未就绪超过5分钟"
"""

        Path("config/prometheus/rules").mkdir(parents=True, exist_ok=True)
        with open("config/prometheus/rules/alerts.yml", "w") as f:
            f.write(alert_rules)

        print("✓ 告警规则已创建: config/prometheus/rules/alerts.yml")
        print()

    def create_grafana_dashboard(self):
        """创建Grafana仪表盘"""
        print("=== 创建Grafana仪表盘 ===\n")

        dashboard = {
            "dashboard": {
                "id": None,
                "title": "Agent应用监控仪表盘",
                "tags": ["agent", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "请求速率",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "{{method}} {{endpoint}}"
                            }
                        ],
                        "yAxes": [{"label": "请求/秒"}]
                    },
                    {
                        "id": 2,
                        "title": "响应时间",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P50"
                            },
                            {
                                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P95"
                            },
                            {
                                "expr": "histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))",
                                "legendFormat": "P99"
                            }
                        ],
                        "yAxes": [{"label": "秒"}]
                    },
                    {
                        "id": 3,
                        "title": "错误率",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m]) * 100"
                            }
                        ],
                        "valueMaps": [
                            {
                                "value": "null",
                                "text": "N/A"
                            }
                        ],
                        "thresholds": "1,5,10"
                    },
                    {
                        "id": 4,
                        "title": "内存使用",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "container_memory_usage_bytes / 1024 / 1024",
                                "legendFormat": "{{name}}"
                            }
                        ],
                        "yAxes": [{"label": "MB"}]
                    },
                    {
                        "id": 5,
                        "title": "CPU使用",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
                                "legendFormat": "{{name}}"
                            }
                        ],
                        "yAxes": [{"label": "%"}]
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }

        Path("config/grafana/dashboards").mkdir(parents=True, exist_ok=True)
        with open("config/grafana/dashboards/agent-dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)

        print("✓ Grafana仪表盘已创建: config/grafana/dashboards/agent-dashboard.json")
        print()


# ============================================================================
# 5. 备份和恢复策略
# ============================================================================

class BackupAndRecovery:
    """备份和恢复策略"""

    def __init__(self):
        self.logger = get_logger("backup_recovery")

    def create_backup_script(self):
        """创建备份脚本"""
        print("=== 创建备份脚本 ===\n")

        backup_script = """#!/bin/bash

# AI Agent系统备份脚本
# 使用方法: ./backup.sh [full|incremental]

set -e

# 配置
BACKUP_DIR="/app/backups"
DATA_DIR="/app/data"
LOG_DIR="/app/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_TYPE=${1:-incremental}

# 创建备份目录
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/memory"
mkdir -p "$BACKUP_DIR/rag"

echo "开始备份 - 类型: $BACKUP_TYPE, 时间: $TIMESTAMP"

# 备份数据库
echo "备份数据库..."
docker exec postgres pg_dump -U postgres agent_db > "$BACKUP_DIR/daily/db_backup_$TIMESTAMP.sql"

# 备份Redis数据
echo "备份Redis数据..."
docker exec redis redis-cli --rdb /data/dump_$TIMESTAMP.rdb
docker cp redis:/data/dump_$TIMESTAMP.rdb "$BACKUP_DIR/daily/"

# 备份Elasticsearch索引
echo "备份Elasticsearch索引..."
curl -X PUT "elasticsearch:9200/_snapshot/backup_repo/snapshot_$TIMESTAMP?wait_for_completion=true" \
  -H 'Content-Type: application/json' \
  -d '{
    "indices": "agent-*",
    "ignore_unavailable": true,
    "include_global_state": false
  }'

# 备份应用数据
echo "备份应用数据..."
if [ "$BACKUP_TYPE" = "full" ]; then
    tar -czf "$BACKUP_DIR/data_backup_full_$TIMESTAMP.tar.gz" -C "$DATA_DIR" .
else
    # 增量备份：只备份最近修改的文件
    find "$DATA_DIR" -mtime -1 -type f -print0 | tar -czf "$BACKUP_DIR/data_backup_incremental_$TIMESTAMP.tar.gz" --null -T -
fi

# 备份日志文件（最近7天）
echo "备份日志文件..."
find "$LOG_DIR" -name "*.log" -mtime -7 -print0 | tar -czf "$BACKUP_DIR/logs_backup_$TIMESTAMP.tar.gz" --null -T -

# 清理旧备份（保留7天）
echo "清理旧备份..."
find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.sql" -mtime +7 -delete
find "$BACKUP_DIR" -name "*.rdb" -mtime +7 -delete

# 备份完成
echo "备份完成: $TIMESTAMP"

# 上传到云存储（可选）
if [ ! -z "$CLOUD_STORAGE_BUCKET" ]; then
    echo "上传到云存储..."
    aws s3 sync "$BACKUP_DIR" "s3://$CLOUD_STORAGE_BUCKET/agent-backups/$(date +%Y/%m/%d)/"
fi

echo "所有备份任务完成"
"""

        with open("scripts/backup.sh", "w") as f:
            f.write(backup_script)

        # 设置执行权限
        os.chmod("scripts/backup.sh", 0o755)

        print("✓ 备份脚本已创建: scripts/backup.sh")
        print()

    def create_recovery_script(self):
        """创建恢复脚本"""
        print("=== 创建恢复脚本 ===\n")

        recovery_script = """#!/bin/bash

# AI Agent系统恢复脚本
# 使用方法: ./recovery.sh <timestamp>

set -e

if [ -z "$1" ]; then
    echo "错误: 请提供备份时间戳"
    echo "使用方法: ./recovery.sh <timestamp>"
    echo "示例: ./recovery.sh 20231225_150000"
    exit 1
fi

TIMESTAMP=$1
BACKUP_DIR="/app/backups"

echo "开始恢复 - 时间戳: $TIMESTAMP"

# 停止应用服务
echo "停止应用服务..."
docker-compose down

# 恢复数据库
echo "恢复数据库..."
if [ -f "$BACKUP_DIR/daily/db_backup_$TIMESTAMP.sql" ]; then
    docker-compose up -d postgres
    sleep 10
    docker exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS agent_db;"
    docker exec postgres psql -U postgres -c "CREATE DATABASE agent_db;"
    docker exec postgres psql -U postgres agent_db < "$BACKUP_DIR/daily/db_backup_$TIMESTAMP.sql"
    echo "数据库恢复完成"
else
    echo "警告: 未找到数据库备份文件"
fi

# 恢复Redis数据
echo "恢复Redis数据..."
if [ -f "$BACKUP_DIR/daily/dump_$TIMESTAMP.rdb" ]; then
    docker-compose up -d redis
    sleep 5
    docker cp "$BACKUP_DIR/daily/dump_$TIMESTAMP.rdb" redis:/data/dump.rdb
    docker-compose restart redis
    echo "Redis数据恢复完成"
else
    echo "警告: 未找到Redis备份文件"
fi

# 恢复Elasticsearch索引
echo "恢复Elasticsearch索引..."
# 这里需要根据实际的快照恢复命令进行调整

# 恢复应用数据
echo "恢复应用数据..."
if [ -f "$BACKUP_DIR/data_backup_full_$TIMESTAMP.tar.gz" ]; then
    # 恢复完整备份
    rm -rf /app/data/*
    tar -xzf "$BACKUP_DIR/data_backup_full_$TIMESTAMP.tar.gz" -C /app/data/
    echo "应用数据恢复完成（完整备份）"
elif [ -f "$BACKUP_DIR/data_backup_incremental_$TIMESTAMP.tar.gz" ]; then
    # 恢复增量备份
    tar -xzf "$BACKUP_DIR/data_backup_incremental_$TIMESTAMP.tar.gz" -C /app/data/
    echo "应用数据恢复完成（增量备份）"
else
    echo "警告: 未找到应用数据备份文件"
fi

# 重启所有服务
echo "重启所有服务..."
docker-compose up -d

# 验证恢复
echo "验证恢复状态..."
sleep 30

# 检查应用健康状态
if curl -f http://localhost:8000/health; then
    echo "✓ 应用恢复成功"
else
    echo "✗ 应用恢复失败"
    exit 1
fi

# 检查数据库连接
if docker exec postgres pg_isready -U postgres; then
    echo "✓ 数据库连接正常"
else
    echo "✗ 数据库连接失败"
fi

# 检查Redis连接
if docker exec redis redis-cli ping; then
    echo "✓ Redis连接正常"
else
    echo "✗ Redis连接失败"
fi

echo "系统恢复完成: $TIMESTAMP"
"""

        with open("scripts/recovery.sh", "w") as f:
            f.write(recovery_script)

        # 设置执行权限
        os.chmod("scripts/recovery.sh", 0o755)

        print("✓ 恢复脚本已创建: scripts/recovery.sh")
        print()

    def create_cron_job(self):
        """创建定时任务"""
        print("=== 创建定时任务 ===\n")

        cron_content = """# AI Agent系统定时任务

# 每天凌晨2点执行完整备份
0 2 * * * /app/scripts/backup.sh full >> /app/logs/backup.log 2>&1

# 每6小时执行增量备份
0 */6 * * * /app/scripts/backup.sh incremental >> /app/logs/backup.log 2>&1

# 每小时检查系统健康状态
0 * * * * curl -f http://localhost:8000/health || echo "Health check failed at $(date)" >> /app/logs/health_check.log 2>&1

# 每天清理临时文件
0 3 * * * find /app/temp -type f -mtime +1 -delete >> /app/logs/cleanup.log 2>&1

# 每周清理旧日志（保留30天）
0 4 * * 0 find /app/logs -name "*.log" -mtime +30 -delete >> /app/logs/cleanup.log 2>&1

# 每月检查磁盘空间
0 5 1 * * df -h >> /app/logs/disk_usage.log 2>&1
"""

        with open("scripts/crontab.txt", "w") as f:
            f.write(cron_content)

        print("✓ 定时任务配置已创建: scripts/crontab.txt")
        print()


# ============================================================================
# 6. 主程序和完整部署指南
# ============================================================================

async def main():
    """主程序入口"""
    print("=== AI Agent生产环境部署指南 ===\n")

    # 1. 生产环境配置
    print("1. 生产环境配置")
    prod_config = ProductionConfig()
    env_setup = ProductionEnvironmentSetup(prod_config)

    env_setup.setup_environment()
    env_setup.create_directory_structure()
    env_setup.setup_logging()

    # 2. Docker部署配置
    print("2. Docker部署配置")
    docker_deployment = DockerDeployment()

    docker_deployment.create_dockerfile()
    docker_deployment.create_docker_compose()
    docker_deployment.create_environment_file()
    docker_deployment.create_dockerignore()

    # 3. Kubernetes部署配置
    print("3. Kubernetes部署配置")
    k8s_deployment = KubernetesDeployment()

    k8s_deployment.create_namespace()
    k8s_deployment.create_configmap()
    k8s_deployment.create_secret()
    k8s_deployment.create_deployment()
    k8s_deployment.create_service()
    k8s_deployment.create_ingress()
    k8s_deployment.create_horizontal_pod_autoscaler()

    # 4. 监控配置
    print("4. 监控配置")
    monitoring = ProductionMonitoring()

    monitoring.create_prometheus_config()
    monitoring.create_alert_rules()
    monitoring.create_grafana_dashboard()

    # 5. 备份和恢复
    print("5. 备份和恢复")
    backup_recovery = BackupAndRecovery()

    Path("scripts").mkdir(exist_ok=True)
    backup_recovery.create_backup_script()
    backup_recovery.create_recovery_script()
    backup_recovery.create_cron_job()

    # 6. 部署指南
    print("\n=== 部署指南 ===\n")

    deployment_guide = {
        "docker_deployment": [
            "1. 修改.env文件中的配置参数",
            "2. 运行 'docker-compose up -d' 启动服务",
            "3. 使用 'docker-compose logs -f' 查看日志",
            "4. 访问 http://localhost:8000/health 检查服务状态"
        ],
        "kubernetes_deployment": [
            "1. 创建命名空间: kubectl apply -f k8s/namespace.yaml",
            "2. 应用配置: kubectl apply -f k8s/",
            "3. 检查状态: kubectl get pods -n agent-system",
            "4. 查看日志: kubectl logs -f deployment/agent-app -n agent-system"
        ],
        "monitoring_setup": [
            "1. 访问 Grafana: http://localhost:3000",
            "2. 访问 Prometheus: http://localhost:9090",
            "3. 配置告警通知渠道",
            "4. 设置仪表盘和告警规则"
        ],
        "backup_procedures": [
            "1. 执行备份: ./scripts/backup.sh full",
            "2. 恢复系统: ./scripts/recovery.sh <timestamp>",
            "3. 设置定时任务: crontab scripts/crontab.txt",
            "4. 测试备份恢复流程"
        ]
    }

    for category, steps in deployment_guide.items():
        print(f"{category}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print()

    print("=== 部署配置生成完成 ===")
    print("请根据实际环境调整配置参数")
    print("在生产环境部署前请进行充分测试")


if __name__ == "__main__":
    # 运行生产环境部署指南
    asyncio.run(main())
