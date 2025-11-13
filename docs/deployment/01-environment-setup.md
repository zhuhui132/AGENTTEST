# 环境设置指南

## 概览

本文档详细介绍了AI Agent系统的环境设置、依赖配置和系统要求。

## 系统要求

### 最低配置

| 组件 | CPU | 内存 | 存储 | 网络 |
|------|-----|------|------|------|
| 基础Agent | 2核 | 4GB | 10GB | 100Mbps |
| 完整系统 | 4核 | 8GB | 50GB | 1Gbps |
| 生产环境 | 8核 | 16GB | 100GB | 10Gbps |

### 推荐配置

| 组件 | CPU | 内存 | 存储 | GPU |
|------|-----|------|------|-----|
| 开发环境 | 4核 | 8GB | 50GB | 可选 |
| 测试环境 | 8核 | 16GB | 100GB | GTX 1660+ |
| 生产环境 | 16核 | 32GB | 500GB+ | RTX 3080+ |

### 操作系统支持

- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+
- **macOS**: 11.0+ (Big Sur+)
- **Windows**: Windows 10/11 (通过WSL2推荐)

## 依赖管理

### Python环境

#### 1. Python版本要求

```bash
# 检查Python版本
python --version  # 需要3.8+

# 推荐使用pyenv管理Python版本
curl https://pyenv.run | bash
pyenv install 3.9.18
pyenv global 3.9.18
```

#### 2. 虚拟环境设置

```bash
# 创建虚拟环境
python -m venv agent-env

# 激活虚拟环境
# Linux/macOS
source agent-env/bin/activate

# Windows
agent-env\\Scripts\\activate

# 升级pip
pip install --upgrade pip
```

#### 3. 核心依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# 开发依赖
pip install -r requirements-dev.txt

# 可选GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 系统依赖

#### Ubuntu/Debian

```bash
# 更新包管理器
sudo apt update && sudo apt upgrade -y

# 基础工具
sudo apt install -y build-essential curl wget git vim htop

# Python开发工具
sudo apt install -y python3-dev python3-pip python3-venv

# 数据库客户端
sudo apt install -y postgresql-client redis-tools

# 多媒体支持（用于多模态处理）
sudo apt install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0

# 图像处理库
sudo apt install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev

# 系统监控工具
sudo apt install -y iotop nethogs
```

#### CentOS/RHEL

```bash
# 启用EPEL仓库
sudo yum install -y epel-release

# 开发工具
sudo yum groupinstall -y "Development Tools"
sudo yum install -y python3-devel python3-pip curl wget git

# 多媒体支持
sudo yum install -y ffmpeg

# 系统监控
sudo yum install -y htop iotop
```

#### macOS

```bash
# 安装Homebrew（如果没有）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 安装依赖
brew install python@3.9 postgresql redis ffmpeg git wget

# 更新shell配置
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

## 数据库设置

### PostgreSQL

#### 安装PostgreSQL

```bash
# Ubuntu/Debian
sudo apt install -y postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install -y postgresql-server postgresql-contrib
sudo postgresql-setup initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql

# macOS (使用Homebrew)
brew install postgresql
brew services start postgresql
```

#### 配置数据库

```sql
-- 连接到PostgreSQL
sudo -u postgres psql

-- 创建数据库和用户
CREATE DATABASE agent_db;
CREATE USER agent_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE agent_db TO agent_user;
ALTER USER agent_user CREATEDB;

-- 创建扩展
\\c agent_db;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

#### 配置文件优化

```bash
# 编辑postgresql.conf
sudo vim /etc/postgresql/13/main/postgresql.conf

# 推荐配置
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

### Redis

#### 安装Redis

```bash
# Ubuntu/Debian
sudo apt install -y redis-server

# CentOS/RHEL
sudo yum install -y redis
sudo systemctl enable redis
sudo systemctl start redis

# macOS
brew install redis
brew services start redis
```

#### 配置Redis

```bash
# 编辑redis.conf
sudo vim /etc/redis/redis.conf

# 推荐配置
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Elasticsearch

#### 安装Elasticsearch

```bash
# 下载并安装Elasticsearch
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.5.0-linux-x86_64.tar.gz
tar -xzf elasticsearch-8.5.0-linux-x86_64.tar.gz
sudo mv elasticsearch-8.5.0 /opt/elasticsearch
sudo chown -R $USER:$USER /opt/elasticsearch

# 启动Elasticsearch
/opt/elasticsearch/bin/elasticsearch -d
```

#### 配置Elasticsearch

```yaml
# config/elasticsearch.yml
cluster.name: agent-cluster
node.name: agent-node-1
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
```

## 环境变量配置

### 基础配置

```bash
# 创建环境文件
cp config/example.env .env

# 编辑环境变量
vim .env
```

#### 环境变量详解

```bash
# ===== 基础配置 =====
ENVIRONMENT=development          # 环境类型: development, testing, production
DEBUG=true                      # 调试模式
LOG_LEVEL=INFO                   # 日志级别: DEBUG, INFO, WARNING, ERROR

# ===== 数据库配置 =====
DATABASE_URL=postgresql://agent_user:password@localhost:5432/agent_db
REDIS_URL=redis://localhost:6379/0

# ===== AI模型配置 =====
LLM_API_KEY=your_openai_api_key
LLM_MODEL=gpt-3.5-turbo
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# ===== 记忆系统配置 =====
MEMORY_ENABLED=true
MEMORY_MAX_MEMORIES=10000
MEMORY_RETRIEVAL_LIMIT=5

# ===== RAG系统配置 =====
RAG_ENABLED=true
RAG_MAX_DOCUMENTS=5000
RAG_SIMILARITY_THRESHOLD=0.7
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ===== 工具系统配置 =====
TOOLS_ENABLED=true
TOOL_TIMEOUT=30

# ===== 监控配置 =====
MONITORING_ENABLED=true
METRICS_PORT=8080
HEALTH_CHECK_INTERVAL=30

# ===== 安全配置 =====
SECRET_KEY=your_very_long_and_secure_secret_key
JWT_SECRET_KEY=your_jwt_secret_key
ENCRYPTION_KEY=your_encryption_key
CORS_ORIGINS=http://localhost:3000,http://localhost:8000

# ===== 外部服务配置 =====
WEATHER_API_KEY=your_weather_api_key
SEARCH_API_KEY=your_search_api_key
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_email_password

# ===== 性能配置 =====
MAX_CONCURRENT_REQUESTS=50
WORKER_PROCESSES=4
REQUEST_TIMEOUT=30

# ===== 文件存储配置 =====
UPLOAD_DIR=./uploads
MAX_FILE_SIZE=10485760      # 10MB
ALLOWED_EXTENSIONS=txt,pdf,doc,docx,jpg,jpeg,png,gif,mp3,wav,mp4

# ===== 缓存配置 =====
CACHE_TYPE=redis
CACHE_TTL=3600               # 1小时
CACHE_MAX_SIZE=10000

# ===== 日志配置 =====
LOG_FILE=./logs/app.log
LOG_MAX_SIZE=10485760         # 10MB
LOG_BACKUP_COUNT=5
```

## 开发环境设置

### IDE配置

#### VS Code

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./agent-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.testing.pytestPath": "./agent-env/bin/pytest",
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "*.egg-info": true
    }
}
```

#### PyCharm

1. 打开项目目录
2. 配置Python解释器为创建的虚拟环境
3. 设置项目结构：
   - 标记`src`为源代码目录
   - 标记`tests`为测试目录
4. 配置代码风格：
   - 安装Black和isort插件
   - 设置行长度为88

### Git配置

```bash
# 全局配置
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 项目配置
git init
git add .
git commit -m "Initial commit"

# .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
agent-env/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 环境变量
.env
.env.local
.env.development

# 日志
logs/
*.log

# 数据文件
data/
backups/
uploads/
temp/

# 缓存
.cache/
*.cache

# 测试
.pytest_cache/
.coverage
htmlcov/
.tox/

# 操作系统
.DS_Store
Thumbs.db
EOF
```

## 性能优化

### 系统优化

#### Linux系统调优

```bash
# 编辑系统限制
sudo vim /etc/security/limits.conf

# 添加以下内容
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768

# 内核参数优化
sudo vim /etc/sysctl.conf

# 添加以下内容
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.tcp_fin_timeout = 30
net.ipv4.tcp_keepalive_time = 1200
net.ipv4.tcp_max_tw_buckets = 5000
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# 应用更改
sudo sysctl -p
```

#### Python性能优化

```bash
# 安装性能分析工具
pip install py-spy memory-profiler line-profiler

# 使用uWSGI作为生产服务器
pip install uwsgi

# uWSGI配置
cat > uwsgi.ini << EOF
[uwsgi]
module = src.server:app
master = true
processes = 4
threads = 2
socket = /tmp/agent.sock
chmod-socket = 666
vacuum = true
die-on-term = true
harakiri = 30
max-requests = 1000
lazy-apps = true
EOF
```

### 数据库优化

#### PostgreSQL优化

```sql
-- 创建索引
CREATE INDEX CONCURRENTLY idx_memory_content_gin ON memory USING gin(to_tsvector('english', content));
CREATE INDEX CONCURRENTLY idx_rag_documents_gin ON rag_documents USING gin(to_tsvector('english', content));
CREATE INDEX CONCURRENTLY idx_conversations_created_at ON conversations(created_at);

-- 分析表统计信息
ANALYZE memory;
ANALYZE rag_documents;
ANALYZE conversations;

-- 配置自动清理
ALTER TABLE memory SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE rag_documents SET (autovacuum_vacuum_scale_factor = 0.1);
```

#### Redis优化

```bash
# Redis配置优化
echo "always-show-logo no" >> /etc/redis/redis.conf
echo "tcp-keepalive 300" >> /etc/redis/redis.conf
echo "tcp-backlog 511" >> /etc/redis/redis.conf
echo "timeout 0" >> /etc/redis/redis.conf
echo "databases 16" >> /etc/redis/redis.conf
```

## 监控和诊断

### 系统监控

#### 安装监控工具

```bash
# 安装htop和iotop
sudo apt install -y htop iotop

# 安装nethogs
sudo apt install -y nethogs

# 安装glances（更全面的监控）
pip install glances
glances
```

#### 应用监控

```python
# config/monitoring.py
import psutil
import logging
from typing import Dict, Any

class SystemMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv
            },
            "process_count": len(psutil.pids())
        }

    def check_system_health(self) -> bool:
        """检查系统健康状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        if cpu_percent > 90:
            self.logger.warning(f"High CPU usage: {cpu_percent}%")

        if memory_percent > 85:
            self.logger.warning(f"High memory usage: {memory_percent}%")

        if disk_percent > 90:
            self.logger.warning(f"High disk usage: {disk_percent}%")

        return cpu_percent < 90 and memory_percent < 85 and disk_percent < 90
```

### 日志管理

#### 日志轮转配置

```bash
# 安装logrotate
sudo apt install -y logrotate

# 配置日志轮转
sudo vim /etc/logrotate.d/agent-app

# 配置内容
/path/to/agent/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 user user
    postrotate
        systemctl reload agent-app
    endscript
}
```

#### 结构化日志

```python
# config/logging.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """设置结构化日志"""

    # 创建日志格式器
    formatter = jsonlogger.JsonFormatter(
        fmt='%(asctime)s %(name)s %(levelname)s %(message)s %(filename)s %(lineno)d'
    )

    # 配置根日志器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # 文件处理器
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
```

## 故障排除

### 常见问题

#### 1. Python依赖问题

```bash
# 清理pip缓存
pip cache purge

# 重新安装依赖
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# 如果仍有问题，使用conda环境
conda create -n agent-env python=3.9
conda activate agent-env
pip install -r requirements.txt
```

#### 2. 数据库连接问题

```bash
# 检查PostgreSQL状态
sudo systemctl status postgresql

# 测试连接
psql -h localhost -U agent_user -d agent_db

# 检查网络连接
netstat -an | grep 5432
```

#### 3. Redis连接问题

```bash
# 检查Redis状态
sudo systemctl status redis

# 测试连接
redis-cli ping

# 检查配置
redis-cli config get "*"
```

#### 4. 权限问题

```bash
# 修复文件权限
sudo chown -R $USER:$USER /path/to/agent
chmod -R 755 /path/to/agent

# 修复目录权限
sudo mkdir -p /var/log/agent
sudo chown -R $USER:$USER /var/log/agent
```

#### 5. 端口占用

```bash
# 查看端口占用
sudo netstat -tlnp | grep :8000
sudo lsof -i :8000

# 杀死占用进程
sudo kill -9 PID
```

### 性能问题诊断

#### CPU使用率高

```bash
# 查看进程CPU使用
top -o %CPU

# 查看Python进程详情
ps aux | grep python

# 使用py-spy分析
sudo py-spy top --pid <PID>
```

#### 内存使用高

```bash
# 查看内存使用
free -h

# 查看进程内存使用
ps aux --sort=-%mem | head

# 使用memory-profiler
python -m memory_profiler your_script.py
```

#### 磁盘I/O问题

```bash
# 查看磁盘使用
df -h

# 查看I/O统计
iotop

# 查看磁盘活动
sudo iotop -o
```

## 安全配置

### 基础安全设置

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 配置防火墙
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp

# 禁用不必要的服务
sudo systemctl disable apache2
sudo systemctl disable mysql
```

### SSL/TLS配置

```bash
# 生成自签名证书（开发环境）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 生产环境使用Let's Encrypt
sudo apt install certbot
sudo certbot --nginx -d your-domain.com
```

### 环境变量安全

```bash
# 使用加密的密钥管理
pip install python-dotenv-vault

# 创建加密的环境文件
dotenv-vault encrypt .env
```

这个环境设置指南提供了完整的环境配置、依赖管理和故障排除方案，确保系统能够在不同环境中稳定运行。
