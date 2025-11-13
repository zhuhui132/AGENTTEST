# Kubernetes部署指南

## 概览

本指南详细介绍如何在Kubernetes集群中部署AI Agent系统，包括资源配置、服务发现、自动扩展、监控告警和故障处理。

## 前置要求

### 集群要求

#### 最低配置

| 组件 | 最小配置 | 推荐配置 |
|------|----------|----------|
| Master节点 | 2核, 4GB, 20GB | 4核, 8GB, 50GB |
| Worker节点 | 2核, 4GB, 50GB | 4核, 16GB, 100GB |
| 集群规模 | 1个Master, 2个Worker | 3个Master, 5+个Worker |

#### 软件要求

- **Kubernetes**: 1.24+
- **Helm**: 3.8+
- **kubectl**: 与Kubernetes版本匹配
- **Container Runtime**: containerd 1.6+ 或 Docker 20.10+

### 存储要求

```bash
# 确保有足够的存储空间
df -h

# 推荐配置
# 数据存储: 100GB+
# 日志存储: 50GB+
# 备份存储: 200GB+
```

## 集群准备

### 安装kubectl

```bash
# Linux (x86_64)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# macOS
brew install kubectl

# 验证安装
kubectl version --client
```

### 配置集群访问

```bash
# 配置kubeconfig
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

# 验证集群连接
kubectl cluster-info
kubectl get nodes
```

### 安装Helm

```bash
# 安装Helm
curl https://get.helm.sh/helm-v3.8.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/helm

# 添加Helm仓库
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update
```

## 命名空间和配置

### 创建命名空间

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-system
  labels:
    name: agent-system
    environment: production
    project: ai-agent
    managed-by: kubernetes
  annotations:
    description: "AI Agent System Namespace"
    created-by: "admin"
```

```bash
# 应用命名空间
kubectl apply -f namespace.yaml

# 设置默认命名空间
kubectl config set-context --current --namespace=agent-system
```

### ConfigMap配置

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: agent-config
  namespace: agent-system
  labels:
    app: agent-system
    component: config
data:
  # 基础配置
  LOG_LEVEL: "INFO"
  ENVIRONMENT: "production"
  DEBUG: "false"

  # 性能配置
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
  DB_POOL_SIZE: "20"
  DB_MAX_CONNECTIONS: "100"

  # Redis配置
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  REDIS_DB: "0"
  REDIS_POOL_SIZE: "50"

  # Elasticsearch配置
  ES_HOST: "elasticsearch-service"
  ES_PORT: "9200"
  ES_INDEX_PREFIX: "agent"
  ES_SHARDS: "3"
  ES_REPLICAS: "1"

  # LLM配置
  LLM_PROVIDER: "openai"
  LLM_MODEL: "gpt-4"
  LLM_TEMPERATURE: "0.7"
  LLM_MAX_TOKENS: "2048"
  LLM_TIMEOUT: "30"
  LLM_RETRY_ATTEMPTS: "3"
  LLM_RETRY_DELAY: "1"

  # 记忆系统配置
  MEMORY_ENABLED: "true"
  MEMORY_MAX_MEMORIES: "10000"
  MEMORY_RETRIEVAL_LIMIT: "10"
  MEMORY_BATCH_SIZE: "100"
  MEMORY_EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"

  # RAG系统配置
  RAG_ENABLED: "true"
  RAG_MAX_DOCUMENTS: "50000"
  RAG_RETRIEVAL_LIMIT: "10"
  RAG_SIMILARITY_THRESHOLD: "0.7"
  RAG_CHUNK_SIZE: "512"
  RAG_CHUNK_OVERLAP: "50"
  RAG_EMBEDDING_BATCH_SIZE: "32"

  # 工具系统配置
  TOOLS_ENABLED: "true"
  TOOL_TIMEOUT: "30"
  TOOL_MAX_CONCURRENT: "10"

  # 监控配置
  MONITORING_ENABLED: "true"
  METRICS_PORT: "9090"
  METRICS_PATH: "/metrics"
  PROMETHEUS_NAMESPACE: "agent"
  PROMETHEUS_SUBSYSTEM: "app"

  # 安全配置
  CORS_ORIGINS: "https://api.your-domain.com,https://app.your-domain.com"
  RATE_LIMIT_ENABLED: "true"
  RATE_LIMIT_REQUESTS_PER_MINUTE: "1000"
  JWT_EXPIRATION_HOURS: "24"
  SESSION_TIMEOUT_MINUTES: "60"

  # 缓存配置
  CACHE_ENABLED: "true"
  CACHE_TYPE: "redis"
  CACHE_TTL: "3600"
  CACHE_MAX_SIZE: "10000"
  CACHE_PREFIX: "agent:"

  # 文件上传配置
  UPLOAD_MAX_FILE_SIZE: "10485760"  # 10MB
  UPLOAD_ALLOWED_EXTENSIONS: "txt,pdf,doc,docx,jpg,jpeg,png,gif,mp3,wav,mp4"
  UPLOAD_TEMP_DIR: "/tmp/uploads"

  # 日志配置
  LOG_FORMAT: "json"
  LOG_MAX_SIZE: "10485760"    # 10MB
  LOG_MAX_BACKUPS: "5"
  LOG_COMPRESS: "true"

  # 应用配置
  APP_NAME: "AI Agent System"
  APP_VERSION: "v3.2.0"
  APP_DESCRIPTION: "Enterprise AI Agent Platform"
  APP_MAINTAINER: "ai-team@yourcompany.com"
```

### Secret配置

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: agent-secrets
  namespace: agent-system
  labels:
    app: agent-system
    component: secrets
type: Opaque
data:
  # Base64编码的密钥
  postgres-password: eW91cl9wb3N0Z3Jlc19wYXNzd29yZA==  # your_postgres_password
  redis-password: eW91cl9yZWRpc19wYXNzd29yZA==             # your_redis_password
  llm-api-key: eW91cl9sbG1fYXBpX2tleQ==                # your_llm_api_key
  jwt-secret: eW91cl9qd3Rfc2VjcmV0X2tleQ==               # your_jwt_secret_key
  encryption-key: eW91cl9lbmNyeXB0aW9uX2tleQ==           # your_encryption_key
  cookie-secret: eW91cl9jb29raWVfc2VjcmV0X2tleQ==        # your_cookie_secret_key

  # 外部服务密钥
  smtp-user: eW91cl9lbWFpbEBnbWFpbC5jb20=              # your_email@gmail.com
  smtp-pass: eW91cl9lbWFpbF9wYXNzd29yZA==               # your_email_password
  weather-api-key: eW91cl93ZWF0aGVyX2FwaV9rZXk=          # your_weather_api_key
  search-api-key: eW91cl9zZWFyY2hfYXBpX2tleQ==             # your_search_api_key
  slack-webhook: aHR0cHM6Ly9ob29rcy5zbGFjay5jb20vc2VydmljZXMv  # slack_webhook_url

  # 证书密钥（如果有）
  tls-crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t  # tls_certificate
  tls-key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t # tls_private_key
```

## 数据存储部署

### PostgreSQL部署

#### 1. PersistentVolume配置

```yaml
# postgres-pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: postgres-pv
  namespace: agent-system
  labels:
    type: local
    app: postgres
spec:
  storageClassName: manual
  capacity:
    storage: 50Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  hostPath:
    path: /data/postgres
```

#### 2. PersistentVolumeClaim

```yaml
# postgres-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: agent-system
  labels:
    app: postgres
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
```

#### 3. ConfigMap配置

```yaml
# postgres-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: postgres-config
  namespace: agent-system
data:
  postgresql.conf: |
    # PostgreSQL配置
    listen_addresses = '*'
    port = 5432
    max_connections = 200
    shared_buffers = 128MB
    effective_cache_size = 512MB
    maintenance_work_mem = 32MB
    checkpoint_completion_target = 0.9
    wal_buffers = 16MB
    default_statistics_target = 100
    random_page_cost = 1.1
    effective_io_concurrency = 200
    work_mem = 4MB
    min_wal_size = 80MB
    max_wal_size = 1GB
    checkpoint_segments = 32
    checkpoint_timeout = 10min
    archive_mode = on
    archive_command = 'cp %p /var/lib/postgresql/archive/%f'

  pg_hba.conf: |
    # PostgreSQL认证配置
    local   all             all                                     trust
    host    all             all             127.0.0.1/32            md5
    host    all             all             0.0.0.0/0               md5
    host    all             all             ::1/128                 md5
```

#### 4. Deployment配置

```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: agent-system
  labels:
    app: postgres
    component: database
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
        component: database
    spec:
      # 节点选择
      nodeSelector:
        node-type: database

      # 容忍度
      tolerations:
      - key: "database"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"

      containers:
      - name: postgres
        image: postgres:13-alpine
        imagePullPolicy: IfNotPresent

        # 端口
        ports:
        - containerPort: 5432
          name: postgres
          protocol: TCP

        # 环境变量
        envFrom:
        - configMapRef:
            name: postgres-config
        - secretRef:
            name: agent-secrets

        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: DB_NAME
        - name: POSTGRES_USER
          valueFrom:
            configMapKeyRef:
              name: agent-config
              key: DB_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: postgres-password
        - name: PGDATA
          value: /var/lib/postgresql/data/pgdata

        # 资源限制
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        # 存储挂载
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        - name: postgres-config-volume
          mountPath: /etc/postgresql/postgresql.conf
          subPath: postgresql.conf
        - name: postgres-config-volume
          mountPath: /etc/postgresql/pg_hba.conf
          subPath: pg_hba.conf
        - name: postgres-archive
          mountPath: /var/lib/postgresql/archive

        # 健康检查
        livenessProbe:
          exec:
            command:
            - sh
            - -c
            - "pg_isready -U $POSTGRES_USER -d $POSTGRES_DB"
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          exec:
            command:
            - sh
            - -c
            - "pg_isready -U $POSTGRES_USER -d $POSTGRES_DB"
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

      # 卷配置
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
      - name: postgres-config-volume
        configMap:
          name: postgres-config
      - name: postgres-archive
        emptyDir: {}
```

#### 5. Service配置

```yaml
# postgres-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: agent-system
  labels:
    app: postgres
    component: database
spec:
  type: ClusterIP
  ports:
  - name: postgres
    port: 5432
    targetPort: 5432
    protocol: TCP
  selector:
    app: postgres

  # 服务发现注解
  annotations:
    prometheus.io/scrape: "false"
    description: "PostgreSQL database service"
```

### Redis部署

```yaml
# redis-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: agent-system
  labels:
    app: redis
    component: cache
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
        component: cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        imagePullPolicy: IfNotPresent

        ports:
        - containerPort: 6379
          name: redis
          protocol: TCP

        command:
        - redis-server
        - /etc/redis/redis.conf
        - --requirepass
        - $(REDIS_PASSWORD)
        - --appendonly
        - "yes"
        - --save
        - "900 1"
        - "300 10"
        - "60 10000"

        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: agent-secrets
              key: redis-password

        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"

        volumeMounts:
        - name: redis-config-volume
          mountPath: /etc/redis
        - name: redis-storage
          mountPath: /data

        livenessProbe:
          exec:
            command:
            - redis-cli
            - -a
            - $(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          exec:
            command:
            - redis-cli
            - -a
            - $(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

      volumes:
      - name: redis-config-volume
        configMap:
          name: redis-config
      - name: redis-storage
        persistentVolumeClaim:
          claimName: redis-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: agent-system
  labels:
    app: redis
    component: cache
spec:
  type: ClusterIP
  ports:
  - name: redis
    port: 6379
    targetPort: 6379
    protocol: TCP
  selector:
    app: redis
```

### Elasticsearch部署

```yaml
# elasticsearch-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: elasticsearch
  namespace: agent-system
  labels:
    app: elasticsearch
    component: search
spec:
  serviceName: elasticsearch
  replicas: 3
  selector:
    matchLabels:
      app: elasticsearch
  template:
    metadata:
      labels:
        app: elasticsearch
        component: search
    spec:
      # 初始化容器
      initContainers:
      - name: increase-vm-max-map
        image: busybox
        command: ["sysctl", "-w", "vm.max_map_count=262144"]
        securityContext:
          privileged: true
      - name: increase-fd-ulimit
        image: busybox
        command: ["sh", "-c", "ulimit -n 65536"]
        securityContext:
          privileged: true

      containers:
      - name: elasticsearch
        image: elasticsearch:8.5.0
        imagePullPolicy: IfNotPresent

        ports:
        - containerPort: 9200
          name: http
          protocol: TCP
        - containerPort: 9300
          name: transport
          protocol: TCP

        env:
        - name: cluster.name
          value: "agent-cluster"
        - name: node.name
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: discovery.seed_hosts
          value: "elasticsearch-0.elasticsearch,elasticsearch-1.elasticsearch,elasticsearch-2.elasticsearch"
        - name: cluster.initial_master_nodes
          value: "elasticsearch-0,elasticsearch-1,elasticsearch-2"
        - name: ES_JAVA_OPTS
          value: "-Xms1g -Xmx1g"
        - name: xpack.security.enabled
          value: "false"
        - name: xpack.security.enrollment.enabled
          value: "false"

        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"

        volumeMounts:
        - name: elasticsearch-data
          mountPath: /usr/share/elasticsearch/data

        livenessProbe:
          httpGet:
            path: /_cluster/health
            port: http
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /_cluster/health?wait_for_status=yellow&timeout=1s
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

      volumeClaimTemplates:
      - metadata:
        name: elasticsearch-data
        labels:
          app: elasticsearch
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-ssd
        resources:
          requests:
            storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: elasticsearch-service
  namespace: agent-system
  labels:
    app: elasticsearch
    component: search
spec:
  type: ClusterIP
  ports:
  - name: http
    port: 9200
    targetPort: 9200
    protocol: TCP
  - name: transport
    port: 9300
    targetPort: 9300
    protocol: TCP
  selector:
    app: elasticsearch
```

## 应用部署

### Agent应用部署

```yaml
# agent-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-app
  namespace: agent-system
  labels:
    app: agent-app
    component: application
    version: v3.2.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: agent-app
  template:
    metadata:
      labels:
        app: agent-app
        component: application
        version: v3.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
        prometheus.io/scheme: "http"
    spec:
      # 服务账户
      serviceAccountName: agent-service-account

      # 节点选择
      nodeSelector:
        node-type: application
        kubernetes.io/arch: amd64

      # 容忍度
      tolerations:
      - key: "application"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      - key: "dedicated"
        operator: "Equal"
        value: "agent-app"
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
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - key: app
                operator: In
                values:
                - agent-app
            topologyKey: kubernetes.io/hostname

      containers:
      - name: agent-app
        image: your-registry/agent-app:v3.2.0
        imagePullPolicy: Always

        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP

        # 环境变量
        envFrom:
        - configMapRef:
            name: agent-config
        - secretRef:
            name: agent-secrets

        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName

        # 资源限制
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        # 存储挂载
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        - name: temp-volume
          mountPath: /app/temp
        - name: cache-volume
          mountPath: /app/cache
        - name: models-volume
          mountPath: /app/models

        # 健康检查
        livenessProbe:
          httpGet:
            path: /health
            port: http
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
          successThreshold: 1

        readinessProbe:
          httpGet:
            path: /ready
            port: http
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
          periodSeconds: 5

        # 启动探针
        startupProbe:
          httpGet:
            path: /startup
            port: http
            scheme: HTTP
          failureThreshold: 30
          periodSeconds: 10
          timeoutSeconds: 5

        # 生命周期
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]

      # 卷配置
      volumes:
      - name: config-volume
        configMap:
          name: agent-config
      - name: logs-volume
        emptyDir:
          sizeLimit: 1Gi
      - name: temp-volume
        emptyDir:
          sizeLimit: 500Mi
      - name: cache-volume
        emptyDir:
          sizeLimit: 2Gi
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
```

### Service配置

```yaml
# agent-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: agent-app-service
  namespace: agent-system
  labels:
    app: agent-app
    component: application
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9090"
    description: "Agent Application Service"
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
# Headless Service for StatefulSet
apiVersion: v1
kind: Service
metadata:
  name: agent-app-headless
  namespace: agent-system
  labels:
    app: agent-app
    component: application
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
```

### Ingress配置

```yaml
# agent-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-app-ingress
  namespace: agent-system
  labels:
    app: agent-app
    component: ingress
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/client-max-body-size: "50m"
    nginx.ingress.kubernetes.io/rate-limit-connections: "100"
    nginx.ingress.kubernetes.io/rate-limit-connections-burst: "200"
    nginx.ingress.kubernetes.io/rate-limit-requests-per-second: "20"
    nginx.ingress.kubernetes.io/rate-limit-burst: "40"
    # 安全头
    nginx.ingress.kubernetes.io/configuration-snippet: |
      more_set_headers "X-Frame-Options: DENY";
      more_set_headers "X-Content-Type-Options: nosniff";
      more_set_headers "X-XSS-Protection: 1; mode=block";
      more_set_headers "Strict-Transport-Security: max-age=31536000; includeSubDomains; preload";
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
      - path: /metrics
        pathType: Exact
        backend:
          service:
            name: agent-app-service
            port:
              number: 9090
```

## 自动扩展配置

### HorizontalPodAutoscaler

```yaml
# agent-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-app-hpa
  namespace: agent-system
  labels:
    app: agent-app
    component: hpa
  annotations:
    description: "Horizontal Pod Autoscaler for Agent Application"
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
  # CPU指标
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70

  # 内存指标
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80

  # 自定义指标
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"

  - type: Pods
    pods:
      metric:
        name: response_time_p95
      target:
        type: AverageValue
        averageValue: "2"

  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Min

    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
      selectPolicy: Max
```

### VerticalPodAutoscaler

```yaml
# agent-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: agent-app-vpa
  namespace: agent-system
  labels:
    app: agent-app
    component: vpa
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: agent-app
      maxAllowed:
        cpu: "2000m"
        memory: "4Gi"
      minAllowed:
        cpu: "100m"
        memory: "256Mi"
      controlledResources: ["cpu", "memory"]
```

### ClusterAutoscaler

```yaml
# cluster-autoscaler.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
  labels:
    app: cluster-autoscaler
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      containers:
      - image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.21.0
        name: cluster-autoscaler
        resources:
          limits:
            cpu: 100m
            memory: 300Mi
          requests:
            cpu: 100m
            memory: 300Mi
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/agent-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
```

## 监控部署

### Prometheus部署

```yaml
# prometheus-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: agent-system
  labels:
    app: prometheus
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
        component: monitoring
    spec:
      serviceAccountName: prometheus
      containers:
      - name: prometheus
        image: prom/prometheus:v2.37.0
        imagePullPolicy: IfNotPresent

        ports:
        - containerPort: 9090
          name: web
          protocol: TCP

        command:
        - /bin/prometheus
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus/
        - --storage.tsdb.retention.time=30d
        - --web.enable-lifecycle
        - --web.enable-admin-api
        - --web.route-prefix=/

        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name

        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"

        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus/
          readOnly: true
        - name: storage-volume
          mountPath: /prometheus/
        - name: rules-volume
          mountPath: /etc/prometheus/rules/
          readOnly: true

        livenessProbe:
          httpGet:
            path: /-/healthy
            port: web
          initialDelaySeconds: 30
          periodSeconds: 15
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /-/ready
            port: web
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: rules-volume
        configMap:
          name: prometheus-rules
      - name: storage-volume
        persistentVolumeClaim:
          claimName: prometheus-pvc
```

### Grafana部署

```yaml
# grafana-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: agent-system
  labels:
    app: grafana
    component: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
        component: monitoring
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:9.3.0
        imagePullPolicy: IfNotPresent

        ports:
        - containerPort: 3000
          name: web
          protocol: TCP

        env:
        - name: GF_SECURITY_ADMIN_USER
          value: "admin"
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secrets
              key: admin-password
        - name: GF_INSTALL_PLUGINS
          value: "grafana-clock-panel,grafana-simple-json-datasource"
        - name: GF_SERVER_DOMAIN
          value: "grafana.your-domain.com"
        - name: GF_SERVER_ROOT_URL
          value: "https://grafana.your-domain.com"

        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"

        volumeMounts:
        - name: data-volume
          mountPath: /var/lib/grafana
        - name: config-volume
          mountPath: /etc/grafana/provisioning/
          readOnly: true
        - name: dashboards-volume
          mountPath: /var/lib/grafana/dashboards/
          readOnly: true

        livenessProbe:
          httpGet:
            path: /api/health
            port: web
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /api/health
            port: web
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: grafana-pvc
      - name: config-volume
        configMap:
          name: grafana-config
      - name: dashboards-volume
        configMap:
          name: grafana-dashboards
```

## 部署脚本

### 自动化部署脚本

```bash
#!/bin/bash
# scripts/k8s-deploy.sh

set -e

# 配置
NAMESPACE="agent-system"
CONTEXT="${KUBE_CONTEXT:-current}"
ENVIRONMENT="${ENVIRONMENT:-production}"
HELM_CHART_PATH="./helm/agent-system"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# 检查依赖
check_dependencies() {
    log "检查依赖..."

    if ! command -v kubectl &> /dev/null; then
        error "kubectl未安装"
        exit 1
    fi

    if ! command -v helm &> /dev/null; then
        error "helm未安装"
        exit 1
    fi

    # 检查集群连接
    if ! kubectl cluster-info &> /dev/null; then
        error "无法连接到Kubernetes集群"
        exit 1
    fi

    log "依赖检查完成"
}

# 创建命名空间
create_namespace() {
    log "创建命名空间: $NAMESPACE"

    kubectl apply -f namespace.yaml

    # 设置默认命名空间
    kubectl config set-context --current --namespace=$NAMESPACE

    log "命名空间创建完成"
}

# 部署配置
deploy_configmaps() {
    log "部署ConfigMap和Secret..."

    kubectl apply -f configmap.yaml
    kubectl apply -f secret.yaml
    kubectl apply -f postgres-config.yaml
    kubectl apply -f redis-config.yaml

    log "配置部署完成"
}

# 部署存储
deploy_storage() {
    log "部署存储资源..."

    # 创建PV和PVC
    kubectl apply -f postgres-pv.yaml
    kubectl apply -f postgres-pvc.yaml
    kubectl apply -f redis-pvc.yaml
    kubectl apply -f elasticsearch-pvc.yaml
    kubectl apply -f prometheus-pvc.yaml
    kubectl apply -f grafana-pvc.yaml
    kubectl apply -f models-pvc.yaml

    log "存储部署完成"
}

# 部署数据库
deploy_databases() {
    log "部署数据库服务..."

    # 部署PostgreSQL
    kubectl apply -f postgres-deployment.yaml
    kubectl apply -f postgres-service.yaml

    # 部署Redis
    kubectl apply -f redis-deployment.yaml
    kubectl apply -f redis-service.yaml

    # 部署Elasticsearch
    kubectl apply -f elasticsearch-statefulset.yaml
    kubectl apply -f elasticsearch-service.yaml

    # 等待数据库就绪
    log "等待数据库服务就绪..."
    kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
    kubectl wait --for=condition=ready pod -l app=elasticsearch --timeout=600s

    log "数据库部署完成"
}

# 部署应用
deploy_application() {
    log "部署应用服务..."

    # 部署应用
    kubectl apply -f agent-deployment.yaml
    kubectl apply -f agent-service.yaml
    kubectl apply -f agent-ingress.yaml

    # 等待应用就绪
    log "等待应用服务就绪..."
    kubectl wait --for=condition=available deployment/agent-app --timeout=600s

    log "应用部署完成"
}

# 部署监控
deploy_monitoring() {
    log "部署监控服务..."

    # 创建监控命名空间
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

    # 部署Prometheus
    kubectl apply -f prometheus-deployment.yaml -n monitoring
    kubectl apply -f prometheus-service.yaml -n monitoring
    kubectl apply -f prometheus-config.yaml -n monitoring
    kubectl apply -f prometheus-rules.yaml -n monitoring

    # 部署Grafana
    kubectl apply -f grafana-deployment.yaml -n monitoring
    kubectl apply -f grafana-service.yaml -n monitoring
    kubectl apply -f grafana-config.yaml -n monitoring

    log "监控部署完成"
}

# 部署自动扩展
deploy_autoscaling() {
    log "部署自动扩展..."

    kubectl apply -f agent-hpa.yaml
    kubectl apply -f agent-vpa.yaml

    # 如果需要Cluster Autoscaler
    if [ "$ENABLE_CLUSTER_AUTOSCALER" = "true" ]; then
        kubectl apply -f cluster-autoscaler.yaml -n kube-system
    fi

    log "自动扩展部署完成"
}

# 健康检查
health_check() {
    log "执行健康检查..."

    # 检查Pod状态
    kubectl get pods -l app=agent-app

    # 检查服务状态
    kubectl get services

    # 检查Ingress状态
    kubectl get ingress

    # 测试应用健康端点
    if kubectl get ingress agent-app-ingress &> /dev/null; then
        INGRESS_URL=$(kubectl get ingress agent-app-ingress -o jsonpath='{.spec.rules[0].host}')
        log "应用访问地址: https://$INGRESS_URL"

        # 测试健康检查
        if curl -f -k "https://$INGRESS_URL/health" &> /dev/null; then
            log "应用健康检查通过"
        else
            warn "应用健康检查失败"
        fi
    fi

    log "健康检查完成"
}

# 主函数
main() {
    local action="${1:-deploy}"

    case "$action" in
        "deploy")
            log "开始部署AI Agent系统到Kubernetes..."

            check_dependencies
            create_namespace
            deploy_configmaps
            deploy_storage
            deploy_databases
            deploy_application
            deploy_monitoring
            deploy_autoscaling
            health_check

            log "Kubernetes部署完成！"
            ;;
        "update")
            log "更新部署..."

            kubectl apply -f .
            kubectl rollout status deployment/agent-app

            log "更新完成"
            ;;
        "scale")
            local replicas="${2:-3}"
            log "扩展应用到 $replicas 个副本..."

            kubectl scale deployment agent-app --replicas=$replicas
            kubectl rollout status deployment/agent-app

            log "扩展完成"
            ;;
        "rollback")
            local revision="${2:-previous}"
            log "回滚到版本: $revision"

            kubectl rollout undo deployment/agent-app --to-revision=$revision
            kubectl rollout status deployment/agent-app

            log "回滚完成"
            ;;
        "status")
            log "查看部署状态..."

            kubectl get all -l app=agent-app
            kubectl top pods -l app=agent-app
            ;;
        "logs")
            local pod="${2:-$(kubectl get pods -l app=agent-app -o jsonpath='{.items[0].metadata.name}')}"
            log "查看Pod日志: $pod"

            kubectl logs -f $pod
            ;;
        "cleanup")
            warn "清理所有资源..."

            kubectl delete -f .
            kubectl delete namespace $NAMESPACE
            kubectl delete namespace monitoring

            log "清理完成"
            ;;
        *)
            echo "用法: $0 {deploy|update|scale|rollback|status|logs|cleanup} [options]"
            echo "  deploy              - 部署完整系统"
            echo "  update              - 更新部署"
            echo "  scale [replicas]    - 扩展应用副本数"
            echo "  rollback [revision] - 回滚到指定版本"
            echo "  status              - 查看部署状态"
            echo "  logs [pod]         - 查看Pod日志"
            echo "  cleanup             - 清理所有资源"
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
```

这个Kubernetes部署指南提供了完整的企业级部署方案，包括高可用配置、自动扩展、监控告警和故障处理，确保系统在生产环境中的稳定运行。
