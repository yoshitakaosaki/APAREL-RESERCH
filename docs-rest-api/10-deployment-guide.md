# デプロイメントガイド

## 1. 概要

REST APIサーバーを本番環境にデプロイするための包括的なガイドです。環境設定、スケーリング戦略、認証サーバー連携、CI/CDパイプラインなど、安定した運用に必要な要素を網羅します。

## 2. 環境構成

### 2.1 推奨システム要件

```yaml
# 最小要件
minimum:
  cpu: 2 cores
  memory: 4GB RAM
  storage: 20GB SSD
  
# 推奨要件
recommended:
  cpu: 4 cores
  memory: 8GB RAM
  storage: 50GB SSD
  
# 高負荷環境
high_load:
  cpu: 8+ cores
  memory: 16GB+ RAM
  storage: 100GB+ SSD
  load_balancer: required
  cache_servers: 2+ Redis instances
```

### 2.2 環境変数設定

```bash
# .env.production

# Django設定
SECRET_KEY=your-production-secret-key-here
DEBUG=False
ALLOWED_HOSTS=api.example.com,api2.example.com
ENVIRONMENT=production

# データベース
DATABASE_URL=postgresql://apiuser:password@db.example.com:5432/apidb
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=40
DB_POOL_TIMEOUT=30

# Redis
REDIS_URL=redis://:password@redis.example.com:6379/0
REDIS_POOL_SIZE=50
REDIS_SOCKET_TIMEOUT=5

# 認証サーバー
AUTH_SERVER_URL=https://auth.example.com
JWKS_URL=https://auth.example.com/.well-known/jwks.json
OAUTH_ISSUER=https://auth.example.com
OAUTH_AUDIENCE=api.example.com

# CORS
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com

# セキュリティ
SESSION_COOKIE_SECURE=True
CSRF_COOKIE_SECURE=True
SECURE_SSL_REDIRECT=True
SECURE_HSTS_SECONDS=31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS=True
SECURE_HSTS_PRELOAD=True

# ログ
LOG_LEVEL=INFO
SENTRY_DSN=https://xxxxx@sentry.io/project-id

# メール
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=notifications@example.com
EMAIL_HOST_PASSWORD=email-password

# ストレージ
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_STORAGE_BUCKET_NAME=api-storage-bucket
AWS_S3_REGION_NAME=ap-northeast-1

# 監視
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx
PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key
PROMETHEUS_METRICS_ENABLED=True
```

## 3. Docker設定

### 3.1 本番用Dockerfile

```dockerfile
# Dockerfile.production
FROM python:3.10-slim as builder

# ビルド引数
ARG DEBIAN_FRONTEND=noninteractive

# システムパッケージ
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /app

# 依存関係インストール（キャッシュ効率化）
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# マルチステージビルド（最終イメージ）
FROM python:3.10-slim

# セキュリティアップデート
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 非rootユーザー作成
RUN useradd -m -u 1000 -s /bin/bash appuser

# 作業ディレクトリ
WORKDIR /app

# ビルダーステージから依存関係コピー
COPY --from=builder /root/.local /home/appuser/.local

# アプリケーションコピー
COPY --chown=appuser:appuser . .

# パス設定
ENV PATH=/home/appuser/.local/bin:$PATH

# 静的ファイル収集
RUN python manage.py collectstatic --noinput

# ユーザー切り替え
USER appuser

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# ポート
EXPOSE 8000

# 起動コマンド
CMD ["gunicorn", "api_server.wsgi:application", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "4", \
     "--threads", "2", \
     "--worker-class", "sync", \
     "--worker-tmp-dir", "/dev/shm", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--log-level", "info", \
     "--timeout", "60", \
     "--graceful-timeout", "30", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100"]
```

### 3.2 Docker Compose設定

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.production
    image: api-server:latest
    container_name: api_server
    restart: unless-stopped
    environment:
      - DJANGO_SETTINGS_MODULE=api_server.settings.production
    env_file:
      - .env.production
    ports:
      - "8000:8000"
    volumes:
      - static_volume:/app/staticfiles
      - media_volume:/app/media
      - logs_volume:/app/logs
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - api_network
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "10"

  db:
    image: postgres:15-alpine
    container_name: api_db
    restart: unless-stopped
    environment:
      POSTGRES_DB: apidb
      POSTGRES_USER: apiuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - api_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U apiuser -d apidb"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: api_redis
    restart: unless-stopped
    command: >
      redis-server
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 1gb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    volumes:
      - redis_data:/data
    networks:
      - api_network
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  nginx:
    image: nginx:alpine
    container_name: api_nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/conf.d:/etc/nginx/conf.d
      - static_volume:/var/www/static
      - media_volume:/var/www/media
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - api
    networks:
      - api_network

volumes:
  postgres_data:
  redis_data:
  static_volume:
  media_volume:
  logs_volume:

networks:
  api_network:
    driver: bridge
```

### 3.3 Nginx設定

```nginx
# nginx/conf.d/api.conf
upstream api_backend {
    least_conn;
    server api:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

# レート制限ゾーン
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/s;

server {
    listen 80;
    server_name api.example.com;
    
    # HTTPSへリダイレクト
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;
    
    # SSL設定
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # セキュリティヘッダー
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # gzip圧縮
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/json application/javascript application/xml+rss;
    
    # 静的ファイル
    location /static/ {
        alias /var/www/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
    
    location /media/ {
        alias /var/www/media/;
        expires 7d;
        add_header Cache-Control "public";
    }
    
    # API
    location /api/ {
        # レート制限
        limit_req zone=api_limit burst=20 nodelay;
        
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # タイムアウト設定
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # バッファリング
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
        
        # キープアライブ
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # ヘルスチェック（レート制限なし）
    location = /api/v1/health {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # メトリクス（内部アクセスのみ）
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://api_backend;
    }
}
```

## 4. デプロイメントプロセス

### 4.1 デプロイメントスクリプト

```bash
#!/bin/bash
# deploy.sh

set -e

# 設定
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
DOCKER_REGISTRY="your-registry.com"
IMAGE_NAME="api-server"

echo "🚀 Starting deployment for $ENVIRONMENT environment..."

# 1. 環境チェック
if [ ! -f ".env.$ENVIRONMENT" ]; then
    echo "❌ Environment file .env.$ENVIRONMENT not found!"
    exit 1
fi

# 2. イメージビルド
echo "🔨 Building Docker image..."
docker build -f Dockerfile.production -t $IMAGE_NAME:$VERSION .

# 3. イメージタグ付けとプッシュ
echo "📤 Pushing image to registry..."
docker tag $IMAGE_NAME:$VERSION $DOCKER_REGISTRY/$IMAGE_NAME:$VERSION
docker tag $IMAGE_NAME:$VERSION $DOCKER_REGISTRY/$IMAGE_NAME:latest
docker push $DOCKER_REGISTRY/$IMAGE_NAME:$VERSION
docker push $DOCKER_REGISTRY/$IMAGE_NAME:latest

# 4. データベースマイグレーション
echo "🗄️ Running database migrations..."
docker-compose -f docker-compose.$ENVIRONMENT.yml run --rm api python manage.py migrate

# 5. 静的ファイル収集
echo "📁 Collecting static files..."
docker-compose -f docker-compose.$ENVIRONMENT.yml run --rm api python manage.py collectstatic --noinput

# 6. ヘルスチェック
echo "🏥 Running pre-deployment health check..."
docker-compose -f docker-compose.$ENVIRONMENT.yml run --rm api python manage.py check --deploy

# 7. デプロイ実行
echo "🚢 Deploying new version..."
docker-compose -f docker-compose.$ENVIRONMENT.yml pull
docker-compose -f docker-compose.$ENVIRONMENT.yml up -d --remove-orphans

# 8. ヘルスチェック待機
echo "⏳ Waiting for health check..."
sleep 10

# 9. デプロイ確認
echo "✅ Verifying deployment..."
curl -f http://localhost/api/v1/health || {
    echo "❌ Health check failed!"
    docker-compose -f docker-compose.$ENVIRONMENT.yml logs --tail=100
    exit 1
}

echo "✨ Deployment completed successfully!"
```

### 4.2 ローリングアップデート

```bash
#!/bin/bash
# rolling-update.sh

set -e

# 設定
REPLICAS=3
SERVICE_NAME="api"
HEALTH_CHECK_URL="http://localhost/api/v1/health"
GRACE_PERIOD=30

echo "🔄 Starting rolling update..."

# 現在の実行中コンテナ取得
OLD_CONTAINERS=$(docker ps -q -f name=${SERVICE_NAME}_)

# 新しいコンテナを順次起動
for i in $(seq 1 $REPLICAS); do
    echo "Starting new container $i/$REPLICAS..."
    
    # 新しいコンテナ起動
    docker-compose up -d --scale $SERVICE_NAME=$((i))
    
    # ヘルスチェック待機
    sleep 10
    
    # ヘルスチェック
    if ! curl -f $HEALTH_CHECK_URL; then
        echo "❌ Health check failed for new container!"
        docker-compose down
        exit 1
    fi
    
    echo "✅ Container $i is healthy"
done

# 古いコンテナを順次停止
echo "Removing old containers..."
for container in $OLD_CONTAINERS; do
    echo "Gracefully stopping container $container..."
    docker stop --time=$GRACE_PERIOD $container
    docker rm $container
    sleep 5
done

echo "✨ Rolling update completed!"
```

## 5. スケーリング戦略

### 5.1 水平スケーリング設定

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  labels:
    app: api-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: your-registry.com/api-server:latest
        ports:
        - containerPort: 8000
        env:
        - name: DJANGO_SETTINGS_MODULE
          value: "api_server.settings.production"
        envFrom:
        - secretRef:
            name: api-secrets
        - configMapRef:
            name: api-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /api/v1/health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health/readiness
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: static-files
          mountPath: /app/staticfiles
        - name: media-files
          mountPath: /app/media
      volumes:
      - name: static-files
        persistentVolumeClaim:
          claimName: static-pvc
      - name: media-files
        persistentVolumeClaim:
          claimName: media-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: api-server
spec:
  selector:
    app: api-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
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
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 60
```

### 5.2 キャッシュ最適化

```python
# api/caching/strategies.py
from django.core.cache import cache
from django.conf import settings
from functools import wraps
import hashlib
import json

class CacheStrategy:
    """キャッシュ戦略クラス"""
    
    @staticmethod
    def cache_key_generator(prefix: str, *args, **kwargs):
        """キャッシュキー生成"""
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True).encode()
        ).hexdigest()
        return f"{prefix}:{key_hash}"
    
    @staticmethod
    def cache_response(
        timeout: int = 300,
        key_prefix: str = None,
        vary_on: list = None
    ):
        """レスポンスキャッシュデコレーター"""
        def decorator(func):
            @wraps(func)
            def wrapper(request, *args, **kwargs):
                # キャッシュキー生成
                cache_key_parts = [key_prefix or func.__name__]
                
                # vary_on パラメータ処理
                if vary_on:
                    for param in vary_on:
                        if param == 'user':
                            cache_key_parts.append(
                                getattr(request, 'user_id', 'anonymous')
                            )
                        elif param == 'query':
                            cache_key_parts.append(
                                request.GET.urlencode()
                            )
                
                cache_key = CacheStrategy.cache_key_generator(
                    ':'.join(cache_key_parts),
                    *args,
                    **kwargs
                )
                
                # キャッシュチェック
                cached_response = cache.get(cache_key)
                if cached_response is not None:
                    return cached_response
                
                # 実行とキャッシュ
                response = func(request, *args, **kwargs)
                
                # 成功レスポンスのみキャッシュ
                if response.status_code == 200:
                    cache.set(cache_key, response, timeout)
                
                return response
            
            return wrapper
        return decorator

# 使用例
@api_view(['GET'])
@CacheStrategy.cache_response(
    timeout=3600,  # 1時間
    key_prefix='user_profile',
    vary_on=['user']
)
def get_user_profile(request):
    # プロフィール取得処理
    pass
```

## 6. 認証サーバー連携

### 6.1 高可用性JWKS取得

```python
# api/authentication/ha_jwks.py
import requests
from typing import Dict, List, Optional
import time
import random
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

class HighAvailabilityJWKS:
    """高可用性JWKS取得クラス"""
    
    def __init__(self):
        # 複数の認証サーバーエンドポイント
        self.auth_servers = settings.AUTH_SERVERS  # リスト形式
        self.timeout = 5
        self.retry_count = 3
        self.cache_timeout = 3600  # 1時間
    
    def get_jwks(self) -> Optional[Dict]:
        """JWKSを取得（フォールバック付き）"""
        
        # キャッシュチェック
        cached_jwks = cache.get('jwks:data')
        if cached_jwks:
            return cached_jwks
        
        # 複数の認証サーバーから取得を試みる
        servers = self.auth_servers.copy()
        random.shuffle(servers)  # ロードバランシング
        
        for server in servers:
            jwks = self._fetch_from_server(server)
            if jwks:
                # キャッシュに保存
                cache.set('jwks:data', jwks, self.cache_timeout)
                return jwks
        
        # すべて失敗した場合、古いキャッシュを使用
        stale_jwks = cache.get('jwks:data:stale')
        if stale_jwks:
            logger.warning("Using stale JWKS cache")
            return stale_jwks
        
        raise Exception("Failed to fetch JWKS from all servers")
    
    def _fetch_from_server(self, server_url: str) -> Optional[Dict]:
        """特定のサーバーからJWKS取得"""
        jwks_url = f"{server_url}/.well-known/jwks.json"
        
        for attempt in range(self.retry_count):
            try:
                response = requests.get(
                    jwks_url,
                    timeout=self.timeout,
                    headers={
                        'User-Agent': 'API-Server/1.0',
                        'Accept': 'application/json'
                    }
                )
                
                if response.status_code == 200:
                    jwks = response.json()
                    
                    # Staleキャッシュも更新
                    cache.set('jwks:data:stale', jwks, None)  # 永続化
                    
                    return jwks
                    
            except requests.RequestException as e:
                logger.warning(
                    f"Failed to fetch JWKS from {server_url} "
                    f"(attempt {attempt + 1}/{self.retry_count}): {str(e)}"
                )
                
                # 指数バックオフ
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)
        
        return None
    
    def preload_jwks(self):
        """起動時にJWKSをプリロード"""
        try:
            self.get_jwks()
            logger.info("JWKS preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload JWKS: {str(e)}")
```

### 6.2 認証サーバー監視

```python
# api/monitoring/auth_server_monitor.py
from django.core.management.base import BaseCommand
import requests
import time
from datetime import datetime
from api.monitoring.alerts import AlertManager

class AuthServerMonitor:
    """認証サーバー監視クラス"""
    
    def __init__(self):
        self.auth_servers = settings.AUTH_SERVERS
        self.alert_manager = AlertManager()
        self.check_interval = 60  # 60秒ごと
    
    def run(self):
        """監視ループ実行"""
        while True:
            for server in self.auth_servers:
                self.check_server(server)
            
            time.sleep(self.check_interval)
    
    def check_server(self, server_url: str):
        """サーバーの健全性チェック"""
        health_url = f"{server_url}/health"
        
        try:
            start_time = time.time()
            response = requests.get(health_url, timeout=10)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                # メトリクス記録
                auth_server_health_gauge.labels(
                    server=server_url
                ).set(1)
                
                auth_server_response_time.labels(
                    server=server_url
                ).observe(response_time)
                
            else:
                # 異常検知
                self.handle_unhealthy_server(
                    server_url,
                    f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            self.handle_unhealthy_server(server_url, str(e))
    
    def handle_unhealthy_server(self, server_url: str, error: str):
        """異常サーバーの処理"""
        # メトリクス記録
        auth_server_health_gauge.labels(server=server_url).set(0)
        
        # アラート送信
        self.alert_manager.send_slack_alert(
            title="Auth Server Unhealthy",
            message=f"Server {server_url} is unhealthy: {error}",
            severity='warning'
        )
        
        # サーバーを一時的に無効化
        cache.set(
            f"auth_server:disabled:{server_url}",
            True,
            300  # 5分間無効化
        )
```

## 7. CI/CDパイプライン

### 7.1 GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
  workflow_dispatch:

env:
  DOCKER_REGISTRY: your-registry.com
  IMAGE_NAME: api-server

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: |
          pytest --cov=api --cov-fail-under=80
      
      - name: Security scan
        run: |
          pip install safety bandit
          safety check
          bandit -r api/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: ./Dockerfile.production
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
          DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
          DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
        run: |
          echo "$DEPLOY_KEY" > deploy_key
          chmod 600 deploy_key
          
          ssh -i deploy_key -o StrictHostKeyChecking=no \
            $DEPLOY_USER@$DEPLOY_HOST \
            "cd /opt/api-server && \
             git pull && \
             docker-compose pull && \
             docker-compose up -d --remove-orphans && \
             docker system prune -f"
      
      - name: Health check
        run: |
          sleep 30
          curl -f https://api.example.com/api/v1/health || exit 1
      
      - name: Notify Slack
        if: always()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          text: 'Production deployment ${{ job.status }}'
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### 7.2 自動ロールバック

```bash
#!/bin/bash
# rollback.sh

set -e

# 設定
PREVIOUS_VERSION=${1:-"previous"}
HEALTH_CHECK_URL="https://api.example.com/api/v1/health"
MAX_RETRIES=5

echo "🔙 Starting rollback to version: $PREVIOUS_VERSION"

# 1. 現在のバージョンをバックアップ
CURRENT_VERSION=$(docker ps --format "table {{.Image}}" | grep api-server | head -1)
echo "Current version: $CURRENT_VERSION"

# 2. ロールバック実行
docker-compose down
docker-compose pull api-server:$PREVIOUS_VERSION
docker-compose up -d

# 3. ヘルスチェック
echo "Waiting for service to be healthy..."
for i in $(seq 1 $MAX_RETRIES); do
    sleep 10
    if curl -f $HEALTH_CHECK_URL; then
        echo "✅ Service is healthy!"
        break
    fi
    
    if [ $i -eq $MAX_RETRIES ]; then
        echo "❌ Health check failed after $MAX_RETRIES attempts"
        exit 1
    fi
done

# 4. 検証
echo "Verifying rollback..."
NEW_VERSION=$(docker ps --format "table {{.Image}}" | grep api-server | head -1)
echo "Rolled back to: $NEW_VERSION"

# 5. アラート
curl -X POST $SLACK_WEBHOOK_URL \
    -H 'Content-Type: application/json' \
    -d "{
        \"text\": \"🔙 Rollback completed\",
        \"attachments\": [{
            \"color\": \"warning\",
            \"fields\": [
                {\"title\": \"From Version\", \"value\": \"$CURRENT_VERSION\", \"short\": true},
                {\"title\": \"To Version\", \"value\": \"$NEW_VERSION\", \"short\": true}
            ]
        }]
    }"

echo "✨ Rollback completed successfully!"
```

## 8. 監視とメンテナンス

### 8.1 プロダクション監視設定

```yaml
# monitoring/prometheus-production.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: 'production'
    service: 'api-server'

rule_files:
  - '/etc/prometheus/rules/*.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

scrape_configs:
  - job_name: 'api-server'
    static_configs:
      - targets: 
          - api1.example.com:8000
          - api2.example.com:8000
          - api3.example.com:8000
    metrics_path: '/metrics'
    
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### 8.2 バックアップ戦略

```bash
#!/bin/bash
# backup.sh

set -e

# 設定
BACKUP_DIR="/backup/api-server"
S3_BUCKET="s3://your-backup-bucket/api-server"
RETENTION_DAYS=30

# タイムスタンプ
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "🔒 Starting backup process..."

# 1. データベースバックアップ
echo "Backing up database..."
docker exec api_db pg_dump -U apiuser apidb | gzip > $BACKUP_DIR/db_$TIMESTAMP.sql.gz

# 2. メディアファイルバックアップ
echo "Backing up media files..."
tar -czf $BACKUP_DIR/media_$TIMESTAMP.tar.gz -C /var/www media/

# 3. 設定ファイルバックアップ
echo "Backing up configuration..."
tar -czf $BACKUP_DIR/config_$TIMESTAMP.tar.gz \
    .env.production \
    docker-compose.production.yml \
    nginx/

# 4. S3アップロード
echo "Uploading to S3..."
aws s3 cp $BACKUP_DIR/db_$TIMESTAMP.sql.gz $S3_BUCKET/db/
aws s3 cp $BACKUP_DIR/media_$TIMESTAMP.tar.gz $S3_BUCKET/media/
aws s3 cp $BACKUP_DIR/config_$TIMESTAMP.tar.gz $S3_BUCKET/config/

# 5. 古いバックアップ削除
echo "Cleaning up old backups..."
find $BACKUP_DIR -type f -mtime +$RETENTION_DAYS -delete
aws s3 ls $S3_BUCKET/ --recursive | \
    awk -v date="$(date -d "-$RETENTION_DAYS days" +%Y-%m-%d)" '$1 < date {print $4}' | \
    xargs -I {} aws s3 rm $S3_BUCKET/{}

echo "✅ Backup completed successfully!"
```

## まとめ

このデプロイメントガイドに従うことで、REST APIサーバーを安全かつ効率的に本番環境にデプロイできます：

1. **環境構成**: 本番環境に適した設定と最適化
2. **コンテナ化**: Dockerによる一貫した実行環境
3. **スケーリング**: 負荷に応じた自動スケーリング
4. **高可用性**: 認証サーバーとの冗長接続
5. **自動化**: CI/CDパイプラインによる継続的デプロイ
6. **監視**: 包括的な監視とアラート体制
7. **バックアップ**: 定期的なバックアップとリカバリ計画