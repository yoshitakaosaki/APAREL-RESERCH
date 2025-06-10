# 監視と診断ガイド

## 1. 概要

REST APIサーバーの安定稼働を実現するための監視・診断戦略と実装方法について説明します。ヘルスチェック、メトリクス収集、ログ設計、アラート設定など、運用に必要な要素を網羅します。

## 2. ヘルスチェック

### 2.1 基本的なヘルスチェックエンドポイント

```python
# api/views/health.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.db import connection
from django.core.cache import cache
from django.conf import settings
import requests
import time
import psutil
import os
from datetime import datetime
import pytz

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """
    基本的なヘルスチェック
    GET /api/v1/health
    """
    start_time = time.time()
    
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'version': settings.API_VERSION,
        'uptime': int(time.time() - settings.SERVER_START_TIME),
        'checks': {}
    }
    
    # データベースチェック
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        health_status['checks']['database'] = {
            'status': 'healthy',
            'latency_ms': int((time.time() - start_time) * 1000)
        }
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['checks']['database'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # Redisチェック
    redis_start = time.time()
    try:
        cache.set('health_check', 'ok', 10)
        cache.get('health_check')
        health_status['checks']['redis'] = {
            'status': 'healthy',
            'latency_ms': int((time.time() - redis_start) * 1000)
        }
    except Exception as e:
        health_status['status'] = 'unhealthy'
        health_status['checks']['redis'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # 認証サーバーチェック
    auth_start = time.time()
    try:
        response = requests.get(
            settings.JWKS_URL,
            timeout=5
        )
        response.raise_for_status()
        health_status['checks']['auth_server'] = {
            'status': 'healthy',
            'latency_ms': int((time.time() - auth_start) * 1000)
        }
    except Exception as e:
        health_status['status'] = 'degraded'  # 劣化状態
        health_status['checks']['auth_server'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return Response(health_status, status=status_code)

@api_view(['GET'])
@permission_classes([AllowAny])
def health_check_detailed(request):
    """
    詳細なヘルスチェック
    GET /api/v1/health/detailed
    """
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now(pytz.UTC).isoformat(),
        'system': {},
        'application': {},
        'dependencies': {}
    }
    
    # システムリソース
    health_status['system'] = {
        'cpu': {
            'usage_percent': psutil.cpu_percent(interval=1),
            'count': psutil.cpu_count()
        },
        'memory': {
            'usage_percent': psutil.virtual_memory().percent,
            'available_mb': psutil.virtual_memory().available // (1024 * 1024),
            'total_mb': psutil.virtual_memory().total // (1024 * 1024)
        },
        'disk': {
            'usage_percent': psutil.disk_usage('/').percent,
            'free_gb': psutil.disk_usage('/').free // (1024 * 1024 * 1024)
        }
    }
    
    # アプリケーション情報
    health_status['application'] = {
        'version': settings.API_VERSION,
        'environment': settings.ENVIRONMENT,
        'debug': settings.DEBUG,
        'pid': os.getpid(),
        'python_version': sys.version.split()[0]
    }
    
    # データベース接続プール
    try:
        from django.db import connections
        db_conn = connections['default']
        health_status['dependencies']['database_pool'] = {
            'active_connections': len(db_conn.queries),
            'max_connections': db_conn.settings_dict.get('CONN_MAX_AGE', 0)
        }
    except Exception as e:
        health_status['dependencies']['database_pool'] = {
            'error': str(e)
        }
    
    return Response(health_status)
```

### 2.2 Liveness/Readiness プローブ

```python
# api/views/probes.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.core.cache import cache
import time

@api_view(['GET'])
@permission_classes([AllowAny])
def liveness_probe(request):
    """
    Kubernetes Liveness Probe
    アプリケーションが生きているかチェック
    """
    return Response({'status': 'alive'})

@api_view(['GET'])
@permission_classes([AllowAny])
def readiness_probe(request):
    """
    Kubernetes Readiness Probe
    トラフィックを受け入れる準備ができているかチェック
    """
    checks_passed = True
    errors = []
    
    # データベース接続チェック
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception as e:
        checks_passed = False
        errors.append(f"Database: {str(e)}")
    
    # キャッシュ接続チェック
    try:
        cache.set('readiness_check', 'ok', 10)
        if cache.get('readiness_check') != 'ok':
            raise Exception("Cache read/write failed")
    except Exception as e:
        checks_passed = False
        errors.append(f"Cache: {str(e)}")
    
    if checks_passed:
        return Response({'status': 'ready'})
    else:
        return Response(
            {'status': 'not ready', 'errors': errors},
            status=503
        )
```

## 3. メトリクス収集

### 3.1 Prometheus メトリクス

```python
# api/metrics/collectors.py
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client.core import CollectorRegistry
import time
from functools import wraps

# メトリクスレジストリ
registry = CollectorRegistry()

# カウンターメトリクス
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

jwt_validations_total = Counter(
    'jwt_validations_total',
    'Total JWT validation attempts',
    ['result'],  # success, failure
    registry=registry
)

# ヒストグラムメトリクス
http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    registry=registry
)

jwt_validation_duration_seconds = Histogram(
    'jwt_validation_duration_seconds',
    'JWT validation duration',
    registry=registry
)

# ゲージメトリクス
active_connections = Gauge(
    'active_connections',
    'Number of active connections',
    registry=registry
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    registry=registry
)

# 情報メトリクス
api_info = Info(
    'api_info',
    'API version and environment info',
    registry=registry
)

# デコレーター
def track_request_metrics(endpoint_name):
    """リクエストメトリクスを追跡するデコレーター"""
    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            method = request.method
            start_time = time.time()
            
            # アクティブ接続数増加
            active_connections.inc()
            
            try:
                response = func(request, *args, **kwargs)
                status = response.status_code
                
                # メトリクス記録
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint_name,
                    status=status
                ).inc()
                
                return response
                
            except Exception as e:
                # エラーメトリクス
                http_requests_total.labels(
                    method=method,
                    endpoint=endpoint_name,
                    status=500
                ).inc()
                raise
                
            finally:
                # レイテンシ記録
                duration = time.time() - start_time
                http_request_duration_seconds.labels(
                    method=method,
                    endpoint=endpoint_name
                ).observe(duration)
                
                # アクティブ接続数減少
                active_connections.dec()
        
        return wrapper
    return decorator

# JWT検証メトリクス
def track_jwt_validation(func):
    """JWT検証メトリクスを追跡するデコレーター"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            jwt_validations_total.labels(result='success').inc()
            return result
            
        except Exception as e:
            jwt_validations_total.labels(result='failure').inc()
            raise
            
        finally:
            duration = time.time() - start_time
            jwt_validation_duration_seconds.observe(duration)
    
    return wrapper
```

### 3.2 カスタムメトリクスビュー

```python
# api/views/metrics.py
from django.http import HttpResponse
from prometheus_client import generate_latest
from api.metrics.collectors import registry, api_info
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

@api_view(['GET'])
@permission_classes([AllowAny])
def metrics_view(request):
    """Prometheusメトリクスエンドポイント"""
    
    # API情報を設定
    api_info.info({
        'version': settings.API_VERSION,
        'environment': settings.ENVIRONMENT,
        'django_version': django.__version__,
    })
    
    # メトリクスを生成
    metrics = generate_latest(registry)
    
    return HttpResponse(
        metrics,
        content_type='text/plain; version=0.0.4; charset=utf-8'
    )

# ビジネスメトリクス
from api.models import UserProfile, Content
from django.db.models import Count, Avg
from django.utils import timezone
from datetime import timedelta

@api_view(['GET'])
@permission_classes([AllowAny])
def business_metrics(request):
    """ビジネスメトリクス"""
    
    now = timezone.now()
    last_24h = now - timedelta(hours=24)
    last_7d = now - timedelta(days=7)
    
    metrics = {
        'users': {
            'total': UserProfile.objects.filter(is_deleted=False).count(),
            'active_24h': UserProfile.objects.filter(
                updated_at__gte=last_24h
            ).count(),
            'new_7d': UserProfile.objects.filter(
                created_at__gte=last_7d
            ).count(),
        },
        'content': {
            'total': Content.objects.filter(is_deleted=False).count(),
            'published': Content.objects.filter(
                status='published'
            ).count(),
            'created_24h': Content.objects.filter(
                created_at__gte=last_24h
            ).count(),
            'avg_views': Content.objects.aggregate(
                Avg('views_count')
            )['views_count__avg'] or 0,
        },
        'engagement': {
            'total_views': Content.objects.aggregate(
                total=Count('views_count')
            )['total'] or 0,
            'total_likes': Content.objects.aggregate(
                total=Count('likes_count')
            )['total'] or 0,
        }
    }
    
    return Response(metrics)
```

### 3.3 メトリクスミドルウェア

```python
# api/middleware/metrics.py
from api.metrics.collectors import (
    http_requests_total,
    http_request_duration_seconds,
    active_connections
)
import time
from django.utils.deprecation import MiddlewareMixin

class MetricsMiddleware(MiddlewareMixin):
    """メトリクス収集ミドルウェア"""
    
    def process_request(self, request):
        """リクエスト開始時の処理"""
        request._metrics_start_time = time.time()
        active_connections.inc()
        
    def process_response(self, request, response):
        """レスポンス時の処理"""
        if hasattr(request, '_metrics_start_time'):
            # パス正規化
            path = request.path
            if path.startswith('/api/v1/'):
                # 動的パラメータを正規化
                path_parts = path.split('/')
                if len(path_parts) > 4 and path_parts[4]:
                    # /api/v1/resource/{id} -> /api/v1/resource/:id
                    path_parts[4] = ':id'
                    path = '/'.join(path_parts)
            
            # メトリクス記録
            duration = time.time() - request._metrics_start_time
            
            http_requests_total.labels(
                method=request.method,
                endpoint=path,
                status=response.status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=request.method,
                endpoint=path
            ).observe(duration)
        
        active_connections.dec()
        return response
    
    def process_exception(self, request, exception):
        """例外時の処理"""
        if hasattr(request, '_metrics_start_time'):
            path = request.path
            
            http_requests_total.labels(
                method=request.method,
                endpoint=path,
                status=500
            ).inc()
        
        active_connections.dec()
        return None
```

## 4. ログ設計

### 4.1 構造化ログ

```python
# api/logging/formatters.py
import json
import logging
from datetime import datetime
import traceback

class JSONFormatter(logging.Formatter):
    """JSON形式のログフォーマッター"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # リクエストコンテキスト
        if hasattr(record, 'request'):
            request = record.request
            log_data['request'] = {
                'method': request.method,
                'path': request.path,
                'user_id': getattr(request, 'user_id', None),
                'request_id': getattr(request, 'request_id', None),
                'ip': self.get_client_ip(request),
                'user_agent': request.META.get('HTTP_USER_AGENT', ''),
            }
        
        # エラー情報
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # カスタムフィールド
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename',
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'thread',
                          'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'request']:
                log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)
    
    def get_client_ip(self, request):
        """クライアントIPアドレスを取得"""
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip
```

### 4.2 ログ設定

```python
# api_server/settings/logging.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'json': {
            '()': 'api.logging.formatters.JSONFormatter',
        },
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'filters': {
        'require_debug_false': {
            '()': 'django.utils.log.RequireDebugFalse',
        },
        'require_debug_true': {
            '()': 'django.utils.log.RequireDebugTrue',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'json' if not DEBUG else 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/api.log',
            'maxBytes': 1024 * 1024 * 100,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/error.log',
            'maxBytes': 1024 * 1024 * 100,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
        },
        'security_file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/security.log',
            'maxBytes': 1024 * 1024 * 100,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
        },
        'performance_file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/performance.log',
            'maxBytes': 1024 * 1024 * 100,  # 100MB
            'backupCount': 10,
            'formatter': 'json',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'api': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG' if DEBUG else 'INFO',
            'propagate': False,
        },
        'api.security': {
            'handlers': ['security_file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'api.performance': {
            'handlers': ['performance_file'],
            'level': 'INFO',
            'propagate': False,
        },
        'django.request': {
            'handlers': ['error_file', 'console'],
            'level': 'ERROR',
            'propagate': False,
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
}
```

### 4.3 コンテキストログ

```python
# api/logging/context.py
import logging
import contextvars
from typing import Optional, Dict, Any

# コンテキスト変数
request_context = contextvars.ContextVar('request_context', default={})

class ContextLogger:
    """コンテキスト付きロガー"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def _add_context(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """コンテキスト情報を追加"""
        context = request_context.get()
        extra.update(context)
        return extra
    
    def debug(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.debug(message, extra=extra)
    
    def info(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.info(message, extra=extra)
    
    def warning(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.warning(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.error(message, extra=extra)
    
    def critical(self, message: str, **kwargs):
        extra = self._add_context(kwargs)
        self.logger.critical(message, extra=extra)

# ログコンテキストミドルウェア
class LogContextMiddleware:
    """リクエストコンテキストをログに追加"""
    
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # コンテキスト設定
        context = {
            'request_id': getattr(request, 'request_id', None),
            'user_id': getattr(request, 'user_id', None),
            'method': request.method,
            'path': request.path,
            'ip': self.get_client_ip(request),
        }
        
        token = request_context.set(context)
        
        try:
            response = self.get_response(request)
            return response
        finally:
            request_context.reset(token)
    
    def get_client_ip(self, request):
        x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
        if x_forwarded_for:
            ip = x_forwarded_for.split(',')[0]
        else:
            ip = request.META.get('REMOTE_ADDR')
        return ip

# 使用例
logger = ContextLogger('api.views')

def some_view(request):
    logger.info('Processing request', extra={
        'action': 'user_profile_update',
        'profile_id': profile_id
    })
```

## 5. アラート設定

### 5.1 アラートルール

```yaml
# monitoring/alerts/api_alerts.yml
groups:
  - name: api_server_alerts
    interval: 30s
    rules:
      # 高エラー率
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m]))
            /
            sum(rate(http_requests_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          service: api_server
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for the last 5 minutes"
      
      # 高レスポンスタイム
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          ) > 1.0
        for: 5m
        labels:
          severity: warning
          service: api_server
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"
      
      # JWT検証失敗率
      - alert: HighJWTValidationFailureRate
        expr: |
          (
            sum(rate(jwt_validations_total{result="failure"}[5m]))
            /
            sum(rate(jwt_validations_total[5m]))
          ) > 0.1
        for: 5m
        labels:
          severity: warning
          service: api_server
        annotations:
          summary: "High JWT validation failure rate"
          description: "JWT validation failure rate is {{ $value | humanizePercentage }}"
      
      # メモリ使用率
      - alert: HighMemoryUsage
        expr: |
          (
            process_resident_memory_bytes
            /
            node_memory_MemTotal_bytes
          ) > 0.8
        for: 5m
        labels:
          severity: warning
          service: api_server
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
      
      # データベース接続数
      - alert: HighDatabaseConnections
        expr: |
          pg_stat_database_numbackends{datname="apidb"} > 80
        for: 5m
        labels:
          severity: warning
          service: api_server
        annotations:
          summary: "High database connection count"
          description: "Database has {{ $value }} active connections"
```

### 5.2 アラート通知設定

```python
# api/monitoring/alerts.py
import requests
from django.conf import settings
import json
from datetime import datetime

class AlertManager:
    """アラート管理クラス"""
    
    def __init__(self):
        self.slack_webhook = settings.SLACK_WEBHOOK_URL
        self.pagerduty_key = settings.PAGERDUTY_INTEGRATION_KEY
    
    def send_slack_alert(self, title: str, message: str, severity: str = 'warning'):
        """Slackアラート送信"""
        color_map = {
            'info': '#36a64f',
            'warning': '#ff9900',
            'error': '#ff0000',
            'critical': '#ff0000'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(severity, '#808080'),
                'title': f':warning: {title}',
                'text': message,
                'fields': [
                    {
                        'title': 'Service',
                        'value': 'API Server',
                        'short': True
                    },
                    {
                        'title': 'Environment',
                        'value': settings.ENVIRONMENT,
                        'short': True
                    },
                    {
                        'title': 'Time',
                        'value': datetime.now().isoformat(),
                        'short': True
                    },
                    {
                        'title': 'Severity',
                        'value': severity.upper(),
                        'short': True
                    }
                ],
                'footer': 'API Monitoring',
                'ts': int(datetime.now().timestamp())
            }]
        }
        
        try:
            response = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
    
    def send_pagerduty_alert(
        self,
        summary: str,
        details: dict,
        severity: str = 'warning'
    ):
        """PagerDutyアラート送信"""
        severity_map = {
            'info': 'info',
            'warning': 'warning',
            'error': 'error',
            'critical': 'critical'
        }
        
        payload = {
            'routing_key': self.pagerduty_key,
            'event_action': 'trigger',
            'payload': {
                'summary': summary,
                'source': 'api-server',
                'severity': severity_map.get(severity, 'warning'),
                'custom_details': details,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                timeout=10
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {str(e)}")

# 使用例
from api.monitoring.alerts import AlertManager

alert_manager = AlertManager()

# 高エラー率検出時
def check_error_rate():
    error_rate = calculate_error_rate()
    
    if error_rate > 0.05:  # 5%以上
        alert_manager.send_slack_alert(
            title="High Error Rate Alert",
            message=f"API error rate is {error_rate:.1%} in the last 5 minutes",
            severity='critical' if error_rate > 0.1 else 'warning'
        )
        
        if error_rate > 0.1:  # 10%以上はPagerDutyにも通知
            alert_manager.send_pagerduty_alert(
                summary="Critical: API error rate exceeds 10%",
                details={
                    'error_rate': f"{error_rate:.1%}",
                    'threshold': '10%',
                    'action_required': 'Immediate investigation needed'
                },
                severity='critical'
            )
```

## 6. ダッシュボード

### 6.1 Grafanaダッシュボード定義

```json
{
  "dashboard": {
    "title": "API Server Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (method)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Response Time (95th percentile)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Connections",
        "targets": [
          {
            "expr": "active_connections"
          }
        ],
        "type": "graph"
      },
      {
        "title": "JWT Validation Success Rate",
        "targets": [
          {
            "expr": "sum(rate(jwt_validations_total{result=\"success\"}[5m])) / sum(rate(jwt_validations_total[5m]))"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Top Endpoints by Request Count",
        "targets": [
          {
            "expr": "topk(10, sum by (endpoint) (rate(http_requests_total[5m])))"
          }
        ],
        "type": "table"
      },
      {
        "title": "Top Endpoints by Latency",
        "targets": [
          {
            "expr": "topk(10, histogram_quantile(0.95, sum by (endpoint, le) (rate(http_request_duration_seconds_bucket[5m]))))"
          }
        ],
        "type": "table"
      },
      {
        "title": "System Resources",
        "panels": [
          {
            "title": "CPU Usage",
            "targets": [
              {
                "expr": "rate(process_cpu_seconds_total[5m])"
              }
            ]
          },
          {
            "title": "Memory Usage",
            "targets": [
              {
                "expr": "process_resident_memory_bytes"
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### 6.2 カスタムダッシュボードAPI

```python
# api/views/dashboard.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.db.models import Count, Avg, Sum, F
from django.utils import timezone
from datetime import timedelta
from api.models import Content, UserProfile, PageView
from django.core.cache import cache
import json

@api_view(['GET'])
def monitoring_dashboard(request):
    """監視ダッシュボード用データ"""
    
    # キャッシュチェック
    cache_key = 'monitoring_dashboard_data'
    cached_data = cache.get(cache_key)
    if cached_data:
        return Response(json.loads(cached_data))
    
    now = timezone.now()
    last_hour = now - timedelta(hours=1)
    last_24h = now - timedelta(hours=24)
    
    # リアルタイムメトリクス
    realtime_metrics = {
        'current_active_users': cache.get('active_users_count', 0),
        'requests_per_minute': cache.get('rpm', 0),
        'average_response_time_ms': cache.get('avg_response_time', 0),
        'error_rate_percent': cache.get('error_rate', 0),
    }
    
    # システムヘルス
    system_health = {
        'api_status': 'healthy',
        'database_status': check_database_health(),
        'cache_status': check_cache_health(),
        'auth_server_status': check_auth_server_health(),
    }
    
    # トラフィック統計
    traffic_stats = PageView.objects.filter(
        created_at__gte=last_24h
    ).aggregate(
        total_views=Count('id'),
        unique_users=Count('user_id', distinct=True),
        avg_load_time=Avg('load_time_ms'),
    )
    
    # エンドポイント統計
    endpoint_stats = PageView.objects.filter(
        created_at__gte=last_hour
    ).values('path').annotate(
        count=Count('id'),
        avg_time=Avg('load_time_ms')
    ).order_by('-count')[:10]
    
    # エラー統計
    error_stats = get_error_statistics(last_24h)
    
    dashboard_data = {
        'timestamp': now.isoformat(),
        'realtime': realtime_metrics,
        'health': system_health,
        'traffic': traffic_stats,
        'endpoints': list(endpoint_stats),
        'errors': error_stats,
    }
    
    # キャッシュに保存（1分間）
    cache.set(cache_key, json.dumps(dashboard_data, default=str), 60)
    
    return Response(dashboard_data)

def check_database_health():
    """データベースヘルスチェック"""
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        return 'healthy'
    except:
        return 'unhealthy'

def check_cache_health():
    """キャッシュヘルスチェック"""
    try:
        cache.set('health_check', 'ok', 10)
        if cache.get('health_check') == 'ok':
            return 'healthy'
    except:
        pass
    return 'unhealthy'

def check_auth_server_health():
    """認証サーバーヘルスチェック"""
    try:
        import requests
        response = requests.get(
            f"{settings.AUTH_SERVER_URL}/health",
            timeout=5
        )
        if response.status_code == 200:
            return 'healthy'
    except:
        pass
    return 'unhealthy'

def get_error_statistics(since):
    """エラー統計取得"""
    # ログから集計（実装例）
    return {
        'total_errors': 0,
        'by_type': {
            '500': 0,
            '502': 0,
            '503': 0,
            '504': 0,
        },
        'top_errors': []
    }
```

## 7. トラブルシューティング

### 7.1 診断エンドポイント

```python
# api/views/diagnostics.py
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser
from rest_framework.response import Response
from django.conf import settings
import sys
import pkg_resources

@api_view(['GET'])
@permission_classes([IsAdminUser])
def diagnostics(request):
    """システム診断情報"""
    
    # Python環境
    python_info = {
        'version': sys.version,
        'executable': sys.executable,
        'path': sys.path,
    }
    
    # インストール済みパッケージ
    installed_packages = {
        pkg.key: pkg.version
        for pkg in pkg_resources.working_set
    }
    
    # Django設定
    django_settings = {
        'DEBUG': settings.DEBUG,
        'DATABASES': {
            name: {
                'ENGINE': db['ENGINE'],
                'HOST': db.get('HOST', ''),
                'PORT': db.get('PORT', ''),
            }
            for name, db in settings.DATABASES.items()
        },
        'CACHES': list(settings.CACHES.keys()),
        'MIDDLEWARE': settings.MIDDLEWARE,
        'INSTALLED_APPS': settings.INSTALLED_APPS,
    }
    
    # 環境変数（機密情報を除外）
    safe_env_vars = {
        k: v for k, v in os.environ.items()
        if not any(sensitive in k.upper() 
                  for sensitive in ['KEY', 'SECRET', 'PASSWORD', 'TOKEN'])
    }
    
    return Response({
        'python': python_info,
        'packages': installed_packages,
        'django': django_settings,
        'environment': safe_env_vars,
    })

@api_view(['POST'])
@permission_classes([IsAdminUser])
def test_connectivity(request):
    """外部サービス接続テスト"""
    
    results = {}
    
    # データベース接続
    try:
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        results['database'] = {'status': 'success'}
    except Exception as e:
        results['database'] = {'status': 'failed', 'error': str(e)}
    
    # Redis接続
    try:
        cache.set('test', 'ok', 10)
        cache.get('test')
        results['redis'] = {'status': 'success'}
    except Exception as e:
        results['redis'] = {'status': 'failed', 'error': str(e)}
    
    # 認証サーバー接続
    try:
        response = requests.get(settings.JWKS_URL, timeout=5)
        response.raise_for_status()
        results['auth_server'] = {'status': 'success'}
    except Exception as e:
        results['auth_server'] = {'status': 'failed', 'error': str(e)}
    
    return Response(results)
```

## まとめ

この監視・診断ガイドに従うことで、REST APIサーバーの健全性を維持できます：

1. **ヘルスチェック**: 多層的な健康状態監視
2. **メトリクス**: ビジネスとシステム両面の指標
3. **ログ**: 構造化された追跡可能なログ
4. **アラート**: 段階的なエスカレーション
5. **診断**: トラブルシューティングツール