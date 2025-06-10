# セキュリティ実装ガイド

## 1. 概要

REST APIサーバーにおけるセキュリティ実装のベストプラクティスと具体的な実装方法を説明します。

## 2. 認証・認可

### 2.1 JWT検証の強化

```python
# api/authentication/enhanced_validator.py
import jwt
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
import hashlib
from django.conf import settings
from django.core.cache import cache
import logging

logger = logging.getLogger(__name__)

class EnhancedJWTValidator:
    """強化されたJWT検証クラス"""
    
    def __init__(self):
        self.max_token_age = timedelta(hours=24)  # 最大トークン年齢
        self.clock_skew = timedelta(minutes=5)   # 時刻のずれ許容範囲
        
    def validate_token_enhanced(self, token: str) -> Dict[str, Any]:
        """
        強化されたトークン検証
        
        追加の検証項目：
        - トークンの最大年齢
        - JTIブラックリスト確認
        - 不審なクレーム検出
        """
        # 基本的なJWT検証
        payload = self.validate_token(token)
        
        # トークン年齢チェック
        iat = datetime.fromtimestamp(payload['iat'], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        token_age = now - iat
        
        if token_age > self.max_token_age:
            raise jwt.InvalidTokenError('Token is too old')
        
        # JTIブラックリスト確認
        jti = payload.get('jti')
        if jti and self._is_token_blacklisted(jti):
            raise jwt.InvalidTokenError('Token has been revoked')
        
        # 不審なクレーム検出
        self._detect_suspicious_claims(payload)
        
        return payload
    
    def _is_token_blacklisted(self, jti: str) -> bool:
        """トークンがブラックリストに含まれているか確認"""
        cache_key = f'blacklist:jwt:{jti}'
        return cache.get(cache_key) is not None
    
    def _detect_suspicious_claims(self, payload: Dict[str, Any]):
        """不審なクレームを検出"""
        # 未来の発行時刻
        iat = datetime.fromtimestamp(payload['iat'], tz=timezone.utc)
        now = datetime.now(timezone.utc)
        
        if iat > now + self.clock_skew:
            logger.warning(f"Token issued in the future: {payload}")
            raise jwt.InvalidTokenError('Token issued in the future')
        
        # 異常に長い有効期限
        exp = datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
        token_lifetime = exp - iat
        
        if token_lifetime > timedelta(days=7):
            logger.warning(f"Token lifetime too long: {token_lifetime}")
            raise jwt.InvalidTokenError('Token lifetime exceeds maximum')
```

### 2.2 スコープベース認可の詳細実装

```python
# api/authentication/permissions.py
from typing import List, Set
from functools import wraps
from django.http import JsonResponse
import re

class ScopePermission:
    """スコープベースの権限管理"""
    
    # スコープ階層定義
    SCOPE_HIERARCHY = {
        'admin': ['profile:write', 'profile:read', 'content:write', 'content:read'],
        'profile:write': ['profile:read'],
        'content:write': ['content:read'],
    }
    
    @classmethod
    def has_scope(cls, user_scopes: Set[str], required_scope: str) -> bool:
        """
        ユーザーが必要なスコープを持っているか確認
        階層的なスコープもサポート
        """
        # 直接のスコープチェック
        if required_scope in user_scopes:
            return True
        
        # 階層的なスコープチェック
        for user_scope in user_scopes:
            if required_scope in cls.SCOPE_HIERARCHY.get(user_scope, []):
                return True
        
        # ワイルドカードスコープチェック（例：profile:*）
        for user_scope in user_scopes:
            if cls._match_wildcard_scope(user_scope, required_scope):
                return True
        
        return False
    
    @staticmethod
    def _match_wildcard_scope(pattern: str, scope: str) -> bool:
        """ワイルドカードスコープのマッチング"""
        # profile:* -> profile:read, profile:write にマッチ
        pattern_regex = pattern.replace('*', '.*')
        return bool(re.match(f'^{pattern_regex}$', scope))

def require_scope_with_fallback(*scopes, fallback_header='X-API-Key'):
    """
    スコープ要求デコレーター（APIキーフォールバック付き）
    
    JWTスコープが不足している場合、APIキーでの認証も試みる
    """
    def decorator(view_func):
        @wraps(view_func)
        def wrapped_view(request, *args, **kwargs):
            # JWT認証チェック
            if hasattr(request, 'user_scopes'):
                for scope in scopes:
                    if ScopePermission.has_scope(request.user_scopes, scope):
                        return view_func(request, *args, **kwargs)
            
            # APIキーフォールバック
            api_key = request.headers.get(fallback_header)
            if api_key and _validate_api_key(api_key, scopes):
                return view_func(request, *args, **kwargs)
            
            # 認証失敗
            return JsonResponse({
                'error': {
                    'code': 'FORBIDDEN',
                    'message': 'Insufficient permissions',
                    'details': {
                        'required_scopes': list(scopes),
                        'authentication_methods': ['JWT', 'API Key']
                    }
                }
            }, status=403)
        
        return wrapped_view
    return decorator

def _validate_api_key(api_key: str, required_scopes: tuple) -> bool:
    """APIキーの検証（実装例）"""
    # APIキーからスコープを取得
    from api.models import APIKey
    try:
        key_obj = APIKey.objects.get(key=api_key, is_active=True)
        key_scopes = set(key_obj.scopes.split())
        
        for scope in required_scopes:
            if ScopePermission.has_scope(key_scopes, scope):
                return True
    except APIKey.DoesNotExist:
        pass
    
    return False
```

## 3. レート制限

### 3.1 カスタムレート制限実装

```python
# api/throttling.py
from rest_framework.throttling import BaseThrottle
from django.core.cache import cache
from django.conf import settings
import time
import hashlib

class ScopedRateThrottle(BaseThrottle):
    """スコープベースのレート制限"""
    
    # スコープ別レート制限設定
    SCOPE_RATES = {
        'admin': '10000/hour',
        'profile:write': '1000/hour',
        'profile:read': '5000/hour',
        'content:write': '500/hour',
        'content:read': '2000/hour',
        'default': '100/hour',
    }
    
    def allow_request(self, request, view):
        """リクエストを許可するか判定"""
        # ユーザー識別
        if hasattr(request, 'user_id'):
            ident = request.user_id
        else:
            ident = self.get_ident(request)
        
        # レート取得
        rate = self.get_rate(request)
        if rate is None:
            return True
        
        self.num_requests, self.duration = self.parse_rate(rate)
        
        # キャッシュキー
        self.key = self.get_cache_key(request, view, ident)
        self.history = cache.get(self.key, [])
        self.now = time.time()
        
        # 古い履歴を削除
        while self.history and self.history[-1] <= self.now - self.duration:
            self.history.pop()
        
        # リクエスト数チェック
        if len(self.history) >= self.num_requests:
            return self.throttle_failure()
        
        return self.throttle_success()
    
    def get_rate(self, request):
        """ユーザーのスコープに基づいてレートを取得"""
        if hasattr(request, 'user_scopes'):
            # 最も寛容なレートを選択
            best_rate = self.SCOPE_RATES['default']
            best_limit = self.parse_rate(best_rate)[0]
            
            for scope in request.user_scopes:
                if scope in self.SCOPE_RATES:
                    rate = self.SCOPE_RATES[scope]
                    limit = self.parse_rate(rate)[0]
                    if limit > best_limit:
                        best_rate = rate
                        best_limit = limit
            
            return best_rate
        
        return self.SCOPE_RATES['default']
    
    def get_cache_key(self, request, view, ident):
        """キャッシュキー生成"""
        return f'throttle:{view.__class__.__name__}:{ident}'
    
    def throttle_success(self):
        """成功時の処理"""
        self.history.insert(0, self.now)
        cache.set(self.key, self.history, self.duration)
        return True
    
    def throttle_failure(self):
        """失敗時の処理"""
        return False
    
    def wait(self):
        """次のリクエストまでの待機時間"""
        if self.history:
            remaining_duration = self.duration - (self.now - self.history[-1])
        else:
            remaining_duration = self.duration
        
        available_requests = self.num_requests - len(self.history) + 1
        if available_requests <= 0:
            return None
        
        return remaining_duration / float(available_requests)

class EndpointRateThrottle(BaseThrottle):
    """エンドポイント別レート制限"""
    
    ENDPOINT_RATES = {
        '/api/v1/auth/login': '10/minute',
        '/api/v1/auth/register': '5/hour',
        '/api/v1/upload': '100/hour',
        '/api/v1/search': '60/minute',
    }
    
    def get_rate(self, request):
        """エンドポイントに基づいてレートを取得"""
        path = request.path
        
        # 完全一致
        if path in self.ENDPOINT_RATES:
            return self.ENDPOINT_RATES[path]
        
        # プレフィックスマッチ
        for endpoint, rate in self.ENDPOINT_RATES.items():
            if path.startswith(endpoint):
                return rate
        
        return None  # 制限なし
```

### 3.2 分散レート制限

```python
# api/distributed_throttling.py
import redis
from django.conf import settings
import time
import json

class DistributedRateLimit:
    """Redis を使用した分散レート制限"""
    
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
        
    def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int,
        endpoint: str = None
    ) -> tuple[bool, dict]:
        """
        分散環境でのレート制限チェック
        
        Returns:
            (allowed, info)
        """
        now = int(time.time())
        window_start = now - window
        
        # Luaスクリプトで原子性を保証
        lua_script = """
        local key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        
        -- 古いエントリを削除
        redis.call('ZREMRANGEBYSCORE', key, 0, window_start)
        
        -- 現在のカウント取得
        local current = redis.call('ZCARD', key)
        
        if current < limit then
            -- リクエスト追加
            redis.call('ZADD', key, now, now)
            redis.call('EXPIRE', key, ARGV[4])
            return {1, current + 1, limit}
        else
            return {0, current, limit}
        end
        """
        
        key = f"rate_limit:{identifier}:{endpoint or 'global'}"
        result = self.redis_client.eval(
            lua_script,
            1,
            key,
            now,
            window_start,
            limit,
            window
        )
        
        allowed = result[0] == 1
        info = {
            'limit': limit,
            'remaining': max(0, limit - result[1]),
            'reset': now + window,
            'retry_after': window if not allowed else None
        }
        
        return allowed, info
```

## 4. CORS設定

### 4.1 詳細なCORS設定

```python
# api/middleware/cors.py
from django.http import HttpResponse
from django.conf import settings
import re

class CustomCORSMiddleware:
    """カスタムCORS実装"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        
        # 許可するオリジンのパターン
        self.allowed_origins = settings.CORS_ALLOWED_ORIGINS
        self.allowed_origin_patterns = [
            re.compile(pattern) for pattern in 
            settings.CORS_ALLOWED_ORIGIN_REGEXES
        ]
        
    def __call__(self, request):
        # プリフライトリクエストの処理
        if request.method == 'OPTIONS':
            response = HttpResponse()
            response = self.add_cors_headers(request, response)
            return response
        
        # 通常のリクエスト処理
        response = self.get_response(request)
        response = self.add_cors_headers(request, response)
        
        return response
    
    def add_cors_headers(self, request, response):
        """CORSヘッダーを追加"""
        origin = request.headers.get('Origin')
        
        if origin and self.is_origin_allowed(origin):
            response['Access-Control-Allow-Origin'] = origin
            response['Access-Control-Allow-Credentials'] = 'true'
            response['Vary'] = 'Origin'
            
            # プリフライトレスポンス
            if request.method == 'OPTIONS':
                response['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
                response['Access-Control-Allow-Headers'] = ', '.join([
                    'Accept',
                    'Accept-Language',
                    'Content-Type',
                    'Authorization',
                    'X-Request-ID',
                    'X-Client-Version',
                ])
                response['Access-Control-Max-Age'] = '86400'  # 24時間
        
        return response
    
    def is_origin_allowed(self, origin: str) -> bool:
        """オリジンが許可されているか確認"""
        # 完全一致
        if origin in self.allowed_origins:
            return True
        
        # パターンマッチ
        for pattern in self.allowed_origin_patterns:
            if pattern.match(origin):
                return True
        
        return False

# settings.py での設定例
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'https://app.example.com',
]

CORS_ALLOWED_ORIGIN_REGEXES = [
    r'^https://.*\.example\.com$',  # サブドメイン許可
    r'^https://deploy-preview-.*\.netlify\.app$',  # Netlifyプレビュー
]
```

## 5. セキュリティヘッダー

### 5.1 セキュリティヘッダーミドルウェア

```python
# api/middleware/security_headers.py
from django.conf import settings
import uuid

class SecurityHeadersMiddleware:
    """セキュリティヘッダーを追加するミドルウェア"""
    
    def __init__(self, get_response):
        self.get_response = get_response
        
    def __call__(self, request):
        # リクエストIDの生成
        request.request_id = request.headers.get(
            'X-Request-ID',
            str(uuid.uuid4())
        )
        
        response = self.get_response(request)
        
        # セキュリティヘッダーの追加
        self.add_security_headers(response, request)
        
        return response
    
    def add_security_headers(self, response, request):
        """セキュリティヘッダーを追加"""
        # 基本的なセキュリティヘッダー
        response['X-Content-Type-Options'] = 'nosniff'
        response['X-Frame-Options'] = 'DENY'
        response['X-XSS-Protection'] = '1; mode=block'
        response['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # リクエストID
        response['X-Request-ID'] = request.request_id
        
        # HSTS（HTTPSの場合のみ）
        if request.is_secure():
            response['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # CSP（Content Security Policy）
        if hasattr(settings, 'CSP_POLICY'):
            response['Content-Security-Policy'] = settings.CSP_POLICY
        
        # Feature Policy
        response['Permissions-Policy'] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        )
        
        return response
```

## 6. 入力検証とサニタイゼーション

### 6.1 カスタムバリデーター

```python
# api/validators.py
from django.core.exceptions import ValidationError
import re
import bleach
from typing import Any, Dict, List

class SecurityValidator:
    """セキュリティ強化バリデーター"""
    
    # SQLインジェクション検出パターン
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
        r"(--|#|/\*|\*/)",
        r"(\bOR\b\s*\d+\s*=\s*\d+)",
        r"(\bAND\b\s*\d+\s*=\s*\d+)",
    ]
    
    # XSS検出パターン
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"onerror\s*=",
        r"onclick\s*=",
        r"<iframe",
    ]
    
    @classmethod
    def validate_no_sql_injection(cls, value: str) -> str:
        """SQLインジェクション攻撃の検出"""
        if not value:
            return value
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError(
                    'Potentially malicious input detected',
                    code='sql_injection_suspected'
                )
        
        return value
    
    @classmethod
    def validate_no_xss(cls, value: str) -> str:
        """XSS攻撃の検出"""
        if not value:
            return value
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValidationError(
                    'Potentially malicious input detected',
                    code='xss_suspected'
                )
        
        return value
    
    @classmethod
    def sanitize_html(cls, html: str, allowed_tags: List[str] = None) -> str:
        """HTMLサニタイゼーション"""
        if allowed_tags is None:
            allowed_tags = [
                'p', 'br', 'span', 'div',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                'strong', 'em', 'u', 's',
                'ul', 'ol', 'li',
                'a', 'img',
                'blockquote', 'code', 'pre',
            ]
        
        allowed_attributes = {
            'a': ['href', 'title', 'target'],
            'img': ['src', 'alt', 'width', 'height'],
            '*': ['class', 'id'],
        }
        
        # bleachを使用してサニタイゼーション
        cleaned = bleach.clean(
            html,
            tags=allowed_tags,
            attributes=allowed_attributes,
            strip=True
        )
        
        return cleaned
    
    @classmethod
    def validate_file_upload(
        cls,
        file,
        allowed_extensions: List[str],
        max_size_mb: int = 10
    ):
        """ファイルアップロードの検証"""
        # ファイルサイズチェック
        if file.size > max_size_mb * 1024 * 1024:
            raise ValidationError(
                f'File size exceeds {max_size_mb}MB limit',
                code='file_too_large'
            )
        
        # 拡張子チェック
        ext = file.name.split('.')[-1].lower()
        if ext not in allowed_extensions:
            raise ValidationError(
                f'File type not allowed. Allowed: {", ".join(allowed_extensions)}',
                code='invalid_file_type'
            )
        
        # MIMEタイプチェック
        import magic
        mime = magic.from_buffer(file.read(1024), mime=True)
        file.seek(0)  # ファイルポインタをリセット
        
        mime_mapping = {
            'jpg': ['image/jpeg'],
            'jpeg': ['image/jpeg'],
            'png': ['image/png'],
            'gif': ['image/gif'],
            'pdf': ['application/pdf'],
        }
        
        allowed_mimes = []
        for ext in allowed_extensions:
            allowed_mimes.extend(mime_mapping.get(ext, []))
        
        if mime not in allowed_mimes:
            raise ValidationError(
                'File content does not match extension',
                code='mime_mismatch'
            )
```

### 6.2 リクエストボディ検証

```python
# api/serializers/validators.py
from rest_framework import serializers
from api.validators import SecurityValidator

class SecureCharField(serializers.CharField):
    """セキュリティ検証付き文字フィールド"""
    
    def __init__(self, *args, **kwargs):
        self.check_sql_injection = kwargs.pop('check_sql_injection', True)
        self.check_xss = kwargs.pop('check_xss', True)
        super().__init__(*args, **kwargs)
    
    def to_internal_value(self, data):
        value = super().to_internal_value(data)
        
        if self.check_sql_injection:
            value = SecurityValidator.validate_no_sql_injection(value)
        
        if self.check_xss:
            value = SecurityValidator.validate_no_xss(value)
        
        return value

class SecureModelSerializer(serializers.ModelSerializer):
    """セキュリティ強化モデルシリアライザー"""
    
    def to_internal_value(self, data):
        # リクエストサイズ制限
        import json
        data_json = json.dumps(data)
        if len(data_json) > 1024 * 1024:  # 1MB
            raise serializers.ValidationError(
                'Request body too large',
                code='request_too_large'
            )
        
        return super().to_internal_value(data)
```

## 7. 監査ログ

### 7.1 監査ログシステム

```python
# api/audit/logger.py
from django.conf import settings
from django.db import models
from django.contrib.contenttypes.models import ContentType
import json
import hashlib

class AuditLog(models.Model):
    """監査ログモデル"""
    
    ACTION_CHOICES = [
        ('CREATE', 'Create'),
        ('UPDATE', 'Update'),
        ('DELETE', 'Delete'),
        ('ACCESS', 'Access'),
        ('LOGIN', 'Login'),
        ('LOGOUT', 'Logout'),
        ('PERMISSION_CHANGE', 'Permission Change'),
    ]
    
    # 基本情報
    action = models.CharField(max_length=20, choices=ACTION_CHOICES)
    user_id = models.CharField(max_length=255, db_index=True)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    
    # リクエスト情報
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    request_id = models.CharField(max_length=255, db_index=True)
    
    # 対象情報
    content_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True)
    object_id = models.CharField(max_length=255, null=True)
    object_repr = models.CharField(max_length=500)
    
    # 変更内容
    changes = models.JSONField(default=dict)
    
    # セキュリティ情報
    risk_score = models.IntegerField(default=0)  # 0-100
    checksum = models.CharField(max_length=64)  # SHA256
    
    class Meta:
        db_table = 'audit_logs'
        indexes = [
            models.Index(fields=['user_id', '-timestamp']),
            models.Index(fields=['action', '-timestamp']),
            models.Index(fields=['content_type', 'object_id']),
        ]
    
    def save(self, *args, **kwargs):
        # チェックサム生成（改竄防止）
        content = f"{self.action}|{self.user_id}|{self.timestamp}|{self.object_repr}|{json.dumps(self.changes)}"
        self.checksum = hashlib.sha256(content.encode()).hexdigest()
        
        super().save(*args, **kwargs)
    
    def verify_integrity(self) -> bool:
        """ログの整合性を検証"""
        content = f"{self.action}|{self.user_id}|{self.timestamp}|{self.object_repr}|{json.dumps(self.changes)}"
        expected_checksum = hashlib.sha256(content.encode()).hexdigest()
        return self.checksum == expected_checksum

class AuditLogger:
    """監査ログ記録クラス"""
    
    @staticmethod
    def log(
        action: str,
        user_id: str,
        request,
        obj=None,
        changes: dict = None,
        risk_score: int = 0
    ):
        """監査ログを記録"""
        log = AuditLog(
            action=action,
            user_id=user_id,
            ip_address=get_client_ip(request),
            user_agent=request.headers.get('User-Agent', ''),
            request_id=getattr(request, 'request_id', ''),
            risk_score=risk_score
        )
        
        if obj:
            log.content_type = ContentType.objects.get_for_model(obj)
            log.object_id = str(obj.pk)
            log.object_repr = str(obj)[:500]
        
        if changes:
            log.changes = changes
        
        log.save()
        
        # 高リスクアクションの通知
        if risk_score >= 80:
            send_security_alert(log)

def get_client_ip(request):
    """クライアントIPアドレスを取得"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
```

## 8. セキュリティ設定例

### 8.1 Django設定

```python
# settings/security.py
from decouple import config

# セキュリティ基本設定
SECRET_KEY = config('SECRET_KEY')
DEBUG = False
ALLOWED_HOSTS = config('ALLOWED_HOSTS', cast=lambda v: [s.strip() for s in v.split(',')])

# セッションセキュリティ
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
SESSION_COOKIE_AGE = 3600  # 1時間

# CSRF設定
CSRF_COOKIE_SECURE = True
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SAMESITE = 'Strict'

# セキュリティミドルウェア
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'api.middleware.security_headers.SecurityHeadersMiddleware',
    'api.middleware.rate_limit.RateLimitMiddleware',
    'api.middleware.audit.AuditMiddleware',
    # ... 他のミドルウェア
]

# コンテンツセキュリティポリシー
CSP_POLICY = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: https:; "
    "connect-src 'self' https://api.example.com;"
)

# ファイルアップロード設定
FILE_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 5242880  # 5MB
FILE_UPLOAD_PERMISSIONS = 0o644

# ロギング設定（セキュリティイベント）
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'security_file': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/security.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 10,
            'formatter': 'security',
        },
    },
    'formatters': {
        'security': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        },
    },
    'loggers': {
        'security': {
            'handlers': ['security_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
```

## 9. セキュリティチェックリスト

### 開発時
- [ ] すべての入力値の検証
- [ ] SQLインジェクション対策（ORMの使用）
- [ ] XSS対策（テンプレートエスケープ）
- [ ] CSRF対策の有効化
- [ ] 適切なエラーハンドリング（情報漏洩防止）

### デプロイ時
- [ ] DEBUG=Falseの確認
- [ ] SECRET_KEYの安全な管理
- [ ] HTTPS強制設定
- [ ] セキュリティヘッダーの設定
- [ ] ファイアウォール設定

### 運用時
- [ ] 定期的なセキュリティパッチ適用
- [ ] 監査ログの定期的な確認
- [ ] 不審なアクセスパターンの監視
- [ ] ペネトレーションテストの実施

## まとめ

このセキュリティ実装ガイドに従うことで、REST APIサーバーに対する主要な脅威から保護できます。セキュリティは継続的なプロセスであり、新しい脅威に対して常に警戒し、対策を更新する必要があります。