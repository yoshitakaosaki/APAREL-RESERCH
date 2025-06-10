# Django REST API実装ガイド（v2.0対応）

## 1. プロジェクト構成（アプリケーション分離後）

### 1.1 実際のディレクトリ構造

```
blead-stamp-svr/
├── manage.py                    # Django管理コマンド
├── requirements.txt             # Python依存関係
├── .env.example                # 環境変数テンプレート
├── config/                     # Django設定
│   ├── __init__.py
│   ├── settings.py             # メイン設定ファイル
│   ├── urls.py                 # URLルーティング
│   ├── wsgi.py                 # WSGI設定
│   └── asgi.py                 # ASGI設定
├── authentication/            # 認証・セキュリティ専用アプリ ✅
│   ├── __init__.py
│   ├── apps.py                # アプリ設定
│   ├── authentication.py     # JWT認証クラス
│   ├── permissions.py         # スコープベース権限管理
│   ├── middleware.py          # セキュリティミドルウェア
│   ├── utils.py              # 標準APIレスポンス
│   ├── admin.py              # 管理画面設定
│   ├── models.py             # 認証関連モデル（必要に応じて）
│   ├── tests.py              # 認証テスト
│   ├── views.py              # 認証関連ビュー（必要に応じて）
│   └── migrations/           # マイグレーション
├── stamp/                    # ビジネスロジック専用アプリ ✅
│   ├── __init__.py
│   ├── apps.py               # アプリ設定
│   ├── models.py             # スタンプラリーモデル
│   ├── models_generic.py     # 汎用コンテンツ管理モデル
│   ├── views.py              # レガシーAPI
│   ├── views_v1.py           # 標準v1エンドポイント
│   ├── views_content.py      # コンテンツ管理API
│   ├── views_upload.py       # ファイルアップロード
│   ├── serializers.py        # レガシーシリアライザー
│   ├── serializers_v1.py     # v1シリアライザー
│   ├── throttles.py          # レート制限設定
│   ├── search.py             # 全文検索機能
│   ├── upload_handlers.py    # ファイルアップロード処理
│   ├── urls.py               # URLパターン
│   ├── admin.py              # 管理画面設定
│   ├── tests.py              # 包括的テストスイート
│   └── migrations/           # マイグレーション
├── static/                   # 静的ファイル
└── docs-app/                # ドキュメント
```

### 1.2 アプリケーション分離の方針

#### authentication アプリ
- **責任**: 認証・認可・セキュリティ関連の機能
- **主要コンポーネント**:
  - JWT検証とJWKS連携
  - スコープベース権限管理
  - セキュリティミドルウェア
  - 標準APIレスポンス形式

#### stamp アプリ
- **責任**: ビジネスロジックとAPI実装
- **主要コンポーネント**:
  - スタンプラリー機能
  - 汎用コンテンツ管理
  - ファイルアップロード
  - 検索機能

## 2. Django設定（config/settings.py）

### 2.1 基本設定

```python
import os
from pathlib import Path
from dotenv import load_dotenv
import dj_database_url

# 環境変数読み込み
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# セキュリティ設定
SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-default-key')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

# アプリケーション登録
DJANGO_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
]

THIRD_PARTY_APPS = [
    'rest_framework',
    'corsheaders',
    'django_redis',
    'drf_spectacular',
]

LOCAL_APPS = [
    'authentication',  # 認証専用アプリ
    'stamp',          # ビジネスロジック専用アプリ
]

INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS
```

### 2.2 ミドルウェア設定

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'authentication.middleware.RequestIDMiddleware',        # リクエストID追跡
    'authentication.middleware.ResponseTimeMiddleware',     # レスポンス時間計測
    'authentication.middleware.SecurityHeadersMiddleware',  # セキュリティヘッダー
    'authentication.middleware.APIExceptionMiddleware',     # 例外処理
    'authentication.middleware.APILoggingMiddleware',       # APIログ
    'authentication.middleware.CORSPolicyMiddleware',       # CORS拡張
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
```

### 2.3 Django REST Framework設定

```python
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'authentication.authentication.JWTAuthentication',
        # 開発用ダミー認証（必要に応じてコメントアウト）
        # 'authentication.authentication.DummyJWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_SCHEMA_CLASS': 'drf_spectacular.openapi.AutoSchema',
    'DEFAULT_THROTTLE_CLASSES': [
        'rest_framework.throttling.AnonRateThrottle',
        'rest_framework.throttling.UserRateThrottle',
    ],
    'DEFAULT_THROTTLE_RATES': {
        'anon': os.getenv('THROTTLE_ANON', '100/hour'),
        'user': os.getenv('THROTTLE_USER', '1000/hour'),
        'auth': os.getenv('THROTTLE_AUTH', '60/min'),
        'content_create': os.getenv('THROTTLE_CONTENT_CREATE', '10/hour'),
        'content_like': os.getenv('THROTTLE_CONTENT_LIKE', '100/hour'),
        'burst': '20/min',
        'ip_strict': '300/hour',
    },
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}
```

### 2.4 認証サーバー設定（Docker対応）

```python
# JWT認証設定
AUTH_SERVER_URL = os.getenv('AUTH_SERVER_URL', 'http://host.docker.internal:8080')
JWKS_URL = os.getenv('JWKS_URL', f'{AUTH_SERVER_URL}/oauth/.well-known/jwks.json')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'RS256')
JWT_AUDIENCE = os.getenv('JWT_AUDIENCE', 'bff-web-client')
JWT_ISSUER = os.getenv('JWT_ISSUER', AUTH_SERVER_URL)

# BFF-Web統合設定
BFF_WEB_URL = os.getenv('BFF_WEB_URL', 'http://localhost:3000')
CORS_ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', BFF_WEB_URL).split(',')
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_ALL_ORIGINS = False  # セキュリティのため明示的にFalse
```

### 2.5 データベース・キャッシュ設定

```python
# データベース設定（PostgreSQL）
DATABASES = {
    'default': dj_database_url.parse(
        os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')
    )
}

# Redis キャッシュ設定
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': os.getenv('REDIS_URL', 'redis://localhost:6379/1'),
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 20,
                'retry_on_timeout': True,
            },
        },
        'KEY_PREFIX': 'blead_stamp',
        'TIMEOUT': 3600,  # 1時間デフォルト
    }
}

# セッションでRedisを使用
SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
SESSION_CACHE_ALIAS = 'default'
```

## 3. 認証実装（authentication/authentication.py）

### 3.1 JWTAuthentication クラス

```python
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from jose import jwt, JWTError
import requests
from django.core.cache import cache
from django.contrib.auth import get_user_model
from django.conf import settings
import logging

logger = logging.getLogger(__name__)
User = get_user_model()

class JWTAuthentication(BaseAuthentication):
    """
    認証サーバーから発行されたJWTトークンを検証する認証クラス
    
    機能:
    - JWKSエンドポイントから公開鍵を取得（キャッシュ付き）
    - JWT署名検証（RS256）
    - Audience, Issuer, Subject クレーム検証
    - Djangoユーザーとの自動連携
    """
    
    def __init__(self):
        self.auth_server_url = settings.AUTH_SERVER_URL
        self.jwks_url = settings.JWKS_URL
        self.jwt_algorithm = settings.JWT_ALGORITHM
        self.jwt_audience = settings.JWT_AUDIENCE
        self.jwt_issuer = settings.JWT_ISSUER

    def authenticate(self, request):
        """
        HTTPリクエストからJWTトークンを取得して認証
        
        Returns:
            tuple: (user, payload) または None
        """
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        
        # キャッシュから検証済みペイロードを確認
        cache_key = f"jwt_verified:{token[:20]}"
        cached_payload = cache.get(cache_key)
        if cached_payload:
            user = self._get_or_create_user(cached_payload)
            return (user, cached_payload)
        
        try:
            # JWKSから公開鍵取得
            public_keys = self._get_public_keys()
            
            # JWT検証
            payload = jwt.decode(
                token,
                public_keys,
                algorithms=[self.jwt_algorithm],
                audience=self.jwt_audience,
                issuer=self.jwt_issuer,
                options={
                    'verify_signature': True,
                    'verify_aud': True,
                    'verify_iss': True,
                    'verify_exp': True,
                    'verify_sub': True,
                }
            )
            
            # 検証結果をキャッシュ（短時間）
            cache.set(cache_key, payload, timeout=60)
            
            # Djangoユーザー取得/作成
            user = self._get_or_create_user(payload)
            return (user, payload)
            
        except JWTError as e:
            logger.warning(f'JWT validation failed: {str(e)}')
            raise exceptions.AuthenticationFailed(f'Invalid JWT: {str(e)}')
        except Exception as e:
            logger.error(f'JWT authentication error: {str(e)}')
            raise exceptions.AuthenticationFailed('JWT authentication failed')

    def _get_public_keys(self):
        """JWKSエンドポイントから公開鍵を取得（1時間キャッシュ）"""
        cache_key = "jwks_public_keys"
        keys = cache.get(cache_key)
        
        if not keys:
            try:
                logger.info(f'Fetching JWKS from {self.jwks_url}')
                response = requests.get(self.jwks_url, timeout=10)
                response.raise_for_status()
                jwks = response.json()
                
                # kid をキーとした辞書に変換
                keys = {key['kid']: key for key in jwks['keys']}
                cache.set(cache_key, keys, timeout=3600)  # 1時間キャッシュ
                logger.info(f'JWKS cached: {len(keys)} keys')
                
            except Exception as e:
                logger.error(f'Failed to fetch JWKS: {str(e)}')
                raise exceptions.AuthenticationFailed(f'Failed to fetch JWKS: {str(e)}')
        
        return keys

    def _get_or_create_user(self, payload):
        """JWTペイロードからDjangoユーザーを取得または作成"""
        user_id = payload.get('sub')
        if not user_id:
            raise exceptions.AuthenticationFailed('JWT missing sub claim')
        
        try:
            user = User.objects.get(username=user_id)
        except User.DoesNotExist:
            # ユーザーが存在しない場合は作成
            user_data = {
                'username': user_id,
                'email': payload.get('email', f'{user_id}@example.com'),
                'first_name': payload.get('given_name', ''),
                'last_name': payload.get('family_name', ''),
                'is_active': True,
            }
            user = User.objects.create(**user_data)
            logger.info(f'Created new user from JWT: {user_id}')
        
        return user

    def authenticate_header(self, request):
        """認証が失敗した場合のHTTPヘッダーを返す"""
        return 'Bearer'
```

### 3.2 開発用ダミー認証（DummyJWTAuthentication）

```python
class DummyJWTAuthentication(BaseAuthentication):
    """
    開発・テスト用のダミーJWT認証
    
    実際の署名検証は行わず、固定ユーザー情報を返す
    """
    
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        
        try:
            from jose import jwt as jose_jwt
            payload = jose_jwt.decode(token, key="", options={"verify_signature": False})
            
            # ダミーユーザー情報
            user_id = payload.get('userid', 'aaaa')
            user = DummyUser(user_id)
            
            return (user, token)
            
        except (jose_jwt.JWTError, ValueError):
            raise exceptions.AuthenticationFailed('Invalid token')
        except Exception:
            raise exceptions.AuthenticationFailed('Token authentication failed')

class DummyUser:
    """開発用ダミーユーザークラス"""
    def __init__(self, user_id):
        self.id = user_id
        self.username = user_id
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False
```

## 4. 権限管理（authentication/permissions.py）

### 4.1 スコープベース権限クラス

```python
from rest_framework.permissions import BasePermission
from rest_framework import exceptions

class HasScope(BasePermission):
    """
    特定のスコープが必要な権限クラス
    
    使用例:
    @permission_classes([IsAuthenticated, HasScope('content:write')])
    """
    
    def __init__(self, required_scope):
        self.required_scope = required_scope
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        
        # JWT ペイロードからスコープを取得
        if hasattr(request, 'auth') and isinstance(request.auth, dict):
            scopes = request.auth.get('scope', '').split()
        else:
            # DummyJWT の場合はデフォルトスコープを許可
            scopes = ['profile:read', 'content:read', 'content:write']
        
        return self.required_scope in scopes

class HasAnyScope(BasePermission):
    """
    複数スコープのいずれかが必要な権限クラス
    
    使用例:
    @permission_classes([IsAuthenticated, HasAnyScope(['content:write', 'admin:all'])])
    """
    
    def __init__(self, required_scopes):
        self.required_scopes = required_scopes
    
    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        
        # JWT ペイロードからスコープを取得
        if hasattr(request, 'auth') and isinstance(request.auth, dict):
            scopes = request.auth.get('scope', '').split()
        else:
            # DummyJWT の場合はデフォルトスコープを許可
            scopes = ['profile:read', 'content:read', 'content:write']
        
        return any(scope in scopes for scope in self.required_scopes)
```

## 5. URLルーティング（config/urls.py）

### 5.1 メインURL設定

```python
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django管理画面
    path('admin/', admin.site.urls),
    
    # API ルーティング
    path('api/', include('stamp.urls')),  # メインAPI
    
    # OpenAPI/Swagger ドキュメント
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),
]

# 開発環境での静的ファイル配信
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
```

### 5.2 アプリURL設定（stamp/urls.py）

```python
from django.urls import path, include
from . import views, views_v1, views_content, views_upload

app_name = 'stamp'

# v1 API エンドポイント
v1_patterns = [
    # 標準エンドポイント
    path('health/', views_v1.health_check, name='health_check_v1'),
    path('users/me/', views_v1.user_profile, name='user_profile'),
    path('users/me/', views_v1.update_user_profile, name='update_user_profile'),
    path('dashboard/', views_v1.dashboard_overview, name='dashboard'),
    path('search/', views_v1.search_content, name='search_content'),
    
    # コンテンツ管理
    path('contents/', views_content.content_list, name='content_list'),
    path('contents/', views_content.content_create, name='content_create'),
    path('contents/<uuid:content_id>/', views_content.content_detail, name='content_detail'),
    path('contents/<uuid:content_id>/', views_content.content_update, name='content_update'),
    path('contents/<uuid:content_id>/like/', views_content.content_like, name='content_like'),
    path('contents/<uuid:content_id>/upload/', views_upload.upload_media, name='upload_content_media'),
    
    # サポートエンドポイント
    path('categories/', views_content.categories_list, name='categories_list'),
    path('tags/', views_content.tags_list, name='tags_list'),
    path('users/me/activities/', views_content.user_activities, name='user_activities'),
    
    # メディア管理
    path('media/<uuid:media_id>/', views_upload.media_info, name='media_info'),
    path('media/<uuid:media_id>/', views_upload.delete_media, name='delete_media'),
]

# レガシーAPI（後方互換性）
legacy_patterns = [
    path('health/', views.health_check_legacy, name='health_check'),
    path('user-info/', views_v1.user_info_legacy, name='user_info'),
    path('stamp-check/', views_v1.stamp_check_legacy, name='stamp_check'),
]

urlpatterns = [
    # v1 API（推奨）
    path('v1/', include(v1_patterns)),
    
    # レガシーAPI（後方互換性）
    path('', include(legacy_patterns)),
]
```

## 6. ビューの実装パターン

### 6.1 標準APIビュー（stamp/views_v1.py）

```python
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from authentication.utils import StandardAPIResponse, APIErrorCodes
from authentication.permissions import HasScope

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_profile(request):
    """
    認証済みユーザーのプロフィール取得
    
    GET /api/v1/users/me/
    """
    user = request.user
    
    # JWT クレーム情報の取得
    jwt_claims = {}
    if hasattr(request, 'auth') and hasattr(request.auth, 'get'):
        jwt_claims = request.auth
    
    user_data = {
        'id': str(user.id),
        'username': user.username,
        'email': user.email,
        'first_name': user.first_name,
        'last_name': user.last_name,
        'is_active': user.is_active,
        'date_joined': user.date_joined.isoformat() if user.date_joined else None,
        'profile': {
            'display_name': user.get_full_name() or user.username,
            'initials': f"{user.first_name[:1]}{user.last_name[:1]}".upper() if user.first_name and user.last_name else user.username[:2].upper()
        }
    }
    
    # JWT メタデータの追加
    if jwt_claims:
        user_data['jwt_metadata'] = {
            'subject': jwt_claims.get('sub'),
            'issued_at': jwt_claims.get('iat'),
            'expires_at': jwt_claims.get('exp'),
            'issuer': jwt_claims.get('iss'),
            'audience': jwt_claims.get('aud'),
            'scopes': jwt_claims.get('scope', '').split() if jwt_claims.get('scope') else []
        }
    
    return StandardAPIResponse.success(
        data=user_data,
        request=request
    )

@api_view(['PATCH'])
@permission_classes([IsAuthenticated, HasScope('profile:write')])
def update_user_profile(request):
    """
    ユーザープロフィール更新
    
    PATCH /api/v1/users/me/
    """
    user = request.user
    data = request.data
    
    # 更新可能フィールドの制限
    updatable_fields = ['first_name', 'last_name', 'email']
    updated_fields = []
    
    for field in updatable_fields:
        if field in data:
            setattr(user, field, data[field])
            updated_fields.append(field)
    
    if updated_fields:
        try:
            user.save(update_fields=updated_fields)
            
            response_data = {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'first_name': user.first_name,
                'last_name': user.last_name,
                'updated_fields': updated_fields,
                'updated_at': timezone.now().isoformat()
            }
            
            return StandardAPIResponse.success(
                data=response_data,
                request=request
            )
            
        except Exception as e:
            return StandardAPIResponse.error(
                code=APIErrorCodes.VALIDATION_ERROR,
                message="プロフィール更新に失敗しました",
                details={'error': str(e)},
                request=request,
                status_code=status.HTTP_400_BAD_REQUEST
            )
    else:
        return StandardAPIResponse.error(
            code=APIErrorCodes.INVALID_INPUT,
            message="更新可能なフィールドが指定されていません",
            details={'allowed_fields': updatable_fields},
            request=request,
            status_code=status.HTTP_400_BAD_REQUEST
        )
```

## 7. テスト実装

### 7.1 認証テスト（stamp/tests.py）

```python
from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch
from authentication.authentication import JWTAuthentication

User = get_user_model()

class JWTAuthenticationTestCase(APITestCase):
    """JWT認証のテスト"""
    
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='test@example.com'
        )
        self.client = APIClient()
    
    @patch('authentication.authentication.JWTAuthentication._get_public_keys')
    def test_jwt_authentication_success(self, mock_get_keys):
        """正常なJWT検証のテスト"""
        # モックJWKS設定
        mock_get_keys.return_value = {"1": {"kty": "RSA", "kid": "1"}}
        
        # モックJWT検証
        with patch('jose.jwt.decode') as mock_decode:
            mock_decode.return_value = {
                'sub': 'testuser',
                'aud': 'bff-web-client',
                'iss': 'http://host.docker.internal:8080',
                'exp': 9999999999,
                'scope': 'profile:read content:write'
            }
            
            # 認証テスト
            self.client.credentials(HTTP_AUTHORIZATION='Bearer valid_token')
            response = self.client.get('/api/v1/users/me/')
            
            self.assertEqual(response.status_code, status.HTTP_200_OK)
            self.assertEqual(response.json()['data']['username'], 'testuser')
    
    def test_missing_authorization_header(self):
        """認証ヘッダーなしのテスト"""
        response = self.client.get('/api/v1/users/me/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
    
    def test_invalid_token_format(self):
        """無効なトークン形式のテスト"""
        self.client.credentials(HTTP_AUTHORIZATION='Invalid token')
        response = self.client.get('/api/v1/users/me/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
```

## 8. 運用・監視

### 8.1 ヘルスチェック実装

```python
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def health_check(request):
    """
    包括的ヘルスチェック
    
    チェック項目:
    - データベース接続
    - Redisキャッシュ
    - 認証サーバー連携（JWKS）
    """
    health_data = {
        'status': 'healthy',
        'timestamp': timezone.now().isoformat(),
        'service': 'REST API Server',
        'version': 'v2.0',
        'environment': 'development'
    }
    
    # データベースヘルスチェック
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            db_status = 'healthy'
    except Exception:
        db_status = 'unhealthy'
        health_data['status'] = 'degraded'
    
    # Redisキャッシュヘルスチェック
    try:
        cache_key = f"health_check_{int(time.time())}"
        cache.set(cache_key, 'test', 10)
        cached_value = cache.get(cache_key)
        cache_status = 'healthy' if cached_value == 'test' else 'unhealthy'
        cache.delete(cache_key)
    except Exception:
        cache_status = 'unhealthy'
        if health_data['status'] == 'healthy':
            health_data['status'] = 'degraded'
    
    # 認証サーバーヘルスチェック（JWKS取得）
    auth_server_status = 'healthy'
    try:
        auth = JWTAuthentication()
        public_keys = auth._get_public_keys()
        if not public_keys:
            auth_server_status = 'unhealthy'
    except Exception:
        auth_server_status = 'unhealthy'
        if health_data['status'] == 'healthy':
            health_data['status'] = 'degraded'
    
    health_data['checks'] = {
        'database': {'status': db_status},
        'cache': {'status': cache_status},
        'authentication_server': {'status': auth_server_status}
    }
    
    # ステータスコード決定
    status_code = status.HTTP_200_OK
    if health_data['status'] == 'degraded':
        status_code = status.HTTP_206_PARTIAL_CONTENT
    elif health_data['status'] == 'unhealthy':
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    
    return StandardAPIResponse.success(
        data=health_data,
        request=request,
        status_code=status_code
    )
```

## 9. まとめ

### 9.1 実装完了項目 ✅

- **アプリケーション分離**: 認証とビジネスロジックの明確な分離
- **JWT認証**: JWKSベースの署名検証と自動ユーザー管理
- **スコープベース認可**: 細かい権限制御
- **標準APIレスポンス**: 統一されたレスポンス形式
- **包括的ミドルウェア**: セキュリティ、ログ、監視
- **Docker環境対応**: `host.docker.internal` による認証サーバー連携
- **汎用コンテンツ管理**: ビジネス非依存のコンテンツシステム

### 9.2 アーキテクチャ特徴

1. **モジュラー設計**: 認証とビジネスロジックの分離により再利用性向上
2. **セキュリティファースト**: JWT検証、スコープ制御、セキュリティヘッダー
3. **運用対応**: ヘルスチェック、ログ、メトリクス、キャッシュ
4. **BFF統合**: BFF-Webとの連携を考慮した設計
5. **テンプレート化**: 他プロジェクトへの適用可能性

この実装により、エンタープライズレベルのREST APIサーバーとして、認証サーバーとBFF-Webの中間層として適切に機能します。