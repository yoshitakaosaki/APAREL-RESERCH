# REST API テスト戦略ガイド

## 1. 概要

REST APIサーバーの品質を保証するための包括的なテスト戦略と実装方法を提供します。JWT検証、APIエンドポイント、統合テストなど、各レイヤーでのテスト手法を詳しく解説します。

## 2. テスト環境構築

### 2.1 PostgreSQL テストデータベース

**🔥 重要**: このプロジェクトでは、テスト時もPostgreSQLを使用します（SQLiteではありません）。これにより本番環境との一貫性を保ち、PostgreSQL固有の機能（位置情報検索等）をテストできます。

#### テストデータベース設定
- **テスト用DB名**: `test_postgres` 
- **エンジン**: PostgreSQL（本番と同じ）
- **自動作成**: テスト実行時に自動作成・削除
- **GIS機能**: PostgreSQLの数学関数を活用した位置情報検索（PostGIS不要）

#### 環境変数設定
```bash
# .env または テスト環境設定
TEST_DB_NAME=test_postgres
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
```

#### PostgreSQL テストデータベースの利点
1. **本番環境との一貫性**: 同じデータベースエンジンを使用
2. **PostgreSQL機能のテスト**: 全文検索、位置情報計算等（PostGIS不要）
3. **SQL方言の統一**: PostgreSQL固有のSQL構文をテスト可能
4. **パフォーマンステスト**: 実際のクエリパフォーマンスを測定

### 2.2 必要な依存関係

```python
# requirements-test.txt
pytest==7.4.4
pytest-django==4.7.0
pytest-cov==4.1.0
pytest-mock==3.12.0
pytest-asyncio==0.21.1
factory-boy==3.3.0
faker==22.0.0
freezegun==1.2.2
responses==0.24.1
pytest-benchmark==4.0.0
```

### 2.2 pytest設定

```ini
# pytest.ini
[tool:pytest]
DJANGO_SETTINGS_MODULE = api_server.settings.test
python_files = tests.py test_*.py *_tests.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --strict-markers
    --tb=short
    --cov=api
    --cov-report=html
    --cov-report=term-missing:skip-covered
    --cov-fail-under=80
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    jwt: marks tests related to JWT validation
testpaths = tests
```

### 2.3 テスト用設定

```python
# api_server/settings/test.py
from .base import *

# テスト用設定
DEBUG = False
TESTING = True

# テスト用データベース (PostgreSQL使用)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'test_postgres',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}

# キャッシュ無効化
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.dummy.DummyCache',
    }
}

# JWT設定（テスト用）
JWKS_URL = 'http://testserver/.well-known/jwks.json'
OAUTH_ISSUER = 'http://testserver'
OAUTH_AUDIENCE = 'test-client'

# メール設定
EMAIL_BACKEND = 'django.core.mail.backends.locmem.EmailBackend'

# メディアファイル
MEDIA_ROOT = '/tmp/test_media/'

# ログ設定（テスト時は最小限）
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'null': {
            'class': 'logging.NullHandler',
        },
    },
    'root': {
        'handlers': ['null'],
    },
}
```

## 3. JWT検証テスト

### 3.1 JWTトークン生成ヘルパー

```python
# tests/helpers/jwt_helper.py
import jwt
import uuid
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import json

class JWTTestHelper:
    """テスト用JWTトークン生成ヘルパー"""
    
    def __init__(self):
        # テスト用RSA鍵ペア生成
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        self.kid = "test-key-2024"
    
    def get_private_key_pem(self):
        """秘密鍵をPEM形式で取得"""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
    
    def get_public_key_pem(self):
        """公開鍵をPEM形式で取得"""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def get_jwks(self):
        """JWKS形式で公開鍵を取得"""
        # RSA公開鍵のコンポーネント取得
        public_numbers = self.public_key.public_numbers()
        
        # Base64URL エンコード
        def to_base64url(num):
            bytes_data = num.to_bytes((num.bit_length() + 7) // 8, 'big')
            return base64.urlsafe_b64encode(bytes_data).decode('utf-8').rstrip('=')
        
        return {
            "keys": [{
                "kty": "RSA",
                "kid": self.kid,
                "use": "sig",
                "alg": "RS256",
                "n": to_base64url(public_numbers.n),
                "e": to_base64url(public_numbers.e)
            }]
        }
    
    def create_token(
        self,
        user_id: str = "test-user-123",
        scopes: list = None,
        exp_minutes: int = 15,
        **kwargs
    ):
        """テスト用JWTトークン生成"""
        now = datetime.now(timezone.utc)
        
        payload = {
            "iss": "http://testserver",
            "sub": user_id,
            "aud": "test-client",
            "exp": now + timedelta(minutes=exp_minutes),
            "iat": now,
            "jti": str(uuid.uuid4()),
            "scope": " ".join(scopes or ["profile:read"]),
            "email": kwargs.get("email", f"{user_id}@example.com"),
            "email_verified": kwargs.get("email_verified", True),
            "name": kwargs.get("name", "Test User"),
        }
        
        # 追加のクレーム
        payload.update(kwargs)
        
        return jwt.encode(
            payload,
            self.get_private_key_pem(),
            algorithm="RS256",
            headers={"kid": self.kid}
        )
    
    def create_expired_token(self, **kwargs):
        """期限切れトークン生成"""
        return self.create_token(exp_minutes=-1, **kwargs)
    
    def create_invalid_signature_token(self):
        """無効な署名のトークン生成"""
        # 異なる秘密鍵で署名
        other_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        payload = {
            "sub": "test-user",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=15)
        }
        
        return jwt.encode(
            payload,
            other_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ),
            algorithm="RS256",
            headers={"kid": self.kid}
        )
```

### 3.2 JWT検証ユニットテスト

```python
# tests/test_jwt_validation.py
import pytest
from unittest.mock import patch, Mock
from django.test import TestCase
from api.authentication.jwt_validator import JWTValidator
from tests.helpers.jwt_helper import JWTTestHelper
import jwt
import responses

@pytest.mark.jwt
class TestJWTValidator(TestCase):
    """JWT検証のユニットテスト"""
    
    def setUp(self):
        self.jwt_helper = JWTTestHelper()
        self.validator = JWTValidator()
        
        # JWKSエンドポイントのモック
        self.mock_jwks_url = "http://testserver/.well-known/jwks.json"
        
    @responses.activate
    def test_valid_token_validation(self):
        """正常なトークンの検証"""
        # JWKSレスポンスのモック
        responses.add(
            responses.GET,
            self.mock_jwks_url,
            json=self.jwt_helper.get_jwks(),
            status=200
        )
        
        # 正常なトークン生成
        token = self.jwt_helper.create_token(
            user_id="user-123",
            scopes=["profile:read", "profile:write"]
        )
        
        # 検証実行
        payload = self.validator.validate_token(token)
        
        # アサーション
        assert payload["sub"] == "user-123"
        assert "profile:read" in payload["scope"]
        assert "profile:write" in payload["scope"]
    
    @responses.activate
    def test_expired_token_rejection(self):
        """期限切れトークンの拒否"""
        responses.add(
            responses.GET,
            self.mock_jwks_url,
            json=self.jwt_helper.get_jwks(),
            status=200
        )
        
        # 期限切れトークン
        expired_token = self.jwt_helper.create_expired_token()
        
        # 検証は失敗すべき
        with pytest.raises(jwt.InvalidTokenError, match="Token has expired"):
            self.validator.validate_token(expired_token)
    
    @responses.activate
    def test_invalid_signature_rejection(self):
        """無効な署名の拒否"""
        responses.add(
            responses.GET,
            self.mock_jwks_url,
            json=self.jwt_helper.get_jwks(),
            status=200
        )
        
        # 無効な署名のトークン
        invalid_token = self.jwt_helper.create_invalid_signature_token()
        
        # 検証は失敗すべき
        with pytest.raises(jwt.InvalidTokenError):
            self.validator.validate_token(invalid_token)
    
    def test_missing_kid_rejection(self):
        """kidヘッダーがないトークンの拒否"""
        # kidなしでトークン作成
        payload = {"sub": "user-123", "exp": 9999999999}
        token = jwt.encode(payload, "secret", algorithm="HS256")
        
        with pytest.raises(jwt.InvalidTokenError, match="Token missing kid"):
            self.validator.validate_token(token)
    
    @responses.activate
    def test_jwks_caching(self):
        """JWKSキャッシングの動作確認"""
        # 1回目のJWKS取得
        responses.add(
            responses.GET,
            self.mock_jwks_url,
            json=self.jwt_helper.get_jwks(),
            status=200
        )
        
        token1 = self.jwt_helper.create_token()
        token2 = self.jwt_helper.create_token()
        
        # 1回目の検証
        self.validator.validate_token(token1)
        
        # 2回目の検証（キャッシュから取得されるはず）
        self.validator.validate_token(token2)
        
        # JWKSエンドポイントは1回しか呼ばれないはず
        assert len(responses.calls) == 1
    
    def test_custom_claims_validation(self):
        """カスタムクレームの検証"""
        with patch.object(self.validator, '_get_public_key') as mock_get_key:
            mock_get_key.return_value = self.jwt_helper.get_public_key_pem()
            
            # email_verified が False のトークン
            token = self.jwt_helper.create_token(
                email_verified=False
            )
            
            # 設定でemail確認を必須にする
            with self.settings(REQUIRE_EMAIL_VERIFIED=True):
                with pytest.raises(jwt.InvalidTokenError, match="Email not verified"):
                    self.validator.validate_token(token)
    
    @responses.activate
    def test_jwks_fetch_failure_handling(self):
        """JWKS取得失敗時の処理"""
        # ネットワークエラーをシミュレート
        responses.add(
            responses.GET,
            self.mock_jwks_url,
            status=500
        )
        
        token = self.jwt_helper.create_token()
        
        with pytest.raises(jwt.InvalidTokenError, match="Failed to fetch public keys"):
            self.validator.validate_token(token)
```

## 4. APIエンドポイントテスト

### 4.1 テストファクトリー

```python
# tests/factories.py
import factory
from factory.django import DjangoModelFactory
from api.models import UserProfile, Content, Category
from faker import Faker

fake = Faker('ja_JP')

class UserProfileFactory(DjangoModelFactory):
    """ユーザープロフィールのファクトリー"""
    
    class Meta:
        model = UserProfile
    
    user_id = factory.Sequence(lambda n: f"user-{n}")
    email = factory.LazyAttribute(lambda obj: f"{obj.user_id}@example.com")
    email_verified = True
    display_name = factory.Faker('name', locale='ja_JP')
    bio = factory.Faker('text', max_nb_chars=200, locale='ja_JP')
    location = factory.Faker('city', locale='ja_JP')
    
    @factory.post_generation
    def preferences(self, create, extracted, **kwargs):
        if not create:
            return
        
        self.preferences = {
            'language': 'ja',
            'timezone': 'Asia/Tokyo',
            'theme': 'light',
            **kwargs
        }

class CategoryFactory(DjangoModelFactory):
    """カテゴリのファクトリー"""
    
    class Meta:
        model = Category
        django_get_or_create = ('slug',)
    
    name = factory.Sequence(lambda n: f"カテゴリ{n}")
    slug = factory.LazyAttribute(lambda obj: obj.name.lower())
    description = factory.Faker('text', max_nb_chars=100, locale='ja_JP')

class ContentFactory(DjangoModelFactory):
    """コンテンツのファクトリー"""
    
    class Meta:
        model = Content
    
    title = factory.Faker('catch_phrase', locale='ja_JP')
    slug = factory.Faker('slug')
    content = factory.Faker('text', max_nb_chars=1000, locale='ja_JP')
    status = 'published'
    author_id = factory.Sequence(lambda n: f"user-{n}")
    category = factory.SubFactory(CategoryFactory)
    created_by = factory.SelfAttribute('author_id')
    updated_by = factory.SelfAttribute('author_id')
    
    @factory.post_generation
    def tags(self, create, extracted, **kwargs):
        if not create:
            return
        
        if extracted:
            self.tags = extracted
        else:
            self.tags = [fake.word() for _ in range(3)]
```

### 4.2 APIビューテスト

```python
# tests/test_api_views.py
import pytest
from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
from unittest.mock import patch
from tests.factories import UserProfileFactory, ContentFactory
from tests.helpers.jwt_helper import JWTTestHelper

@pytest.mark.django_db
class TestUserProfileAPI(TestCase):
    """ユーザープロフィールAPIのテスト"""
    
    def setUp(self):
        self.client = APIClient()
        self.jwt_helper = JWTTestHelper()
        self.user_id = "test-user-123"
        
        # テスト用プロフィール作成
        self.profile = UserProfileFactory(user_id=self.user_id)
    
    def _authorize_request(self, scopes=None):
        """認証ヘッダーを設定"""
        token = self.jwt_helper.create_token(
            user_id=self.user_id,
            scopes=scopes or ["profile:read"]
        )
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
    
    @patch('api.authentication.jwt_validator.JWTValidator.validate_token')
    def test_get_profile_success(self, mock_validate):
        """プロフィール取得成功"""
        # JWT検証のモック
        mock_validate.return_value = {
            "sub": self.user_id,
            "scope": "profile:read",
            "email": "test@example.com",
            "email_verified": True,
        }
        
        self._authorize_request()
        
        response = self.client.get(reverse('v1:user-profile'))
        
        assert response.status_code == 200
        assert response.json()['data']['id'] == str(self.profile.id)
        assert response.json()['data']['display_name'] == self.profile.display_name
    
    @patch('api.authentication.jwt_validator.JWTValidator.validate_token')
    def test_update_profile_success(self, mock_validate):
        """プロフィール更新成功"""
        mock_validate.return_value = {
            "sub": self.user_id,
            "scope": "profile:write",
        }
        
        self._authorize_request(scopes=["profile:write"])
        
        update_data = {
            "display_name": "新しい名前",
            "bio": "新しい自己紹介",
            "preferences": {
                "language": "en",
                "theme": "dark"
            }
        }
        
        response = self.client.patch(
            reverse('v1:user-profile'),
            data=update_data,
            format='json'
        )
        
        assert response.status_code == 200
        
        # DBから再取得して確認
        self.profile.refresh_from_db()
        assert self.profile.display_name == "新しい名前"
        assert self.profile.bio == "新しい自己紹介"
        assert self.profile.preferences['language'] == 'en'
        assert self.profile.preferences['theme'] == 'dark'
    
    def test_unauthorized_access(self):
        """認証なしでのアクセス拒否"""
        response = self.client.get(reverse('v1:user-profile'))
        
        assert response.status_code == 401
        assert response.json()['error']['code'] == 'UNAUTHORIZED'
    
    @patch('api.authentication.jwt_validator.JWTValidator.validate_token')
    def test_insufficient_scope(self, mock_validate):
        """スコープ不足でのアクセス拒否"""
        mock_validate.return_value = {
            "sub": self.user_id,
            "scope": "profile:read",  # writeスコープがない
        }
        
        self._authorize_request(scopes=["profile:read"])
        
        response = self.client.patch(
            reverse('v1:user-profile'),
            data={"display_name": "新しい名前"}
        )
        
        assert response.status_code == 403
        assert response.json()['error']['code'] == 'FORBIDDEN'
        assert 'profile:write' in response.json()['error']['details']['required_scopes']
```

### 4.3 統合テスト

```python
# tests/test_integration.py
import pytest
from django.test import TestCase, TransactionTestCase
from django.db import transaction
from rest_framework.test import APIClient
from tests.factories import UserProfileFactory, ContentFactory, CategoryFactory
from tests.helpers.jwt_helper import JWTTestHelper
import responses
from unittest.mock import patch

@pytest.mark.integration
class TestContentWorkflow(TransactionTestCase):
    """コンテンツワークフローの統合テスト"""
    
    def setUp(self):
        self.client = APIClient()
        self.jwt_helper = JWTTestHelper()
        self.author_id = "author-123"
        
        # テストデータ準備
        self.author = UserProfileFactory(user_id=self.author_id)
        self.category = CategoryFactory(name="技術", slug="tech")
    
    @patch('api.authentication.jwt_validator.JWTValidator._get_public_key')
    def test_complete_content_lifecycle(self, mock_get_key):
        """コンテンツの作成から削除までの完全なライフサイクル"""
        mock_get_key.return_value = self.jwt_helper.get_public_key_pem()
        
        # 認証設定
        token = self.jwt_helper.create_token(
            user_id=self.author_id,
            scopes=["content:write", "content:read"]
        )
        self.client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
        
        # 1. コンテンツ作成
        create_data = {
            "title": "統合テスト記事",
            "content": "# 見出し\n\nこれはテスト記事です。",
            "category_id": str(self.category.id),
            "tags": ["test", "integration"],
            "status": "draft"
        }
        
        create_response = self.client.post(
            '/api/v1/contents',
            data=create_data,
            format='json'
        )
        
        assert create_response.status_code == 201
        content_id = create_response.json()['data']['id']
        
        # 2. コンテンツ取得
        get_response = self.client.get(f'/api/v1/contents/{content_id}')
        
        assert get_response.status_code == 200
        content_data = get_response.json()['data']
        assert content_data['title'] == "統合テスト記事"
        assert content_data['status'] == 'draft'
        assert '<h1>見出し</h1>' in content_data['content_html']
        
        # 3. コンテンツ更新（公開）
        update_data = {
            "status": "published",
            "published_at": "2024-01-10T10:00:00Z"
        }
        
        update_response = self.client.patch(
            f'/api/v1/contents/{content_id}',
            data=update_data,
            format='json'
        )
        
        assert update_response.status_code == 200
        assert update_response.json()['data']['status'] == 'published'
        
        # 4. 公開コンテンツ一覧で確認
        list_response = self.client.get(
            '/api/v1/contents?status=published'
        )
        
        assert list_response.status_code == 200
        contents = list_response.json()['data']
        assert any(c['id'] == content_id for c in contents)
        
        # 5. コンテンツ削除（論理削除）
        delete_response = self.client.delete(
            f'/api/v1/contents/{content_id}'
        )
        
        assert delete_response.status_code == 204
        
        # 6. 削除後は取得できない
        get_deleted_response = self.client.get(
            f'/api/v1/contents/{content_id}'
        )
        
        assert get_deleted_response.status_code == 404
```

## 5. パフォーマンステスト

### 5.1 ベンチマークテスト

```python
# tests/test_performance.py
import pytest
from django.test import TestCase
from tests.factories import ContentFactory, UserProfileFactory
from tests.helpers.jwt_helper import JWTTestHelper
import time

@pytest.mark.slow
class TestAPIPerformance(TestCase):
    """APIパフォーマンステスト"""
    
    @classmethod
    def setUpTestData(cls):
        # 大量のテストデータ作成
        cls.users = UserProfileFactory.create_batch(100)
        cls.contents = ContentFactory.create_batch(1000)
    
    def test_content_list_performance(self):
        """コンテンツ一覧のパフォーマンス"""
        from django.test import Client
        client = Client()
        
        # N+1問題のチェック
        with self.assertNumQueries(3):  # 期待されるクエリ数
            response = client.get('/api/v1/contents?page=1&per_page=20')
            assert response.status_code == 200
    
    @pytest.mark.benchmark
    def test_jwt_validation_benchmark(self, benchmark):
        """JWT検証のベンチマーク"""
        jwt_helper = JWTTestHelper()
        validator = JWTValidator()
        
        token = jwt_helper.create_token()
        
        # キャッシュをウォームアップ
        with patch.object(validator, '_get_public_key') as mock:
            mock.return_value = jwt_helper.get_public_key_pem()
            validator.validate_token(token)
        
        # ベンチマーク実行
        def validate():
            with patch.object(validator, '_get_public_key') as mock:
                mock.return_value = jwt_helper.get_public_key_pem()
                return validator.validate_token(token)
        
        result = benchmark(validate)
        
        # パフォーマンス基準
        assert benchmark.stats['mean'] < 0.001  # 1ms以下
```

### 5.2 負荷テスト

```python
# tests/test_load.py
import pytest
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import statistics

@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_requests():
    """並行リクエストのテスト"""
    
    async def make_request(session, url, headers):
        async with session.get(url, headers=headers) as response:
            return response.status, await response.json()
    
    async def run_load_test():
        # テスト設定
        url = "http://localhost:8001/api/v1/health"
        concurrent_requests = 100
        total_requests = 1000
        
        jwt_helper = JWTTestHelper()
        token = jwt_helper.create_token()
        headers = {"Authorization": f"Bearer {token}"}
        
        # 結果収集
        response_times = []
        errors = 0
        
        async with aiohttp.ClientSession() as session:
            for batch in range(0, total_requests, concurrent_requests):
                start_time = time.time()
                
                # 並行リクエスト実行
                tasks = [
                    make_request(session, url, headers)
                    for _ in range(concurrent_requests)
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                batch_time = time.time() - start_time
                response_times.extend([batch_time / concurrent_requests] * concurrent_requests)
                
                # エラーカウント
                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                    elif result[0] != 200:
                        errors += 1
        
        # 統計計算
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        error_rate = errors / total_requests
        
        # アサーション
        assert avg_response_time < 0.1  # 平均100ms以下
        assert p95_response_time < 0.2  # 95%が200ms以下
        assert error_rate < 0.01  # エラー率1%以下
        
        print(f"Average response time: {avg_response_time:.3f}s")
        print(f"95th percentile: {p95_response_time:.3f}s")
        print(f"Error rate: {error_rate:.1%}")
    
    await run_load_test()
```

## 6. モックとスタブ

### 6.1 外部サービスのモック

```python
# tests/mocks/external_services.py
from unittest.mock import Mock, MagicMock
import responses

class MockAuthServer:
    """認証サーバーのモック"""
    
    def __init__(self):
        self.jwt_helper = JWTTestHelper()
    
    def setup_jwks_endpoint(self):
        """JWKSエンドポイントのセットアップ"""
        responses.add(
            responses.GET,
            "http://localhost:8000/.well-known/jwks.json",
            json=self.jwt_helper.get_jwks(),
            status=200
        )
    
    def setup_token_introspection(self, token_info):
        """トークンイントロスペクションのセットアップ"""
        responses.add(
            responses.POST,
            "http://localhost:8000/oauth/introspect",
            json={
                "active": token_info.get("active", True),
                "scope": token_info.get("scope", "profile:read"),
                "sub": token_info.get("sub", "test-user"),
                "exp": token_info.get("exp", 9999999999),
            },
            status=200
        )

class MockRedis:
    """Redisのモック"""
    
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value, timeout=None):
        self.data[key] = value
        return True
    
    def delete(self, key):
        return self.data.pop(key, None) is not None
    
    def exists(self, key):
        return key in self.data
    
    def clear(self):
        self.data.clear()

# コンテキストマネージャー
from contextlib import contextmanager

@contextmanager
def mock_auth_server():
    """認証サーバーモックのコンテキストマネージャー"""
    mock_server = MockAuthServer()
    
    with responses.RequestsMock() as rsps:
        mock_server.setup_jwks_endpoint()
        yield mock_server

@contextmanager
def mock_redis():
    """Redisモックのコンテキストマネージャー"""
    mock_redis_instance = MockRedis()
    
    with patch('django.core.cache.cache', mock_redis_instance):
        yield mock_redis_instance
```

### 6.2 テストでの使用例

```python
# tests/test_with_mocks.py
from tests.mocks.external_services import mock_auth_server, mock_redis

class TestWithMocks(TestCase):
    """モックを使用したテスト"""
    
    def test_jwt_validation_with_mock_auth_server(self):
        """モック認証サーバーでのJWT検証"""
        with mock_auth_server() as auth_server:
            # トークン生成
            token = auth_server.jwt_helper.create_token(
                user_id="mock-user-123",
                scopes=["admin"]
            )
            
            # API呼び出し
            response = self.client.get(
                '/api/v1/users/me',
                HTTP_AUTHORIZATION=f'Bearer {token}'
            )
            
            assert response.status_code == 200
    
    def test_caching_with_mock_redis(self):
        """モックRedisでのキャッシング"""
        with mock_redis() as redis:
            # キャッシュ設定
            redis.set('test_key', 'test_value')
            
            # キャッシュ取得
            value = redis.get('test_key')
            assert value == 'test_value'
            
            # キャッシュクリア
            redis.clear()
            assert redis.get('test_key') is None
```

## 7. テストデータ管理

### 7.1 フィクスチャ

```python
# tests/fixtures/test_data.py
import pytest
from tests.factories import UserProfileFactory, ContentFactory, CategoryFactory

@pytest.fixture
def test_user():
    """テストユーザーフィクスチャ"""
    return UserProfileFactory(
        user_id="fixture-user-123",
        display_name="テストユーザー",
        email="test@example.com"
    )

@pytest.fixture
def test_categories():
    """テストカテゴリフィクスチャ"""
    return [
        CategoryFactory(name="技術", slug="tech"),
        CategoryFactory(name="ビジネス", slug="business"),
        CategoryFactory(name="ライフスタイル", slug="lifestyle"),
    ]

@pytest.fixture
def test_contents(test_user, test_categories):
    """テストコンテンツフィクスチャ"""
    contents = []
    for i, category in enumerate(test_categories):
        for j in range(3):
            content = ContentFactory(
                title=f"{category.name}記事{j+1}",
                author_id=test_user.user_id,
                category=category,
                status="published" if j < 2 else "draft"
            )
            contents.append(content)
    return contents

@pytest.fixture
def authenticated_client(test_user):
    """認証済みAPIクライアント"""
    from rest_framework.test import APIClient
    from tests.helpers.jwt_helper import JWTTestHelper
    
    client = APIClient()
    jwt_helper = JWTTestHelper()
    
    token = jwt_helper.create_token(
        user_id=test_user.user_id,
        scopes=["profile:read", "profile:write", "content:read", "content:write"]
    )
    
    client.credentials(HTTP_AUTHORIZATION=f'Bearer {token}')
    return client
```

### 7.2 データベースシーディング

```python
# tests/management/commands/seed_test_data.py
from django.core.management.base import BaseCommand
from tests.factories import UserProfileFactory, ContentFactory, CategoryFactory
import random

class Command(BaseCommand):
    help = 'テストデータのシーディング'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--users',
            type=int,
            default=10,
            help='作成するユーザー数'
        )
        parser.add_argument(
            '--contents',
            type=int,
            default=50,
            help='作成するコンテンツ数'
        )
    
    def handle(self, *args, **options):
        # カテゴリ作成
        categories = [
            CategoryFactory(name="技術", slug="tech"),
            CategoryFactory(name="ビジネス", slug="business"),
            CategoryFactory(name="ライフスタイル", slug="lifestyle"),
            CategoryFactory(name="エンターテイメント", slug="entertainment"),
        ]
        
        # ユーザー作成
        users = UserProfileFactory.create_batch(options['users'])
        self.stdout.write(
            self.style.SUCCESS(f'{len(users)} users created')
        )
        
        # コンテンツ作成
        contents = []
        for _ in range(options['contents']):
            content = ContentFactory(
                author_id=random.choice(users).user_id,
                category=random.choice(categories),
                status=random.choice(['draft', 'published', 'published', 'published']),
            )
            contents.append(content)
        
        self.stdout.write(
            self.style.SUCCESS(f'{len(contents)} contents created')
        )
```

## 8. CI/CD統合

### 8.1 GitHub Actions設定

```yaml
# .github/workflows/test.yml
name: API Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run migrations
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/1
      run: |
        python manage.py migrate
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379/1
        JWKS_URL: http://testserver/.well-known/jwks.json
        OAUTH_ISSUER: http://testserver
      run: |
        pytest -v --cov=api --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### 8.2 テストカバレッジレポート

```python
# tests/conftest.py
import pytest
from django.conf import settings

def pytest_configure(config):
    """pytest設定"""
    settings.configure(
        DATABASES={
            'default': {
                'ENGINE': 'django.db.backends.postgresql',
                'NAME': 'test_postgres',
                'USER': 'postgres',
                'PASSWORD': 'postgres',
                'HOST': 'localhost',
                'PORT': '5432',
            }
        },
        INSTALLED_APPS=[
            'django.contrib.contenttypes',
            'django.contrib.auth',
            'api',
        ],
        MIDDLEWARE=[],
        ROOT_URLCONF='api.urls',
        SECRET_KEY='test-secret-key',
    )

@pytest.fixture(scope='session')
def django_db_setup():
    """データベースセットアップ"""
    from django.core.management import call_command
    call_command('migrate', '--run-syncdb')

# カバレッジ設定
coverage_config = {
    'source': ['api'],
    'omit': [
        '*/tests/*',
        '*/migrations/*',
        '*/__init__.py',
        '*/admin.py',
        '*/apps.py',
    ],
    'report': {
        'exclude_lines': [
            'pragma: no cover',
            'def __repr__',
            'raise AssertionError',
            'raise NotImplementedError',
            'if __name__ == .__main__.:',
        ],
    },
}
```

## 9. テストのベストプラクティス

### 9.1 テスト命名規則

```python
class TestNamingConventions:
    """テスト命名規則の例"""
    
    def test_should_return_user_profile_when_valid_token_provided(self):
        """正常系: 有効なトークンでユーザープロフィールを返す"""
        pass
    
    def test_should_raise_401_when_token_expired(self):
        """異常系: トークン期限切れで401エラー"""
        pass
    
    def test_should_update_profile_when_valid_data_provided(self):
        """正常系: 有効なデータでプロフィール更新"""
        pass
```

### 9.2 テストの構造化

```python
class TestStructure:
    """AAAパターンでのテスト構造"""
    
    def test_example(self):
        # Arrange (準備)
        user = UserProfileFactory()
        token = create_test_token(user.user_id)
        
        # Act (実行)
        response = self.client.get(
            '/api/v1/users/me',
            HTTP_AUTHORIZATION=f'Bearer {token}'
        )
        
        # Assert (検証)
        assert response.status_code == 200
        assert response.json()['data']['id'] == str(user.id)
```

## まとめ

このテスト戦略ガイドに従うことで、REST APIサーバーの品質を確保できます：

1. **包括的なテスト**: ユニットテストから統合テストまで全レイヤーをカバー
2. **JWT検証**: 認証フローの完全なテスト
3. **パフォーマンス**: ベンチマークと負荷テストによる性能保証
4. **自動化**: CI/CDパイプラインでの継続的なテスト実行
5. **保守性**: ファクトリーとフィクスチャによる効率的なテストデータ管理