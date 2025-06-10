# REST APIサーバー実装ドキュメント

## 概要

このディレクトリには、認証サーバーから発行されたJWTトークンを検証し、ビジネスロジックを提供するREST APIサーバー（Django）の実装に必要なドキュメントが含まれています。

## 🏗️ アーキテクチャでの位置づけ

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  BFF-Web    │────▶│ REST API    │ ← このサーバー
│             │◀────│  (Next.js)  │◀────│  (Django)   │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                         ┌─────────────┐
                                         │Auth Server  │
                                         │   (JWKS)    │
                                         └─────────────┘
```

## 📚 ドキュメント構成

### 🚀 セットアップガイド

0. **[インストールガイド](./00-installation-guide.md)** ⭐ **START HERE**
   - 開発環境セットアップ
   - 依存関係インストール
   - venv環境構築
   - VS Code設定

### 必須ドキュメント（実装前に必ず読む）

1. **[実装要件書](./01-implementation-requirements.md)**
   - REST APIサーバーの役割と責任
   - JWT検証要件
   - 必須実装機能

2. **[JWT検証実装ガイド](./02-jwt-validation-guide.md)**
   - JWKSを使用した公開鍵取得
   - JWT検証ロジック
   - キャッシュ戦略

3. **[API設計書](./03-api-specification.md)**
   - エンドポイント仕様
   - リクエスト/レスポンス形式
   - エラーハンドリング

### 実装ガイド

4. **[Django実装ガイド](./04-django-implementation.md)**
   - プロジェクト構成
   - 認証ミドルウェア
   - 権限管理

5. **[セキュリティ実装](./05-security-implementation.md)**
   - スコープベース認可
   - レート制限
   - CORS設定

6. **[データモデル設計](./06-data-models.md)**
   - ビジネスロジック用モデル
   - ユーザー関連データ
   - 監査ログ

### 統合とテスト

7. **[BFF統合ガイド](./07-bff-integration.md)**
   - BFF-Webとの通信仕様
   - 認証ヘッダー処理
   - エラー伝播

8. **[テスト戦略](./08-testing-guide.md)**
   - JWT検証テスト
   - API統合テスト
   - モックJWT生成

### 運用

9. **[監視と診断](./09-monitoring-guide.md)**
   - ヘルスチェック
   - メトリクス
   - ログ設計

10. **[デプロイメント](./10-deployment-guide.md)**
    - 環境設定
    - スケーリング戦略
    - 認証サーバー連携

## 🚀 クイックスタート

> **⚠️ 詳細なインストール手順は [00-installation-guide.md](./00-installation-guide.md) をご覧ください**

### 1. 開発環境準備

```bash
# 1. リポジトリクローン
git clone <repository-url>
cd blead-stamp-svr

# 2. Python仮想環境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 依存関係インストール
pip install --upgrade pip
pip install -r requirements.txt

# 4. 環境変数設定
cp .env.example .env
# .envファイルを編集

# 5. データベース初期化
python manage.py migrate
python manage.py createsuperuser

# 6. 静的ファイル収集
python manage.py collectstatic

# 7. 開発サーバー起動
python manage.py runserver 0.0.0.0:8000
```

### 2. 必要な環境変数

```env
# Django設定
SECRET_KEY=django-insecure-replace-this-in-production
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0

# データベース（PostgreSQL）
DATABASE_URL=postgresql://postgres:postgres@host.docker.internal:5432/postgres

# 認証サーバー（Docker環境用）
AUTH_SERVER_URL=http://host.docker.internal:8080
JWKS_URL=http://host.docker.internal:8080/oauth/.well-known/jwks.json
JWT_ALGORITHM=RS256
JWT_AUDIENCE=bff-web-client
JWT_ISSUER=http://host.docker.internal:8080/oauth

# Redis キャッシュ
REDIS_URL=redis://localhost:6379/1

# BFF-Web統合
BFF_WEB_URL=http://localhost:3000
CORS_ALLOWED_ORIGINS=http://localhost:3000

# Docker環境設定
DOCKER_ENVIRONMENT=False

# API設定
API_VERSION=v1
API_TITLE=REST API Server
API_DESCRIPTION=JWT authentication enabled REST API server for BFF-Web integration

# レート制限設定
THROTTLE_ANON=100/hour
THROTTLE_USER=1000/hour
THROTTLE_AUTH=60/min
THROTTLE_CONTENT_CREATE=10/hour
THROTTLE_CONTENT_LIKE=100/hour

# ログレベル
LOG_LEVEL=INFO
```

### 3. アプリケーション構造（分離後）

```
blead-stamp-svr/
├── authentication/          # 認証・セキュリティ専用アプリ
│   ├── authentication.py   # JWTAuthentication, DummyJWTAuthentication
│   ├── permissions.py      # HasScope, HasAnyScope権限クラス
│   ├── middleware.py       # セキュリティ、リクエストID、ログ関連
│   └── utils.py           # StandardAPIResponse, APIErrorCodes
├── stamp/                  # ビジネスロジック専用アプリ
│   ├── models.py          # スタンプラリーモデル
│   ├── models_generic.py  # 汎用コンテンツ管理モデル
│   ├── views_v1.py        # 標準API v1エンドポイント
│   ├── views_content.py   # 汎用コンテンツ管理API
│   ├── views_upload.py    # ファイルアップロード機能
│   ├── serializers_v1.py  # v1 API用シリアライザー
│   ├── throttles.py       # レート制限設定
│   ├── search.py          # 全文検索機能
│   └── upload_handlers.py # ファイルアップロード処理
└── config/                # Django設定
    ├── settings.py        # メイン設定ファイル
    └── urls.py           # URLルーティング
```

### 4. JWT検証の実装（authentication/authentication.py）

```python
from rest_framework.authentication import BaseAuthentication
from rest_framework import exceptions
from jose import jwt, JWTError
import requests
from django.core.cache import cache
from django.contrib.auth import get_user_model

class JWTAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.META.get('HTTP_AUTHORIZATION')
        if not auth_header or not auth_header.startswith('Bearer '):
            return None
            
        token = auth_header.split(' ')[1]
        
        try:
            # JWKSから公開鍵取得（キャッシュ付き）
            public_keys = self._get_public_keys()
            
            # JWT署名検証・クレーム検証
            payload = jwt.decode(
                token,
                public_keys,
                algorithms=['RS256'],
                audience=settings.JWT_AUDIENCE,
                issuer=settings.JWT_ISSUER
            )
            
            # Djangoユーザー取得/作成
            user = self._get_or_create_user(payload)
            return (user, payload)
            
        except JWTError as e:
            raise exceptions.AuthenticationFailed(f'Invalid JWT: {str(e)}')
    
    def _get_public_keys(self):
        """JWKSエンドポイントから公開鍵取得（1時間キャッシュ）"""
        cache_key = "jwks_public_keys"
        keys = cache.get(cache_key)
        
        if not keys:
            response = requests.get(settings.JWKS_URL, timeout=10)
            response.raise_for_status()
            jwks = response.json()
            keys = {key['kid']: key for key in jwks['keys']}
            cache.set(cache_key, keys, timeout=3600)  # 1時間キャッシュ
        
        return keys
```

## 📋 実装チェックリスト

### 基本機能 ✅
- [x] JWT検証認証クラス (`authentication/authentication.py`)
- [x] JWKS公開鍵取得とキャッシュ（1時間）
- [x] スコープベース権限管理 (`authentication/permissions.py`)
- [x] APIエンドポイント実装（v1構造）
- [x] 標準化エラーハンドリング (`authentication/utils.py`)

### セキュリティ ✅
- [x] Bearer Token検証
- [x] スコープ検証（HasScope, HasAnyScope）
- [x] レート制限（`stamp/throttles.py`）
- [x] CORS設定（BFF-Web統合対応）
- [x] セキュリティヘッダー（CSP, XSS, HSTS）

### 統合 ✅
- [x] BFF-Webからのリクエスト受信
- [x] 認証サーバーJWKS連携（`host.docker.internal:8080`）
- [x] ユーザー情報自動同期
- [x] 標準APIレスポンス統一

### 運用 ✅
- [x] ヘルスチェックエンドポイント (`/api/v1/health/`)
- [x] リクエストID追跡
- [x] 包括的ログ出力
- [x] Redis キャッシュ対応

### 汎用機能 ✅
- [x] コンテンツ管理システム（`models_generic.py`）
- [x] ファイルアップロード機能
- [x] 全文検索機能
- [x] ユーザー活動追跡
- [x] API バージョニング（/api/v1/, /api/）

## 🔑 重要な考慮事項

### JWT検証のポイント

1. **オフライン検証**: 認証サーバーへの問い合わせなしでJWT検証
2. **公開鍵キャッシュ**: JWKSエンドポイントへのアクセスを最小化
3. **署名検証**: RS256アルゴリズムでの署名検証必須
4. **有効期限確認**: exp クレームの検証

### スコープによる認可（実装済み）

```python
# authentication/permissions.py の権限クラスを使用
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from authentication.permissions import HasScope, HasAnyScope

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_profile(request):
    # 基本認証のみ（全認証ユーザーがアクセス可能）
    pass

@api_view(['PATCH'])
@permission_classes([IsAuthenticated, HasScope('profile:write')])
def update_user_profile(request):
    # profile:writeスコープが必要
    pass

@api_view(['POST'])
@permission_classes([IsAuthenticated, HasScope('content:write')])
def create_content(request):
    # content:writeスコープが必要
    pass

@api_view(['POST'])
@permission_classes([IsAuthenticated, HasAnyScope(['content:write', 'admin:all'])])
def admin_content_action(request):
    # content:writeまたはadmin:allスコープのいずれかが必要
    pass
```

### エラーレスポンス標準化（実装済み）

```json
{
    "error": {
        "code": "AUTHENTICATION_FAILED",
        "message": "認証に失敗しました",
        "details": {
            "reason": "invalid_token",
            "description": "JWT署名が無効です"
        }
    },
    "meta": {
        "request_id": "req_123456789",
        "timestamp": "2024-01-10T12:00:00Z",
        "version": "v1"
    }
}
```

利用可能なエラーコード（`authentication/utils.py`）：
- `AUTHENTICATION_FAILED`: 認証失敗
- `PERMISSION_DENIED`: 権限不足
- `VALIDATION_ERROR`: バリデーションエラー
- `NOT_FOUND`: リソース未発見
- `INTERNAL_ERROR`: サーバー内部エラー
- `RATE_LIMIT_EXCEEDED`: レート制限超過

## 🔗 BFF-Webとの連携

### 実装済みAPIエンドポイント

#### 標準エンドポイント（`stamp/views_v1.py`）
```http
GET /api/v1/health/          # ヘルスチェック（詳細システム状態）
GET /api/v1/users/me/        # ユーザープロフィール取得
PATCH /api/v1/users/me/      # ユーザープロフィール更新
GET /api/v1/dashboard/       # ダッシュボード概要
GET /api/v1/search/          # 全文検索（位置情報対応）
```

#### コンテンツ管理（`stamp/views_content.py`）
```http
GET /api/v1/contents/        # コンテンツ一覧（フィルタ・ページネーション）
POST /api/v1/contents/       # コンテンツ作成
GET /api/v1/contents/{id}/   # コンテンツ詳細
PATCH /api/v1/contents/{id}/ # コンテンツ更新
POST /api/v1/contents/{id}/like/ # いいね/いいね解除

GET /api/v1/categories/      # カテゴリ一覧
GET /api/v1/tags/           # タグ一覧
GET /api/v1/users/me/activities/ # ユーザー活動履歴
```

#### ファイルアップロード（`stamp/views_upload.py`）
```http
POST /api/v1/contents/{id}/upload/ # ファイルアップロード
DELETE /api/v1/media/{id}/         # メディアファイル削除
GET /api/v1/media/{id}/           # メディアファイル情報
```

### 標準レスポンス形式

#### 成功レスポンス
```json
{
    "data": {
        "id": "user_123",
        "username": "john_doe",
        "email": "john@example.com",
        "profile": {
            "display_name": "John Doe",
            "initials": "JD"
        },
        "jwt_metadata": {
            "subject": "user_123",
            "issued_at": 1641811200,
            "expires_at": 1641814800,
            "scopes": ["profile:read", "content:write"]
        }
    },
    "meta": {
        "request_id": "req_123456789",
        "timestamp": "2024-01-10T12:00:00Z",
        "version": "v1"
    }
}
```

#### ページネーション対応レスポンス
```json
{
    "data": [...],
    "meta": {
        "request_id": "req_123456789",
        "timestamp": "2024-01-10T12:00:00Z",
        "version": "v1",
        "pagination": {
            "page": 1,
            "per_page": 20,
            "total": 150,
            "total_pages": 8
        }
    },
    "links": {
        "next": "/api/v1/contents/?page=2",
        "previous": null
    }
}
```

## 🆘 サポート

### トラブルシューティング

**Q: JWT検証が失敗する（AUTHENTICATION_FAILED）**
A: 以下を確認してください：
- JWKSエンドポイント: `http://host.docker.internal:8080/oauth/.well-known/jwks.json`
- Redis が起動しているか（`redis-cli ping`）
- 環境変数 `JWKS_URL`, `JWT_ISSUER`, `JWT_AUDIENCE` が正しく設定されているか

**Q: 403 Forbiddenが返される（PERMISSION_DENIED）**
A: JWTペイロードに必要なスコープが含まれているか確認：
- 開発時は `DummyJWTAuthentication` を有効化
- スコープ例: `content:write`, `profile:write`, `admin:all`

**Q: Docker環境でアクセスできない**
A: 以下を確認：
- `host.docker.internal` が正しく名前解決されているか
- 認証サーバーが8080ポートで稼働しているか
- CORS設定でBFF-WebのURLが許可されているか

**Q: Redisキャッシュエラー**
A: Redis サーバーを起動：
```bash
sudo service redis-server start
redis-cli ping  # PONG が返ることを確認
```

**Q: パフォーマンスが悪い**
A: 以下を確認：
- JWKSキャッシュが有効（1時間キャッシュ）
- JWT検証結果キャッシュが有効（1分キャッシュ）
- データベースコネクションプールが適切に設定

### 関連ドキュメント

- [認証サーバードキュメント](../docs-auth/)
- [BFF-Webドキュメント](../docs-bff/)
- [OAuth2.0仕様](https://tools.ietf.org/html/rfc6749)
- [JWT仕様](https://tools.ietf.org/html/rfc7519)

## 📝 更新履歴

| 日付 | バージョン | 内容 |
|------|-----------|---------|
| 2025-06-09 | 2.0.0 | アプリケーション分離完了・Docker対応・汎用テンプレート化 |
| 2024-01-10 | 1.0.0 | 初版作成 |

### v2.0.0 主要変更点
- **アプリケーション分離**: `authentication` アプリと `stamp` アプリに分離
- **Docker環境対応**: `host.docker.internal` 対応、環境変数による動的設定
- **汎用テンプレート化**: 汎用コンテンツ管理システム、標準APIレスポンス形式
- **包括的ミドルウェア**: セキュリティ、ログ、リクエスト追跡
- **JWKS連携確認**: 認証サーバーとの正常な公開鍵取得を確認
- **Redis統合**: キャッシュシステム完全対応

---

**重要**: 
- このREST APIサーバーは認証サーバー（`host.docker.internal:8080`）のJWKSエンドポイントから公開鍵を取得してJWT検証を行います
- JWTの発行は認証サーバーが担当し、このサーバーは検証のみを行います
- アプリケーション分離により、認証機能とビジネスロジックが明確に分離されています