# REST API 仕様書

## 1. 概要

このドキュメントは、JWT認証を使用するREST APIサーバーのエンドポイント仕様を定義します。

### 1.1 基本情報

- **ベースURL**: `https://api.example.com`
- **バージョン**: v1
- **形式**: JSON
- **認証**: Bearer Token (JWT)
- **文字エンコーディング**: UTF-8

### 1.2 共通仕様

#### リクエストヘッダー

```http
Authorization: Bearer {jwt_token}
Content-Type: application/json
Accept: application/json
X-Request-ID: {unique-request-id}
Accept-Language: ja,en;q=0.9
```

#### レスポンスヘッダー

```http
Content-Type: application/json; charset=utf-8
X-Request-ID: {unique-request-id}
X-Response-Time: {milliseconds}ms
X-Rate-Limit-Limit: 1000
X-Rate-Limit-Remaining: 999
X-Rate-Limit-Reset: 1641024000
```

### 1.3 JWT認証について

認証サーバーから発行されたJWTトークンを使用します。トークンには以下の情報が含まれます：

```json
{
    "iss": "http://localhost:8000",
    "sub": "user-id",
    "aud": "bff-web-client",
    "exp": 1704931200,
    "iat": 1704927600,
    "scope": "profile:read profile:write dashboard:read",
    "email": "user@example.com",
    "email_verified": true,
    "name": "ユーザー名"
}
```

## 2. エンドポイント一覧

### 2.1 ヘルスチェック

#### `GET /api/v1/health`

**説明**: APIサーバーの稼働状態を確認

**認証**: 不要

**レスポンス例**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-10T10:00:00Z",
    "version": "1.0.0",
    "uptime": 3600,
    "checks": {
        "database": {
            "status": "healthy",
            "latency_ms": 5
        },
        "redis": {
            "status": "healthy",
            "latency_ms": 1
        },
        "auth_server": {
            "status": "healthy",
            "latency_ms": 50,
            "jwks_cached": true
        }
    }
}
```

**エラーレスポンス**:
```json
{
    "status": "unhealthy",
    "timestamp": "2024-01-10T10:00:00Z",
    "version": "1.0.0",
    "checks": {
        "database": {
            "status": "unhealthy",
            "error": "Connection timeout"
        }
    }
}
```

### 2.2 ユーザー関連

#### `GET /api/v1/users/me`

**説明**: 現在のユーザー情報を取得

**認証**: 必須

**必要スコープ**: `profile:read`

**レスポンス例**:
```json
{
    "data": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "email": "user@example.com",
        "email_verified": true,
        "name": "山田太郎",
        "preferred_username": "yamada",
        "avatar_url": "https://cdn.example.com/avatars/123.jpg",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-10T10:00:00Z",
        "profile": {
            "bio": "ソフトウェアエンジニア",
            "location": "東京",
            "website": "https://example.com",
            "company": "Example Corp",
            "preferences": {
                "language": "ja",
                "timezone": "Asia/Tokyo",
                "theme": "dark",
                "notifications": {
                    "email": true,
                    "push": false
                }
            }
        },
        "stats": {
            "posts_count": 42,
            "followers_count": 100,
            "following_count": 50
        },
        "linked_providers": ["google", "github"]
    }
}
```

**エラーレスポンス**:
```json
{
    "error": {
        "code": "UNAUTHORIZED",
        "message": "Authentication required",
        "details": {
            "reason": "invalid_token"
        },
        "request_id": "req_123abc"
    }
}
```

#### `PATCH /api/v1/users/me`

**説明**: 現在のユーザー情報を更新

**認証**: 必須

**必要スコープ**: `profile:write`

**リクエストボディ**:
```json
{
    "name": "山田太郎",
    "profile": {
        "bio": "フルスタックエンジニア",
        "location": "大阪",
        "website": "https://my-blog.example.com",
        "company": "New Company Inc"
    },
    "preferences": {
        "language": "en",
        "theme": "light",
        "notifications": {
            "email": false
        }
    }
}
```

**レスポンス例**:
```json
{
    "data": {
        "id": "123e4567-e89b-12d3-a456-426614174000",
        "email": "user@example.com",
        "name": "山田太郎",
        "updated_at": "2024-01-10T10:30:00Z",
        "profile": {
            "bio": "フルスタックエンジニア",
            "location": "大阪",
            "website": "https://my-blog.example.com",
            "company": "New Company Inc"
        }
    },
    "message": "Profile updated successfully"
}
```

**バリデーションエラー**:
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid request data",
        "details": {
            "fields": {
                "profile.website": ["Invalid URL format"],
                "preferences.language": ["Unsupported language code"]
            }
        },
        "request_id": "req_456def"
    }
}
```

#### `POST /api/v1/users/me/avatar`

**説明**: ユーザーアバターをアップロード

**認証**: 必須

**必要スコープ**: `profile:write`

**リクエスト**: `multipart/form-data`
- `avatar`: 画像ファイル（JPEG, PNG, GIF）
- 最大サイズ: 5MB
- 推奨サイズ: 400x400px

**レスポンス例**:
```json
{
    "data": {
        "avatar_url": "https://cdn.example.com/avatars/123e4567.jpg",
        "thumbnail_url": "https://cdn.example.com/avatars/123e4567_thumb.jpg"
    },
    "message": "Avatar uploaded successfully"
}
```

### 2.3 ダッシュボード

#### `GET /api/v1/dashboard`

**説明**: ダッシュボードデータを取得

**認証**: 必須

**必要スコープ**: `dashboard:read` または `admin`

**クエリパラメータ**:
- `period`: 期間（`today`, `week`, `month`, `year`）デフォルト: `month`
- `timezone`: タイムゾーン（例: `Asia/Tokyo`）デフォルト: UTC

**レスポンス例**:
```json
{
    "data": {
        "summary": {
            "total_views": 12345,
            "total_interactions": 567,
            "active_users": 89,
            "growth_rate": 15.5
        },
        "charts": {
            "daily_views": [
                {
                    "date": "2024-01-01",
                    "views": 420,
                    "unique_users": 150
                },
                {
                    "date": "2024-01-02",
                    "views": 380,
                    "unique_users": 140
                }
            ],
            "top_content": [
                {
                    "id": "content_123",
                    "title": "人気の記事",
                    "views": 1500,
                    "engagement_rate": 25.5
                }
            ]
        },
        "recent_activities": [
            {
                "id": "activity_789",
                "type": "comment",
                "user": {
                    "id": "user_456",
                    "name": "田中花子",
                    "avatar_url": "https://cdn.example.com/avatars/456.jpg"
                },
                "content": "素晴らしい記事でした！",
                "created_at": "2024-01-10T09:30:00Z"
            }
        ]
    },
    "meta": {
        "period": "month",
        "timezone": "Asia/Tokyo",
        "generated_at": "2024-01-10T10:00:00Z"
    }
}
```

### 2.4 コンテンツ管理

#### `GET /api/v1/contents`

**説明**: コンテンツ一覧を取得

**認証**: 必須

**必要スコープ**: `content:read`

**クエリパラメータ**:
- `page`: ページ番号（デフォルト: 1）
- `per_page`: 1ページあたりの件数（デフォルト: 20、最大: 100）
- `sort`: ソート順（`created_at`, `updated_at`, `title`）
- `order`: 並び順（`asc`, `desc`）デフォルト: `desc`
- `status`: ステータスフィルター（`draft`, `published`, `archived`）
- `search`: 検索キーワード

**レスポンス例**:
```json
{
    "data": [
        {
            "id": "content_123",
            "title": "サンプル記事",
            "slug": "sample-article",
            "excerpt": "この記事は...",
            "status": "published",
            "author": {
                "id": "user_123",
                "name": "山田太郎",
                "avatar_url": "https://cdn.example.com/avatars/123.jpg"
            },
            "tags": ["技術", "プログラミング", "Python"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-05T10:00:00Z",
            "published_at": "2024-01-02T00:00:00Z",
            "stats": {
                "views": 1500,
                "likes": 45,
                "comments": 12
            }
        }
    ],
    "meta": {
        "current_page": 1,
        "per_page": 20,
        "total_pages": 5,
        "total_count": 95
    },
    "links": {
        "first": "/api/v1/contents?page=1",
        "last": "/api/v1/contents?page=5",
        "prev": null,
        "next": "/api/v1/contents?page=2"
    }
}
```

#### `GET /api/v1/contents/{id}`

**説明**: 特定のコンテンツを取得

**認証**: 必須

**必要スコープ**: `content:read`

**パスパラメータ**:
- `id`: コンテンツID

**レスポンス例**:
```json
{
    "data": {
        "id": "content_123",
        "title": "サンプル記事",
        "slug": "sample-article",
        "content": "# 見出し\n\nこれは本文です...",
        "content_html": "<h1>見出し</h1><p>これは本文です...</p>",
        "status": "published",
        "author": {
            "id": "user_123",
            "name": "山田太郎",
            "avatar_url": "https://cdn.example.com/avatars/123.jpg",
            "bio": "ソフトウェアエンジニア"
        },
        "tags": ["技術", "プログラミング", "Python"],
        "metadata": {
            "reading_time": 5,
            "word_count": 1200,
            "language": "ja"
        },
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-05T10:00:00Z",
        "published_at": "2024-01-02T00:00:00Z"
    }
}
```

#### `POST /api/v1/contents`

**説明**: 新規コンテンツを作成

**認証**: 必須

**必要スコープ**: `content:write`

**リクエストボディ**:
```json
{
    "title": "新しい記事",
    "slug": "new-article",
    "content": "# 見出し\n\n本文...",
    "tags": ["技術", "新機能"],
    "status": "draft",
    "metadata": {
        "featured_image": "https://example.com/image.jpg",
        "seo_title": "SEO用タイトル",
        "seo_description": "SEO用説明文"
    }
}
```

**レスポンス例**:
```json
{
    "data": {
        "id": "content_456",
        "title": "新しい記事",
        "slug": "new-article",
        "status": "draft",
        "created_at": "2024-01-10T10:00:00Z"
    },
    "message": "Content created successfully"
}
```

### 2.5 検索

#### `GET /api/v1/search`

**説明**: 全文検索

**認証**: 必須

**必要スコープ**: `search:read` または `content:read`

**クエリパラメータ**:
- `q`: 検索クエリ（必須）
- `type`: 検索対象（`all`, `content`, `user`, `tag`）デフォルト: `all`
- `page`: ページ番号
- `per_page`: 1ページあたりの件数

**レスポンス例**:
```json
{
    "data": {
        "results": [
            {
                "type": "content",
                "score": 0.95,
                "item": {
                    "id": "content_123",
                    "title": "Python プログラミング入門",
                    "excerpt": "Pythonの基礎を学ぶ...",
                    "url": "/contents/content_123"
                },
                "highlights": {
                    "title": "<mark>Python</mark> プログラミング入門",
                    "content": "...<mark>Python</mark>の基礎を学ぶ..."
                }
            },
            {
                "type": "user",
                "score": 0.80,
                "item": {
                    "id": "user_456",
                    "name": "Python太郎",
                    "avatar_url": "https://cdn.example.com/avatars/456.jpg"
                }
            }
        ],
        "facets": {
            "type": {
                "content": 15,
                "user": 3,
                "tag": 7
            },
            "tags": {
                "Python": 10,
                "プログラミング": 8,
                "入門": 5
            }
        }
    },
    "meta": {
        "query": "Python",
        "total_results": 25,
        "search_time_ms": 42
    }
}
```

## 3. エラーレスポンス

### 3.1 エラー形式

すべてのエラーレスポンスは以下の形式に従います：

```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable error message",
        "details": {
            // エラー詳細（オプション）
        },
        "request_id": "req_unique_id",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

### 3.2 エラーコード一覧

| HTTPステータス | エラーコード | 説明 |
|--------------|-------------|------|
| 400 | `BAD_REQUEST` | 不正なリクエスト |
| 400 | `VALIDATION_ERROR` | バリデーションエラー |
| 401 | `UNAUTHORIZED` | 認証が必要 |
| 401 | `TOKEN_EXPIRED` | トークンの有効期限切れ（15分） |
| 401 | `INVALID_TOKEN` | 無効なトークン |
| 401 | `TOKEN_REVOKED` | トークンが無効化されている |
| 403 | `FORBIDDEN` | アクセス権限なし |
| 403 | `INSUFFICIENT_SCOPE` | スコープ不足 |
| 404 | `NOT_FOUND` | リソースが見つからない |
| 409 | `CONFLICT` | リソースの競合 |
| 429 | `RATE_LIMIT_EXCEEDED` | レート制限超過 |
| 500 | `INTERNAL_ERROR` | サーバー内部エラー |
| 503 | `SERVICE_UNAVAILABLE` | サービス利用不可 |

### 3.3 エラーレスポンス例

#### バリデーションエラー
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid request data",
        "details": {
            "fields": {
                "email": ["Invalid email format"],
                "age": ["Must be greater than 0"]
            }
        },
        "request_id": "req_abc123",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

#### 認証エラー
```json
{
    "error": {
        "code": "UNAUTHORIZED",
        "message": "Authentication required",
        "details": {
            "reason": "Missing Authorization header"
        },
        "request_id": "req_def456",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

#### トークン期限切れエラー
```json
{
    "error": {
        "code": "TOKEN_EXPIRED",
        "message": "The access token expired",
        "details": {
            "expired_at": "2024-01-10T09:45:00Z",
            "issued_at": "2024-01-10T09:30:00Z"
        },
        "request_id": "req_token789",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

#### スコープ不足エラー
```json
{
    "error": {
        "code": "INSUFFICIENT_SCOPE",
        "message": "Insufficient permissions",
        "details": {
            "required_scopes": ["content:write"],
            "user_scopes": ["content:read"]
        },
        "request_id": "req_ghi789",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

## 4. レート制限

### 4.1 制限値

| エンドポイント | 制限 | ウィンドウ |
|-------------|------|-----------|
| 認証なしエンドポイント | 60回/時 | 1時間 |
| 認証ありエンドポイント | 1000回/時 | 1時間 |
| 検索エンドポイント | 100回/時 | 1時間 |
| ファイルアップロード | 10回/時 | 1時間 |

### 4.2 レート制限ヘッダー

```http
X-Rate-Limit-Limit: 1000
X-Rate-Limit-Remaining: 999
X-Rate-Limit-Reset: 1641024000
```

### 4.3 制限超過時のレスポンス

```json
{
    "error": {
        "code": "RATE_LIMIT_EXCEEDED",
        "message": "Too many requests",
        "details": {
            "retry_after": 3600,
            "limit": 1000,
            "reset_at": "2024-01-10T11:00:00Z"
        },
        "request_id": "req_jkl012",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

## 5. ページネーション

### 5.1 リクエストパラメータ

- `page`: ページ番号（1から開始）
- `per_page`: 1ページあたりの件数（デフォルト: 20、最大: 100）

### 5.2 レスポンス形式

```json
{
    "data": [...],
    "meta": {
        "current_page": 2,
        "per_page": 20,
        "total_pages": 10,
        "total_count": 195,
        "from": 21,
        "to": 40
    },
    "links": {
        "first": "/api/v1/resource?page=1&per_page=20",
        "last": "/api/v1/resource?page=10&per_page=20",
        "prev": "/api/v1/resource?page=1&per_page=20",
        "next": "/api/v1/resource?page=3&per_page=20"
    }
}
```

## 6. フィルタリングとソート

### 6.1 フィルタリング

クエリパラメータを使用してフィルタリング：

```
GET /api/v1/contents?status=published&tags=Python,Django
```

### 6.2 ソート

`sort`と`order`パラメータを使用：

```
GET /api/v1/contents?sort=created_at&order=desc
```

複数フィールドでのソート：

```
GET /api/v1/contents?sort=status,created_at&order=asc,desc
```

## 7. バージョニング

### 7.1 URLパスベース

現在のバージョン: `v1`

```
https://api.example.com/api/v1/users/me
```

### 7.2 廃止予定の通知

廃止予定のエンドポイントには以下のヘッダーが含まれます：

```http
Deprecation: true
Sunset: Wed, 31 Dec 2024 23:59:59 GMT
Link: <https://api.example.com/docs/migrations>; rel="deprecation"
```

## 8. スコープ一覧

### 8.1 利用可能なスコープ

| スコープ | 説明 | 権限 |
|---------|------|------|
| `profile:read` | プロフィール読み取り | ユーザー情報の閲覧 |
| `profile:write` | プロフィール書き込み | ユーザー情報の編集 |
| `content:read` | コンテンツ読み取り | コンテンツの閲覧 |
| `content:write` | コンテンツ書き込み | コンテンツの作成・編集 |
| `dashboard:read` | ダッシュボード読み取り | 統計情報の閲覧 |
| `search:read` | 検索実行 | 検索機能の利用 |
| `admin` | 管理者権限 | すべての操作 |

### 8.2 スコープの階層

- `admin`スコープはすべての操作を許可
- 書き込みスコープは対応する読み取り権限を含む（例：`profile:write`は`profile:read`を含む）

## 9. WebSocket サポート（オプション）

### 9.1 接続

```javascript
const ws = new WebSocket('wss://api.example.com/ws');
ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer {jwt_token}'
}));
```

### 9.2 メッセージ形式

```json
{
    "type": "notification",
    "data": {
        "id": "notif_123",
        "message": "新しいコメントがあります",
        "timestamp": "2024-01-10T10:00:00Z"
    }
}
```

## 10. 開発者向け情報

### 10.1 サンドボックス環境

- URL: `https://sandbox-api.example.com`
- 制限: 本番環境の1/10のレート制限
- データ: 毎日リセット

### 10.2 APIクライアントライブラリ

- JavaScript/TypeScript: `npm install @example/api-client`
- Python: `pip install example-api-client`
- Go: `go get github.com/example/api-client-go`

### 10.3 OpenAPI仕様

OpenAPI 3.0仕様書: `https://api.example.com/openapi.json`

## 11. JWT検証に関する注意事項

### 11.1 トークン有効期限

- アクセストークン: 15分
- トークン期限切れの場合は401エラー
- BFF-Webでリフレッシュトークンを使用して新しいトークンを取得

### 11.2 JWKS取得

- エンドポイント: `http://localhost:8000/oauth/.well-known/jwks.json`
- キャッシュ期間: 1時間
- 鍵ローテーション: 月次

### 11.3 必須クレーム

- `sub`: ユーザーID
- `scope`: 権限スコープ
- `client_id`: クライアントID
- `jti`: JWT ID（トークン無効化チェック用）

## まとめ

このAPI仕様書に従って実装することで、一貫性のある、使いやすいREST APIを提供できます。認証にはJWTトークンを使用し、適切なスコープによる認可を行います。認証サーバーとの連携により、セキュアなAPIアクセスを実現します。