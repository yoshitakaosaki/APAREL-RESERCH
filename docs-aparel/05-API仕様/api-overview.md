# API仕様概要

## 1. 概要

本ドキュメントは、テックパック生成アプリケーションのAPI仕様を定義します。RESTful APIアーキテクチャに基づき、マイクロサービス間の通信を標準化します。

## 2. API設計原則

### 2.1 RESTful設計

- **リソース指向**: URLはリソースを表現
- **HTTPメソッド**: 標準的なCRUD操作
- **ステートレス**: 各リクエストは独立
- **統一インターフェース**: 一貫性のあるAPI設計

### 2.2 命名規則

```
# リソース（複数形）
GET    /api/v1/projects
GET    /api/v1/projects/{id}
POST   /api/v1/projects
PUT    /api/v1/projects/{id}
DELETE /api/v1/projects/{id}

# サブリソース
GET    /api/v1/projects/{id}/sections
POST   /api/v1/projects/{id}/sections

# アクション（動詞）
POST   /api/v1/projects/{id}/publish
POST   /api/v1/projects/{id}/duplicate
```

### 2.3 バージョニング

- URLパスベース: `/api/v1/`
- 後方互換性の維持
- 非推奨APIの段階的廃止

## 3. 認証・認可

### 3.1 認証方式

```typescript
// JWT Bearer Token
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

// API Key (特定用途)
X-API-Key: sk_live_abcdef123456
```

### 3.2 トークン仕様

```typescript
interface JWTPayload {
  sub: string;          // ユーザーID
  email: string;        // メールアドレス
  roles: string[];      // ロール
  permissions: string[]; // 権限
  iat: number;          // 発行時刻
  exp: number;          // 有効期限
  jti: string;          // トークンID
}
```

## 4. リクエスト・レスポンス形式

### 4.1 標準レスポンス形式

#### 成功レスポンス

```json
{
  "success": true,
  "data": {
    // リソースデータ
  },
  "meta": {
    "timestamp": "2024-03-20T10:30:00Z",
    "version": "1.0"
  }
}
```

#### エラーレスポンス

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "入力データが無効です",
    "details": [
      {
        "field": "style_number",
        "message": "スタイル番号は必須です"
      }
    ]
  },
  "meta": {
    "timestamp": "2024-03-20T10:30:00Z",
    "request_id": "req_123456"
  }
}
```

### 4.2 ページネーション

```json
{
  "success": true,
  "data": [...],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 100,
    "total_pages": 5,
    "has_next": true,
    "has_prev": false
  },
  "links": {
    "self": "/api/v1/projects?page=1&per_page=20",
    "next": "/api/v1/projects?page=2&per_page=20",
    "prev": null,
    "first": "/api/v1/projects?page=1&per_page=20",
    "last": "/api/v1/projects?page=5&per_page=20"
  }
}
```

### 4.3 フィルタリング・ソート

```
# フィルタリング
GET /api/v1/projects?status=active&created_after=2024-01-01

# ソート
GET /api/v1/projects?sort=-created_at,style_number

# 検索
GET /api/v1/projects?q=summer+collection

# フィールド選択
GET /api/v1/projects?fields=id,style_number,status
```

## 5. エラーハンドリング

### 5.1 HTTPステータスコード

| コード | 意味 | 使用場面 |
|--------|------|----------|
| 200 | OK | 成功（GET, PUT） |
| 201 | Created | リソース作成成功（POST） |
| 204 | No Content | 成功（DELETE） |
| 400 | Bad Request | 不正なリクエスト |
| 401 | Unauthorized | 認証エラー |
| 403 | Forbidden | 権限エラー |
| 404 | Not Found | リソースが存在しない |
| 409 | Conflict | 競合エラー |
| 422 | Unprocessable Entity | バリデーションエラー |
| 429 | Too Many Requests | レート制限 |
| 500 | Internal Server Error | サーバーエラー |
| 503 | Service Unavailable | メンテナンス中 |

### 5.2 エラーコード体系

```typescript
enum ErrorCode {
  // 認証・認可
  AUTH_INVALID_TOKEN = 'AUTH_INVALID_TOKEN',
  AUTH_TOKEN_EXPIRED = 'AUTH_TOKEN_EXPIRED',
  AUTH_INSUFFICIENT_PERMISSIONS = 'AUTH_INSUFFICIENT_PERMISSIONS',
  
  // バリデーション
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  VALIDATION_REQUIRED_FIELD = 'VALIDATION_REQUIRED_FIELD',
  VALIDATION_INVALID_FORMAT = 'VALIDATION_INVALID_FORMAT',
  
  // リソース
  RESOURCE_NOT_FOUND = 'RESOURCE_NOT_FOUND',
  RESOURCE_ALREADY_EXISTS = 'RESOURCE_ALREADY_EXISTS',
  RESOURCE_LOCKED = 'RESOURCE_LOCKED',
  
  // ビジネスロジック
  BUSINESS_INVALID_STATE = 'BUSINESS_INVALID_STATE',
  BUSINESS_LIMIT_EXCEEDED = 'BUSINESS_LIMIT_EXCEEDED',
  
  // システム
  SYSTEM_ERROR = 'SYSTEM_ERROR',
  SYSTEM_MAINTENANCE = 'SYSTEM_MAINTENANCE',
  SYSTEM_RATE_LIMIT = 'SYSTEM_RATE_LIMIT'
}
```

## 6. レート制限

### 6.1 制限ポリシー

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1679308800
X-RateLimit-Policy: sliding-window
```

### 6.2 制限レベル

| プラン | リクエスト/時 | バースト | 同時接続数 |
|--------|---------------|----------|------------|
| Free | 100 | 10 | 5 |
| Basic | 1,000 | 50 | 20 |
| Pro | 10,000 | 200 | 100 |
| Enterprise | カスタム | カスタム | カスタム |

## 7. WebSocket API

### 7.1 接続エンドポイント

```
wss://api.example.com/ws/v1/realtime
```

### 7.2 メッセージ形式

```typescript
interface WebSocketMessage {
  type: 'subscribe' | 'unsubscribe' | 'event' | 'ping' | 'pong';
  channel?: string;
  event?: string;
  data?: any;
  id?: string;
  timestamp: string;
}
```

### 7.3 イベントタイプ

```typescript
enum RealtimeEvent {
  // プロジェクト
  PROJECT_UPDATED = 'project.updated',
  PROJECT_DELETED = 'project.deleted',
  
  // コラボレーション
  USER_JOINED = 'collaboration.user_joined',
  USER_LEFT = 'collaboration.user_left',
  CURSOR_MOVED = 'collaboration.cursor_moved',
  CONTENT_CHANGED = 'collaboration.content_changed',
  
  // 通知
  NOTIFICATION_NEW = 'notification.new',
  COMMENT_ADDED = 'comment.added',
  REVIEW_STATUS_CHANGED = 'review.status_changed'
}
```

## 8. バッチ操作

### 8.1 バルク作成

```json
POST /api/v1/projects/bulk
{
  "operations": [
    {
      "method": "create",
      "data": {
        "style_number": "ST-001",
        "style_name": "Summer Shirt"
      }
    },
    {
      "method": "create",
      "data": {
        "style_number": "ST-002",
        "style_name": "Winter Coat"
      }
    }
  ]
}
```

### 8.2 バルク更新

```json
PATCH /api/v1/projects/bulk
{
  "ids": ["proj_123", "proj_456"],
  "update": {
    "status": "approved",
    "approved_by": "user_789",
    "approved_at": "2024-03-20T10:30:00Z"
  }
}
```

## 9. ファイルアップロード

### 9.1 マルチパートアップロード

```
POST /api/v1/files/upload
Content-Type: multipart/form-data

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="design.svg"
Content-Type: image/svg+xml

<svg>...</svg>
------WebKitFormBoundary
Content-Disposition: form-data; name="metadata"

{"type": "svg_part", "category": "button"}
------WebKitFormBoundary--
```

### 9.2 署名付きURL

```json
POST /api/v1/files/presigned-url
{
  "filename": "techpack.pdf",
  "content_type": "application/pdf",
  "size": 1048576
}

// レスポンス
{
  "upload_url": "https://storage.example.com/upload?signature=...",
  "file_id": "file_123",
  "expires_at": "2024-03-20T11:30:00Z"
}
```

## 10. 非同期処理

### 10.1 ジョブの作成

```json
POST /api/v1/jobs
{
  "type": "generate_techpack",
  "priority": "high",
  "data": {
    "project_id": "proj_123",
    "format": "pdf",
    "include_attachments": true
  }
}

// レスポンス
{
  "job_id": "job_456",
  "status": "queued",
  "created_at": "2024-03-20T10:30:00Z",
  "estimated_completion": "2024-03-20T10:35:00Z"
}
```

### 10.2 ジョブのステータス確認

```json
GET /api/v1/jobs/job_456

{
  "job_id": "job_456",
  "status": "processing",
  "progress": 65,
  "current_step": "Generating PDF",
  "created_at": "2024-03-20T10:30:00Z",
  "started_at": "2024-03-20T10:30:30Z"
}
```

## 11. API監視・ロギング

### 11.1 ヘルスチェック

```json
GET /api/v1/health

{
  "status": "healthy",
  "timestamp": "2024-03-20T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy",
    "ai_service": "degraded"
  },
  "version": "1.2.3",
  "uptime": 864000
}
```

### 11.2 メトリクス

```json
GET /api/v1/metrics

{
  "requests": {
    "total": 1000000,
    "rate_per_minute": 1500,
    "error_rate": 0.001
  },
  "latency": {
    "p50": 45,
    "p95": 120,
    "p99": 250
  },
  "resources": {
    "cpu_usage": 65,
    "memory_usage": 78,
    "disk_usage": 45
  }
}
```

## 12. SDK・クライアントライブラリ

### 12.1 対応言語

- JavaScript/TypeScript (公式)
- Python
- Java
- Go
- Ruby

### 12.2 使用例（TypeScript）

```typescript
import { TechPackAPI } from '@techpack/api-client';

const api = new TechPackAPI({
  apiKey: process.env.TECHPACK_API_KEY,
  baseURL: 'https://api.example.com',
  timeout: 30000
});

// プロジェクト作成
const project = await api.projects.create({
  style_number: 'ST-001',
  style_name: 'Summer Collection Shirt',
  season: '2024SS'
});

// セクション更新
await api.sections.update(project.id, 'cover_page', {
  brand: 'Example Brand',
  designer: 'John Doe'
});

// リアルタイム接続
api.realtime.on('project.updated', (event) => {
  console.log('Project updated:', event.data);
});
```