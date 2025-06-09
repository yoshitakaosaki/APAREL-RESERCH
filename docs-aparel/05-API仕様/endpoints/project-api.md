# プロジェクトAPI仕様

## 1. 概要

プロジェクトAPIは、テックパックプロジェクトの作成、管理、更新、削除を行うためのエンドポイントを提供します。

## 2. エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/projects` | プロジェクト一覧取得 |
| GET | `/api/v1/projects/{id}` | プロジェクト詳細取得 |
| POST | `/api/v1/projects` | プロジェクト新規作成 |
| PUT | `/api/v1/projects/{id}` | プロジェクト更新 |
| PATCH | `/api/v1/projects/{id}` | プロジェクト部分更新 |
| DELETE | `/api/v1/projects/{id}` | プロジェクト削除 |
| POST | `/api/v1/projects/{id}/duplicate` | プロジェクト複製 |
| POST | `/api/v1/projects/{id}/archive` | プロジェクトアーカイブ |
| POST | `/api/v1/projects/{id}/restore` | プロジェクト復元 |

## 3. 詳細仕様

### 3.1 プロジェクト一覧取得

```http
GET /api/v1/projects
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 | 例 |
|------------|------|------|------|------|
| page | integer | No | ページ番号 | 1 |
| per_page | integer | No | 1ページあたりの件数 | 20 |
| status | string | No | ステータスフィルター | active |
| search | string | No | 検索キーワード | summer |
| sort | string | No | ソート順 | -created_at |
| created_after | datetime | No | 作成日時（以降） | 2024-01-01 |
| created_before | datetime | No | 作成日時（以前） | 2024-12-31 |
| tags | string[] | No | タグフィルター | ["SS24", "shirt"] |

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "proj_abc123",
      "style_number": "ST-001",
      "style_name": "Summer Collection Shirt",
      "brand": "Example Brand",
      "season": "2024SS",
      "status": "in_progress",
      "version": "1.2",
      "tags": ["shirt", "summer", "casual"],
      "created_by": {
        "id": "user_123",
        "name": "John Doe",
        "email": "john@example.com"
      },
      "created_at": "2024-03-15T10:30:00Z",
      "updated_at": "2024-03-20T15:45:00Z",
      "thumbnail_url": "https://storage.example.com/thumbnails/proj_abc123.jpg",
      "sections_completed": 15,
      "sections_total": 20,
      "completion_rate": 75
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 45,
    "total_pages": 3,
    "has_next": true,
    "has_prev": false
  },
  "meta": {
    "timestamp": "2024-03-20T10:30:00Z"
  }
}
```

### 3.2 プロジェクト詳細取得

```http
GET /api/v1/projects/{id}
```

#### パスパラメータ

| パラメータ | 型 | 説明 |
|------------|------|------|
| id | string | プロジェクトID |

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| include | string[] | No | 追加情報を含む (例: sections, history, collaborators) |

#### レスポンス

```json
{
  "success": true,
  "data": {
    "id": "proj_abc123",
    "style_number": "ST-001",
    "style_name": "Summer Collection Shirt",
    "style_description": "A casual summer shirt with modern design elements",
    "brand": "Example Brand",
    "label": "Main Line",
    "season": "2024SS",
    "delivery": "2024-05-01",
    "status": "in_progress",
    "version": "1.2",
    "target_gender": "unisex",
    "product_type": "shirt",
    "fit_type": "regular",
    "size_range": ["XS", "S", "M", "L", "XL", "XXL"],
    "fabric_type": "100% Cotton",
    "colorways": [
      {
        "id": "color_001",
        "name": "White",
        "code": "WHT",
        "pantone": "11-0601 TPX"
      },
      {
        "id": "color_002",
        "name": "Navy",
        "code": "NVY",
        "pantone": "19-3933 TPX"
      }
    ],
    "tags": ["shirt", "summer", "casual"],
    "sections": [
      {
        "id": "sect_001",
        "type": "cover_page",
        "status": "completed",
        "last_updated": "2024-03-20T10:00:00Z"
      },
      {
        "id": "sect_002",
        "type": "technical_flat_front",
        "status": "in_progress",
        "last_updated": "2024-03-20T15:30:00Z"
      }
    ],
    "collaborators": [
      {
        "id": "user_123",
        "name": "John Doe",
        "role": "designer",
        "permissions": ["edit", "comment"]
      },
      {
        "id": "user_456",
        "name": "Jane Smith",
        "role": "technical_designer",
        "permissions": ["edit", "comment", "approve"]
      }
    ],
    "created_by": {
      "id": "user_123",
      "name": "John Doe",
      "email": "john@example.com"
    },
    "created_at": "2024-03-15T10:30:00Z",
    "updated_at": "2024-03-20T15:45:00Z",
    "settings": {
      "measurement_unit": "cm",
      "currency": "USD",
      "language": "en",
      "auto_save": true,
      "version_control": true
    }
  },
  "meta": {
    "timestamp": "2024-03-20T16:00:00Z"
  }
}
```

### 3.3 プロジェクト新規作成

```http
POST /api/v1/projects
```

#### リクエストボディ

```json
{
  "style_number": "ST-002",
  "style_name": "Winter Collection Coat",
  "style_description": "A warm winter coat with premium materials",
  "brand": "Example Brand",
  "label": "Premium Line",
  "season": "2024FW",
  "delivery": "2024-09-01",
  "target_gender": "women",
  "product_type": "coat",
  "fit_type": "oversized",
  "size_range": ["XS", "S", "M", "L", "XL"],
  "fabric_type": "Wool Blend",
  "colorways": [
    {
      "name": "Black",
      "code": "BLK",
      "pantone": "19-0303 TPX"
    }
  ],
  "tags": ["coat", "winter", "premium"],
  "template_id": "tmpl_winter_coat",
  "settings": {
    "measurement_unit": "cm",
    "currency": "USD",
    "language": "en"
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "id": "proj_def456",
    "style_number": "ST-002",
    "style_name": "Winter Collection Coat",
    "status": "draft",
    "version": "1.0",
    "created_at": "2024-03-20T16:30:00Z",
    "sections": [
      // テンプレートから生成されたセクション
    ]
  },
  "meta": {
    "timestamp": "2024-03-20T16:30:00Z"
  }
}
```

### 3.4 プロジェクト更新

```http
PUT /api/v1/projects/{id}
```

#### リクエストボディ

全てのフィールドを含む完全なプロジェクトデータ

### 3.5 プロジェクト部分更新

```http
PATCH /api/v1/projects/{id}
```

#### リクエストボディ

```json
{
  "status": "review",
  "tags": ["coat", "winter", "premium", "bestseller"]
}
```

### 3.6 プロジェクト削除

```http
DELETE /api/v1/projects/{id}
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| permanent | boolean | No | 完全削除（デフォルト: false） |

#### レスポンス

```json
{
  "success": true,
  "message": "Project archived successfully",
  "meta": {
    "timestamp": "2024-03-20T17:00:00Z"
  }
}
```

### 3.7 プロジェクト複製

```http
POST /api/v1/projects/{id}/duplicate
```

#### リクエストボディ

```json
{
  "style_number": "ST-003",
  "style_name": "Winter Collection Coat - Variation",
  "copy_sections": true,
  "copy_collaborators": false,
  "copy_attachments": true
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "id": "proj_ghi789",
    "style_number": "ST-003",
    "style_name": "Winter Collection Coat - Variation",
    "status": "draft",
    "version": "1.0",
    "source_project_id": "proj_def456",
    "created_at": "2024-03-20T17:30:00Z"
  },
  "meta": {
    "timestamp": "2024-03-20T17:30:00Z"
  }
}
```

## 4. プロジェクトデータモデル

### 4.1 Project

```typescript
interface Project {
  id: string;
  style_number: string;
  style_name: string;
  style_description?: string;
  brand: string;
  label?: string;
  season: string;
  delivery?: string;
  status: ProjectStatus;
  version: string;
  
  // 製品情報
  target_gender: 'men' | 'women' | 'unisex' | 'kids';
  product_type: string;
  fit_type: string;
  size_range: string[];
  fabric_type: string;
  colorways: Colorway[];
  
  // メタデータ
  tags: string[];
  sections: Section[];
  collaborators: Collaborator[];
  attachments?: Attachment[];
  
  // 監査情報
  created_by: User;
  created_at: string;
  updated_at: string;
  deleted_at?: string;
  
  // 設定
  settings: ProjectSettings;
}
```

### 4.2 ProjectStatus

```typescript
enum ProjectStatus {
  DRAFT = 'draft',
  IN_PROGRESS = 'in_progress',
  REVIEW = 'review',
  APPROVED = 'approved',
  PRODUCTION = 'production',
  COMPLETED = 'completed',
  ARCHIVED = 'archived'
}
```

### 4.3 Colorway

```typescript
interface Colorway {
  id: string;
  name: string;
  code: string;
  pantone?: string;
  hex?: string;
  is_primary?: boolean;
}
```

### 4.4 Section

```typescript
interface Section {
  id: string;
  type: SectionType;
  status: 'empty' | 'in_progress' | 'completed' | 'approved';
  data?: any;
  last_updated: string;
  last_updated_by?: string;
  locked?: boolean;
  locked_by?: string;
}
```

## 5. エラーレスポンス

### 5.1 バリデーションエラー (422)

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "入力データが無効です",
    "details": [
      {
        "field": "style_number",
        "message": "スタイル番号は既に使用されています"
      },
      {
        "field": "delivery",
        "message": "納期は将来の日付である必要があります"
      }
    ]
  },
  "meta": {
    "timestamp": "2024-03-20T18:00:00Z",
    "request_id": "req_xyz789"
  }
}
```

### 5.2 権限エラー (403)

```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_PERMISSIONS",
    "message": "この操作を実行する権限がありません",
    "required_permission": "project:delete"
  },
  "meta": {
    "timestamp": "2024-03-20T18:00:00Z",
    "request_id": "req_xyz790"
  }
}
```

### 5.3 リソースロックエラー (409)

```json
{
  "success": false,
  "error": {
    "code": "RESOURCE_LOCKED",
    "message": "プロジェクトは他のユーザーによって編集中です",
    "locked_by": {
      "id": "user_456",
      "name": "Jane Smith"
    },
    "locked_until": "2024-03-20T18:30:00Z"
  },
  "meta": {
    "timestamp": "2024-03-20T18:00:00Z",
    "request_id": "req_xyz791"
  }
}
```

## 6. Webhookイベント

### 6.1 イベントタイプ

```typescript
enum ProjectWebhookEvent {
  PROJECT_CREATED = 'project.created',
  PROJECT_UPDATED = 'project.updated',
  PROJECT_DELETED = 'project.deleted',
  PROJECT_STATUS_CHANGED = 'project.status_changed',
  PROJECT_APPROVED = 'project.approved',
  PROJECT_REJECTED = 'project.rejected',
  PROJECT_EXPORTED = 'project.exported'
}
```

### 6.2 Webhookペイロード

```json
{
  "event": "project.status_changed",
  "timestamp": "2024-03-20T18:30:00Z",
  "data": {
    "project_id": "proj_abc123",
    "old_status": "in_progress",
    "new_status": "review",
    "changed_by": {
      "id": "user_123",
      "name": "John Doe"
    }
  },
  "metadata": {
    "webhook_id": "webhook_123",
    "delivery_attempt": 1
  }
}
```