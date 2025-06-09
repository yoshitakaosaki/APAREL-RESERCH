# リソース管理API仕様

## 1. 概要

リソース管理APIは、素材ライブラリ、SVGパーツ、用語集、テンプレートなどの共有リソースを管理するためのエンドポイントを提供します。

## 2. SVGパーツAPI

### 2.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/svg-parts` | SVGパーツ一覧取得 |
| GET | `/api/v1/svg-parts/{id}` | SVGパーツ詳細取得 |
| POST | `/api/v1/svg-parts` | SVGパーツ新規作成 |
| PUT | `/api/v1/svg-parts/{id}` | SVGパーツ更新 |
| DELETE | `/api/v1/svg-parts/{id}` | SVGパーツ削除 |
| POST | `/api/v1/svg-parts/generate` | AIによるSVG生成 |
| GET | `/api/v1/svg-parts/categories` | カテゴリー一覧 |

### 2.2 SVGパーツ一覧取得

```http
GET /api/v1/svg-parts
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| category | string | No | カテゴリーフィルター |
| type | string | No | パーツタイプ |
| search | string | No | 検索キーワード |
| tags | string[] | No | タグフィルター |
| status | string | No | ステータス |
| page | integer | No | ページ番号 |
| per_page | integer | No | 1ページあたりの件数 |

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "svg_btn_001",
      "code": "BTN-ROUND-001",
      "name": {
        "ja": "丸ボタン",
        "en": "Round Button"
      },
      "type": "button",
      "status": "approved",
      "version": 2,
      "svgData": {
        "source": "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 100 100\">...</svg>",
        "viewBox": {
          "x": 0,
          "y": 0,
          "width": 100,
          "height": 100
        },
        "width": 20,
        "height": 20
      },
      "parameters": [
        {
          "id": "diameter",
          "name": {
            "ja": "直径",
            "en": "Diameter"
          },
          "type": "number",
          "constraints": {
            "min": 10,
            "max": 30,
            "default": 20,
            "unit": "mm"
          }
        }
      ],
      "categories": ["button", "basic"],
      "tags": ["round", "2-hole", "standard"],
      "usage": {
        "count": 156,
        "lastUsed": "2024-03-20T10:00:00Z",
        "rating": 4.5
      },
      "thumbnail": "https://storage.example.com/thumbnails/svg_btn_001.png",
      "createdAt": "2024-01-15T08:00:00Z",
      "updatedAt": "2024-03-10T14:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 245,
    "total_pages": 13
  }
}
```

### 2.3 AIによるSVG生成

```http
POST /api/v1/svg-parts/generate
```

#### リクエストボディ

```json
{
  "type": "image_to_svg",
  "input": {
    "image_url": "https://storage.example.com/uploads/button_photo.jpg",
    "part_type": "button",
    "requirements": {
      "remove_background": true,
      "optimize_paths": true,
      "generate_parameters": true,
      "target_complexity": "medium"
    }
  },
  "metadata": {
    "name": {
      "ja": "シェルボタン",
      "en": "Shell Button"
    },
    "categories": ["button", "natural"],
    "tags": ["shell", "4-hole", "decorative"]
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "task_id": "task_svg_gen_123",
    "status": "processing",
    "estimated_time": 30,
    "progress_url": "/api/v1/tasks/task_svg_gen_123"
  }
}
```

## 3. 用語集API

### 3.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/terms` | 用語一覧取得 |
| GET | `/api/v1/terms/{id}` | 用語詳細取得 |
| POST | `/api/v1/terms` | 用語新規作成 |
| PUT | `/api/v1/terms/{id}` | 用語更新 |
| DELETE | `/api/v1/terms/{id}` | 用語削除 |
| POST | `/api/v1/terms/collect` | AI用語収集タスク |
| GET | `/api/v1/terms/search` | 用語検索 |

### 3.2 用語検索

```http
GET /api/v1/terms/search
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| q | string | Yes | 検索クエリ |
| lang | string | No | 言語コード |
| category | string[] | No | カテゴリーフィルター |
| fuzzy | boolean | No | あいまい検索 |
| limit | integer | No | 結果数制限 |

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "term_col_001",
      "code": "COL-001",
      "translations": {
        "ja": {
          "name": "襟",
          "reading": "えり",
          "description": "衣服の首周りの部分",
          "usageExamples": [
            "シャツの襟を立てる",
            "丸襟のデザイン"
          ]
        },
        "en": {
          "name": "collar",
          "description": "The part of a garment that fastens around or frames the neck",
          "usageExamples": [
            "Button-down collar",
            "Spread collar design"
          ]
        }
      },
      "svgParts": [
        {
          "partId": "svg_col_001",
          "isPrimary": true,
          "usageContext": "Basic collar shape"
        }
      ],
      "categories": ["garment_parts", "upper_body"],
      "tags": ["collar", "neckline", "basic"],
      "relatedTerms": ["term_nck_001", "term_btn_001"],
      "score": 0.95
    }
  ],
  "meta": {
    "query": "襟",
    "total_results": 15,
    "search_time": 0.023
  }
}
```

### 3.3 AI用語収集タスク

```http
POST /api/v1/terms/collect
```

#### リクエストボディ

```json
{
  "source": {
    "type": "document",
    "url": "https://example.com/fashion-glossary.pdf"
  },
  "configuration": {
    "languages": ["ja", "en"],
    "categories": ["garment_parts"],
    "confidence_threshold": 0.7,
    "auto_approve": false
  }
}
```

## 4. 素材ライブラリAPI

### 4.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/materials` | 素材一覧取得 |
| GET | `/api/v1/materials/{id}` | 素材詳細取得 |
| POST | `/api/v1/materials` | 素材新規登録 |
| PUT | `/api/v1/materials/{id}` | 素材情報更新 |
| DELETE | `/api/v1/materials/{id}` | 素材削除 |
| GET | `/api/v1/materials/{id}/suppliers` | サプライヤー一覧 |
| GET | `/api/v1/materials/{id}/pricing` | 価格情報 |

### 4.2 素材一覧取得

```http
GET /api/v1/materials
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| type | string | No | 素材タイプ（fabric, trim, accessory） |
| category | string | No | カテゴリー |
| composition | string | No | 組成フィルター |
| supplier | string | No | サプライヤーID |
| min_moq | integer | No | 最小ロット以下 |
| max_price | number | No | 最大価格 |

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "mat_fab_001",
      "code": "FB-COT-100",
      "name": "Premium Cotton Oxford",
      "type": "fabric",
      "category": "woven",
      "composition": "100% Cotton",
      "specifications": {
        "weight": "140gsm",
        "width": "148cm",
        "weave": "Oxford",
        "thread_count": "40s",
        "finish": "Easy Care"
      },
      "colors": [
        {
          "code": "WHT",
          "name": "White",
          "pantone": "11-0601 TPX",
          "stock_status": "in_stock"
        },
        {
          "code": "NVY",
          "name": "Navy",
          "pantone": "19-3933 TPX",
          "stock_status": "in_stock"
        }
      ],
      "suppliers": [
        {
          "id": "sup_001",
          "name": "ABC Textiles",
          "lead_time": 14,
          "moq": 500,
          "pricing": {
            "currency": "USD",
            "unit": "meter",
            "tiers": [
              {"min_qty": 500, "price": 8.50},
              {"min_qty": 1000, "price": 7.80},
              {"min_qty": 5000, "price": 7.20}
            ]
          }
        }
      ],
      "certifications": [
        "OEKO-TEX Standard 100",
        "GOTS Certified"
      ],
      "images": [
        "https://storage.example.com/materials/mat_fab_001_1.jpg",
        "https://storage.example.com/materials/mat_fab_001_2.jpg"
      ],
      "usage_count": 234,
      "rating": 4.7,
      "tags": ["cotton", "oxford", "shirt", "premium"],
      "created_at": "2024-01-20T10:00:00Z",
      "updated_at": "2024-03-15T14:00:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 456,
    "total_pages": 23
  }
}
```

## 5. テンプレートAPI

### 5.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/templates` | テンプレート一覧 |
| GET | `/api/v1/templates/{id}` | テンプレート詳細 |
| POST | `/api/v1/templates` | テンプレート作成 |
| PUT | `/api/v1/templates/{id}` | テンプレート更新 |
| DELETE | `/api/v1/templates/{id}` | テンプレート削除 |
| POST | `/api/v1/templates/{id}/clone` | テンプレート複製 |

### 5.2 テンプレート一覧取得

```http
GET /api/v1/templates
```

#### クエリパラメータ

| パラメータ | 型 | 必須 | 説明 |
|------------|------|------|------|
| category | string | No | カテゴリー |
| product_type | string | No | 製品タイプ |
| visibility | string | No | public, private, organization |
| search | string | No | 検索キーワード |

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "tmpl_shirt_basic",
      "name": "Basic Shirt Template",
      "description": "Standard template for shirts with all essential sections",
      "category": "tops",
      "product_type": "shirt",
      "visibility": "public",
      "version": "2.0",
      "sections": [
        {
          "type": "cover_page",
          "required": true,
          "prefilled": false
        },
        {
          "type": "technical_flat_front",
          "required": true,
          "prefilled": true,
          "default_data": {
            "svg_template": "svg_shirt_front_001"
          }
        }
        // ... other sections
      ],
      "default_settings": {
        "measurement_unit": "cm",
        "size_range": ["XS", "S", "M", "L", "XL"],
        "grading_rules": "standard_mens"
      },
      "tags": ["shirt", "basic", "mens", "standard"],
      "usage_count": 1523,
      "rating": 4.8,
      "created_by": {
        "id": "user_admin",
        "name": "System Administrator"
      },
      "is_system": true,
      "created_at": "2023-06-01T00:00:00Z",
      "updated_at": "2024-02-15T10:30:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 78,
    "total_pages": 4
  }
}
```

## 6. ファイル管理API

### 6.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| POST | `/api/v1/files/upload` | ファイルアップロード |
| GET | `/api/v1/files/{id}` | ファイル情報取得 |
| DELETE | `/api/v1/files/{id}` | ファイル削除 |
| POST | `/api/v1/files/presigned-url` | 署名付きURL取得 |
| POST | `/api/v1/files/process` | ファイル処理 |

### 6.2 ファイルアップロード

```http
POST /api/v1/files/upload
```

#### マルチパートフォームデータ

```
Content-Type: multipart/form-data

------WebKitFormBoundary
Content-Disposition: form-data; name="file"; filename="button_design.svg"
Content-Type: image/svg+xml

<svg>...</svg>
------WebKitFormBoundary
Content-Disposition: form-data; name="type"

svg_part
------WebKitFormBoundary
Content-Disposition: form-data; name="metadata"

{"category": "button", "tags": ["decorative", "metal"]}
------WebKitFormBoundary--
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "file_id": "file_xyz789",
    "filename": "button_design.svg",
    "size": 4567,
    "mime_type": "image/svg+xml",
    "url": "https://storage.example.com/files/file_xyz789.svg",
    "thumbnail_url": "https://storage.example.com/thumbnails/file_xyz789.png",
    "uploaded_at": "2024-03-20T16:45:00Z"
  }
}
```

## 7. エラーハンドリング

### 7.1 共通エラーコード

| コード | 説明 |
|------|------|
| RESOURCE_NOT_FOUND | リソースが存在しない |
| RESOURCE_ALREADY_EXISTS | リソースが既に存在する |
| INVALID_FILE_TYPE | 無効なファイルタイプ |
| FILE_TOO_LARGE | ファイルサイズが大きすぎる |
| PROCESSING_ERROR | 処理エラー |
| QUOTA_EXCEEDED | 割り当て超過 |

### 7.2 エラーレスポンス例

```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE_TYPE",
    "message": "サポートされていないファイル形式です",
    "details": {
      "allowed_types": ["image/jpeg", "image/png", "image/svg+xml"],
      "provided_type": "application/pdf"
    }
  },
  "meta": {
    "timestamp": "2024-03-20T17:00:00Z",
    "request_id": "req_abc123"
  }
}
```