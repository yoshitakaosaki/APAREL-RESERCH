# AI/MCP統合API仕様

## 1. 概要

AI/MCP統合APIは、AIエージェントによる自動処理と外部システム連携（MCP: Model Context Protocol）を管理するためのエンドポイントを提供します。

## 2. AIエージェントAPI

### 2.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/ai/agents` | エージェント一覧 |
| GET | `/api/v1/ai/agents/{id}` | エージェント詳細 |
| POST | `/api/v1/ai/tasks` | AIタスク作成 |
| GET | `/api/v1/ai/tasks/{id}` | タスク状態確認 |
| DELETE | `/api/v1/ai/tasks/{id}` | タスクキャンセル |
| GET | `/api/v1/ai/models` | 利用可能モデル |
| POST | `/api/v1/ai/analyze` | コンテンツ分析 |

### 2.2 AIタスク作成

```http
POST /api/v1/ai/tasks
```

#### リクエストボディ

```json
{
  "agent_type": "term_collector",
  "priority": "high",
  "configuration": {
    "source": {
      "type": "document",
      "url": "https://storage.example.com/docs/fashion_guide.pdf"
    },
    "extraction": {
      "languages": ["ja", "en"],
      "confidence_threshold": 0.75,
      "categories": ["garment_parts", "materials"],
      "extract_images": true
    },
    "processing": {
      "auto_translate": true,
      "link_svg_parts": true,
      "validate_with_mcp": true
    }
  },
  "callback_url": "https://api.example.com/webhooks/ai-task-complete"
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "task_id": "task_ai_001",
    "agent_type": "term_collector",
    "status": "queued",
    "priority": "high",
    "created_at": "2024-03-20T10:00:00Z",
    "estimated_completion": "2024-03-20T10:15:00Z",
    "progress": {
      "current_step": "initialization",
      "percentage": 0,
      "steps_completed": 0,
      "total_steps": 5
    }
  }
}
```

### 2.3 タスク状態確認

```http
GET /api/v1/ai/tasks/{id}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "task_id": "task_ai_001",
    "agent_type": "term_collector",
    "status": "processing",
    "priority": "high",
    "created_at": "2024-03-20T10:00:00Z",
    "started_at": "2024-03-20T10:02:00Z",
    "progress": {
      "current_step": "extraction",
      "percentage": 45,
      "steps_completed": 2,
      "total_steps": 5,
      "details": {
        "pages_processed": 23,
        "total_pages": 50,
        "terms_extracted": 156,
        "confidence_avg": 0.82
      }
    },
    "partial_results": {
      "extracted_terms": 156,
      "validated_terms": 102,
      "new_terms": 34,
      "updated_terms": 68
    },
    "logs": [
      {
        "timestamp": "2024-03-20T10:02:00Z",
        "level": "info",
        "message": "Task started"
      },
      {
        "timestamp": "2024-03-20T10:02:30Z",
        "level": "info",
        "message": "Document loaded successfully"
      }
    ]
  }
}
```

### 2.4 コンテンツ分析

```http
POST /api/v1/ai/analyze
```

#### リクエストボディ

```json
{
  "content_type": "image",
  "content_url": "https://storage.example.com/images/garment_photo.jpg",
  "analysis_types": [
    "garment_detection",
    "color_extraction",
    "style_classification",
    "detail_identification"
  ],
  "options": {
    "return_confidence": true,
    "extract_measurements": true,
    "identify_materials": true
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_001",
    "results": {
      "garment_detection": {
        "type": "shirt",
        "confidence": 0.95,
        "style": "casual",
        "gender": "unisex"
      },
      "color_extraction": {
        "primary": {
          "hex": "#FFFFFF",
          "name": "White",
          "pantone_closest": "11-0601 TPX"
        },
        "secondary": [
          {
            "hex": "#000080",
            "name": "Navy",
            "pantone_closest": "19-3933 TPX"
          }
        ]
      },
      "style_classification": {
        "categories": ["casual", "business_casual"],
        "fit": "regular",
        "season": "all_season"
      },
      "detail_identification": {
        "collar": {
          "type": "button_down",
          "confidence": 0.88
        },
        "buttons": {
          "count": 7,
          "type": "plastic",
          "color": "white"
        },
        "pocket": {
          "present": true,
          "type": "patch",
          "position": "left_chest"
        }
      }
    },
    "metadata": {
      "processing_time": 2.34,
      "model_version": "fashion-detect-v2.1"
    }
  }
}
```

## 3. MCP連携API

### 3.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/mcp/providers` | MCPプロバイダー一覧 |
| POST | `/api/v1/mcp/providers` | プロバイダー登録 |
| PUT | `/api/v1/mcp/providers/{id}` | プロバイダー更新 |
| DELETE | `/api/v1/mcp/providers/{id}` | プロバイダー削除 |
| POST | `/api/v1/mcp/search` | 外部検索 |
| POST | `/api/v1/mcp/sync` | データ同期 |
| GET | `/api/v1/mcp/status` | 連携状態 |

### 3.2 MCPプロバイダー登録

```http
POST /api/v1/mcp/providers
```

#### リクエストボディ

```json
{
  "name": "Fashion Terms Database",
  "type": "fashion_db",
  "endpoint": "https://api.fashion-terms.com/v2",
  "capabilities": {
    "search": true,
    "retrieve": true,
    "translate": true,
    "validate": false,
    "generate": false
  },
  "authentication": {
    "type": "api_key",
    "credentials": {
      "api_key": "encrypted_key_here"
    }
  },
  "mapping": {
    "request_format": "fashion_db_v2",
    "response_format": "fashion_db_v2",
    "field_mappings": {
      "term_name": "$.name",
      "term_description": "$.definition",
      "translations": "$.translations[*]"
    }
  },
  "rate_limit": {
    "requests_per_second": 10,
    "daily_quota": 10000,
    "burst_limit": 50
  },
  "is_active": true
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "provider_id": "mcp_001",
    "name": "Fashion Terms Database",
    "type": "fashion_db",
    "status": "active",
    "health_check_url": "/api/v1/mcp/providers/mcp_001/health",
    "created_at": "2024-03-20T11:00:00Z"
  }
}
```

### 3.3 外部検索

```http
POST /api/v1/mcp/search
```

#### リクエストボディ

```json
{
  "query": "button types fashion",
  "providers": ["mcp_001", "mcp_002"],
  "options": {
    "language": "en",
    "limit": 50,
    "include_images": true,
    "merge_results": true
  },
  "filters": {
    "categories": ["accessories", "fasteners"],
    "confidence_min": 0.7
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "query": "button types fashion",
    "total_results": 42,
    "results": [
      {
        "id": "ext_term_001",
        "source": "mcp_001",
        "confidence": 0.92,
        "data": {
          "name": "Shank Button",
          "description": "A button with a hollow protrusion on the back",
          "translations": {
            "ja": "シャンクボタン",
            "zh": "脚钉"
          },
          "categories": ["button", "fastener"],
          "images": [
            "https://external.example.com/images/shank_button.jpg"
          ],
          "related_terms": ["button", "fastener", "attachment"]
        }
      }
    ],
    "providers_status": {
      "mcp_001": {
        "status": "success",
        "results_count": 28,
        "response_time": 234
      },
      "mcp_002": {
        "status": "success",
        "results_count": 14,
        "response_time": 456
      }
    }
  }
}
```

### 3.4 データ同期

```http
POST /api/v1/mcp/sync
```

#### リクエストボディ

```json
{
  "sync_type": "incremental",
  "providers": ["mcp_001"],
  "data_types": ["terms", "materials"],
  "options": {
    "since": "2024-03-15T00:00:00Z",
    "conflict_resolution": "prefer_external",
    "validate_before_import": true,
    "batch_size": 100
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "sync_id": "sync_001",
    "status": "in_progress",
    "started_at": "2024-03-20T12:00:00Z",
    "progress": {
      "terms": {
        "total": 500,
        "processed": 0,
        "imported": 0,
        "updated": 0,
        "errors": 0
      },
      "materials": {
        "total": 200,
        "processed": 0,
        "imported": 0,
        "updated": 0,
        "errors": 0
      }
    },
    "monitoring_url": "/api/v1/mcp/sync/sync_001"
  }
}
```

## 4. AIモデル管理API

### 4.1 エンドポイント一覧

| メソッド | エンドポイント | 説明 |
|----------|--------------|------|
| GET | `/api/v1/ai/models` | モデル一覧 |
| GET | `/api/v1/ai/models/{id}` | モデル詳細 |
| POST | `/api/v1/ai/models/{id}/deploy` | モデルデプロイ |
| POST | `/api/v1/ai/models/{id}/test` | モデルテスト |
| GET | `/api/v1/ai/models/{id}/metrics` | モデルメトリクス |

### 4.2 モデル一覧取得

```http
GET /api/v1/ai/models
```

#### レスポンス

```json
{
  "success": true,
  "data": [
    {
      "id": "model_001",
      "name": "YOLOv8-Fashion",
      "type": "object_detection",
      "version": "1.2.0",
      "purpose": "Apparel parts detection",
      "status": "deployed",
      "performance": {
        "accuracy": 0.92,
        "latency": 45,
        "throughput": 22
      },
      "deployment": {
        "endpoint": "https://ml.example.com/models/yolo-fashion",
        "runtime": "onnx",
        "hardware": "gpu",
        "instances": 3
      },
      "training": {
        "dataset": "apparel_parts_10k",
        "epochs": 100,
        "last_trained": "2024-03-01T00:00:00Z"
      },
      "classes": [
        "button", "pocket", "collar", "zipper", 
        "stitch_line", "hem", "cuff", "placket"
      ]
    },
    {
      "id": "model_002",
      "name": "DeepSVG-Custom",
      "type": "vectorization",
      "version": "2.0.1",
      "purpose": "Neural vectorization",
      "status": "deployed",
      "performance": {
        "quality_score": 0.88,
        "latency": 2300,
        "success_rate": 0.95
      }
    }
  ]
}
```

## 5. オーケストレーションAPI

### 5.1 ワークフロー実行

```http
POST /api/v1/ai/workflows
```

#### リクエストボディ

```json
{
  "workflow_type": "complete_term_extraction",
  "input": {
    "documents": [
      "https://storage.example.com/docs/catalog_2024.pdf"
    ],
    "images": [
      "https://storage.example.com/images/collection_photos.zip"
    ]
  },
  "configuration": {
    "agents": [
      {
        "type": "document_parser",
        "config": {
          "extract_images": true,
          "ocr_enabled": true
        }
      },
      {
        "type": "term_collector",
        "config": {
          "languages": ["ja", "en"],
          "confidence_threshold": 0.7
        }
      },
      {
        "type": "svg_generator",
        "config": {
          "generate_for_new_terms": true,
          "quality": "high"
        }
      },
      {
        "type": "translator",
        "config": {
          "target_languages": ["zh", "ko"]
        }
      }
    ],
    "orchestration": {
      "parallel_execution": true,
      "error_handling": "continue_on_error",
      "timeout": 3600
    }
  }
}
```

#### レスポンス

```json
{
  "success": true,
  "data": {
    "workflow_id": "wf_001",
    "status": "running",
    "created_at": "2024-03-20T13:00:00Z",
    "agents": [
      {
        "agent_id": "agent_001",
        "type": "document_parser",
        "status": "completed",
        "progress": 100
      },
      {
        "agent_id": "agent_002",
        "type": "term_collector",
        "status": "running",
        "progress": 45
      },
      {
        "agent_id": "agent_003",
        "type": "svg_generator",
        "status": "waiting",
        "progress": 0
      },
      {
        "agent_id": "agent_004",
        "type": "translator",
        "status": "waiting",
        "progress": 0
      }
    ],
    "monitoring_url": "/api/v1/ai/workflows/wf_001"
  }
}
```

## 6. エラーハンドリング

### 6.1 AI特有のエラーコード

| コード | 説明 |
|------|------|
| AI_MODEL_NOT_AVAILABLE | AIモデルが利用不可 |
| AI_PROCESSING_FAILED | AI処理失敗 |
| AI_CONFIDENCE_TOO_LOW | 信頼度が闾値以下 |
| MCP_PROVIDER_ERROR | MCPプロバイダーエラー |
| MCP_RATE_LIMIT | MCPレート制限 |
| WORKFLOW_TIMEOUT | ワークフロータイムアウト |

### 6.2 エラーレスポンス例

```json
{
  "success": false,
  "error": {
    "code": "AI_CONFIDENCE_TOO_LOW",
    "message": "AIの信頼度が闾値を下回っています",
    "details": {
      "required_confidence": 0.7,
      "actual_confidence": 0.45,
      "suggestion": "より高品質な画像を使用するか、手動で確認してください"
    }
  }
}
```