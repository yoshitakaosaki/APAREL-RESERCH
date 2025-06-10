# REST APIサーバー実装要件書

## 1. システム概要

### 1.1 目的
BFF-Web（Next.js）からのAPIリクエストを受け付け、認証サーバーが発行したJWTトークンを検証し、ビジネスロジックを実行するREST APIサーバー。

### 1.2 システム構成での位置づけ

```
認証フロー:
1. Browser → BFF-Web: ユーザーリクエスト
2. BFF-Web → Auth Server: 認証・JWT取得
3. BFF-Web → REST API: JWTを含むAPIリクエスト（ココ）
4. REST API → Auth Server (JWKS): 公開鍵取得・JWT検証
```

### 1.3 主要な責務

1. **JWT検証**: 認証サーバー発行のJWTを検証
2. **認可制御**: スコープベースのアクセス制御
3. **ビジネスロジック**: アプリケーション固有の処理
4. **データ管理**: ビジネスデータの永続化
5. **API提供**: RESTful APIエンドポイント

## 2. 機能要件

### 2.1 認証・認可

#### JWT検証
- [ ] RS256アルゴリズムでの署名検証
- [ ] JWKSエンドポイントからの公開鍵取得（`/oauth/.well-known/jwks.json`）
- [ ] 公開鍵のキャッシュ（1時間）
- [ ] トークン有効期限の検証（15分）
- [ ] Issuer（発行者）の検証
- [ ] Audience（対象者）の検証（client_id）
- [ ] JTIブラックリスト確認（トークン無効化対応）

#### スコープベース認可
- [ ] トークン内のスコープ抽出
- [ ] エンドポイント別必要スコープ定義
- [ ] スコープ不足時の403エラー
- [ ] 階層的スコープ対応（例：admin > user）

### 2.2 APIエンドポイント

#### 必須エンドポイント
| エンドポイント | メソッド | 説明 | 必要スコープ |
|-------------|---------|------|------------|
| `/api/v1/health` | GET | ヘルスチェック | なし |
| `/api/v1/users/me` | GET | 自分の情報取得 | `profile:read` |
| `/api/v1/users/me` | PATCH | 自分の情報更新 | `profile:write` |
| `/api/v1/dashboard` | GET | ダッシュボード情報 | `dashboard:read` |

#### エンドポイント設計原則
- RESTful設計（リソース指向）
- バージョニング（/api/v1/）
- 一貫した命名規則（複数形、ケバブケース）
- 適切なHTTPステータスコード
- すべてのエンドポイントは末尾スラッシュ不要（Django設定で調整）

### 2.3 データ管理

#### ユーザー関連データ
- ユーザープロフィール拡張情報
- ユーザー設定・プリファレンス
- アクティビティ履歴

#### ビジネスデータ
- アプリケーション固有のエンティティ
- リレーショナルデータ設計
- 監査証跡の保持

### 2.4 エラーハンドリング

#### 標準エラーレスポンス
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Human readable message",
        "details": {
            "field": "specific_field",
            "reason": "validation_error"
        },
        "request_id": "unique-request-id"
    }
}
```

#### エラーコード体系
- `UNAUTHORIZED`: 認証エラー（401）
- `FORBIDDEN`: 認可エラー（403）
- `NOT_FOUND`: リソース不在（404）
- `VALIDATION_ERROR`: バリデーションエラー（400）
- `INTERNAL_ERROR`: サーバーエラー（500）

## 3. 非機能要件

### 3.1 パフォーマンス
- API応答時間: 95パーセンタイルで200ms以内
- 同時接続数: 1000クライアント以上
- スループット: 1000 req/sec以上

### 3.2 可用性
- 稼働率: 99.9%以上
- 計画停止: 月1回、最大2時間
- グレースフルシャットダウン対応

### 3.3 セキュリティ
- すべての通信でHTTPS必須
- JWTの適切な検証
- SQLインジェクション対策
- XSS対策（出力エスケープ）
- CORS適切な設定

### 3.4 スケーラビリティ
- 水平スケール可能な設計
- ステートレスな実装
- データベース接続プーリング
- キャッシュ戦略

## 4. 技術要件

### 4.1 技術スタック
```
言語: Python 3.10+
フレームワーク: Django 4.2+ / Django REST Framework 3.14+
データベース: PostgreSQL 13+
キャッシュ: Redis 6+
非同期処理: Celery 5.2+（オプション）
```

### 4.2 必須ライブラリ
```python
# requirements.txt
Django>=4.2,<5.0
djangorestframework>=3.14
psycopg2-binary>=2.9
redis>=4.5
PyJWT>=2.7
cryptography>=40.0
requests>=2.30
django-cors-headers>=4.0
gunicorn>=20.1
python-decouple>=3.8
```

### 4.3 推奨ライブラリ
```python
# 開発・デバッグ
django-debug-toolbar>=4.0
django-extensions>=3.2

# テスト
pytest-django>=4.5
factory-boy>=3.2
coverage>=7.2

# 監視・ログ
django-prometheus>=2.3
sentry-sdk>=1.25
python-json-logger>=2.0
```

## 5. インターフェース仕様

### 5.1 認証ヘッダー
```http
Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCIsImtpZCI6IjIwMjQtMDEifQ...
```

### 5.2 共通リクエストヘッダー
```http
Content-Type: application/json
Accept: application/json
X-Request-ID: unique-request-id
X-Client-Version: 1.0.0
```

### 5.3 共通レスポンスヘッダー
```http
Content-Type: application/json
X-Request-ID: unique-request-id
X-Response-Time: 123ms
X-Rate-Limit-Remaining: 999
```

## 6. JWT検証仕様

### 6.1 期待されるJWTペイロード
```json
{
    "iss": "http://localhost:8000",
    "sub": "user-id-123",
    "aud": "bff-web-client",
    "exp": 1704931200,
    "nbf": 1704927600,
    "iat": 1704927600,
    "jti": "unique-jwt-id",
    "scope": "profile:read profile:write dashboard:read",
    "client_id": "bff-web-client",
    "grant_type": "authorization_code",
    "email": "user@example.com",
    "email_verified": true,
    "name": "John Doe",
    "preferred_username": "johndoe"
}
```

### 6.2 検証項目
1. **署名検証**: JWKSから取得した公開鍵で検証（RS256）
2. **有効期限**: `exp` が現在時刻より未来
3. **Not Before**: `nbf` が現在時刻より過去
4. **発行者**: `iss` が設定値と一致（`http://localhost:8000`）
5. **対象者**: `aud` がクライアントIDと一致
6. **発行時刻**: `iat` が妥当な範囲内
7. **JTI**: ブラックリストに含まれていないこと

## 7. データベース設計方針

### 7.1 基本方針
- 認証情報は保持しない（JWTで完結）
- ユーザーIDは認証サーバーのものを使用
- ビジネスデータのみ管理
- 適切なインデックス設計

### 7.2 必須テーブル
```sql
-- ユーザープロフィール拡張
CREATE TABLE user_profiles (
    user_id VARCHAR(255) PRIMARY KEY,  -- 認証サーバーのユーザーID
    display_name VARCHAR(255),
    bio TEXT,
    avatar_url VARCHAR(500),
    preferences JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- APIアクセスログ
CREATE TABLE api_access_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms INTEGER,
    request_id VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## 8. セキュリティ要件

### 8.1 認証・認可
- Bearerトークン以外の認証方式は受け付けない
- 無効なトークンは即座に401エラー
- スコープ不足は403エラー
- トークンのJTIブラックリスト確認（オプション）

### 8.2 通信セキュリティ
- HTTPS必須（HTTP Strict Transport Security）
- 適切なCORS設定（BFF-Webのオリジンのみ許可）
- セキュリティヘッダーの付与

### 8.3 データ保護
- SQLインジェクション対策（パラメータバインディング）
- 出力時のHTMLエスケープ
- 機密情報のログ出力禁止
- 個人情報の適切な暗号化

## 9. 監視・運用要件

### 9.1 ヘルスチェック
```json
GET /api/v1/health

{
    "status": "healthy",
    "timestamp": "2024-01-10T10:00:00Z",
    "version": "1.0.0",
    "checks": {
        "database": "ok",
        "redis": "ok",
        "auth_server": "ok"
    }
}
```

### 9.2 メトリクス
- リクエスト数（エンドポイント別）
- レスポンスタイム（パーセンタイル）
- エラー率
- JWT検証失敗率
- データベース接続数

### 9.3 ログ出力
- アクセスログ（全リクエスト）
- エラーログ（警告以上）
- 監査ログ（データ変更操作）
- パフォーマンスログ（遅いクエリ）

## 10. 制約事項

### 10.1 認証に関する制約
- JWT発行は行わない（検証のみ）
- ユーザー登録・ログインは扱わない
- パスワード情報は保持しない

### 10.2 技術的制約
- Python 3.10以上必須
- PostgreSQL 13以上必須
- 認証サーバーとの通信が必要

### 10.3 運用上の制約
- JWKSキャッシュのため、鍵ローテーション時に最大1時間の遅延
- 認証サーバー停止時も既存JWTは検証可能

## 11. 移行・統合要件

### 11.1 既存システムからの移行
- 段階的移行をサポート
- 旧APIとの並行稼働期間を設定
- データ移行ツールの提供

### 11.2 BFF-Webとの統合
- OpenAPI仕様書の提供
- SDKまたはクライアントライブラリ
- 統合テスト環境

## 12. 成果物

### 12.1 ソースコード
- GitHubリポジトリ
- 適切なブランチ戦略
- CI/CDパイプライン

### 12.2 ドキュメント
- API仕様書（OpenAPI 3.0）
- 実装ガイド
- 運用マニュアル

### 12.3 テスト
- 単体テストカバレッジ80%以上
- 統合テストシナリオ
- 負荷テスト結果

## 13. スケジュール目安

| フェーズ | 期間 | 内容 |
|---------|------|------|
| 設計 | 1週間 | API設計、データモデル設計 |
| JWT検証実装 | 1週間 | 認証ミドルウェア、JWKS連携 |
| API実装 | 2週間 | エンドポイント実装 |
| テスト | 1週間 | 単体・統合テスト |
| 統合 | 1週間 | BFF-Web連携テスト |

## まとめ

このREST APIサーバーは、認証サーバーとBFF-Webの間で、セキュアで高性能なビジネスロジック層として機能します。JWT検証を中心としたステートレスな設計により、スケーラブルなシステムを実現します。