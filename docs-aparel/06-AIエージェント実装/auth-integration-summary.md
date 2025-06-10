# AIエージェントシステム 認証基盤統合サマリー

## 1. 概要

本ドキュメントは、既存の認証基盤（BFF-Web + REST API）をAIエージェントシステムに統合した変更内容をまとめたものです。

## 2. 既存認証基盤の特徴

### 2.1 アーキテクチャ
```
ブラウザ → BFF-Web (Next.js) → REST API (Django) → AIエージェント
              ↓                     ↓
        認証サーバー (OAuth)   JWKS エンドポイント
```

### 2.2 主要機能
- **OAuth 2.0 + PKCE**: Google、LINE、GitHub、メール認証
- **JWT (RS256)**: 非対称暗号化による安全なトークン
- **スコープベース認可**: 細かい権限制御
- **セッション管理**: Redis使用、7日間TTL

## 3. AIエージェント用の新規スコープ

```yaml
# AIエージェントシステム専用スコープ
ai_agent_scopes:
  # 基本操作
  - agent:read          # エージェント状態・結果の読み取り
  - agent:execute       # エージェントタスクの実行
  - agent:admin         # エージェント管理
  
  # テックパック機能
  - techpack:read       # テックパック閲覧
  - techpack:write      # テックパック作成・編集
  - techpack:generate   # AI生成機能の使用
  - techpack:approve    # 生成結果の承認
  
  # 専門機能
  - terms:extract       # 用語抽出
  - terms:manage        # 用語集管理
  - svg:generate        # SVG生成
  - svg:edit           # SVGパラメータ編集
  
  # モニタリング
  - task:monitor        # タスク監視
  - metrics:view        # メトリクス閲覧
```

## 4. 主な変更内容

### 4.1 削除した機能
- カスタムJWT実装（`SecureKeyManager`クラス）
- 独自の認証フロー
- 個別のユーザー管理

### 4.2 追加した統合ポイント

#### FastAPI（AIエージェントAPI）での認証
```python
from authentication.jwt_validator import JWTValidator
from authentication.permissions import HasScope

class AIAgentAuth:
    def __init__(self):
        self.jwt_validator = JWTValidator(
            jwks_endpoint=settings.JWKS_ENDPOINT,
            cache_ttl=3600
        )

@app.post("/api/v1/ai/tasks/generate-techpack")
async def generate_techpack(
    request: TechPackRequest,
    auth_payload: Dict = Depends(auth.require_scopes(['techpack:generate']))
):
    user_id = auth_payload['sub']
    # 認証されたユーザーコンテキストで処理
```

#### エージェント基底クラスの更新
```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        # JWT検証器の初期化
        self.jwt_validator = JWTValidator(
            jwks_endpoint=settings.JWKS_ENDPOINT,
            cache_ttl=3600
        )
```

### 4.3 サービス間通信
```python
# AIエージェント間の通信用サービストークン
service_tokens = {
    'term_collector': ['terms:extract', 'terms:manage'],
    'svg_generator': ['svg:generate', 'storage:write'],
    'orchestrator': ['agent:execute', 'agent:admin']
}
```

## 5. 統合のメリット

1. **セキュリティ向上**
   - 実績のあるOAuth 2.0 + PKCE実装
   - RS256署名による強固なトークン検証

2. **開発効率**
   - 認証コードの重複排除
   - 統一されたエラーハンドリング

3. **保守性**
   - 認証システムの一元管理
   - セキュリティアップデートの簡素化

4. **ユーザー体験**
   - シングルサインオン（SSO）
   - 統一されたログイン体験

## 6. 環境変数設定

```env
# AIエージェント用環境変数
JWKS_ENDPOINT=https://auth.example.com/.well-known/jwks.json
AUTH_SERVICE_URL=https://auth.example.com
SERVICE_TOKEN_ENDPOINT=https://auth.example.com/api/v1/service-token

# Redis（セッション管理）
REDIS_URL=redis://localhost:6379/0
```

## 7. デプロイメント構成の変更

### Docker Compose
```yaml
services:
  ai-agent:
    environment:
      - JWKS_ENDPOINT=${JWKS_ENDPOINT}
      - AUTH_SERVICE_URL=${AUTH_SERVICE_URL}
    depends_on:
      - redis  # 認証キャッシュ用
```

## 8. 移行チェックリスト

- [x] カスタム認証コードの削除
- [x] JWT検証を既存基盤に置き換え
- [x] スコープベース認可の実装
- [x] APIレスポンスの標準化
- [x] エラーハンドリングの統一
- [x] 使用例の更新（認証ヘッダー付き）
- [x] セキュリティフローの更新
- [x] サービス間通信の認証追加

## 9. 今後の作業

1. **テスト実装**
   - 認証フローのE2Eテスト
   - スコープ検証のユニットテスト

2. **監視設定**
   - 認証エラーのモニタリング
   - トークン有効期限のアラート

3. **ドキュメント整備**
   - API仕様書の認証セクション追加
   - 開発者ガイドの更新