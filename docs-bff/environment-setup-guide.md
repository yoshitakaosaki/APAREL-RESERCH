# BFF OAuth2.0+PKCE認証システム - 環境構築ガイド

## 概要

このガイドでは、blead-stamp BFF（Backend For Frontend）OAuth2.0+PKCE認証システムの開発環境構築手順を説明します。

## システム要件

### 必要なソフトウェア
- **Node.js**: 18.x以上
- **npm**: 9.x以上
- **Python**: 3.11以上
- **Redis**: 6.x以上
- **VS Code**: 最新版（推奨）
- **Docker**: 20.x以上（Docker環境の場合）

### 推奨環境
- **OS**: Linux、macOS、Windows（WSL2推奨）
- **メモリ**: 8GB以上
- **ストレージ**: 20GB以上の空き容量

## プロジェクト構成

```
blead-stamp/                    # BFF-Webアプリケーション
├── src/
│   ├── app/
│   │   ├── api/auth/          # OAuth2.0認証エンドポイント
│   │   ├── components/        # UIコンポーネント
│   │   └── dashboard/         # 認証後ダッシュボード
│   ├── lib/                   # ライブラリとユーティリティ
│   └── hooks/                 # React Hooks
├── docs/                      # システム設計ドキュメント
├── docs-bff/                  # BFF実装ドキュメント
├── test-*.js                  # E2Eテストファイル
└── .env.local                 # 環境設定ファイル

blead-auth-svr/                # Django認証サーバー（別リポジトリ）
├── manage.py
├── venv/                      # Python仮想環境
└── settings.py
```

## 環境構築手順

### 1. BFF-Webアプリケーション環境構築

#### 1.1 プロジェクトクローン
```bash
git clone <repository-url> blead-stamp
cd blead-stamp
```

#### 1.2 依存関係のインストール
```bash
npm install
```

#### 1.3 環境設定ファイルの作成
`.env.local` ファイルを作成：

```env
# BFF-Web OAuth2.0+PKCE 認証システム開発環境設定

# 認証サーバー設定（Django）
AUTH_SERVER_URL=http://host.docker.internal:8080
# ブラウザリダイレクト用認証サーバーURL（外部アクセス可能）
AUTH_SERVER_PUBLIC_URL=http://host.docker.internal:8080
AUTH_CLIENT_ID=bff-web-client
AUTH_CLIENT_SECRET=7Y9bC3dE5fG7hJ9kL3mN5pQ7rS9tU3vW5xY7zA3bC5dE7f
AUTH_REDIRECT_URI=http://localhost:3000/api/auth/callback

# Redis設定
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# セッション設定
SESSION_SECRET=b56e9255fa64268c22a6e795bf4d61722994e59a64c590429f8b74205f104e07
SESSION_COOKIE_NAME=bff_session
SESSION_EXPIRY=604800  # 7日間（秒単位）
AUTH_SESSION_EXPIRY=600  # 認証プロセス用一時セッション：10分

# Cookie設定
COOKIE_SECURE=false  # 開発環境では false
COOKIE_SAME_SITE=lax  # 開発環境では lax

# アプリケーション設定
NEXT_PUBLIC_BASE_URL=http://localhost:3000
NODE_ENV=development

# ログ設定
LOG_LEVEL=debug

# セキュリティ設定
ENCRYPTION_KEY=9a8c5850216e2ac4f325e2d89149a641cf25ccde794f22062ba0257971c3b0fc
RATE_LIMIT_MAX=1000  # 開発環境では制限を緩く
RATE_LIMIT_WINDOW=3600

# 開発用設定
NEXT_PUBLIC_DEBUG=true
```

#### 1.4 BFF-Webアプリケーション起動
```bash
npm run dev
```

アプリケーションは `http://localhost:3000` で起動します。

### 2. Redis環境構築

#### 2.1 Redisサーバー起動
```bash
# バックグラウンドで起動
redis-server --daemonize yes --port 6379

# 起動確認
redis-cli ping
# 結果: PONG
```

#### 2.2 Redis接続テスト
```bash
# Redis CLIでテスト
redis-cli
127.0.0.1:6379> set test "Hello Redis"
127.0.0.1:6379> get test
127.0.0.1:6379> exit
```

### 3. Django認証サーバー環境構築

#### 3.1 認証サーバープロジェクトの準備
```bash
# 認証サーバーのディレクトリへ移動
cd /workspaces/blead-auth-svr

# Python仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate

# 依存関係のインストール
pip install -r requirements.txt
```

#### 3.2 Django設定の確認
`settings.py` で以下の設定を確認：

```python
ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    'host.docker.internal',  # Docker container communication用
    # 本番環境では適切なドメインを追加
]

# OAuth2.0設定
OAUTH2_PROVIDER = {
    'SCOPES': {
        'openid': 'OpenID Connect scope',
        'profile': 'Access to your profile',
        'email': 'Access to your email',
    },
    'ACCESS_TOKEN_EXPIRE_SECONDS': 3600,
    'REFRESH_TOKEN_EXPIRE_SECONDS': 3600 * 24 * 7,  # 7日間
}
```

#### 3.3 Django認証サーバー起動

**VS Codeデバッガーを使用する場合**:

`.vscode/launch.json` を作成：
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Django Server",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/manage.py",
            "args": [
                "runserver",
                "0.0.0.0:8080"
            ],
            "django": true,
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "DJANGO_SETTINGS_MODULE": "djrestauthPrj.settings"
            },
            "python": "${workspaceFolder}/venv/bin/python"
        }
    ]
}
```

**コマンドラインから起動する場合**:
```bash
source venv/bin/activate
python manage.py runserver 0.0.0.0:8080
```

### 4. VS Code環境設定

#### 4.1 ポートフォワーディング設定
1. VS Code下部の「ポート」タブを開く
2. 「ポートの転送」をクリック
3. ポート `8080` を追加
4. 可視性を「パブリック」に設定

または、コマンドパレット（`Ctrl+Shift+P`）で：
1. "Forward a Port" を検索
2. `8080` を入力

#### 4.2 推奨VS Code拡張機能
```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.debugpy",
        "bradlc.vscode-tailwindcss",
        "esbenp.prettier-vscode",
        "ms-vscode.vscode-typescript-next",
        "redhat.vscode-yaml"
    ]
}
```

## Docker環境での構築

### Docker Compose設定例

`docker-compose.yml`:
```yaml
version: '3.8'
services:
  bff-web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - AUTH_SERVER_URL=http://auth-server:8080
      - AUTH_SERVER_PUBLIC_URL=http://localhost:8080
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - auth-server
    volumes:
      - .:/app
      - /app/node_modules

  auth-server:
    build: ../blead-auth-svr
    ports:
      - "8080:8080"
    environment:
      - DJANGO_SETTINGS_MODULE=djrestauthPrj.settings
    volumes:
      - ../blead-auth-svr:/app

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Docker環境でのネットワーク設定

**重要な注意事項**:
- Docker環境では `localhost` の代わりに `host.docker.internal` を使用
- コンテナ間通信とブラウザアクセスでURLが異なる場合がある
- Puppeteerテストもコンテナ内で実行される場合は内部URLを使用

## 動作確認

### 1. 基本動作確認
```bash
# BFF-Webアプリケーションへアクセス
curl http://localhost:3000

# 認証状態確認エンドポイント
curl http://localhost:3000/api/auth/me

# Django認証サーバー疎通確認
curl http://localhost:8080/admin/
```

### 2. E2Eテスト実行
```bash
# 完全認証フローテスト
node test-complete-auth-flow.js

# Cookie設定検証テスト
node test-cookie-success-verification.js

# URL詳細トレーステスト
node test-url-trace.js
```

### 3. Redis動作確認
```bash
# Redis接続テスト
redis-cli ping

# セッションデータ確認
redis-cli keys "*"
redis-cli get "auth_session:sample-session-id"
```

## トラブルシューティング

### よくある問題と解決策

#### 1. Cookieが設定されない
**症状**: 認証フローでCookieが設定されず、セッションが維持されない

**原因と解決策**:
- JavaScript リダイレクトを使用している → 中間ページ経由のサーバーサイドリダイレクトに変更
- Cookie設定が不正 → HttpOnly, SameSite, Path設定を確認

#### 2. Django認証サーバーに接続できない
**症状**: `chrome-error://chromewebdata/` エラーまたは接続拒否

**原因と解決策**:
- ポートフォワーディング未設定 → VS Codeでポート8080を転送
- ALLOWED_HOSTS設定不備 → `host.docker.internal` を追加
- Djangoサーバー未起動 → サーバー起動状態を確認

#### 3. Redis接続エラー
**症状**: セッション保存・取得でエラー

**原因と解決策**:
- Redisサーバー未起動 → `redis-server --daemonize yes` で起動
- 接続設定間違い → REDIS_URL、REDIS_HOST設定を確認

#### 4. 環境変数が読み込まれない
**症状**: 設定が反映されない

**原因と解決策**:
- .env.localファイル不存在 → ファイル作成と配置確認
- Next.js再起動不足 → `npm run dev` を再実行

### ログ確認方法

#### BFF-Webアプリケーションログ
```bash
# 開発サーバーコンソール出力を確認
npm run dev

# ブラウザ開発者ツールのコンソールタブ
```

#### Django認証サーバーログ
```bash
# VS Codeデバッグコンソール
# または統合ターミナル出力を確認
```

#### Redisログ
```bash
# Redis CLIでモニタリング
redis-cli monitor

# セッションキー一覧表示
redis-cli keys "*session*"
```

## 開発コマンド一覧

### BFF-Web開発コマンド
```bash
npm run dev          # 開発サーバー起動
npm run build        # プロダクションビルド
npm run start        # プロダクションサーバー起動
npm run lint         # ESLint実行
```

### テストコマンド
```bash
node test-complete-auth-flow.js           # 完全認証フローテスト
node test-cookie-success-verification.js  # Cookie設定テスト
node test-url-trace.js                    # URL詳細トレース
```

### Redis管理コマンド
```bash
redis-server --daemonize yes --port 6379  # バックグラウンド起動
redis-cli ping                            # 疎通確認
redis-cli flushdb                         # データベースクリア
redis-cli shutdown                        # サーバー停止
```

## セキュリティ考慮事項

### 開発環境での注意点
- `.env.local` ファイルはGitにコミットしない（`.gitignore`に追加済み）
- 開発用の認証情報は本番環境で使用しない
- HTTPSではない環境でのCookie設定（Secure=false）

### 本番環境への移行時
- 環境変数の適切な設定
- HTTPS強制設定
- セキュリティヘッダーの追加
- Rate Limitingの強化
- ログ監視の設定

---

**作成日**: 2025-06-06  
**更新日**: 2025-06-06  
**作成者**: Claude Code Assistant  
**プロジェクト**: blead-stamp BFF Authentication System