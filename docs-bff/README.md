# BFF-Web 実装ドキュメント

## 概要

このディレクトリには、BFF-Web (Next.js) の実装に必要なすべてのドキュメントが含まれています。

## 📚 ドキュメント構成

### 必須ドキュメント（実装前に必ず読む）

1. **[実装指示書](./01-implementation-instructions.md)**
   - BFF-Webの役割と責任
   - 必須実装エンドポイント
   - セキュリティ要件

2. **[API仕様書](./02-api-specification.md)**
   - 認証サーバーAPIの完全な仕様
   - リクエスト/レスポンス形式
   - エラーコード一覧

3. **[シーケンス図集](./03-sequence-diagrams.md)**
   - 認証フローの視覚的説明
   - 各ステップの詳細
   - エラーケースの処理

### 実装ガイド

4. **[実装ガイド](./04-implementation-guide.md)**
   - Next.jsプロジェクトのセットアップ
   - 各エンドポイントの実装コード
   - Redisセッション管理

5. **[セキュリティガイド](./05-security-guide.md)**
   - PKCE実装の詳細
   - Cookie設定
   - セキュリティベストプラクティス

### 追加機能

6. **[マルチプロバイダー対応](./06-multi-provider.md)**
   - 複数認証プロバイダーの実装
   - プロバイダー別の考慮事項

7. **[ログアウト実装](./07-logout-implementation.md)**
   - 完全なログアウト処理
   - トークン無効化

### リファレンス

8. **[環境設定](./08-environment-setup.md)**
   - 環境変数の設定
   - Docker Compose設定
   - 開発環境構築

9. **[トラブルシューティング](./09-troubleshooting.md)**
   - よくある問題と解決方法
   - デバッグ方法

10. **[テストガイド](./10-testing-guide.md)**
    - テストシナリオ
    - テストコード例

## 🚀 クイックスタート

### 1. 環境準備

```bash
# Node.js 18+ が必要
node --version

# プロジェクト作成
npx create-next-app@latest bff-web --typescript --app --tailwind
cd bff-web

# 依存関係インストール
npm install redis ioredis jose uuid
npm install @types/uuid --save-dev
```

### 2. 環境変数設定

`.env.local` ファイルを作成:

```env
# 認証サーバー設定（Django）
AUTH_SERVER_URL=http://host.docker.internal:8080
AUTH_SERVER_PUBLIC_URL=http://localhost:8080
AUTH_CLIENT_ID=bff-web-client
AUTH_CLIENT_SECRET=7Y9bC3dE5fG7hJ9kL3mN5pQ7rS9tU3vW5xY7zA3bC5dE7f
AUTH_REDIRECT_URI=http://localhost:3000/api/auth/callback

# Redis設定
REDIS_URL=redis://localhost:6379
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# セッション設定
SESSION_SECRET=b56e9255fa64268c22a6e795bf4d61722994e59a64c590429f8b74205f104e07
SESSION_COOKIE_NAME=bff_session
SESSION_EXPIRY=604800
AUTH_SESSION_EXPIRY=600

# セキュリティ設定
ENCRYPTION_KEY=9a8c5850216e2ac4f325e2d89149a641cf25ccde794f22062ba0257971c3b0fc
COOKIE_SECURE=false
COOKIE_SAME_SITE=lax

# URL
NEXT_PUBLIC_BASE_URL=http://localhost:3000
NODE_ENV=development
```

### 3. 実装開始

1. **[実装指示書](./01-implementation-instructions.md)** を読む
2. **[シーケンス図集](./03-sequence-diagrams.md)** でフローを理解
3. **[実装ガイド](./04-implementation-guide.md)** のコードを参考に実装

## 📋 実装チェックリスト

- [x] ドキュメントをすべて読んだ
- [x] 開発環境をセットアップした
- [x] 認証サーバーにアクセスできることを確認
- [x] Redisが起動していることを確認
- [x] 基本的な認証フロー（Google）を実装
- [x] PKCE (code_verifier/challenge) の実装
- [x] state パラメータの生成と検証
- [x] JWT トークンデコード機能
- [x] Docker内部ネットワーク対応
- [x] Cookie設定の実装（中間ページ経由）
- [x] エラーハンドリングを実装
- [x] セキュリティヘッダーの設定
- [ ] トークンリフレッシュを実装
- [ ] ログアウトを実装
- [ ] テストを作成した

## 🆘 サポート

### 認証チームへの連絡

- **Slack**: #auth-support
- **Email**: auth-team@example.com
- **Wiki**: https://wiki.example.com/auth

### よくある質問

**Q: PKCEって何？**
A: Proof Key for Code Exchange。認可コード横取り攻撃を防ぐセキュリティ機能です。詳細は[セキュリティガイド](./05-security-guide.md)を参照。

**Q: セッションの有効期限は？**
A: デフォルト7日間。`SESSION_EXPIRY`環境変数で変更可能。

**Q: エラーが発生したらどうすれば？**
A: まず[トラブルシューティング](./09-troubleshooting.md)を確認。解決しない場合は認証チームへ。

## 📝 更新履歴

| 日付 | バージョン | 内容 |
|------|-----------|------|
| 2024-01-10 | 1.0.0 | 初版作成 |
| 2025-01-07 | 1.1.0 | 現在の実装に合わせて更新:<br/>- Docker内部ネットワーク対応<br/>- JWT トークンデコード<br/>- 中間ページ経由Cookie設定<br/>- 実装チェックリスト更新 |

---

**重要**: セキュリティに関わる実装は、推測せずに必ず認証チームに確認してください。