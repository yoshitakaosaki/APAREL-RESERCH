# 環境設定ガイド

## 概要

BFF-Web の環境設定と構築手順を説明します。

## 1. 開発環境のセットアップ

### 必要なソフトウェア

- Node.js 18.0.0 以上
- npm 9.0.0 以上
- Redis 6.0 以上
- Docker & Docker Compose（オプション）

### バージョン確認

```bash
# Node.js
node --version  # v18.0.0以上

# npm
npm --version   # 9.0.0以上

# Redis
redis-server --version  # 6.0以上
```

## 2. プロジェクトのセットアップ

### プロジェクト作成

```bash
# Next.js プロジェクトの作成
npx create-next-app@latest bff-web \
  --typescript \
  --app \
  --tailwind \
  --eslint \
  --no-src-dir \
  --import-alias "@/*"

cd bff-web
```

### 依存関係のインストール

```bash
# 本番依存関係
npm install \
  ioredis \
  jose \
  uuid \
  zod

# 開発依存関係
npm install --save-dev \
  @types/uuid \
  @types/node \
  prettier \
  eslint-config-prettier \
  @testing-library/react \
  @testing-library/jest-dom \
  jest \
  jest-environment-jsdom
```

## 3. 環境変数の設定

### .env.local ファイルの作成

```bash
# .env.local
# 認証サーバー設定（Django）
AUTH_SERVER_URL=http://host.docker.internal:8080
# ブラウザリダイレクト用認証サーバーURL（外部アクセス可能）
AUTH_SERVER_PUBLIC_URL=http://localhost:8080
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

### 環境変数の生成

```bash
# セッション暗号化キーの生成（32バイト）
openssl rand -hex 32

# セッションシークレットの生成
openssl rand -base64 32

# CSRFシークレットの生成
openssl rand -base64 32
```

### 環境変数のバリデーション

```typescript
// lib/env.ts
import { z } from 'zod';

const envSchema = z.object({
  // 認証サーバー
  AUTH_SERVER_URL: z.string().url(),
  AUTH_CLIENT_ID: z.string().min(1),
  AUTH_CLIENT_SECRET: z.string().min(1),
  AUTH_REDIRECT_URI: z.string().url(),
  
  // Redis
  REDIS_URL: z.string().url(),
  REDIS_PASSWORD: z.string().optional(),
  
  // セッション
  SESSION_SECRET: z.string().min(32),
  SESSION_COOKIE_NAME: z.string().min(1),
  SESSION_ENCRYPTION_KEY: z.string().regex(/^[0-9a-f]{64}$/i),
  
  // アプリケーション
  NEXT_PUBLIC_BASE_URL: z.string().url(),
  NODE_ENV: z.enum(['development', 'test', 'production']),
  
  // セキュリティ
  CSRF_SECRET: z.string().min(32),
});

export function validateEnv() {
  try {
    return envSchema.parse(process.env);
  } catch (error) {
    console.error('❌ Invalid environment variables:', error);
    process.exit(1);
  }
}

// アプリケーション起動時に実行
export const env = validateEnv();
```

## 4. Docker Compose 設定

### docker-compose.yml

```yaml
version: '3.8'

services:
  # BFF-Web
  bff-web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=development
    env_file:
      - .env.local
    depends_on:
      - redis
      - auth-server
    volumes:
      - .:/app
      - /app/node_modules
      - /app/.next
    command: npm run dev

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  # 認証サーバー（モック）
  auth-server:
    image: stoplight/prism:4
    ports:
      - "8000:4010"
    command: mock -h 0.0.0.0 /tmp/api.yaml
    volumes:
      - ./mock/auth-api.yaml:/tmp/api.yaml:ro

volumes:
  redis_data:
```

### Dockerfile

```dockerfile
FROM node:18-alpine AS base

# 依存関係のインストール
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package*.json ./
RUN npm ci

# ビルド
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npm run build

# 実行
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000

CMD ["node", "server.js"]
```

### 開発用 Dockerfile.dev

```dockerfile
FROM node:18-alpine

WORKDIR /app

# 開発に必要なツール
RUN apk add --no-cache git

# package.json のコピーと依存関係のインストール
COPY package*.json ./
RUN npm install

# アプリケーションファイルのコピー
COPY . .

# ポート公開
EXPOSE 3000

# 開発サーバー起動
CMD ["npm", "run", "dev"]
```

## 5. 開発サーバーの起動

### ローカル環境

```bash
# Redis の起動（別ターミナル）
redis-server --daemonize yes --port 6379

# 認証サーバーの起動（別ターミナル）
cd ../blead-auth-svr
python manage.py runserver 0.0.0.0:8080

# BFF-Web の起動
npm run dev
```

### Docker Compose 使用

```bash
# すべてのサービスを起動
docker-compose up

# バックグラウンドで起動
docker-compose up -d

# ログの確認
docker-compose logs -f bff-web

# サービスの停止
docker-compose down
```

## 6. 環境別設定

### 開発環境 (.env.development)

```bash
NODE_ENV=development
AUTH_SERVER_URL=http://host.docker.internal:8080
AUTH_SERVER_PUBLIC_URL=http://localhost:8080
NEXT_PUBLIC_BASE_URL=http://localhost:3000
SESSION_COOKIE_NAME=bff_session_dev
COOKIE_SECURE=false
COOKIE_SAME_SITE=lax
```

### ステージング環境 (.env.staging)

```bash
NODE_ENV=production
AUTH_SERVER_URL=https://auth-staging.example.com
NEXT_PUBLIC_BASE_URL=https://app-staging.example.com
SESSION_COOKIE_NAME=__Secure-session
```

### 本番環境 (.env.production)

```bash
NODE_ENV=production
AUTH_SERVER_URL=https://auth.example.com
NEXT_PUBLIC_BASE_URL=https://app.example.com
SESSION_COOKIE_NAME=__Host-session
```

## 7. SSL/TLS 設定（開発環境）

### mkcert を使用したローカルHTTPS

```bash
# mkcert のインストール
brew install mkcert  # macOS
# または
sudo apt install libnss3-tools  # Ubuntu
wget https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64
sudo mv mkcert-v1.4.4-linux-amd64 /usr/local/bin/mkcert
sudo chmod +x /usr/local/bin/mkcert

# ローカルCA の作成
mkcert -install

# 証明書の生成
mkcert localhost 127.0.0.1 ::1

# Next.js の設定
```

### next.config.js の HTTPS 設定

```javascript
// next.config.js
const fs = require('fs');
const path = require('path');

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

// 開発環境でHTTPSを有効化
if (process.env.NODE_ENV === 'development') {
  nextConfig.serverOptions = {
    https: {
      key: fs.readFileSync(path.join(__dirname, 'localhost-key.pem')),
      cert: fs.readFileSync(path.join(__dirname, 'localhost.pem')),
    },
  };
}

module.exports = nextConfig;
```

## 8. Redis の設定

### Redis 設定ファイル (redis.conf)

```conf
# バインドアドレス
bind 127.0.0.1 ::1

# ポート
port 6379

# パスワード（本番環境では必須）
# requirepass your-redis-password

# 永続化設定
save 900 1
save 300 10
save 60 10000

# AOF（Append Only File）有効化
appendonly yes

# 最大メモリ使用量
maxmemory 256mb
maxmemory-policy allkeys-lru

# ログレベル
loglevel notice
```

### Redis クライアント設定

```typescript
// lib/redis.ts
import { Redis } from 'ioredis';

const redisConfig = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,
  db: parseInt(process.env.REDIS_DB || '0'),
  
  // 接続設定
  connectTimeout: 10000,
  maxRetriesPerRequest: 3,
  enableReadyCheck: true,
  
  // 本番環境向け設定
  ...(process.env.NODE_ENV === 'production' && {
    tls: {},
    sentinels: process.env.REDIS_SENTINELS
      ? JSON.parse(process.env.REDIS_SENTINELS)
      : undefined,
    name: process.env.REDIS_SENTINEL_NAME,
  }),
};

const redis = new Redis(redisConfig);

// 接続イベント
redis.on('connect', () => {
  console.log('✅ Redis connected');
});

redis.on('error', (err) => {
  console.error('❌ Redis error:', err);
});

export default redis;
```

## 9. ヘルスチェック設定

### ヘルスチェックエンドポイント

```typescript
// app/api/health/route.ts
import { NextResponse } from 'next/server';
import redis from '@/lib/redis';

export async function GET() {
  const health = {
    status: 'ok',
    timestamp: new Date().toISOString(),
    services: {
      redis: 'unknown',
      authServer: 'unknown',
    },
  };

  // Redis チェック
  try {
    await redis.ping();
    health.services.redis = 'healthy';
  } catch (error) {
    health.services.redis = 'unhealthy';
    health.status = 'degraded';
  }

  // 認証サーバーチェック
  try {
    const response = await fetch(`${process.env.AUTH_SERVER_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    });
    health.services.authServer = response.ok ? 'healthy' : 'unhealthy';
  } catch (error) {
    health.services.authServer = 'unhealthy';
    health.status = 'degraded';
  }

  const statusCode = health.status === 'ok' ? 200 : 503;
  return NextResponse.json(health, { status: statusCode });
}
```

## 10. トラブルシューティング

### よくある問題と解決方法

#### Redis 接続エラー

```bash
# エラー: Redis connection refused
# 解決方法:
redis-cli ping  # Redis が起動しているか確認
sudo systemctl start redis  # Redis を起動
```

#### 環境変数が読み込まれない

```bash
# エラー: Environment variable not found
# 解決方法:
cp .env.local.example .env.local  # 環境変数ファイルを作成
source .env.local  # 環境変数を読み込む（シェルの場合）
```

#### ポート競合

```bash
# エラー: Port 3000 is already in use
# 解決方法:
lsof -i :3000  # ポートを使用しているプロセスを確認
kill -9 <PID>  # プロセスを終了
# または
PORT=3001 npm run dev  # 別のポートで起動
```

### デバッグモード

```typescript
// lib/debug.ts
export const DEBUG = {
  auth: process.env.DEBUG_AUTH === 'true',
  redis: process.env.DEBUG_REDIS === 'true',
  api: process.env.DEBUG_API === 'true',
};

export function debugLog(category: keyof typeof DEBUG, ...args: any[]) {
  if (DEBUG[category]) {
    console.log(`[${category.toUpperCase()}]`, ...args);
  }
}
```

## まとめ

適切な環境設定は、セキュアで安定したBFF-Webの基盤となります。特に：

1. **環境変数の管理**: 機密情報の安全な管理
2. **Docker化**: 一貫した開発・本番環境
3. **Redis設定**: セッション管理の信頼性
4. **ヘルスチェック**: システムの可用性監視

これらを適切に設定することで、本番環境でも安心して運用できるシステムを構築できます。