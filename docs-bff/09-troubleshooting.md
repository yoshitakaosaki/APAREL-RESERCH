# トラブルシューティングガイド

## 概要

BFF-Web 実装時によく発生する問題と解決方法をまとめています。

## 1. 認証フロー関連

### 問題: "invalid_state" エラー

**症状**
```
Error: invalid_state - Security validation failed
```

**原因**
- Redisセッションの有効期限切れ
- stateパラメータの不一致
- 複数タブでの認証試行

**解決方法**
```typescript
// 1. Redis TTL を確認
await redis.ttl(`auth:${sessionId}`); // 残り時間を確認

// 2. state の生成と保存を確認
const state = generateState();
console.log('Generated state:', state);
await redis.setex(`auth:${sessionId}`, 600, JSON.stringify({
  state,
  codeVerifier,
  timestamp: Date.now(),
}));

// 3. デバッグログを追加
console.log('Saved state:', savedState);
console.log('Received state:', receivedState);
console.log('Match:', savedState === receivedState);
```

### 問題: "code_verifier" が無効

**症状**
```
Error: invalid_grant - PKCE verification failed
```

**原因**
- code_verifierの長さが不足（43文字未満）
- base64パディングの問題
- verifierとchallengeの不一致

**解決方法**
```typescript
// 正しいPKCE実装を確認
function generateCodeVerifier(): string {
  // 必ず43文字以上になるように32バイト使用
  const verifier = crypto.randomBytes(32).toString('base64url');
  console.log('Verifier length:', verifier.length); // 43以上を確認
  return verifier;
}

function generateCodeChallenge(verifier: string): string {
  const challenge = crypto
    .createHash('sha256')
    .update(verifier)
    .digest('base64url'); // base64urlを使用（パディングなし）
  console.log('Challenge:', challenge);
  return challenge;
}

// テスト
const verifier = generateCodeVerifier();
const challenge = generateCodeChallenge(verifier);
console.log({
  verifier,
  verifierLength: verifier.length,
  challenge,
  challengeLength: challenge.length,
});
```

### 問題: Cookieが設定されない

**症状**
- ログイン後もログイン画面にリダイレクトされる
- `bff_session` Cookieが見つからない

**原因**
- SameSite属性の設定ミス
- HTTPSとHTTPの混在
- ドメインの不一致

**解決方法**
```typescript
// 1. Cookie設定を確認
const cookieOptions = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production', // 開発環境ではfalse
  sameSite: process.env.NODE_ENV === 'production' ? 'strict' : 'lax',
  path: '/',
  maxAge: 60 * 60 * 24 * 7,
};

// 2. レスポンスヘッダーを確認
response.headers.forEach((value, key) => {
  console.log(`${key}: ${value}`);
});

// 3. ブラウザの開発者ツールで確認
// Application > Cookies で以下を確認：
// - Domain
// - Path
// - SameSite
// - Secure
// - HttpOnly
```

## 2. Redis 関連

### 問題: Redis接続エラー

**症状**
```
Error: connect ECONNREFUSED 127.0.0.1:6379
```

**原因**
- Redisサーバーが起動していない
- 接続設定の誤り
- ファイアウォール/ネットワークの問題

**解決方法**
```bash
# 1. Redisの起動確認
redis-cli ping
# PONG が返ってくるか確認

# 2. Redisサーバーを起動
redis-server
# または
brew services start redis  # macOS
sudo systemctl start redis  # Linux

# 3. 接続テスト
node -e "
const redis = require('ioredis');
const client = new redis('redis://localhost:6379');
client.ping().then(() => {
  console.log('Redis connected');
  process.exit(0);
}).catch(err => {
  console.error('Redis error:', err);
  process.exit(1);
});
"
```

### 問題: Redisメモリ不足

**症状**
```
Error: OOM command not allowed when used memory > 'maxmemory'
```

**解決方法**
```bash
# 1. 現在のメモリ使用量を確認
redis-cli info memory

# 2. 不要なキーを削除
redis-cli --scan --pattern "auth:*" | xargs redis-cli del
redis-cli --scan --pattern "session:*" | head -100 | xargs redis-cli del

# 3. メモリ設定を調整
redis-cli config set maxmemory 512mb
redis-cli config set maxmemory-policy allkeys-lru
```

## 3. トークン関連

### 問題: アクセストークンの期限切れ

**症状**
- APIリクエストが401エラーを返す
- "token_expired" エラー

**解決方法**
```typescript
// 自動リフレッシュミドルウェアの実装
async function withTokenRefresh<T>(
  sessionId: string,
  apiCall: (token: string) => Promise<T>
): Promise<T> {
  const session = await getSession(sessionId);
  
  // トークン有効期限チェック
  const expiresAt = new Date(session.expiresAt);
  const now = new Date();
  const bufferTime = 5 * 60 * 1000; // 5分前
  
  if (expiresAt.getTime() - now.getTime() < bufferTime) {
    // リフレッシュが必要
    console.log('Token expiring soon, refreshing...');
    await refreshToken(sessionId);
    const updatedSession = await getSession(sessionId);
    return apiCall(updatedSession.accessToken);
  }
  
  try {
    return await apiCall(session.accessToken);
  } catch (error: any) {
    if (error.status === 401) {
      // トークンが無効な場合はリフレッシュして再試行
      console.log('Token invalid, refreshing...');
      await refreshToken(sessionId);
      const updatedSession = await getSession(sessionId);
      return apiCall(updatedSession.accessToken);
    }
    throw error;
  }
}
```

### 問題: リフレッシュトークンが無効

**症状**
```
Error: invalid_grant - Refresh token is invalid or expired
```

**解決方法**
```typescript
// エラーハンドリングの改善
async function handleRefreshError(error: any, sessionId: string) {
  console.error('Refresh token error:', error);
  
  // セッションをクリア
  await deleteSession(sessionId);
  
  // ユーザーを再ログインに誘導
  return NextResponse.json(
    { 
      error: 'session_expired',
      message: 'Please login again',
      redirect: '/login'
    },
    { status: 401 }
  );
}
```

## 4. CORS エラー

### 問題: CORS policy エラー

**症状**
```
Access to fetch at 'http://localhost:8000/oauth/token' from origin 
'http://localhost:3000' has been blocked by CORS policy
```

**原因**
- 認証サーバーのCORS設定不足
- credentialsの設定漏れ

**解決方法**

**認証サーバー側（Django）**
```python
# settings.py
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]
```

**BFF-Web側**
```typescript
// fetchにcredentialsを追加
const response = await fetch(url, {
  method: 'POST',
  credentials: 'include', // 重要
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(data),
});
```

## 5. 環境変数関連

### 問題: 環境変数が undefined

**症状**
```
TypeError: Cannot read property 'AUTH_CLIENT_ID' of undefined
```

**解決方法**
```typescript
// 1. .env.local ファイルの存在確認
if (!fs.existsSync('.env.local')) {
  console.error('❌ .env.local file not found');
  console.log('Creating from template...');
  fs.copyFileSync('.env.local.example', '.env.local');
}

// 2. 環境変数の検証
const required = [
  'AUTH_SERVER_URL',
  'AUTH_CLIENT_ID',
  'AUTH_CLIENT_SECRET',
  'REDIS_URL',
];

const missing = required.filter(key => !process.env[key]);
if (missing.length > 0) {
  console.error('❌ Missing environment variables:', missing);
  console.log('\nPlease set these in your .env.local file:');
  missing.forEach(key => {
    console.log(`${key}=your-value-here`);
  });
  process.exit(1);
}

// 3. Next.js の環境変数読み込み確認
console.log('Loaded env vars:', Object.keys(process.env).filter(k => k.startsWith('AUTH_')));
```

## 6. ネットワークエラー

### 問題: fetch timeout

**症状**
```
Error: network timeout at: http://localhost:8000/oauth/token
```

**解決方法**
```typescript
// タイムアウト付きfetchの実装
async function fetchWithTimeout(
  url: string,
  options: RequestInit,
  timeout: number = 5000
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeout);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    return response;
  } catch (error: any) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeout}ms`);
    }
    throw error;
  }
}

// リトライ付きリクエスト
async function fetchWithRetry(
  url: string,
  options: RequestInit,
  maxRetries: number = 3
): Promise<Response> {
  let lastError;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetchWithTimeout(url, options);
      return response;
    } catch (error) {
      console.log(`Attempt ${i + 1} failed:`, error);
      lastError = error;
      
      // 指数バックオフ
      if (i < maxRetries - 1) {
        const delay = Math.pow(2, i) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }
  
  throw lastError;
}
```

## 7. デバッグ手法

### 詳細ログの有効化

```typescript
// lib/logger.ts
export class DebugLogger {
  private category: string;
  private enabled: boolean;
  
  constructor(category: string) {
    this.category = category;
    this.enabled = process.env[`DEBUG_${category.toUpperCase()}`] === 'true';
  }
  
  log(...args: any[]) {
    if (this.enabled) {
      console.log(`[${this.category}]`, new Date().toISOString(), ...args);
    }
  }
  
  error(...args: any[]) {
    // エラーは常に出力
    console.error(`[${this.category}:ERROR]`, new Date().toISOString(), ...args);
  }
}

// 使用例
const authLogger = new DebugLogger('auth');
authLogger.log('Starting authentication flow');
authLogger.log('State:', state);
authLogger.log('Code verifier length:', codeVerifier.length);
```

### HTTPリクエストの監視

```typescript
// lib/http-debug.ts
export function debugHttpRequest(req: Request) {
  console.log('=== HTTP Request ===');
  console.log('URL:', req.url);
  console.log('Method:', req.method);
  console.log('Headers:', Object.fromEntries(req.headers.entries()));
  if (req.body) {
    req.clone().text().then(body => {
      console.log('Body:', body);
    });
  }
}

export function debugHttpResponse(res: Response, body?: any) {
  console.log('=== HTTP Response ===');
  console.log('Status:', res.status, res.statusText);
  console.log('Headers:', Object.fromEntries(res.headers.entries()));
  if (body) {
    console.log('Body:', JSON.stringify(body, null, 2));
  }
}
```

### Redisデバッグ

```bash
# Redis Monitor モード（すべてのコマンドを表示）
redis-cli monitor

# 特定のパターンのキーを確認
redis-cli --scan --pattern "session:*" | head -10

# キーの内容を確認
redis-cli get "session:your-session-id"

# TTLを確認
redis-cli ttl "auth:your-auth-id"
```

## 8. パフォーマンス問題

### 問題: レスポンスが遅い

**診断方法**
```typescript
// パフォーマンス測定
async function measurePerformance<T>(
  name: string,
  operation: () => Promise<T>
): Promise<T> {
  const start = performance.now();
  try {
    const result = await operation();
    const duration = performance.now() - start;
    console.log(`[PERF] ${name}: ${duration.toFixed(2)}ms`);
    return result;
  } catch (error) {
    const duration = performance.now() - start;
    console.error(`[PERF] ${name} failed: ${duration.toFixed(2)}ms`);
    throw error;
  }
}

// 使用例
const tokens = await measurePerformance('token-exchange', async () => {
  return await exchangeCodeForTokens(code, codeVerifier);
});
```

### Redis接続プールの最適化

```typescript
// 接続プール設定
const redis = new Redis({
  host: process.env.REDIS_HOST,
  port: parseInt(process.env.REDIS_PORT || '6379'),
  
  // パフォーマンス設定
  enableOfflineQueue: false,
  maxRetriesPerRequest: 3,
  connectTimeout: 5000,
  
  // 接続プール
  lazyConnect: true,
  keepAlive: 30000,
  
  // パフォーマンス最適化
  enableReadyCheck: false,
});
```

## 9. セキュリティ問題の診断

### セキュリティヘッダーの確認

```bash
# セキュリティヘッダーの確認
curl -I http://localhost:3000 | grep -E "(X-Frame-Options|X-Content-Type|Strict-Transport)"

# または Next.js のセキュリティヘッダー設定
```

```javascript
// next.config.js
const securityHeaders = [
  {
    key: 'X-Frame-Options',
    value: 'DENY',
  },
  {
    key: 'X-Content-Type-Options',
    value: 'nosniff',
  },
  {
    key: 'Referrer-Policy',
    value: 'strict-origin-when-cross-origin',
  },
];

module.exports = {
  async headers() {
    return [
      {
        source: '/:path*',
        headers: securityHeaders,
      },
    ];
  },
};
```

## 10. チェックリスト

問題が発生した場合の確認事項：

### 基本チェック
- [ ] Node.js バージョンは 18 以上か
- [ ] すべての依存関係がインストールされているか
- [ ] .env.local ファイルが存在するか
- [ ] 環境変数がすべて設定されているか

### サービスチェック
- [ ] Redis が起動しているか
- [ ] 認証サーバーが起動しているか
- [ ] ポートの競合はないか

### 設定チェック
- [ ] AUTH_REDIRECT_URI が正しいか
- [ ] Cookie のドメイン設定が正しいか
- [ ] CORS 設定が適切か

### デバッグ
- [ ] ブラウザの開発者ツールでエラーを確認
- [ ] ネットワークタブでリクエスト/レスポンスを確認
- [ ] アプリケーションタブで Cookie を確認
- [ ] サーバーログを確認

## サポート

解決しない場合は、以下の情報を添えて認証チームに連絡してください：

1. エラーメッセージの全文
2. 実行した操作の手順
3. 関連するログ
4. 環境情報（OS、Node.jsバージョンなど）