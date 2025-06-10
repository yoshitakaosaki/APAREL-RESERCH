# BFF-Web セキュリティガイド

## 概要

このガイドでは、BFF-Web実装におけるセキュリティベストプラクティスを説明します。

## 1. PKCE (Proof Key for Code Exchange) の実装

### PKCEとは？

PKCEは、OAuth 2.0の認可コード横取り攻撃を防ぐセキュリティ拡張です。特にパブリッククライアント（SPAやモバイルアプリ）で重要ですが、BFFでも実装することでセキュリティを強化できます。

### 正しい実装

```typescript
// ✅ 正しいPKCE実装
import crypto from 'crypto';

export function generateCodeVerifier(): string {
  // RFC 7636: 43-128文字の[A-Z][a-z][0-9]-._~
  // 32バイト = 256ビット = base64urlで43文字
  return crypto.randomBytes(32).toString('base64url');
}

export function generateCodeChallenge(verifier: string): string {
  // S256 method: SHA256ハッシュのbase64url
  return crypto
    .createHash('sha256')
    .update(verifier)
    .digest('base64url');
}
```

### よくある間違い

```typescript
// ❌ 短すぎるverifier
const verifier = crypto.randomBytes(16).toString('base64url'); // 22文字しかない

// ❌ base64（パディングあり）を使用
const verifier = crypto.randomBytes(32).toString('base64'); // = パディングが含まれる

// ❌ plainメソッドの使用（非推奨）
const challenge = verifier; // SHA256を使わない

// ❌ verifierの再利用
let globalVerifier = generateCodeVerifier(); // 使い回しは危険
```

### セキュリティ要件

1. **code_verifierの要件**
   - 最低43文字（推奨: 43-128文字）
   - 暗号学的に安全な乱数を使用
   - 各認証フローで新規生成

2. **code_challengeの要件**
   - S256メソッド（SHA256）を使用
   - plainメソッドは使用しない
   - base64urlエンコーディング（パディングなし）

## 2. State パラメータによるCSRF対策

### Stateの役割

stateパラメータは、認証リクエストとコールバックを紐付けることで、CSRF攻撃を防ぎます。

### 実装例

```typescript
// ✅ セキュアなstate生成
export function generateState(): string {
  // 最低128ビット（16バイト）の乱数
  return crypto.randomBytes(16).toString('base64url');
}

// 認証フローでの使用
const state = generateState();
await redis.setex(`auth:${sessionId}`, 600, JSON.stringify({
  state,
  codeVerifier,
  timestamp: Date.now(),
}));

// コールバックでの検証
if (savedState !== receivedState) {
  throw new Error('Invalid state - possible CSRF attack');
}
```

### 追加のセキュリティ対策

```typescript
// タイムスタンプによる有効期限チェック
const MAX_AGE = 10 * 60 * 1000; // 10分
if (Date.now() - savedData.timestamp > MAX_AGE) {
  throw new Error('State expired');
}

// 使用済みstateの記録（リプレイ攻撃対策）
await redis.setex(`used_state:${state}`, 3600, '1');
if (await redis.exists(`used_state:${state}`)) {
  throw new Error('State already used');
}
```

## 3. セキュアなCookie設定

### 推奨Cookie設定

```typescript
// ✅ セキュアなCookie設定
const SECURE_COOKIE_OPTIONS = {
  httpOnly: true,        // XSS対策: JavaScriptからアクセス不可
  secure: true,          // HTTPS必須
  sameSite: 'strict',    // CSRF対策: 厳格な同一サイト制限
  path: '/',             // パス制限
  maxAge: 60 * 60 * 24 * 7, // 有効期限: 7日
};

// 環境別の設定
const getCookieOptions = () => {
  const base = {
    httpOnly: true,
    path: '/',
    maxAge: 60 * 60 * 24 * 7,
  };

  if (process.env.NODE_ENV === 'production') {
    return {
      ...base,
      secure: true,
      sameSite: 'strict' as const,
      domain: '.example.com', // サブドメイン共有時
    };
  }

  // 開発環境
  return {
    ...base,
    secure: false,      // HTTPで動作
    sameSite: 'lax' as const, // 開発時は緩和
  };
};
```

### Cookie名の命名規則

```typescript
// ✅ 推奨される命名
const COOKIE_NAMES = {
  session: '__Host-session',      // __Host- プレフィックス
  auth: '__Secure-auth_session',  // __Secure- プレフィックス
};

// __Host- プレフィックスの要件:
// - secure: true
// - path: '/'
// - domain属性なし

// __Secure- プレフィックスの要件:
// - secure: true
```

## 4. トークン管理のセキュリティ

### トークンの保存

```typescript
// ✅ サーバーサイドでのトークン管理
// Redisに暗号化して保存
import crypto from 'crypto';

const ENCRYPTION_KEY = process.env.SESSION_ENCRYPTION_KEY!;

function encrypt(text: string): string {
  const iv = crypto.randomBytes(16);
  const cipher = crypto.createCipheriv(
    'aes-256-gcm',
    Buffer.from(ENCRYPTION_KEY, 'hex'),
    iv
  );
  
  let encrypted = cipher.update(text, 'utf8', 'hex');
  encrypted += cipher.final('hex');
  
  const authTag = cipher.getAuthTag();
  
  return iv.toString('hex') + ':' + authTag.toString('hex') + ':' + encrypted;
}

function decrypt(text: string): string {
  const parts = text.split(':');
  const iv = Buffer.from(parts[0], 'hex');
  const authTag = Buffer.from(parts[1], 'hex');
  const encrypted = parts[2];
  
  const decipher = crypto.createDecipheriv(
    'aes-256-gcm',
    Buffer.from(ENCRYPTION_KEY, 'hex'),
    iv
  );
  
  decipher.setAuthTag(authTag);
  
  let decrypted = decipher.update(encrypted, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  
  return decrypted;
}

// 使用例
await redis.setex(
  `session:${sessionId}`,
  ttl,
  encrypt(JSON.stringify(sessionData))
);
```

### トークンのローテーション

```typescript
// ✅ リフレッシュ時のトークンローテーション
async function rotateTokens(oldRefreshToken: string): Promise<TokenPair> {
  // 古いトークンをブラックリストに追加
  await redis.setex(
    `blacklist:${oldRefreshToken}`,
    60 * 60 * 24 * 7, // 元の有効期限まで保持
    '1'
  );
  
  // 新しいトークンを取得
  const newTokens = await exchangeRefreshToken(oldRefreshToken);
  
  // 使用済みトークンの再利用を検知
  if (await redis.exists(`blacklist:${oldRefreshToken}`)) {
    // セキュリティ侵害の可能性
    await revokeAllUserTokens(userId);
    throw new Error('Token reuse detected');
  }
  
  return newTokens;
}
```

## 5. ネットワークセキュリティ

### HTTPSの強制

```typescript
// ✅ HTTPS強制のミドルウェア
export function httpsRedirect(req: NextRequest): NextResponse | null {
  if (
    process.env.NODE_ENV === 'production' &&
    req.headers.get('x-forwarded-proto') !== 'https'
  ) {
    return NextResponse.redirect(
      `https://${req.headers.get('host')}${req.nextUrl.pathname}${req.nextUrl.search}`,
      301
    );
  }
  return null;
}
```

### セキュリティヘッダー

```typescript
// ✅ 推奨セキュリティヘッダー
export function addSecurityHeaders(response: NextResponse): NextResponse {
  // XSS対策
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  
  // コンテンツセキュリティポリシー
  response.headers.set(
    'Content-Security-Policy',
    "default-src 'self'; " +
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; " +
    "style-src 'self' 'unsafe-inline'; " +
    "img-src 'self' data: https:; " +
    "font-src 'self'; " +
    "connect-src 'self' https://auth.example.com; " +
    "frame-ancestors 'none';"
  );
  
  // HTTPS強制
  response.headers.set(
    'Strict-Transport-Security',
    'max-age=31536000; includeSubDomains; preload'
  );
  
  // リファラーポリシー
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  
  // 権限ポリシー
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=(), interest-cohort=()'
  );
  
  return response;
}
```

## 6. 入力検証とサニタイゼーション

### パラメータ検証

```typescript
// ✅ 厳格な入力検証
import { z } from 'zod';

const LoginRequestSchema = z.object({
  provider_hint: z.enum(['google', 'github', 'line', 'email']).optional(),
  login_hint: z.string().email().optional(),
  redirect_after: z.string().regex(/^\/[a-zA-Z0-9\-_\/]*$/).optional(),
});

export async function validateLoginRequest(body: unknown) {
  try {
    return LoginRequestSchema.parse(body);
  } catch (error) {
    throw new Error('Invalid request parameters');
  }
}

// URLパラメータの検証
const CallbackParamsSchema = z.object({
  code: z.string().regex(/^[a-zA-Z0-9\-_]+$/),
  state: z.string().regex(/^[a-zA-Z0-9\-_]+$/),
  error: z.string().optional(),
  error_description: z.string().optional(),
});
```

### XSS対策

```typescript
// ✅ HTMLエスケープ
function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#x27;',
    '/': '&#x2F;',
  };
  return text.replace(/[&<>"'/]/g, (char) => map[char]);
}

// エラーメッセージの表示
const safeErrorMessage = escapeHtml(error.message);
```

## 7. レート制限

### 実装例

```typescript
// ✅ IPベースのレート制限
const RATE_LIMIT = {
  windowMs: 60 * 1000, // 1分
  maxRequests: 10,      // 最大10リクエスト
};

export async function checkRateLimit(ip: string): Promise<boolean> {
  const key = `rate_limit:${ip}`;
  const current = await redis.incr(key);
  
  if (current === 1) {
    await redis.expire(key, RATE_LIMIT.windowMs / 1000);
  }
  
  if (current > RATE_LIMIT.maxRequests) {
    return false;
  }
  
  return true;
}

// ミドルウェアでの使用
export async function rateLimitMiddleware(req: NextRequest) {
  const ip = req.headers.get('x-forwarded-for') || 'unknown';
  
  if (!(await checkRateLimit(ip))) {
    return NextResponse.json(
      { error: 'Too many requests' },
      { 
        status: 429,
        headers: {
          'Retry-After': '60',
        },
      }
    );
  }
}
```

## 8. ログとモニタリング

### セキュアなログ記録

```typescript
// ✅ センシティブ情報を除外したログ
interface LogData {
  level: 'info' | 'warn' | 'error';
  message: string;
  context?: Record<string, any>;
}

function sanitizeLogData(data: any): any {
  const sensitive = [
    'password',
    'token',
    'access_token',
    'refresh_token',
    'code_verifier',
    'client_secret',
  ];
  
  if (typeof data !== 'object' || data === null) {
    return data;
  }
  
  const sanitized: any = Array.isArray(data) ? [] : {};
  
  for (const [key, value] of Object.entries(data)) {
    if (sensitive.some(s => key.toLowerCase().includes(s))) {
      sanitized[key] = '[REDACTED]';
    } else if (typeof value === 'object') {
      sanitized[key] = sanitizeLogData(value);
    } else {
      sanitized[key] = value;
    }
  }
  
  return sanitized;
}

export function secureLog(data: LogData) {
  console.log(JSON.stringify({
    ...data,
    context: sanitizeLogData(data.context),
    timestamp: new Date().toISOString(),
  }));
}
```

### セキュリティイベントの監視

```typescript
// ✅ セキュリティイベントの記録
async function logSecurityEvent(event: {
  type: 'login_attempt' | 'token_refresh' | 'logout' | 'suspicious_activity';
  userId?: string;
  ip: string;
  userAgent: string;
  success: boolean;
  details?: any;
}) {
  await redis.lpush(
    'security_events',
    JSON.stringify({
      ...event,
      timestamp: new Date().toISOString(),
    })
  );
  
  // 不審なアクティビティの検出
  if (event.type === 'suspicious_activity' || !event.success) {
    await alertSecurityTeam(event);
  }
}
```

## 9. 環境変数の管理

### セキュアな環境変数

```bash
# .env.local.example
# 機密情報は含めない

# 必須の環境変数
AUTH_SERVER_URL=
AUTH_CLIENT_ID=
AUTH_CLIENT_SECRET= # 本番環境では環境変数管理サービスを使用
AUTH_REDIRECT_URI=

# セッション暗号化キー（32バイトの16進数）
SESSION_ENCRYPTION_KEY= # openssl rand -hex 32

# Redis
REDIS_URL=
REDIS_PASSWORD= # 本番環境では必須

# セッション設定
SESSION_SECRET= # openssl rand -base64 32
SESSION_COOKIE_NAME=__Host-session
```

### 環境変数の検証

```typescript
// ✅ 起動時の環境変数チェック
const requiredEnvVars = [
  'AUTH_SERVER_URL',
  'AUTH_CLIENT_ID',
  'AUTH_CLIENT_SECRET',
  'SESSION_ENCRYPTION_KEY',
  'REDIS_URL',
];

export function validateEnvironment() {
  const missing = requiredEnvVars.filter(
    (key) => !process.env[key]
  );
  
  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missing.join(', ')}`
    );
  }
  
  // 形式チェック
  if (!/^[0-9a-f]{64}$/i.test(process.env.SESSION_ENCRYPTION_KEY!)) {
    throw new Error('SESSION_ENCRYPTION_KEY must be 32 bytes hex');
  }
}
```

## 10. セキュリティチェックリスト

### 実装時の確認事項

- [ ] PKCEを正しく実装している（43文字以上のverifier）
- [ ] stateパラメータで CSRF対策をしている
- [ ] Cookieに適切なセキュリティ属性を設定している
- [ ] トークンをサーバーサイドのみで管理している
- [ ] HTTPS を強制している（本番環境）
- [ ] セキュリティヘッダーを設定している
- [ ] 入力値を検証している
- [ ] レート制限を実装している
- [ ] センシティブ情報をログに出力していない
- [ ] 環境変数を安全に管理している

### デプロイ前の確認事項

- [ ] 本番環境の環境変数を設定した
- [ ] Redisにパスワードを設定した
- [ ] HTTPSが有効になっている
- [ ] セキュリティヘッダーが正しく設定されている
- [ ] エラーメッセージが詳細すぎない
- [ ] ログにセンシティブ情報が含まれていない
- [ ] レート制限が適切に機能している
- [ ] セッション有効期限が適切である

## まとめ

BFF-Webのセキュリティは、複数の層で実装する必要があります：

1. **認証フロー**: PKCE、state による保護
2. **データ保護**: 暗号化、セキュアなCookie
3. **ネットワーク**: HTTPS、セキュリティヘッダー
4. **アプリケーション**: 入力検証、レート制限
5. **運用**: ログ監視、インシデント対応

これらすべてを適切に実装することで、安全な認証システムを構築できます。