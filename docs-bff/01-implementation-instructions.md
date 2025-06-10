# BFF-Web (Next.js) 実装指示書

## 概要

このドキュメントは、認証サーバー（Django）と連携するBFF-Web（Next.js）の実装者向けの指示書です。

## 1. BFF-Webの役割と責任

### 主な役割
1. **認証プロキシ**: ブラウザと認証サーバー間の仲介
2. **セッション管理**: JWTトークンをサーバー側で安全に管理
3. **PKCE実装**: OAuth 2.0のセキュリティ強化
4. **プロバイダー選択**: 必要に応じてUI提供

### 責任範囲
- ❌ **ユーザー認証を行わない**（認証サーバーの責任）
- ❌ **ユーザー情報を永続化しない**（認証サーバーの責任）
- ✅ **セッション管理**（Redis使用）
- ✅ **トークンのセキュアな保管**
- ✅ **認証状態の維持**

## 2. 必須実装エンドポイント

### 2.1 認証開始: `POST /api/auth/login` / `GET /api/auth/login`

```typescript
// リクエストパラメータ
interface LoginRequest {
  provider_hint?: 'google' | 'line' | 'github' | 'email' | null;
  login_hint?: string;     // メールアドレス（オプション）
  redirect_after?: string; // 認証後のリダイレクト先
  identifier?: string;     // ユーザー識別子（メールアドレス等）
  password?: string;       // パスワード（email認証の場合）
}

// レスポンス
interface LoginResponse {
  redirectUrl: string;     // 認証サーバーへのリダイレクトURL
  sessionId: string;       // 認証セッションID（Cookie設定用）
}

// 実装要件
1. PKCE code_verifier を生成（43-128文字）
2. code_challenge を計算（SHA256 + Base64URL）
3. state を生成（CSRF対策、最低16バイト）
4. Redis に保存:
   - key: `bff:auth:{sessionId}`
   - value: { state, codeVerifier, providerHint, loginHint, redirectAfter, createdAt }
   - TTL: 600秒（10分）
5. HttpOnly Cookie設定: `bff_auth_session={sessionId}`
6. 認証サーバーURLを構築:
   - base: `${AUTH_SERVER_PUBLIC_URL || AUTH_SERVER_URL}/oauth/authorize`
   - params: {
       response_type: 'code',
       client_id: CLIENT_ID,
       redirect_uri: REDIRECT_URI,
       state: state,
       code_challenge: challenge,
       code_challenge_method: 'S256',
       scope: 'openid profile email',
       provider_hint?: provider_hint,
       login_hint?: login_hint
     }
7. POSTの場合: JSONレスポンス（JS Redirect用）
8. GETの場合: 中間ページ経由でサーバーサイドリダイレクト
```

### 2.2 コールバック処理: `GET /api/auth/callback`

```typescript
// クエリパラメータ
interface CallbackQuery {
  code?: string;
  state?: string;
  error?: string;
  error_description?: string;
}

// 実装要件
1. エラーチェック:
   - error パラメータが存在 → エラーページへ
   - code または state が欠落 → エラー
2. state 検証:
   - Redis から `auth:{sessionId}` を取得
   - state が一致しない → エラー
3. トークン交換:
   - POST `${AUTH_SERVER_URL}/oauth/token/`
   - Content-Type: 'application/x-www-form-urlencoded'
   - body (URLSearchParams): {
       grant_type: 'authorization_code',
       code: code,
       redirect_uri: REDIRECT_URI,
       client_id: CLIENT_ID,
       client_secret: CLIENT_SECRET,
       code_verifier: savedCodeVerifier
     }
4. JWTアクセストークンからユーザー情報を抽出:
   - JWTペイロードをデコード（Base64URL）
   - ユーザー情報を抽出: { id, email, name, provider }
5. セッション作成:
   - key: `bff:session:{newSessionId}`
   - value: {
       userId,
       accessToken,
       refreshToken,
       expiresAt,
       user: { id, email, name, provider },
       provider,
       createdAt
     }
   - TTL: 7日間（SESSION_EXPIRY環境変数）
6. セキュアCookie設定とリダイレクト:
   - HttpOnly, Secure（本番）, SameSite=Lax, Path=/, Max-Age=7日間
```

### 2.3 トークンリフレッシュ: `POST /api/auth/refresh`

```typescript
// 実装要件
1. 現在のセッションを確認
2. refreshToken を使用してトークン更新
3. 新しいトークンでセッション更新
4. エラー時は401を返す
```

### 2.4 ログアウト: `POST /api/auth/logout`

```typescript
// 実装要件
1. セッションデータを取得
2. 認証サーバーでトークン無効化:
   - POST `${AUTH_SERVER_URL}/oauth/api/tokens/revoke/`
3. Redisセッション削除
4. Cookie削除
5. Clear-Site-Dataヘッダー設定
```

### 2.5 認証状態確認: `GET /api/auth/me`

```typescript
// レスポンス
interface AuthStatusResponse {
  authenticated: boolean;
  user?: {
    id: string;
    email: string;
    name: string;
    provider: string;
    linkedAccounts: string[];
  };
}
```

## 3. セキュリティ要件

### 3.1 必須セキュリティ設定

```typescript
// Cookie設定
const COOKIE_OPTIONS = {
  httpOnly: true,           // 必須
  secure: true,             // 本番環境で必須
  sameSite: 'strict',       // CSRF対策
  maxAge: 3600 * 24 * 7,    // 7日間
  path: '/'
};

// セキュリティヘッダー
const SECURITY_HEADERS = {
  'X-Frame-Options': 'DENY',
  'X-Content-Type-Options': 'nosniff',
  'X-XSS-Protection': '1; mode=block',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Content-Security-Policy': "default-src 'self'"
};
```

### 3.2 Redis セキュリティ

```typescript
// Redis接続設定
const redisConfig = {
  host: process.env.REDIS_HOST || 'localhost',
  port: parseInt(process.env.REDIS_PORT || '6379'),
  password: process.env.REDIS_PASSWORD,  // 本番環境で必須
  tls: process.env.NODE_ENV === 'production' ? {} : undefined,
  db: parseInt(process.env.REDIS_DB || '0'),
  // keyPrefixは個別に設定
};

// キー命名規則
const REDIS_KEYS = {
  authSession: (id: string) => `bff:auth:${id}`,
  userSession: (id: string) => `bff:session:${id}`,
  blacklist: (token: string) => `bff:blacklist:${token}`,
  rateLimit: (ip: string) => `bff:rate:${ip}`,
};
```

## 4. エラーハンドリング

### 4.1 エラーレスポンス形式

```typescript
interface ErrorResponse {
  error: string;           // エラーコード
  error_description?: string;  // 詳細説明
  error_uri?: string;      // エラー情報URL
  state?: string;          // 元のstate（該当する場合）
}

// エラーコード
const ERROR_CODES = {
  INVALID_REQUEST: 'invalid_request',
  UNAUTHORIZED: 'unauthorized',
  ACCESS_DENIED: 'access_denied',
  INVALID_STATE: 'invalid_state',
  SERVER_ERROR: 'server_error',
  TEMPORARILY_UNAVAILABLE: 'temporarily_unavailable',
};
```

### 4.2 エラー処理フロー

```typescript
// 認証エラーの統一処理
function handleAuthError(error: any): ErrorResponse {
  // ログ記録（センシティブ情報は除外）
  logger.error('Auth error', {
    error: error.code,
    timestamp: new Date().toISOString(),
    // トークンやパスワードは記録しない
  });
  
  // ユーザー向けエラーメッセージ
  const userMessage = getUserFriendlyMessage(error);
  
  return {
    error: error.code || 'unknown_error',
    error_description: userMessage,
  };
}
```

## 5. 認証サーバーAPI仕様

### 5.1 エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|-------------|---------|------|
| `/oauth/authorize` | GET | 認証開始 |
| `/oauth/token/` | POST | トークン交換/更新 |
| `/oauth/api/tokens/revoke/` | POST | トークン無効化 |
| `/.well-known/jwks.json` | GET | 公開鍵取得 |

### 5.2 認可リクエスト

```http
GET /oauth/authorize?
  response_type=code&
  client_id={CLIENT_ID}&
  redirect_uri={REDIRECT_URI}&
  state={STATE}&
  code_challenge={CODE_CHALLENGE}&
  code_challenge_method=S256&
  scope=openid+profile+email&
  provider_hint={PROVIDER}&  # オプション
  login_hint={EMAIL}         # オプション
```

### 5.3 トークン交換

```http
POST /oauth/token
Content-Type: application/json

{
  "grant_type": "authorization_code",
  "code": "{AUTHORIZATION_CODE}",
  "redirect_uri": "{REDIRECT_URI}",
  "client_id": "{CLIENT_ID}",
  "client_secret": "{CLIENT_SECRET}",
  "code_verifier": "{CODE_VERIFIER}"
}

# レスポンス
{
  "access_token": "eyJ...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "...",
  "scope": "openid profile email",
  "id_token": "eyJ...",  # OpenID Connect
  "user": {
    "id": "123",
    "email": "user@example.com",
    "name": "User Name",
    "provider": "google"
  }
}
```

## 6. テストシナリオ

### 6.1 正常系テスト

```typescript
describe('Authentication Flow', () => {
  test('Google認証フロー', async () => {
    // 1. ログイン開始
    const loginRes = await request(app)
      .post('/api/auth/login')
      .send({ provider_hint: 'google' });
    
    expect(loginRes.body.redirectUrl).toMatch(/\/oauth\/authorize/);
    
    // 2. コールバック処理
    const callbackRes = await request(app)
      .get('/api/auth/callback')
      .query({ code: 'test-code', state: 'test-state' });
    
    expect(callbackRes.status).toBe(302);
    expect(callbackRes.headers['set-cookie']).toBeDefined();
  });
});
```

### 6.2 異常系テスト

```typescript
describe('Error Handling', () => {
  test('無効なstate', async () => {
    const res = await request(app)
      .get('/api/auth/callback')
      .query({ code: 'test-code', state: 'invalid-state' });
    
    expect(res.status).toBe(400);
    expect(res.body.error).toBe('invalid_state');
  });
  
  test('トークン期限切れ', async () => {
    // 期限切れセッションでアクセス
    const res = await request(app)
      .get('/api/auth/me')
      .set('Cookie', 'session=expired-session');
    
    expect(res.status).toBe(401);
  });
});
```

## 7. 実装チェックリスト

### 必須実装
- [x] PKCE (code_verifier/challenge) の実装
- [x] state パラメータの生成と検証
- [x] Redis セッション管理
- [x] HttpOnly Cookie の設定
- [x] JWT トークンデコード機能
- [x] Docker内部ネットワーク対応
- [x] エラーハンドリング
- [x] セキュリティヘッダー
- [x] 中間ページ経由のCookie設定
- [ ] トークンリフレッシュ機能
- [ ] ログアウト時のトークン無効化

### 推奨実装
- [ ] レート制限
- [ ] ログ記録（センシティブ情報除外）
- [ ] ヘルスチェックエンドポイント
- [ ] メトリクス収集
- [ ] 全デバイスログアウト

### オプション
- [ ] プロバイダー選択UI
- [ ] アカウントリンク機能
- [ ] MFA対応
- [ ] セッション延長機能

## 8. トラブルシューティング

### よくある問題

1. **「invalid_state」エラー**
   - 原因: Redisセッションの有効期限切れ
   - 対策: TTLを適切に設定（最低10分）

2. **Cookie が設定されない**
   - 原因: SameSite属性の設定ミス
   - 対策: 開発環境では'lax'、本番は'strict'

3. **CORS エラー**
   - 原因: 認証サーバーのCORS設定
   - 対策: credentials: 'include' を設定

4. **トークンリフレッシュ失敗**
   - 原因: refresh_tokenの期限切れ
   - 対策: 再ログインを促す

## 9. サンプル実装

完全なサンプル実装は以下を参照:
- `/docs/implementation-guide-bff-oauth.md`
- GitHub: [サンプルリポジトリURL]

## 10. 連絡先

実装で不明な点がある場合:
- 認証サーバーチーム: auth-team@example.com
- ドキュメント: https://auth-docs.example.com
- Slack: #auth-support