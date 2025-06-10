# BFF-Web 実装者向けシーケンス図集

## 概要

このドキュメントは、BFF-Web実装者が認証フローを視覚的に理解するためのシーケンス図集です。

## 1. 基本認証フロー（プロバイダー選択あり）

```mermaid
sequenceDiagram
    autonumber
    participant U as User<br/>(Browser)
    participant B as BFF-Web<br/>(Next.js)
    participant R as Redis
    participant A as Auth Server<br/>(Django)
    participant P as Provider<br/>(Google/LINE/GitHub)
    
    Note over U,P: 【初回ログイン - プロバイダー選択フロー】
    
    U->>B: GET /dashboard（未認証）
    B->>R: Check session
    R-->>B: No session
    B->>U: 302 Redirect to /login
    
    U->>B: POST /api/auth/login
    Note over B: provider_hint = null（未指定）
    B->>B: Generate:<br/>- sessionId (UUID)<br/>- state (16+ bytes)<br/>- code_verifier (43+ chars)<br/>- code_challenge (SHA256)
    B->>R: SET bff:auth:{sessionId}<br/>TTL: 600s
    Note over R: {<br/>  state,<br/>  codeVerifier,<br/>  providerHint,<br/>  redirectAfter,<br/>  createdAt<br/>}
    B->>U: Set-Cookie: bff_auth_session={sessionId}<br/>Return: { redirectUrl, sessionId }
    
    U->>A: GET /oauth/authorize<br/>?response_type=code<br/>&client_id=bff-web<br/>&redirect_uri=...<br/>&state={state}<br/>&code_challenge={challenge}<br/>&code_challenge_method=S256
    A->>A: Validate params<br/>Store PKCE
    A->>U: 200 Provider Selection Page
    
    rect rgba(200, 200, 255, 0.3)
        Note over U,A: プロバイダー選択画面
        U->>A: Select Provider (e.g., LINE)
        A->>U: 302 Redirect to LINE
    end
    
    U->>P: Provider Auth Page
    U->>P: Enter Credentials
    P->>A: Callback with provider code
    A->>P: Exchange for tokens
    P-->>A: User info
    A->>A: Create/Update user<br/>Generate auth code
    A->>U: 302 Redirect to BFF<br/>?code={AUTH_CODE}&state={state}
    
    U->>B: GET /api/auth/callback<br/>?code={AUTH_CODE}&state={state}
    B->>R: GET bff:auth:{sessionId}
    R-->>B: { state, codeVerifier }
    B->>B: Verify state match
    B->>A: POST /oauth/token/<br/>Content-Type: form-urlencoded<br/>{<br/>  grant_type: "authorization_code",<br/>  code: AUTH_CODE,<br/>  code_verifier: codeVerifier<br/>}
    A->>A: Verify PKCE
    A-->>B: {<br/>  access_token (JWT),<br/>  refresh_token,<br/>  expires_in<br/>}
    
    B->>B: Decode JWT to extract:<br/>{ id, email, name, provider }
    B->>B: Generate new sessionId
    B->>R: SET bff:session:{sessionId}<br/>TTL: 7 days
    Note over R: {<br/>  userId,<br/>  accessToken,<br/>  refreshToken,<br/>  expiresAt,<br/>  user: { id, email, name, provider },<br/>  provider<br/>}
    B->>R: DEL bff:auth:{old_sessionId}
    B->>U: Set-Cookie: bff_session={sessionId}<br/>Delete-Cookie: bff_auth_session<br/>302 Redirect to /dashboard
```

## 2. 直接プロバイダー指定フロー（高速ログイン）

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant B as BFF-Web
    participant R as Redis
    participant A as Auth Server
    participant G as Google
    
    Note over U,G: 【プロバイダー直接指定フロー】
    
    U->>B: POST /api/auth/login<br/>{ provider_hint: "google" }
    B->>B: Generate PKCE & state
    B->>R: Save auth session
    B->>U: { redirectUrl }
    
    U->>A: GET /oauth/authorize<br/>?provider_hint=google&...
    Note over A: Skip provider selection
    A->>U: 302 Direct to Google
    
    U->>G: Google Auth
    G->>A: Callback
    A->>U: 302 to BFF with code
    
    U->>B: Callback processing
    B->>A: Token exchange
    A-->>B: Tokens
    B->>R: Create session
    B->>U: Set session cookie
```

## 3. トークンリフレッシュフロー

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant B as BFF-Web
    participant R as Redis
    participant A as Auth Server
    
    Note over U,A: 【アクセストークン期限切れ時の自動更新】
    
    U->>B: GET /api/protected/data<br/>Cookie: bff_session=xxx
    B->>R: GET session:xxx
    R-->>B: Session data
    B->>B: Check token expiry
    Note over B: Token expired!
    
    rect rgba(255, 200, 200, 0.3)
        Note over B,A: 自動リフレッシュ処理
        B->>A: POST /oauth/token<br/>{<br/>  grant_type: "refresh_token",<br/>  refresh_token: "..."<br/>}
        A->>A: Validate refresh token
        A-->>B: {<br/>  access_token: "new_token",<br/>  expires_in: 3600<br/>}
        B->>R: Update session<br/>New access_token
    end
    
    B->>B: Continue with request
    B-->>U: Protected data
```

## 4. ログアウトフロー

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant B as BFF-Web
    participant R as Redis
    participant A as Auth Server
    
    Note over U,A: 【完全ログアウト処理】
    
    U->>B: POST /api/auth/logout
    B->>R: GET session:xxx
    R-->>B: { refreshToken, ... }
    
    par Token Revocation
        B->>A: POST /oauth/api/tokens/revoke/<br/>{ token: refreshToken }
        A->>A: Blacklist tokens
        A-->>B: 200 OK
    and Session Cleanup
        B->>R: DEL session:xxx
        B->>R: SADD logout:user123
    end
    
    B->>U: Clear-Site-Data: "cache", "cookies"<br/>Delete-Cookie: bff_session<br/>200 { success: true }
    
    U->>U: Clear local storage<br/>Redirect to /login
```

## 5. エラーハンドリングフロー

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant B as BFF-Web
    participant R as Redis
    participant A as Auth Server
    
    Note over U,A: 【各種エラーケースの処理】
    
    alt Invalid State
        U->>B: GET /api/auth/callback<br/>?code=xxx&state=invalid
        B->>R: GET bff:auth:sessionId
        R-->>B: { state: "different" }
        B->>U: 400 { error: "invalid_state" }
    else Code Exchange Failure
        U->>B: GET /api/auth/callback<br/>?code=expired&state=valid
        B->>A: POST /oauth/token
        A-->>B: 400 { error: "invalid_grant" }
        B->>U: Redirect to /login?error=expired
    else User Cancelled
        U->>B: GET /api/auth/callback<br/>?error=access_denied
        B->>U: Redirect to /login?error=cancelled
    else Network Error
        U->>B: GET /api/auth/callback
        B->>A: POST /oauth/token
        Note over B,A: Network timeout
        B->>U: 500 { error: "network_error" }<br/>Retry available
    end
```

## 6. セッション管理詳細

```mermaid
sequenceDiagram
    participant B as BFF-Web
    participant R as Redis
    
    Note over B,R: 【Redis キー構造と管理】
    
    Note over R: bff:auth:{uuid}<br/>一時認証セッション<br/>TTL: 10分
    Note over R: bff:session:{uuid}<br/>ユーザーセッション<br/>TTL: 7日
    Note over R: bff:blacklist:{jti}<br/>無効化トークン<br/>TTL: トークン有効期限まで
    Note over R: bff:rate:{ip}<br/>レート制限カウンター<br/>TTL: 1分
    
    B->>R: SET bff:auth:123 EX 600<br/>{ state, codeVerifier, providerHint }
    B->>R: SET bff:session:456 EX 604800<br/>{ userId, tokens, user }
    B->>R: INCR bff:rate:192.168.1.1<br/>EXPIRE bff:rate:192.168.1.1 60
```

## 7. 実装チェックポイント

### 各ステップでの確認事項

```mermaid
graph TD
    A[1. Login Start] -->|Check| A1[sessionId生成?<br/>state生成?<br/>PKCE生成?]
    A --> B[2. Auth Redirect]
    B -->|Check| B1[全パラメータ含む?<br/>Cookie設定?]
    B --> C[3. Callback]
    C -->|Check| C1[state検証?<br/>エラーチェック?]
    C --> D[4. Token Exchange]
    D -->|Check| D1[code_verifier送信?<br/>client認証?]
    D --> E[5. Session Create]
    E -->|Check| E1[新sessionId?<br/>適切なTTL?<br/>古いauth削除?]
    E --> F[6. Cookie Set]
    F -->|Check| F1[HttpOnly?<br/>Secure?<br/>SameSite?]
```

## 実装の重要ポイント

### 1. PKCE 実装の詳細

```typescript
// 正しい実装
const verifier = crypto.randomBytes(32).toString('base64url'); // 43文字以上
const challenge = crypto
  .createHash('sha256')
  .update(verifier)
  .digest('base64url'); // パディング除去

// よくある間違い
// ❌ base64（パディングあり）
// ❌ 短すぎるverifier（32文字未満）
// ❌ verifierの使い回し
```

### 2. State 管理

```typescript
// セッションごとにユニーク
const state = crypto.randomBytes(16).toString('base64url');

// Redis保存時は有効期限必須
await redis.setex(`bff:auth:${sessionId}`, 600, JSON.stringify({
  state,
  codeVerifier,
  providerHint,
  redirectAfter,
  createdAt: new Date().toISOString()
}));
```

### 3. エラー時のクリーンアップ

```typescript
// エラー発生時も必ずクリーンアップ
try {
  // 認証処理
} catch (error) {
  // 一時セッション削除
  await redis.del(`bff:auth:${sessionId}`);
  // エラーレスポンス
} finally {
  // 一時Cookieは必ず削除
  response.cookies.delete('bff_auth_session');
}
```

## まとめ

これらのシーケンス図は、BFF-Web実装の各フェーズで参照してください：

1. **設計フェーズ**: 全体フローの理解
2. **実装フェーズ**: 各ステップの詳細確認
3. **テストフェーズ**: エラーケースの網羅
4. **レビューフェーズ**: セキュリティチェック

特に重要なのは：
- **状態管理**: Redis キーの適切な管理
- **エラー処理**: 全エラーケースの考慮
- **セキュリティ**: PKCE、state、Cookie設定の正確な実装