# テストガイド

## 概要

BFF-Web の包括的なテスト戦略とテストコードの実装方法を説明します。

## 1. テスト環境のセットアップ

### 必要なパッケージのインストール

```bash
# テスト関連パッケージ
npm install --save-dev \
  jest \
  @testing-library/react \
  @testing-library/jest-dom \
  @testing-library/user-event \
  jest-environment-jsdom \
  @types/jest \
  ts-jest \
  msw \
  supertest \
  @types/supertest
```

### Jest 設定 (jest.config.js)

```javascript
const nextJest = require('next/jest');

const createJestConfig = nextJest({
  dir: './',
});

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/$1',
  },
  testEnvironment: 'jest-environment-jsdom',
  testPathIgnorePatterns: ['/node_modules/', '/.next/'],
  collectCoverageFrom: [
    'app/**/*.{js,jsx,ts,tsx}',
    'lib/**/*.{js,jsx,ts,tsx}',
    '!**/*.d.ts',
    '!**/node_modules/**',
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
};

module.exports = createJestConfig(customJestConfig);
```

### Jest セットアップ (jest.setup.js)

```javascript
import '@testing-library/jest-dom';

// グローバルモック
global.fetch = jest.fn();

// 環境変数のモック
process.env = {
  ...process.env,
  AUTH_SERVER_URL: 'http://localhost:8000',
  AUTH_CLIENT_ID: 'test-client',
  AUTH_CLIENT_SECRET: 'test-secret',
  AUTH_REDIRECT_URI: 'http://localhost:3000/api/auth/callback',
  SESSION_COOKIE_NAME: 'test_session',
  REDIS_URL: 'redis://localhost:6379',
};

// Redis モック
jest.mock('ioredis', () => {
  const Redis = jest.fn(() => ({
    get: jest.fn(),
    set: jest.fn(),
    setex: jest.fn(),
    del: jest.fn(),
    exists: jest.fn(),
    incr: jest.fn(),
    expire: jest.fn(),
    ttl: jest.fn(),
    ping: jest.fn().mockResolvedValue('PONG'),
    on: jest.fn(),
  }));
  return Redis;
});
```

## 2. ユニットテスト

### PKCE ユーティリティのテスト

```typescript
// lib/auth/__tests__/pkce.test.ts
import { generateCodeVerifier, generateCodeChallenge, generateState } from '../pkce';
import crypto from 'crypto';

describe('PKCE Utilities', () => {
  describe('generateCodeVerifier', () => {
    it('should generate a code verifier with correct length', () => {
      const verifier = generateCodeVerifier();
      expect(verifier).toHaveLength(43); // 32 bytes base64url = 43 chars
    });

    it('should generate unique verifiers', () => {
      const verifier1 = generateCodeVerifier();
      const verifier2 = generateCodeVerifier();
      expect(verifier1).not.toBe(verifier2);
    });

    it('should only contain base64url characters', () => {
      const verifier = generateCodeVerifier();
      expect(verifier).toMatch(/^[A-Za-z0-9\-_]+$/);
    });
  });

  describe('generateCodeChallenge', () => {
    it('should generate correct SHA256 challenge', () => {
      const verifier = 'test-verifier';
      const expectedChallenge = crypto
        .createHash('sha256')
        .update(verifier)
        .digest('base64url');
      
      const challenge = generateCodeChallenge(verifier);
      expect(challenge).toBe(expectedChallenge);
    });

    it('should generate consistent challenge for same verifier', () => {
      const verifier = 'consistent-verifier';
      const challenge1 = generateCodeChallenge(verifier);
      const challenge2 = generateCodeChallenge(verifier);
      expect(challenge1).toBe(challenge2);
    });
  });

  describe('generateState', () => {
    it('should generate state with minimum length', () => {
      const state = generateState();
      expect(state.length).toBeGreaterThanOrEqual(22); // 16 bytes base64url
    });

    it('should generate unique states', () => {
      const states = new Set();
      for (let i = 0; i < 100; i++) {
        states.add(generateState());
      }
      expect(states.size).toBe(100);
    });
  });
});
```

### セッション管理のテスト

```typescript
// lib/__tests__/redis.test.ts
import { createSession, getSession, updateSession, deleteSession } from '../redis';
import Redis from 'ioredis';

jest.mock('ioredis');

describe('Session Management', () => {
  let redis: jest.Mocked<Redis>;

  beforeEach(() => {
    redis = new Redis() as jest.Mocked<Redis>;
    jest.clearAllMocks();
  });

  describe('createSession', () => {
    it('should create session with correct TTL', async () => {
      const sessionId = 'test-session-id';
      const data = { user: 'test-user' };
      const ttl = 3600;

      await createSession(sessionId, data, ttl);

      expect(redis.setex).toHaveBeenCalledWith(
        `session:${sessionId}`,
        ttl,
        JSON.stringify(data)
      );
    });

    it('should use default TTL if not provided', async () => {
      const sessionId = 'test-session-id';
      const data = { user: 'test-user' };

      await createSession(sessionId, data);

      expect(redis.setex).toHaveBeenCalledWith(
        `session:${sessionId}`,
        3600 * 24 * 7, // 7 days
        JSON.stringify(data)
      );
    });
  });

  describe('getSession', () => {
    it('should return parsed session data', async () => {
      const sessionData = { user: 'test-user' };
      redis.get.mockResolvedValue(JSON.stringify(sessionData));

      const result = await getSession('test-session-id');

      expect(result).toEqual(sessionData);
      expect(redis.get).toHaveBeenCalledWith('session:test-session-id');
    });

    it('should return null for non-existent session', async () => {
      redis.get.mockResolvedValue(null);

      const result = await getSession('non-existent');

      expect(result).toBeNull();
    });
  });

  describe('deleteSession', () => {
    it('should delete session', async () => {
      await deleteSession('test-session-id');

      expect(redis.del).toHaveBeenCalledWith('session:test-session-id');
    });
  });
});
```

## 3. API ルートテスト

### ログインエンドポイントのテスト

```typescript
// app/api/auth/login/__tests__/route.test.ts
import { POST } from '../route';
import { NextRequest } from 'next/server';
import { generateCodeVerifier, generateCodeChallenge, generateState } from '@/lib/auth/pkce';
import * as redis from '@/lib/redis';

jest.mock('@/lib/auth/pkce');
jest.mock('@/lib/redis');

describe('POST /api/auth/login', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (generateState as jest.Mock).mockReturnValue('test-state');
    (generateCodeVerifier as jest.Mock).mockReturnValue('test-verifier');
    (generateCodeChallenge as jest.Mock).mockReturnValue('test-challenge');
  });

  it('should initiate login flow with provider hint', async () => {
    const request = new NextRequest('http://localhost:3000/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        provider_hint: 'google',
        redirect_after: '/dashboard',
      }),
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.redirectUrl).toContain('/oauth/authorize');
    expect(data.redirectUrl).toContain('provider_hint=google');
    expect(data.redirectUrl).toContain('code_challenge=test-challenge');
  });

  it('should set auth session cookie', async () => {
    const request = new NextRequest('http://localhost:3000/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({}),
    });

    const response = await POST(request);

    const setCookieHeader = response.headers.get('set-cookie');
    expect(setCookieHeader).toContain('auth_session_id=');
    expect(setCookieHeader).toContain('HttpOnly');
  });

  it('should handle errors gracefully', async () => {
    (redis.createAuthSession as jest.Mock).mockRejectedValue(new Error('Redis error'));

    const request = new NextRequest('http://localhost:3000/api/auth/login', {
      method: 'POST',
      body: JSON.stringify({}),
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(500);
    expect(data.error).toBe('server_error');
  });
});
```

### コールバックエンドポイントのテスト

```typescript
// app/api/auth/callback/__tests__/route.test.ts
import { GET } from '../route';
import { NextRequest } from 'next/server';
import * as redis from '@/lib/redis';

// モックの設定
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('GET /api/auth/callback', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should handle successful callback', async () => {
    // Redis モック
    (redis.getAuthSession as jest.Mock).mockResolvedValue({
      state: 'valid-state',
      codeVerifier: 'test-verifier',
      redirectAfter: '/dashboard',
    });

    // Token exchange モック
    mockFetch.mockResolvedValue({
      ok: true,
      json: async () => ({
        access_token: 'test-access-token',
        refresh_token: 'test-refresh-token',
        expires_in: 3600,
        user: {
          id: '123',
          email: 'test@example.com',
          name: 'Test User',
        },
      }),
    });

    const request = new NextRequest(
      'http://localhost:3000/api/auth/callback?code=test-code&state=valid-state',
      {
        headers: {
          cookie: 'auth_session_id=test-session-id',
        },
      }
    );

    const response = await GET(request);

    expect(response.status).toBe(307); // Redirect
    expect(response.headers.get('location')).toBe('/dashboard');
    expect(response.headers.get('set-cookie')).toContain('bff_session=');
  });

  it('should handle state mismatch', async () => {
    (redis.getAuthSession as jest.Mock).mockResolvedValue({
      state: 'different-state',
      codeVerifier: 'test-verifier',
    });

    const request = new NextRequest(
      'http://localhost:3000/api/auth/callback?code=test-code&state=invalid-state',
      {
        headers: {
          cookie: 'auth_session_id=test-session-id',
        },
      }
    );

    const response = await GET(request);

    expect(response.status).toBe(307);
    expect(response.headers.get('location')).toContain('/login?error=invalid_state');
  });

  it('should handle OAuth errors', async () => {
    const request = new NextRequest(
      'http://localhost:3000/api/auth/callback?error=access_denied&error_description=User+cancelled',
      {}
    );

    const response = await GET(request);

    expect(response.status).toBe(307);
    expect(response.headers.get('location')).toContain('/login?error=access_denied');
  });
});
```

## 4. 統合テスト

### 認証フロー全体のテスト

```typescript
// __tests__/integration/auth-flow.test.ts
import { createMocks } from 'node-mocks-http';
import { POST as loginPOST } from '@/app/api/auth/login/route';
import { GET as callbackGET } from '@/app/api/auth/callback/route';
import { GET as meGET } from '@/app/api/auth/me/route';

describe('Authentication Flow Integration', () => {
  it('should complete full authentication flow', async () => {
    // Step 1: Initiate login
    const loginReq = createMocks({
      method: 'POST',
      body: {
        provider_hint: 'google',
      },
    }).req;

    const loginRes = await loginPOST(loginReq);
    const loginData = await loginRes.json();
    
    expect(loginData.redirectUrl).toBeDefined();
    
    // Extract state and session from login response
    const authSessionId = loginRes.headers.get('set-cookie')
      ?.match(/auth_session_id=([^;]+)/)?.[1];
    
    expect(authSessionId).toBeDefined();

    // Step 2: Simulate OAuth callback
    // In real scenario, user would authenticate with provider
    const callbackReq = createMocks({
      method: 'GET',
      query: {
        code: 'mock-auth-code',
        state: 'mock-state', // This should match what was saved
      },
      headers: {
        cookie: `auth_session_id=${authSessionId}`,
      },
    }).req;

    const callbackRes = await callbackGET(callbackReq);
    
    expect(callbackRes.status).toBe(307);
    
    // Extract session cookie
    const sessionId = callbackRes.headers.get('set-cookie')
      ?.match(/bff_session=([^;]+)/)?.[1];
    
    expect(sessionId).toBeDefined();

    // Step 3: Check authentication status
    const meReq = createMocks({
      method: 'GET',
      headers: {
        cookie: `bff_session=${sessionId}`,
      },
    }).req;

    const meRes = await meGET(meReq);
    const meData = await meRes.json();
    
    expect(meData.authenticated).toBe(true);
    expect(meData.user).toBeDefined();
  });
});
```

## 5. E2E テスト

### Playwright を使用したE2Eテスト

```bash
npm install --save-dev @playwright/test
npx playwright install
```

```typescript
// e2e/auth.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Authentication E2E', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000');
  });

  test('should login with Google', async ({ page }) => {
    // Login page
    await page.goto('/login');
    
    // Click Google login button
    await page.click('button:has-text("Continue with Google")');
    
    // Mock OAuth provider response
    // In real E2E test, you would handle actual OAuth flow
    await page.route('**/oauth/authorize*', async route => {
      const url = new URL(route.request().url());
      const state = url.searchParams.get('state');
      const redirectUri = url.searchParams.get('redirect_uri');
      
      // Simulate successful OAuth callback
      await route.fulfill({
        status: 302,
        headers: {
          Location: `${redirectUri}?code=test-code&state=${state}`,
        },
      });
    });
    
    // Should be redirected to dashboard
    await expect(page).toHaveURL('/dashboard');
    
    // Should see user info
    await expect(page.locator('text=test@example.com')).toBeVisible();
  });

  test('should logout', async ({ page }) => {
    // Assume already logged in
    await page.goto('/dashboard');
    
    // Click logout button
    await page.click('button:has-text("Logout")');
    
    // Should be redirected to login
    await expect(page).toHaveURL('/login');
    
    // Should not have session cookie
    const cookies = await page.context().cookies();
    const sessionCookie = cookies.find(c => c.name === 'bff_session');
    expect(sessionCookie).toBeUndefined();
  });

  test('should handle session expiry', async ({ page }) => {
    // Set expired session cookie
    await page.context().addCookies([{
      name: 'bff_session',
      value: 'expired-session-id',
      domain: 'localhost',
      path: '/',
      expires: Date.now() / 1000 - 3600, // 1 hour ago
    }]);
    
    // Try to access protected page
    await page.goto('/dashboard');
    
    // Should be redirected to login
    await expect(page).toHaveURL('/login');
  });
});
```

## 6. パフォーマンステスト

```typescript
// __tests__/performance/auth-performance.test.ts
import { performance } from 'perf_hooks';

describe('Authentication Performance', () => {
  it('should complete login flow within acceptable time', async () => {
    const iterations = 100;
    const times: number[] = [];
    
    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      
      // Simulate login flow
      await simulateLoginFlow();
      
      const duration = performance.now() - start;
      times.push(duration);
    }
    
    const avgTime = times.reduce((a, b) => a + b) / times.length;
    const maxTime = Math.max(...times);
    
    console.log(`Average time: ${avgTime.toFixed(2)}ms`);
    console.log(`Max time: ${maxTime.toFixed(2)}ms`);
    
    expect(avgTime).toBeLessThan(100); // Average under 100ms
    expect(maxTime).toBeLessThan(500); // Max under 500ms
  });

  it('should handle concurrent requests', async () => {
    const concurrentRequests = 50;
    const start = performance.now();
    
    const promises = Array(concurrentRequests).fill(null).map(() => 
      simulateLoginFlow()
    );
    
    await Promise.all(promises);
    
    const duration = performance.now() - start;
    console.log(`${concurrentRequests} concurrent requests completed in ${duration.toFixed(2)}ms`);
    
    expect(duration).toBeLessThan(5000); // Under 5 seconds
  });
});

async function simulateLoginFlow() {
  // Mock implementation
  await new Promise(resolve => setTimeout(resolve, Math.random() * 50));
}
```

## 7. セキュリティテスト

```typescript
// __tests__/security/security.test.ts
describe('Security Tests', () => {
  describe('PKCE Security', () => {
    it('should reject invalid code_verifier', async () => {
      const response = await fetch('/oauth/token', {
        method: 'POST',
        body: JSON.stringify({
          grant_type: 'authorization_code',
          code: 'valid-code',
          code_verifier: 'invalid-verifier', // Wrong verifier
        }),
      });
      
      expect(response.status).toBe(400);
      const data = await response.json();
      expect(data.error).toBe('invalid_grant');
    });

    it('should reject missing code_challenge', async () => {
      const response = await fetch('/oauth/authorize?response_type=code&client_id=test');
      
      expect(response.status).toBe(400);
    });
  });

  describe('State Security', () => {
    it('should reject reused state', async () => {
      const state = 'test-state';
      
      // First use
      await useStateInAuthFlow(state);
      
      // Try to reuse
      const response = await useStateInAuthFlow(state);
      expect(response.status).toBe(400);
    });
  });

  describe('Cookie Security', () => {
    it('should set secure cookie attributes', async () => {
      const response = await initiateLogin();
      const setCookie = response.headers.get('set-cookie');
      
      expect(setCookie).toContain('HttpOnly');
      expect(setCookie).toContain('SameSite=Strict');
      if (process.env.NODE_ENV === 'production') {
        expect(setCookie).toContain('Secure');
      }
    });
  });
});
```

## 8. テスト実行とカバレッジ

### package.json スクリプト

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "test:e2e": "playwright test",
    "test:e2e:headed": "playwright test --headed",
    "test:security": "jest __tests__/security",
    "test:performance": "jest __tests__/performance",
    "test:all": "npm run test:coverage && npm run test:e2e"
  }
}
```

### カバレッジレポート

```bash
# カバレッジレポートの生成
npm run test:coverage

# HTMLレポートを開く
open coverage/lcov-report/index.html
```

## 9. CI/CD 統合

### GitHub Actions 設定

```yaml
# .github/workflows/test.yml
name: Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'
      
      - name: Install dependencies
        run: npm ci
      
      - name: Run unit tests
        run: npm run test:coverage
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage/lcov.info
      
      - name: Run E2E tests
        run: |
          npm run build
          npm run start &
          npx wait-on http://localhost:3000
          npm run test:e2e
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: |
            coverage/
            test-results/
```

## 10. ベストプラクティス

### テストの原則

1. **AAA パターン**
   - Arrange: テストデータの準備
   - Act: テスト対象の実行
   - Assert: 結果の検証

2. **独立性**
   - テストは他のテストに依存しない
   - 実行順序に関係なく成功する

3. **再現性**
   - 同じ条件で常に同じ結果
   - 外部依存はモック化

4. **可読性**
   - テスト名は何をテストしているか明確に
   - 複雑なセットアップは関数化

### モックの使い方

```typescript
// 良い例：必要最小限のモック
jest.mock('@/lib/redis', () => ({
  createSession: jest.fn(),
  getSession: jest.fn(),
}));

// 悪い例：過度なモック
jest.mock('entire-module'); // すべてをモック化
```

### データ生成

```typescript
// テストデータファクトリー
export const createTestUser = (overrides = {}) => ({
  id: '123',
  email: 'test@example.com',
  name: 'Test User',
  ...overrides,
});

export const createTestSession = (overrides = {}) => ({
  accessToken: 'test-access-token',
  refreshToken: 'test-refresh-token',
  expiresAt: new Date(Date.now() + 3600 * 1000).toISOString(),
  user: createTestUser(),
  ...overrides,
});
```

## まとめ

包括的なテスト戦略により：

1. **品質保証**: バグの早期発見
2. **リファクタリング**: 安心してコード改善
3. **ドキュメント**: テストが仕様書の役割
4. **信頼性**: 本番環境での安定稼働

継続的にテストを追加・改善することで、堅牢なBFF-Webを維持できます。