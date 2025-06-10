# BFF-Web 実装ガイド

## 目次
1. [プロジェクトセットアップ](#1-プロジェクトセットアップ)
2. [ディレクトリ構造](#2-ディレクトリ構造)
3. [基本ライブラリの設定](#3-基本ライブラリの設定)
4. [認証エンドポイントの実装](#4-認証エンドポイントの実装)
5. [ミドルウェアの実装](#5-ミドルウェアの実装)
6. [フロントエンドコンポーネント](#6-フロントエンドコンポーネント)

## 1. プロジェクトセットアップ

### Next.js プロジェクトの作成

```bash
npx create-next-app@latest bff-web \
  --typescript \
  --app \
  --tailwind \
  --eslint

cd bff-web
```

### 必要なパッケージのインストール

```bash
# 認証関連
npm install jose uuid

# Redis
npm install ioredis

# 型定義
npm install --save-dev @types/uuid

# 開発ツール
npm install --save-dev prettier eslint-config-prettier
```

## 2. ディレクトリ構造

```
bff-web/
├── app/
│   ├── api/
│   │   └── auth/
│   │       ├── login/
│   │       │   └── route.ts
│   │       ├── callback/
│   │       │   └── route.ts
│   │       ├── refresh/
│   │       │   └── route.ts
│   │       ├── logout/
│   │       │   └── route.ts
│   │       └── me/
│   │           └── route.ts
│   ├── login/
│   │   └── page.tsx
│   ├── dashboard/
│   │   └── page.tsx
│   └── layout.tsx
├── lib/
│   ├── auth/
│   │   ├── pkce.ts
│   │   ├── session.ts
│   │   └── types.ts
│   ├── redis.ts
│   └── constants.ts
├── middleware.ts
├── .env.local
└── docker-compose.yml
```

## 3. 基本ライブラリの設定

### Redis クライアント (`lib/redis.ts`)

```typescript
import { Redis } from 'ioredis';

const getRedisUrl = () => {
  if (process.env.REDIS_URL) {
    return process.env.REDIS_URL;
  }
  return 'redis://localhost:6379';
};

const redis = new Redis(getRedisUrl(), {
  maxRetriesPerRequest: 3,
  retryStrategy: (times) => {
    if (times > 3) {
      console.error('Redis connection failed after 3 retries');
      return null;
    }
    return Math.min(times * 200, 1000);
  },
});

redis.on('error', (err) => {
  console.error('Redis Client Error:', err);
});

redis.on('connect', () => {
  console.log('Redis Client Connected');
});

// セッション管理用のヘルパー関数
export async function createSession(
  sessionId: string,
  data: any,
  ttl: number = 3600 * 24 * 7 // 7 days
): Promise<void> {
  await redis.setex(`session:${sessionId}`, ttl, JSON.stringify(data));
}

export async function getSession(sessionId: string): Promise<any | null> {
  const data = await redis.get(`session:${sessionId}`);
  return data ? JSON.parse(data) : null;
}

export async function updateSession(
  sessionId: string,
  data: any,
  ttl: number = 3600 * 24 * 7
): Promise<void> {
  await redis.setex(`session:${sessionId}`, ttl, JSON.stringify(data));
}

export async function deleteSession(sessionId: string): Promise<void> {
  await redis.del(`session:${sessionId}`);
}

// 認証セッション用（一時的）
export async function createAuthSession(
  sessionId: string,
  data: any,
  ttl: number = 600 // 10 minutes
): Promise<void> {
  await redis.setex(`auth:${sessionId}`, ttl, JSON.stringify(data));
}

export async function getAuthSession(sessionId: string): Promise<any | null> {
  const data = await redis.get(`auth:${sessionId}`);
  return data ? JSON.parse(data) : null;
}

export async function deleteAuthSession(sessionId: string): Promise<void> {
  await redis.del(`auth:${sessionId}`);
}

export default redis;
```

### PKCE ユーティリティ (`lib/auth/pkce.ts`)

```typescript
import crypto from 'crypto';

export function generateCodeVerifier(): string {
  // RFC 7636 準拠: 43-128文字の base64url 文字列
  return crypto.randomBytes(32).toString('base64url');
}

export function generateCodeChallenge(verifier: string): string {
  // S256 method: SHA256(verifier) を base64url エンコード
  return crypto
    .createHash('sha256')
    .update(verifier)
    .digest('base64url');
}

export function generateState(): string {
  // CSRF対策: 最低16バイトのランダム文字列
  return crypto.randomBytes(16).toString('base64url');
}
```

### 型定義 (`lib/auth/types.ts`)

```typescript
export interface AuthSession {
  state: string;
  codeVerifier: string;
  createdAt: string;
  redirectAfter?: string;
}

export interface UserSession {
  accessToken: string;
  refreshToken: string;
  expiresAt: string;
  user: User;
  createdAt: string;
  updatedAt?: string;
}

export interface User {
  id: string;
  email: string;
  name: string;
  picture?: string;
  provider: string;
  providerId: string;
  linkedAccounts: string[];
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  refresh_token: string;
  scope: string;
  id_token?: string;
  user?: User;
}

export interface ErrorResponse {
  error: string;
  error_description?: string;
  error_uri?: string;
}
```

### 定数 (`lib/constants.ts`)

```typescript
// 環境変数
export const AUTH_SERVER_URL = process.env.AUTH_SERVER_URL || 'http://localhost:8000';
export const AUTH_CLIENT_ID = process.env.AUTH_CLIENT_ID!;
export const AUTH_CLIENT_SECRET = process.env.AUTH_CLIENT_SECRET!;
export const AUTH_REDIRECT_URI = process.env.AUTH_REDIRECT_URI || 'http://localhost:3000/api/auth/callback';

// Cookie設定
export const SESSION_COOKIE_NAME = process.env.SESSION_COOKIE_NAME || 'bff_session';
export const AUTH_SESSION_COOKIE_NAME = 'auth_session_id';

export const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: process.env.NODE_ENV === 'production',
  sameSite: 'lax' as const,
  path: '/',
};

export const SESSION_COOKIE_OPTIONS = {
  ...COOKIE_OPTIONS,
  maxAge: 60 * 60 * 24 * 7, // 7 days
};

export const AUTH_COOKIE_OPTIONS = {
  ...COOKIE_OPTIONS,
  maxAge: 60 * 10, // 10 minutes
};

// エラーコード
export const ERROR_CODES = {
  INVALID_REQUEST: 'invalid_request',
  UNAUTHORIZED: 'unauthorized',
  ACCESS_DENIED: 'access_denied',
  INVALID_STATE: 'invalid_state',
  SERVER_ERROR: 'server_error',
  NETWORK_ERROR: 'network_error',
} as const;
```

## 4. 認証エンドポイントの実装

### ログイン開始 (`app/api/auth/login/route.ts`)

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';
import { generateCodeVerifier, generateCodeChallenge, generateState } from '@/lib/auth/pkce';
import { createAuthSession } from '@/lib/redis';
import { AUTH_SERVER_URL, AUTH_CLIENT_ID, AUTH_REDIRECT_URI, AUTH_COOKIE_OPTIONS } from '@/lib/constants';
import type { AuthSession } from '@/lib/auth/types';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json().catch(() => ({}));
    const { provider_hint, login_hint, redirect_after = '/dashboard' } = body;

    // PKCE と state の生成
    const sessionId = uuidv4();
    const state = generateState();
    const codeVerifier = generateCodeVerifier();
    const codeChallenge = generateCodeChallenge(codeVerifier);

    // 認証セッション作成
    const authSession: AuthSession = {
      state,
      codeVerifier,
      createdAt: new Date().toISOString(),
      redirectAfter,
    };

    await createAuthSession(sessionId, authSession);

    // 認証サーバーへのリダイレクトURL構築
    const authUrl = new URL('/oauth/authorize', AUTH_SERVER_URL);
    const params = {
      response_type: 'code',
      client_id: AUTH_CLIENT_ID,
      redirect_uri: AUTH_REDIRECT_URI,
      state,
      code_challenge: codeChallenge,
      code_challenge_method: 'S256',
      scope: 'openid profile email',
      ...(provider_hint && { provider_hint }),
      ...(login_hint && { login_hint }),
    };

    Object.entries(params).forEach(([key, value]) => {
      if (value) authUrl.searchParams.append(key, value);
    });

    // レスポンス作成
    const response = NextResponse.json({
      redirectUrl: authUrl.toString(),
    });

    // 認証セッションIDをCookieに設定
    response.cookies.set('auth_session_id', sessionId, AUTH_COOKIE_OPTIONS);

    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'server_error', error_description: 'Failed to initiate login' },
      { status: 500 }
    );
  }
}
```

### コールバック処理 (`app/api/auth/callback/route.ts`)

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { v4 as uuidv4 } from 'uuid';
import { getAuthSession, deleteAuthSession, createSession } from '@/lib/redis';
import {
  AUTH_SERVER_URL,
  AUTH_CLIENT_ID,
  AUTH_CLIENT_SECRET,
  AUTH_REDIRECT_URI,
  SESSION_COOKIE_NAME,
  SESSION_COOKIE_OPTIONS,
  ERROR_CODES,
} from '@/lib/constants';
import type { AuthSession, UserSession, TokenResponse, ErrorResponse } from '@/lib/auth/types';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const code = searchParams.get('code');
  const state = searchParams.get('state');
  const error = searchParams.get('error');
  const errorDescription = searchParams.get('error_description');

  // エラーチェック
  if (error) {
    console.error('OAuth error:', { error, errorDescription });
    return NextResponse.redirect(
      new URL(`/login?error=${error}&message=${encodeURIComponent(errorDescription || error)}`, request.url)
    );
  }

  if (!code || !state) {
    return NextResponse.redirect(
      new URL('/login?error=invalid_request&message=Missing+required+parameters', request.url)
    );
  }

  try {
    // 認証セッション取得
    const authSessionId = request.cookies.get('auth_session_id')?.value;
    if (!authSessionId) {
      throw new Error('No auth session found');
    }

    const authSession: AuthSession | null = await getAuthSession(authSessionId);
    if (!authSession) {
      throw new Error('Auth session expired');
    }

    // state 検証
    if (authSession.state !== state) {
      throw new Error('State mismatch');
    }

    // トークン交換
    const tokenResponse = await exchangeCodeForTokens(code, authSession.codeVerifier);

    // ユーザーセッション作成
    const sessionId = uuidv4();
    const userSession: UserSession = {
      accessToken: tokenResponse.access_token,
      refreshToken: tokenResponse.refresh_token,
      expiresAt: new Date(Date.now() + tokenResponse.expires_in * 1000).toISOString(),
      user: tokenResponse.user!,
      createdAt: new Date().toISOString(),
    };

    await createSession(sessionId, userSession);

    // 認証セッション削除
    await deleteAuthSession(authSessionId);

    // リダイレクト先
    const redirectTo = authSession.redirectAfter || '/dashboard';

    // レスポンス作成
    const response = NextResponse.redirect(new URL(redirectTo, request.url));
    response.cookies.set(SESSION_COOKIE_NAME, sessionId, SESSION_COOKIE_OPTIONS);
    response.cookies.delete('auth_session_id');

    return response;
  } catch (error: any) {
    console.error('Callback error:', error);
    
    // エラーメッセージの判定
    let errorCode = ERROR_CODES.SERVER_ERROR;
    let errorMessage = 'Authentication failed';
    
    if (error.message.includes('State mismatch')) {
      errorCode = ERROR_CODES.INVALID_STATE;
      errorMessage = 'Security validation failed. Please try again.';
    } else if (error.message.includes('session')) {
      errorMessage = 'Session expired. Please try again.';
    }

    return NextResponse.redirect(
      new URL(`/login?error=${errorCode}&message=${encodeURIComponent(errorMessage)}`, request.url)
    );
  }
}

async function exchangeCodeForTokens(code: string, codeVerifier: string): Promise<TokenResponse> {
  const tokenUrl = new URL('/oauth/token', AUTH_SERVER_URL);
  
  const response = await fetch(tokenUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      grant_type: 'authorization_code',
      code,
      redirect_uri: AUTH_REDIRECT_URI,
      client_id: AUTH_CLIENT_ID,
      client_secret: AUTH_CLIENT_SECRET,
      code_verifier: codeVerifier,
    }),
  });

  if (!response.ok) {
    const error: ErrorResponse = await response.json();
    console.error('Token exchange failed:', error);
    throw new Error(error.error_description || error.error || 'Token exchange failed');
  }

  return response.json();
}
```

### トークンリフレッシュ (`app/api/auth/refresh/route.ts`)

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { getSession, updateSession } from '@/lib/redis';
import { AUTH_SERVER_URL, AUTH_CLIENT_ID, AUTH_CLIENT_SECRET, SESSION_COOKIE_NAME } from '@/lib/constants';
import type { UserSession, TokenResponse } from '@/lib/auth/types';

export async function POST(request: NextRequest) {
  try {
    const sessionId = request.cookies.get(SESSION_COOKIE_NAME)?.value;
    if (!sessionId) {
      return NextResponse.json(
        { error: 'unauthorized', error_description: 'No session found' },
        { status: 401 }
      );
    }

    const session: UserSession | null = await getSession(sessionId);
    if (!session) {
      return NextResponse.json(
        { error: 'unauthorized', error_description: 'Session not found' },
        { status: 401 }
      );
    }

    // トークンリフレッシュ
    const response = await fetch(new URL('/oauth/token', AUTH_SERVER_URL), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        grant_type: 'refresh_token',
        refresh_token: session.refreshToken,
        client_id: AUTH_CLIENT_ID,
        client_secret: AUTH_CLIENT_SECRET,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('Token refresh failed:', error);
      return NextResponse.json(
        { error: 'invalid_grant', error_description: 'Failed to refresh token' },
        { status: 401 }
      );
    }

    const tokens: TokenResponse = await response.json();

    // セッション更新
    const updatedSession: UserSession = {
      ...session,
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token,
      expiresAt: new Date(Date.now() + tokens.expires_in * 1000).toISOString(),
      updatedAt: new Date().toISOString(),
    };

    await updateSession(sessionId, updatedSession);

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Refresh error:', error);
    return NextResponse.json(
      { error: 'server_error', error_description: 'Internal server error' },
      { status: 500 }
    );
  }
}
```

### ログアウト (`app/api/auth/logout/route.ts`)

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { getSession, deleteSession } from '@/lib/redis';
import { AUTH_SERVER_URL, AUTH_CLIENT_ID, AUTH_CLIENT_SECRET, SESSION_COOKIE_NAME } from '@/lib/constants';
import type { UserSession } from '@/lib/auth/types';

export async function POST(request: NextRequest) {
  try {
    const sessionId = request.cookies.get(SESSION_COOKIE_NAME)?.value;
    
    if (sessionId) {
      const session: UserSession | null = await getSession(sessionId);
      
      if (session?.refreshToken) {
        // トークン無効化（エラーでも続行）
        try {
          await revokeToken(session.refreshToken);
        } catch (error) {
          console.error('Token revocation failed:', error);
        }
      }
      
      // セッション削除
      await deleteSession(sessionId);
    }

    // レスポンス作成
    const response = NextResponse.json({
      success: true,
      message: 'Logged out successfully',
    });

    // Cookie削除とセキュリティヘッダー
    response.cookies.delete(SESSION_COOKIE_NAME);
    response.cookies.delete('auth_session_id');
    response.headers.set('Clear-Site-Data', '"cache", "cookies", "storage"');

    return response;
  } catch (error) {
    console.error('Logout error:', error);
    
    // エラーでもCookieは削除
    const response = NextResponse.json(
      { error: 'logout_failed', error_description: 'Logout encountered an error' },
      { status: 500 }
    );
    response.cookies.delete(SESSION_COOKIE_NAME);
    response.cookies.delete('auth_session_id');
    
    return response;
  }
}

async function revokeToken(refreshToken: string): Promise<void> {
  const response = await fetch(new URL('/oauth/api/tokens/revoke/', AUTH_SERVER_URL), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({
      token: refreshToken,
      token_type_hint: 'refresh_token',
      client_id: AUTH_CLIENT_ID,
      client_secret: AUTH_CLIENT_SECRET,
    }),
  });

  if (!response.ok) {
    throw new Error('Token revocation failed');
  }
}
```

### 認証状態確認 (`app/api/auth/me/route.ts`)

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/redis';
import { SESSION_COOKIE_NAME } from '@/lib/constants';
import type { UserSession } from '@/lib/auth/types';

export async function GET(request: NextRequest) {
  try {
    const sessionId = request.cookies.get(SESSION_COOKIE_NAME)?.value;
    
    if (!sessionId) {
      return NextResponse.json({
        authenticated: false,
      });
    }

    const session: UserSession | null = await getSession(sessionId);
    
    if (!session) {
      return NextResponse.json({
        authenticated: false,
      });
    }

    // トークン有効期限チェック
    const isExpired = new Date(session.expiresAt) < new Date();
    
    if (isExpired) {
      return NextResponse.json({
        authenticated: false,
        reason: 'token_expired',
      });
    }

    return NextResponse.json({
      authenticated: true,
      user: session.user,
      expiresAt: session.expiresAt,
    });
  } catch (error) {
    console.error('Auth check error:', error);
    return NextResponse.json({
      authenticated: false,
      error: 'check_failed',
    });
  }
}
```

## 5. ミドルウェアの実装

### 認証ミドルウェア (`middleware.ts`)

```typescript
import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// 認証が必要なパス
const protectedPaths = [
  '/dashboard',
  '/profile',
  '/settings',
  '/api/protected',
];

// 認証不要なパス
const publicPaths = [
  '/login',
  '/api/auth',
  '/_next',
  '/favicon.ico',
];

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  
  // 公開パスはスキップ
  if (publicPaths.some(path => pathname.startsWith(path))) {
    return NextResponse.next();
  }
  
  // 保護されたパスの確認
  const isProtectedPath = protectedPaths.some(path => pathname.startsWith(path));
  
  if (isProtectedPath) {
    const sessionCookie = request.cookies.get(process.env.SESSION_COOKIE_NAME || 'bff_session');
    
    if (!sessionCookie?.value) {
      // ログインページへリダイレクト
      const loginUrl = new URL('/login', request.url);
      loginUrl.searchParams.set('redirect', pathname);
      return NextResponse.redirect(loginUrl);
    }
    
    // オプション: APIで認証状態を確認
    if (pathname.startsWith('/api/protected')) {
      try {
        const authCheckUrl = new URL('/api/auth/me', request.url);
        const authResponse = await fetch(authCheckUrl, {
          headers: {
            cookie: request.headers.get('cookie') || '',
          },
        });
        
        const authData = await authResponse.json();
        
        if (!authData.authenticated) {
          return NextResponse.json(
            { error: 'unauthorized', message: 'Authentication required' },
            { status: 401 }
          );
        }
      } catch (error) {
        console.error('Auth check failed:', error);
        return NextResponse.json(
          { error: 'auth_check_failed' },
          { status: 500 }
        );
      }
    }
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder
     */
    '/((?!_next/static|_next/image|favicon.ico|public).*)',
  ],
};
```

## 6. フロントエンドコンポーネント

### ログインページ (`app/login/page.tsx`)

```tsx
'use client';

import { useState, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';

export default function LoginPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    // URLからエラーメッセージを取得
    const errorParam = searchParams.get('error');
    const messageParam = searchParams.get('message');
    if (errorParam || messageParam) {
      setError(messageParam || errorParam || 'An error occurred');
    }
  }, [searchParams]);

  const handleLogin = async (providerHint?: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider_hint: providerHint,
          redirect_after: searchParams.get('redirect') || '/dashboard',
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        window.location.href = data.redirectUrl;
      } else {
        const errorData = await response.json();
        setError(errorData.error_description || 'Login failed');
      }
    } catch (err) {
      setError('Network error. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            Sign in to your account
          </h2>
          {searchParams.get('redirect') && (
            <p className="mt-2 text-center text-sm text-gray-600">
              You need to sign in to access that page
            </p>
          )}
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
            {error}
          </div>
        )}
        
        <div className="space-y-4">
          <button
            onClick={() => handleLogin('google')}
            disabled={isLoading}
            className="w-full flex justify-center items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <img src="/google-icon.svg" alt="Google" className="w-5 h-5 mr-2" />
            Continue with Google
          </button>
          
          <button
            onClick={() => handleLogin('github')}
            disabled={isLoading}
            className="w-full flex justify-center items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <img src="/github-icon.svg" alt="GitHub" className="w-5 h-5 mr-2" />
            Continue with GitHub
          </button>
          
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300" />
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-gray-50 text-gray-500">Or</span>
            </div>
          </div>
          
          <button
            onClick={() => handleLogin()}
            disabled={isLoading}
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Loading...' : 'Show all login options'}
          </button>
        </div>
      </div>
    </div>
  );
}
```

### ダッシュボード (`app/dashboard/page.tsx`)

```tsx
import { cookies } from 'next/headers';
import { redirect } from 'next/navigation';
import { getSession } from '@/lib/redis';
import { SESSION_COOKIE_NAME } from '@/lib/constants';
import LogoutButton from '@/components/LogoutButton';

async function getUser() {
  const cookieStore = cookies();
  const sessionId = cookieStore.get(SESSION_COOKIE_NAME)?.value;
  
  if (!sessionId) {
    redirect('/login');
  }
  
  const session = await getSession(sessionId);
  if (!session) {
    redirect('/login');
  }
  
  // トークン有効期限チェック
  if (new Date(session.expiresAt) < new Date()) {
    redirect('/login?message=Session+expired');
  }
  
  return session;
}

export default async function DashboardPage() {
  const session = await getUser();
  
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold">Dashboard</h1>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-700">
                {session.user.email}
              </span>
              <LogoutButton />
            </div>
          </div>
        </div>
      </nav>
      
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                User Information
              </h2>
              <dl className="grid grid-cols-1 gap-x-4 gap-y-6 sm:grid-cols-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Name</dt>
                  <dd className="mt-1 text-sm text-gray-900">{session.user.name}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Email</dt>
                  <dd className="mt-1 text-sm text-gray-900">{session.user.email}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Provider</dt>
                  <dd className="mt-1 text-sm text-gray-900 capitalize">{session.user.provider}</dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Linked Accounts</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {session.user.linkedAccounts.join(', ') || 'None'}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
          
          <div className="mt-6 bg-white overflow-hidden shadow rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                Session Information
              </h2>
              <dl className="space-y-2">
                <div>
                  <dt className="text-sm font-medium text-gray-500">Session Created</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {new Date(session.createdAt).toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-gray-500">Token Expires</dt>
                  <dd className="mt-1 text-sm text-gray-900">
                    {new Date(session.expiresAt).toLocaleString()}
                  </dd>
                </div>
              </dl>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
```

### ログアウトボタン (`components/LogoutButton.tsx`)

```tsx
'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function LogoutButton() {
  const [isLoading, setIsLoading] = useState(false);
  const router = useRouter();

  const handleLogout = async () => {
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/auth/logout', {
        method: 'POST',
      });
      
      if (response.ok) {
        // Clear any client-side state
        if (typeof window !== 'undefined') {
          localStorage.clear();
          sessionStorage.clear();
        }
        
        // Redirect to login
        router.push('/login');
      } else {
        console.error('Logout failed');
        // Even on error, redirect to login
        router.push('/login');
      }
    } catch (error) {
      console.error('Logout error:', error);
      router.push('/login');
    }
  };

  return (
    <button
      onClick={handleLogout}
      disabled={isLoading}
      className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"
    >
      {isLoading ? 'Logging out...' : 'Logout'}
    </button>
  );
}
```

## まとめ

この実装ガイドに従うことで、セキュアな BFF-Web を構築できます。重要なポイント：

1. **PKCE の正確な実装**
2. **state による CSRF 対策**
3. **適切なエラーハンドリング**
4. **セキュアな Cookie 設定**
5. **トークンの適切な管理**

不明な点があれば、認証チームまでお問い合わせください。