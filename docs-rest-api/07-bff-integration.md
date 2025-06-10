# BFF-Web統合ガイド

## 1. 概要

このガイドでは、REST APIサーバーとBFF-Web（Next.js）間の統合方法について説明します。認証フロー、通信仕様、エラーハンドリングなど、両システム間の連携に必要な実装を詳しく解説します。

## 2. アーキテクチャ概要

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  BFF-Web    │────▶│ REST API    │
│             │◀────│  (Next.js)  │◀────│  (Django)   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                     │
                           │                     │
                           ▼                     ▼
                    ┌─────────────┐     ┌─────────────┐
                    │Auth Server  │     │  Database   │
                    │   (OAuth)   │     │(PostgreSQL) │
                    └─────────────┘     └─────────────┘
```

## 3. 通信仕様

### 3.1 基本的な通信フロー

```typescript
// BFF-Web側の実装例
// app/lib/api-client.ts

import { getServerSession } from 'next-auth/next';

class APIClient {
  private baseURL: string;
  
  constructor() {
    this.baseURL = process.env.API_SERVER_URL || 'http://localhost:8001';
  }
  
  async request(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<Response> {
    // セッションからJWTトークン取得
    const session = await getServerSession();
    
    if (!session?.accessToken) {
      throw new Error('Authentication required');
    }
    
    // リクエストヘッダー設定
    const headers = {
      'Authorization': `Bearer ${session.accessToken}`,
      'Content-Type': 'application/json',
      'X-Request-ID': generateRequestId(),
      'X-Client-Version': process.env.CLIENT_VERSION || '1.0.0',
      ...options.headers,
    };
    
    // APIリクエスト送信
    const response = await fetch(`${this.baseURL}${endpoint}`, {
      ...options,
      headers,
    });
    
    // トークン期限切れの処理
    if (response.status === 401) {
      // リフレッシュトークンで更新を試みる
      const newToken = await this.refreshToken(session.refreshToken);
      
      if (newToken) {
        // 新しいトークンでリトライ
        headers['Authorization'] = `Bearer ${newToken}`;
        return fetch(`${this.baseURL}${endpoint}`, {
          ...options,
          headers,
        });
      }
    }
    
    return response;
  }
  
  async refreshToken(refreshToken: string): Promise<string | null> {
    try {
      const response = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ refreshToken }),
      });
      
      if (response.ok) {
        const data = await response.json();
        return data.accessToken;
      }
    } catch (error) {
      console.error('Token refresh failed:', error);
    }
    
    return null;
  }
}

function generateRequestId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export const apiClient = new APIClient();
```

### 3.2 API呼び出しラッパー

```typescript
// app/lib/api-wrapper.ts

import { apiClient } from './api-client';

interface APIResponse<T> {
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
}

export class APIWrapper {
  // ユーザープロフィール取得
  static async getUserProfile(): Promise<APIResponse<UserProfile>> {
    try {
      const response = await apiClient.request('/api/v1/users/me');
      
      if (!response.ok) {
        const error = await response.json();
        return { error };
      }
      
      const data = await response.json();
      return { data: data.data };
      
    } catch (error) {
      return {
        error: {
          code: 'NETWORK_ERROR',
          message: 'Failed to fetch user profile',
          details: error,
        },
      };
    }
  }
  
  // ユーザープロフィール更新
  static async updateUserProfile(
    profile: Partial<UserProfile>
  ): Promise<APIResponse<UserProfile>> {
    try {
      const response = await apiClient.request('/api/v1/users/me', {
        method: 'PATCH',
        body: JSON.stringify(profile),
      });
      
      if (!response.ok) {
        const error = await response.json();
        return { error };
      }
      
      const data = await response.json();
      return { data: data.data };
      
    } catch (error) {
      return {
        error: {
          code: 'NETWORK_ERROR',
          message: 'Failed to update user profile',
          details: error,
        },
      };
    }
  }
  
  // ダッシュボードデータ取得
  static async getDashboardData(
    period: string = 'month'
  ): Promise<APIResponse<DashboardData>> {
    try {
      const response = await apiClient.request(
        `/api/v1/dashboard?period=${period}`
      );
      
      if (!response.ok) {
        const error = await response.json();
        return { error };
      }
      
      const data = await response.json();
      return { data: data.data };
      
    } catch (error) {
      return {
        error: {
          code: 'NETWORK_ERROR',
          message: 'Failed to fetch dashboard data',
          details: error,
        },
      };
    }
  }
}

// 型定義
interface UserProfile {
  id: string;
  email: string;
  name: string;
  avatar_url: string;
  profile: {
    bio: string;
    location: string;
    website: string;
    company: string;
  };
  preferences: {
    language: string;
    timezone: string;
    theme: string;
  };
}

interface DashboardData {
  summary: {
    total_views: number;
    total_interactions: number;
    active_users: number;
    growth_rate: number;
  };
  charts: {
    daily_views: Array<{
      date: string;
      views: number;
      unique_users: number;
    }>;
  };
}
```

## 4. 認証ヘッダー処理

### 4.1 Next.js Middleware での自動付与

```typescript
// middleware.ts

import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';
import { getToken } from 'next-auth/jwt';

export async function middleware(request: NextRequest) {
  // API Routeへのリクエストの場合
  if (request.nextUrl.pathname.startsWith('/api/proxy/')) {
    // JWTトークン取得
    const token = await getToken({ req: request });
    
    if (!token?.accessToken) {
      return NextResponse.json(
        { error: 'Authentication required' },
        { status: 401 }
      );
    }
    
    // プロキシ先URL構築
    const apiPath = request.nextUrl.pathname.replace('/api/proxy', '');
    const apiUrl = `${process.env.API_SERVER_URL}${apiPath}${request.nextUrl.search}`;
    
    // ヘッダー準備
    const headers = new Headers(request.headers);
    headers.set('Authorization', `Bearer ${token.accessToken}`);
    headers.set('X-Request-ID', crypto.randomUUID());
    
    // APIサーバーへリクエスト転送
    const response = await fetch(apiUrl, {
      method: request.method,
      headers,
      body: request.body,
    });
    
    // レスポンス転送
    return new NextResponse(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: '/api/proxy/:path*',
};
```

### 4.2 Server Components での API 呼び出し

```typescript
// app/dashboard/page.tsx

import { headers } from 'next/headers';
import { getServerSession } from 'next-auth/next';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

async function fetchDashboardData() {
  const session = await getServerSession(authOptions);
  
  if (!session?.accessToken) {
    throw new Error('Unauthorized');
  }
  
  const response = await fetch(
    `${process.env.API_SERVER_URL}/api/v1/dashboard`,
    {
      headers: {
        'Authorization': `Bearer ${session.accessToken}`,
        'X-Request-ID': crypto.randomUUID(),
      },
      next: {
        revalidate: 60, // 1分間キャッシュ
      },
    }
  );
  
  if (!response.ok) {
    throw new Error('Failed to fetch dashboard data');
  }
  
  return response.json();
}

export default async function DashboardPage() {
  const data = await fetchDashboardData();
  
  return (
    <div className="dashboard">
      <h1>Dashboard</h1>
      <div className="summary">
        <div>Total Views: {data.data.summary.total_views}</div>
        <div>Active Users: {data.data.summary.active_users}</div>
      </div>
    </div>
  );
}
```

## 5. エラーハンドリング

### 5.1 統一エラー処理

```typescript
// app/lib/error-handler.ts

export class APIError extends Error {
  code: string;
  statusCode: number;
  details?: any;
  
  constructor(
    message: string,
    code: string,
    statusCode: number,
    details?: any
  ) {
    super(message);
    this.name = 'APIError';
    this.code = code;
    this.statusCode = statusCode;
    this.details = details;
  }
}

export async function handleAPIResponse<T>(
  response: Response
): Promise<T> {
  // 成功レスポンス
  if (response.ok) {
    const data = await response.json();
    return data.data || data;
  }
  
  // エラーレスポンス
  let errorData;
  try {
    errorData = await response.json();
  } catch {
    errorData = { error: { message: response.statusText } };
  }
  
  const error = errorData.error || errorData;
  
  // エラータイプ別処理
  switch (response.status) {
    case 401:
      throw new APIError(
        error.message || 'Authentication required',
        error.code || 'UNAUTHORIZED',
        401,
        error.details
      );
      
    case 403:
      throw new APIError(
        error.message || 'Permission denied',
        error.code || 'FORBIDDEN',
        403,
        error.details
      );
      
    case 404:
      throw new APIError(
        error.message || 'Resource not found',
        error.code || 'NOT_FOUND',
        404,
        error.details
      );
      
    case 422:
    case 400:
      throw new APIError(
        error.message || 'Validation error',
        error.code || 'VALIDATION_ERROR',
        response.status,
        error.details
      );
      
    case 429:
      throw new APIError(
        error.message || 'Too many requests',
        error.code || 'RATE_LIMIT_EXCEEDED',
        429,
        error.details
      );
      
    case 500:
    case 502:
    case 503:
    case 504:
      throw new APIError(
        error.message || 'Server error',
        error.code || 'INTERNAL_ERROR',
        response.status,
        error.details
      );
      
    default:
      throw new APIError(
        error.message || 'Unknown error',
        error.code || 'UNKNOWN_ERROR',
        response.status,
        error.details
      );
  }
}
```

### 5.2 React Error Boundary

```typescript
// app/components/error-boundary.tsx

'use client';

import { Component, ReactNode } from 'react';
import { APIError } from '@/lib/error-handler';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  
  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }
  
  componentDidCatch(error: Error, errorInfo: any) {
    // エラーログ送信
    console.error('Error caught by boundary:', error, errorInfo);
    
    // エラートラッキング（Sentry等）
    if (error instanceof APIError) {
      // API エラーの場合
      trackAPIError(error);
    } else {
      // その他のエラー
      trackGeneralError(error);
    }
  }
  
  reset = () => {
    this.setState({ hasError: false, error: null });
  };
  
  render() {
    if (this.state.hasError && this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback(this.state.error, this.reset);
      }
      
      // デフォルトエラー表示
      if (this.state.error instanceof APIError) {
        return <APIErrorDisplay error={this.state.error} onRetry={this.reset} />;
      }
      
      return <GenericErrorDisplay error={this.state.error} onRetry={this.reset} />;
    }
    
    return this.props.children;
  }
}

function APIErrorDisplay({ 
  error, 
  onRetry 
}: { 
  error: APIError; 
  onRetry: () => void;
}) {
  return (
    <div className="error-container">
      <h2>エラーが発生しました</h2>
      <p>{error.message}</p>
      {error.code === 'VALIDATION_ERROR' && error.details?.fields && (
        <ul>
          {Object.entries(error.details.fields).map(([field, errors]) => (
            <li key={field}>
              {field}: {(errors as string[]).join(', ')}
            </li>
          ))}
        </ul>
      )}
      <button onClick={onRetry}>再試行</button>
    </div>
  );
}

function trackAPIError(error: APIError) {
  // Sentry等へのエラー送信
  console.error('API Error:', {
    code: error.code,
    message: error.message,
    statusCode: error.statusCode,
    details: error.details,
  });
}

function trackGeneralError(error: Error) {
  // 一般的なエラー追跡
  console.error('General Error:', error);
}
```

## 6. リアルタイム通信

### 6.1 WebSocket 統合

```typescript
// app/lib/websocket-client.ts

import { getSession } from 'next-auth/react';

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 5;
  private reconnectAttempts: number = 0;
  private listeners: Map<string, Set<Function>> = new Map();
  
  constructor(private url: string) {}
  
  async connect(): Promise<void> {
    const session = await getSession();
    
    if (!session?.accessToken) {
      throw new Error('Authentication required');
    }
    
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        
        // 認証
        this.send('auth', { token: session.accessToken });
        
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          this.handleMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.handleReconnect();
      };
    });
  }
  
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectInterval);
    } else {
      console.error('Max reconnection attempts reached');
      this.emit('connection_failed', {});
    }
  }
  
  private handleMessage(message: any): void {
    const { type, data } = message;
    this.emit(type, data);
  }
  
  send(type: string, data: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, data }));
    } else {
      console.error('WebSocket is not connected');
    }
  }
  
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }
  
  off(event: string, callback: Function): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }
  
  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }
  
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// 使用例
const wsClient = new WebSocketClient('wss://api.example.com/ws');

// React Hook
export function useWebSocket() {
  useEffect(() => {
    wsClient.connect();
    
    // イベントリスナー登録
    const handleNotification = (data: any) => {
      console.log('New notification:', data);
    };
    
    wsClient.on('notification', handleNotification);
    
    return () => {
      wsClient.off('notification', handleNotification);
      wsClient.disconnect();
    };
  }, []);
  
  return wsClient;
}
```

## 7. ファイルアップロード

### 7.1 マルチパートフォームデータ

```typescript
// app/lib/file-upload.ts

import { apiClient } from './api-client';

export async function uploadFile(
  file: File,
  onProgress?: (percentage: number) => void
): Promise<{ url: string; id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('type', file.type);
  
  // XMLHttpRequest を使用して進捗を追跡
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    // 進捗イベント
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const percentage = Math.round((event.loaded / event.total) * 100);
        onProgress(percentage);
      }
    });
    
    // 完了イベント
    xhr.addEventListener('load', async () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const response = JSON.parse(xhr.responseText);
          resolve(response.data);
        } catch (error) {
          reject(new Error('Invalid response format'));
        }
      } else {
        reject(new Error(`Upload failed: ${xhr.statusText}`));
      }
    });
    
    // エラーイベント
    xhr.addEventListener('error', () => {
      reject(new Error('Network error'));
    });
    
    // リクエスト設定
    xhr.open('POST', '/api/proxy/v1/upload');
    
    // 認証ヘッダー設定（セッションから取得）
    getSession().then(session => {
      if (session?.accessToken) {
        xhr.setRequestHeader('Authorization', `Bearer ${session.accessToken}`);
      }
      xhr.send(formData);
    });
  });
}

// React Component での使用例
export function FileUploadComponent() {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  
  const handleFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setUploading(true);
    setProgress(0);
    
    try {
      const result = await uploadFile(file, setProgress);
      console.log('Upload successful:', result);
      
      // アップロード完了後の処理
      toast.success('ファイルをアップロードしました');
      
    } catch (error) {
      console.error('Upload failed:', error);
      toast.error('アップロードに失敗しました');
      
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };
  
  return (
    <div>
      <input
        type="file"
        onChange={handleFileSelect}
        disabled={uploading}
        accept="image/*"
      />
      
      {uploading && (
        <div className="progress">
          <div 
            className="progress-bar"
            style={{ width: `${progress}%` }}
          />
          <span>{progress}%</span>
        </div>
      )}
    </div>
  );
}
```

## 8. キャッシュ戦略

### 8.1 SWR を使用したデータフェッチ

```typescript
// app/hooks/use-api.ts

import useSWR, { SWRConfiguration } from 'swr';
import { apiClient } from '@/lib/api-client';
import { handleAPIResponse } from '@/lib/error-handler';

const defaultOptions: SWRConfiguration = {
  revalidateOnFocus: false,
  revalidateOnReconnect: true,
  shouldRetryOnError: true,
  errorRetryCount: 3,
  errorRetryInterval: 5000,
};

// 汎用フェッチャー
async function fetcher<T>(url: string): Promise<T> {
  const response = await apiClient.request(url);
  return handleAPIResponse<T>(response);
}

// カスタムフック
export function useAPI<T>(
  endpoint: string | null,
  options?: SWRConfiguration
) {
  return useSWR<T>(
    endpoint,
    fetcher,
    { ...defaultOptions, ...options }
  );
}

// 使用例
export function useUserProfile() {
  return useAPI<UserProfile>('/api/v1/users/me', {
    refreshInterval: 60000, // 1分ごとに更新
  });
}

export function useContents(page: number = 1) {
  return useAPI<ContentList>(`/api/v1/contents?page=${page}`, {
    revalidateOnFocus: true,
  });
}

// ミューテーション付きフック
export function useUpdateProfile() {
  const { data, error, mutate } = useUserProfile();
  
  const updateProfile = async (updates: Partial<UserProfile>) => {
    try {
      // 楽観的更新
      mutate(
        { ...data!, ...updates },
        false // 再検証しない
      );
      
      // API更新
      const response = await apiClient.request('/api/v1/users/me', {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
      
      const updated = await handleAPIResponse<UserProfile>(response);
      
      // 成功時は再検証
      mutate(updated, true);
      
      return updated;
      
    } catch (error) {
      // エラー時は元に戻す
      mutate(data, false);
      throw error;
    }
  };
  
  return {
    profile: data,
    isLoading: !error && !data,
    isError: error,
    updateProfile,
  };
}
```

### 8.2 React Query を使用したデータフェッチ

```typescript
// app/hooks/use-react-query.ts

import { 
  useQuery, 
  useMutation, 
  useQueryClient,
  UseQueryOptions,
} from '@tanstack/react-query';
import { apiClient } from '@/lib/api-client';
import { handleAPIResponse } from '@/lib/error-handler';

// クエリキー管理
export const queryKeys = {
  user: ['user'] as const,
  userProfile: () => [...queryKeys.user, 'profile'] as const,
  contents: (filters?: any) => ['contents', filters] as const,
  content: (id: string) => ['content', id] as const,
};

// プロフィール取得
export function useProfile(
  options?: UseQueryOptions<UserProfile>
) {
  return useQuery({
    queryKey: queryKeys.userProfile(),
    queryFn: async () => {
      const response = await apiClient.request('/api/v1/users/me');
      return handleAPIResponse<UserProfile>(response);
    },
    staleTime: 5 * 60 * 1000, // 5分
    ...options,
  });
}

// プロフィール更新
export function useUpdateProfile() {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (updates: Partial<UserProfile>) => {
      const response = await apiClient.request('/api/v1/users/me', {
        method: 'PATCH',
        body: JSON.stringify(updates),
      });
      return handleAPIResponse<UserProfile>(response);
    },
    onSuccess: (data) => {
      // キャッシュ更新
      queryClient.setQueryData(queryKeys.userProfile(), data);
      
      // 関連するクエリを無効化
      queryClient.invalidateQueries({ 
        queryKey: queryKeys.user 
      });
    },
    onError: (error) => {
      console.error('Profile update failed:', error);
    },
  });
}

// 無限スクロール対応
export function useInfiniteContents() {
  return useInfiniteQuery({
    queryKey: queryKeys.contents(),
    queryFn: async ({ pageParam = 1 }) => {
      const response = await apiClient.request(
        `/api/v1/contents?page=${pageParam}&per_page=20`
      );
      return handleAPIResponse<ContentList>(response);
    },
    getNextPageParam: (lastPage) => {
      return lastPage.meta.current_page < lastPage.meta.total_pages
        ? lastPage.meta.current_page + 1
        : undefined;
    },
  });
}
```

## 9. 型安全性

### 9.1 API レスポンス型生成

```typescript
// scripts/generate-api-types.ts

import { generateApi } from 'swagger-typescript-api';
import path from 'path';

// OpenAPI仕様からTypeScript型を生成
generateApi({
  name: 'APITypes.ts',
  output: path.resolve(process.cwd(), './app/types'),
  url: 'http://localhost:8001/openapi.json',
  httpClientType: 'fetch',
  generateClient: false,
  generateRouteTypes: true,
  generateResponses: true,
  extractRequestParams: true,
  extractRequestBody: true,
  prettier: {
    printWidth: 120,
    tabWidth: 2,
    singleQuote: true,
    trailingComma: 'all',
  },
});
```

### 9.2 型安全なAPIクライアント

```typescript
// app/lib/typed-api-client.ts

import { paths } from '@/types/APITypes';

type APIPath = keyof paths;
type APIMethod<P extends APIPath> = keyof paths[P];
type APIParameters<
  P extends APIPath,
  M extends APIMethod<P>
> = paths[P][M] extends { parameters: infer Params } ? Params : never;
type APIRequestBody<
  P extends APIPath,
  M extends APIMethod<P>
> = paths[P][M] extends { requestBody: { content: { 'application/json': infer Body } } }
  ? Body
  : never;
type APIResponse<
  P extends APIPath,
  M extends APIMethod<P>
> = paths[P][M] extends { responses: { 200: { content: { 'application/json': infer Res } } } }
  ? Res
  : never;

export class TypedAPIClient {
  async request<
    P extends APIPath,
    M extends APIMethod<P>
  >(
    path: P,
    method: M,
    options?: {
      params?: APIParameters<P, M>;
      body?: APIRequestBody<P, M>;
    }
  ): Promise<APIResponse<P, M>> {
    // 実装
    const response = await apiClient.request(path as string, {
      method: method as string,
      body: options?.body ? JSON.stringify(options.body) : undefined,
    });
    
    return handleAPIResponse(response);
  }
}

// 使用例（型安全）
const client = new TypedAPIClient();

// 正しい使用
const profile = await client.request(
  '/api/v1/users/me',
  'get'
);

// エラー: パスが存在しない
const invalid = await client.request(
  '/api/v1/invalid',  // TypeScript Error
  'get'
);
```

## 10. モニタリングとデバッグ

### 10.1 リクエストロギング

```typescript
// app/lib/api-logger.ts

interface APILog {
  requestId: string;
  method: string;
  url: string;
  headers: Record<string, string>;
  body?: any;
  response?: {
    status: number;
    statusText: string;
    body: any;
  };
  duration: number;
  error?: Error;
}

class APILogger {
  private logs: APILog[] = [];
  private maxLogs: number = 100;
  
  logRequest(log: APILog): void {
    this.logs.unshift(log);
    
    // 最大件数を超えたら古いログを削除
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(0, this.maxLogs);
    }
    
    // 開発環境でコンソール出力
    if (process.env.NODE_ENV === 'development') {
      console.group(`API ${log.method} ${log.url}`);
      console.log('Request ID:', log.requestId);
      console.log('Duration:', `${log.duration}ms`);
      
      if (log.error) {
        console.error('Error:', log.error);
      } else if (log.response) {
        console.log('Status:', log.response.status);
        console.log('Response:', log.response.body);
      }
      
      console.groupEnd();
    }
    
    // エラーの場合は追加処理
    if (log.error || (log.response && log.response.status >= 400)) {
      this.handleError(log);
    }
  }
  
  private handleError(log: APILog): void {
    // エラー追跡サービスに送信
    if (typeof window !== 'undefined' && window.Sentry) {
      window.Sentry.captureException(log.error || new Error('API Error'), {
        extra: {
          requestId: log.requestId,
          url: log.url,
          status: log.response?.status,
        },
      });
    }
  }
  
  getLogs(): APILog[] {
    return [...this.logs];
  }
  
  clearLogs(): void {
    this.logs = [];
  }
}

export const apiLogger = new APILogger();

// デバッグ用React DevTools
if (process.env.NODE_ENV === 'development') {
  if (typeof window !== 'undefined') {
    (window as any).__API_LOGS__ = apiLogger;
  }
}
```

## まとめ

このBFF-Web統合ガイドに従うことで、Next.jsアプリケーションとDjango REST APIサーバー間の安全で効率的な通信を実現できます。重要なポイント：

1. **認証**: JWTトークンの適切な管理と自動更新
2. **エラーハンドリング**: 統一されたエラー処理とユーザーフィードバック
3. **型安全性**: TypeScriptによる型安全なAPI通信
4. **パフォーマンス**: 適切なキャッシュ戦略とデータフェッチ
5. **監視**: リクエストログとエラー追跡