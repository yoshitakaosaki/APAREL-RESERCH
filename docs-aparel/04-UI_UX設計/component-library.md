# コンポーネントライブラリ仕様書

## 1. 概要

本ドキュメントは、テックパック生成アプリケーションのコンポーネントライブラリの詳細仕様を定義します。

## 2. 基本コンポーネント

### 2.1 Button（ボタン）

#### 仕様

```typescript
interface ButtonProps {
  // 外観
  variant: 'primary' | 'secondary' | 'tertiary' | 'danger' | 'ghost';
  size: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  
  // コンテンツ
  children: React.ReactNode;
  icon?: IconName;
  iconPosition?: 'left' | 'right';
  
  // 状態
  loading?: boolean;
  disabled?: boolean;
  selected?: boolean;
  
  // レイアウト
  fullWidth?: boolean;
  align?: 'left' | 'center' | 'right';
  
  // イベント
  onClick?: (event: MouseEvent) => void;
  onFocus?: (event: FocusEvent) => void;
  onBlur?: (event: FocusEvent) => void;
  
  // その他
  type?: 'button' | 'submit' | 'reset';
  className?: string;
  dataTestId?: string;
}
```

#### バリエーション

| Variant | 用途 | 背景色 | テキスト色 |
|---------|------|--------|------------|
| primary | 主要アクション | $primary-500 | white |
| secondary | 副次アクション | $gray-100 | $gray-900 |
| tertiary | 第三アクション | transparent | $primary-500 |
| danger | 破壊的アクション | $error | white |
| ghost | 最小限のUI | transparent | $gray-600 |

### 2.2 Input（入力フィールド）

#### 仕様

```typescript
interface InputProps {
  // タイプ
  type?: 'text' | 'email' | 'password' | 'number' | 'tel' | 'url' | 'search';
  
  // 値
  value?: string | number;
  defaultValue?: string | number;
  placeholder?: string;
  
  // ラベル
  label?: string;
  labelPosition?: 'top' | 'left' | 'floating';
  required?: boolean;
  
  // バリデーション
  error?: string;
  success?: boolean;
  warning?: string;
  helpText?: string;
  
  // 制約
  maxLength?: number;
  minLength?: number;
  pattern?: string;
  min?: number;
  max?: number;
  step?: number;
  
  // 装飾
  prefix?: React.ReactNode;
  suffix?: React.ReactNode;
  icon?: IconName;
  
  // 状態
  disabled?: boolean;
  readOnly?: boolean;
  
  // イベント
  onChange?: (value: string | number) => void;
  onBlur?: (event: FocusEvent) => void;
  onFocus?: (event: FocusEvent) => void;
  onKeyDown?: (event: KeyboardEvent) => void;
}
```

### 2.3 Select（セレクトボックス）

#### 仕様

```typescript
interface SelectOption {
  value: string | number;
  label: string;
  disabled?: boolean;
  icon?: IconName;
  description?: string;
}

interface SelectProps {
  // オプション
  options: SelectOption[];
  value?: string | number | (string | number)[];
  defaultValue?: string | number | (string | number)[];
  
  // モード
  multiple?: boolean;
  searchable?: boolean;
  clearable?: boolean;
  
  // ラベル
  label?: string;
  placeholder?: string;
  required?: boolean;
  
  // バリデーション
  error?: string;
  helpText?: string;
  
  // カスタマイズ
  renderOption?: (option: SelectOption) => React.ReactNode;
  groupBy?: (option: SelectOption) => string;
  
  // 状態
  disabled?: boolean;
  loading?: boolean;
  
  // イベント
  onChange?: (value: string | number | (string | number)[]) => void;
  onSearch?: (query: string) => void;
}
```

### 2.4 Checkbox / Radio

#### 仕様

```typescript
interface CheckboxProps {
  // 値
  checked?: boolean;
  defaultChecked?: boolean;
  indeterminate?: boolean;
  
  // ラベル
  label?: string;
  description?: string;
  
  // 状態
  disabled?: boolean;
  error?: boolean;
  
  // イベント
  onChange?: (checked: boolean) => void;
}

interface RadioGroupProps {
  // オプション
  options: RadioOption[];
  value?: string | number;
  defaultValue?: string | number;
  
  // レイアウト
  orientation?: 'horizontal' | 'vertical';
  
  // ラベル
  label?: string;
  required?: boolean;
  
  // 状態
  disabled?: boolean;
  error?: string;
  
  // イベント
  onChange?: (value: string | number) => void;
}
```

### 2.5 TextArea（テキストエリア）

#### 仕様

```typescript
interface TextAreaProps extends Omit<InputProps, 'type' | 'prefix' | 'suffix'> {
  // サイズ
  rows?: number;
  minRows?: number;
  maxRows?: number;
  
  // 機能
  autoResize?: boolean;
  showCount?: boolean;
  maxLength?: number;
  
  // リサイズ
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
}
```

## 3. レイアウトコンポーネント

### 3.1 Grid（グリッド）

```typescript
interface GridProps {
  // グリッド設定
  columns?: number | { xs?: number; sm?: number; md?: number; lg?: number; xl?: number };
  gap?: number | string;
  rowGap?: number | string;
  columnGap?: number | string;
  
  // アラインメント
  alignItems?: 'start' | 'center' | 'end' | 'stretch';
  justifyItems?: 'start' | 'center' | 'end' | 'stretch';
  
  // その他
  className?: string;
  children: React.ReactNode;
}
```

### 3.2 Stack（スタック）

```typescript
interface StackProps {
  // 方向
  direction?: 'row' | 'column';
  
  // 間隔
  spacing?: number | string;
  
  // アラインメント
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around' | 'evenly';
  
  // ラップ
  wrap?: boolean;
  
  // その他
  divider?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}
```

### 3.3 Card（カード）

```typescript
interface CardProps {
  // バリエーション
  variant?: 'flat' | 'elevated' | 'outlined';
  
  // パディング
  padding?: 'none' | 'sm' | 'md' | 'lg';
  
  // 装飾
  hoverable?: boolean;
  clickable?: boolean;
  selected?: boolean;
  
  // コンテンツ
  header?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  
  // イベント
  onClick?: () => void;
}
```

## 4. ナビゲーションコンポーネント

### 4.1 Tabs（タブ）

```typescript
interface TabItem {
  key: string;
  label: string;
  icon?: IconName;
  badge?: number | string;
  disabled?: boolean;
}

interface TabsProps {
  // タブ
  items: TabItem[];
  activeKey?: string;
  defaultActiveKey?: string;
  
  // スタイル
  variant?: 'line' | 'card' | 'button';
  size?: 'sm' | 'md' | 'lg';
  
  // レイアウト
  orientation?: 'horizontal' | 'vertical';
  align?: 'start' | 'center' | 'end';
  
  // イベント
  onChange?: (key: string) => void;
  
  // コンテンツ
  children: (activeKey: string) => React.ReactNode;
}
```

### 4.2 Breadcrumb（パンくずリスト）

```typescript
interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: IconName;
  onClick?: () => void;
}

interface BreadcrumbProps {
  items: BreadcrumbItem[];
  separator?: React.ReactNode;
  maxItems?: number;
  itemsBeforeCollapse?: number;
  itemsAfterCollapse?: number;
}
```

### 4.3 Pagination（ページネーション）

```typescript
interface PaginationProps {
  // ページ情報
  current: number;
  total: number;
  pageSize?: number;
  
  // 表示
  showSizeChanger?: boolean;
  pageSizeOptions?: number[];
  showTotal?: boolean;
  showJumper?: boolean;
  
  // スタイル
  size?: 'sm' | 'md' | 'lg';
  simple?: boolean;
  
  // イベント
  onChange?: (page: number, pageSize: number) => void;
  onShowSizeChange?: (current: number, size: number) => void;
}
```

## 5. データ表示コンポーネント

### 5.1 Table（テーブル）

```typescript
interface Column<T> {
  key: string;
  title: string;
  dataIndex?: string;
  width?: number | string;
  align?: 'left' | 'center' | 'right';
  fixed?: 'left' | 'right';
  sortable?: boolean;
  filterable?: boolean;
  render?: (value: any, record: T, index: number) => React.ReactNode;
}

interface TableProps<T> {
  // データ
  columns: Column<T>[];
  data: T[];
  rowKey: string | ((record: T) => string);
  
  // 機能
  selectable?: boolean;
  expandable?: boolean;
  editable?: boolean;
  
  // ページネーション
  pagination?: false | PaginationProps;
  
  // スタイル
  size?: 'sm' | 'md' | 'lg';
  bordered?: boolean;
  striped?: boolean;
  hoverable?: boolean;
  
  // イベント
  onRowClick?: (record: T, index: number) => void;
  onSelectionChange?: (selectedKeys: string[]) => void;
}
```

### 5.2 List（リスト）

```typescript
interface ListItemProps {
  avatar?: React.ReactNode;
  title: React.ReactNode;
  description?: React.ReactNode;
  extra?: React.ReactNode;
  actions?: React.ReactNode[];
  onClick?: () => void;
}

interface ListProps<T> {
  dataSource: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  
  // レイアウト
  layout?: 'horizontal' | 'vertical';
  size?: 'sm' | 'md' | 'lg';
  
  // 機能
  loading?: boolean;
  loadMore?: React.ReactNode;
  
  // スタイル
  bordered?: boolean;
  split?: boolean;
}
```

## 6. フィードバックコンポーネント

### 6.1 Alert（アラート）

```typescript
interface AlertProps {
  // タイプ
  type: 'info' | 'success' | 'warning' | 'error';
  
  // コンテンツ
  title?: string;
  description?: React.ReactNode;
  icon?: IconName | boolean;
  
  // アクション
  closable?: boolean;
  action?: React.ReactNode;
  
  // イベント
  onClose?: () => void;
}
```

### 6.2 Toast（トースト）

```typescript
interface ToastOptions {
  type?: 'info' | 'success' | 'warning' | 'error';
  title: string;
  description?: string;
  duration?: number; // ms
  position?: 'top' | 'top-right' | 'top-left' | 'bottom' | 'bottom-right' | 'bottom-left';
  closable?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// 使用例
toast.success({
  title: '保存しました',
  description: 'テックパックが正常に保存されました',
  duration: 3000
});
```

### 6.3 Modal（モーダル）

```typescript
interface ModalProps {
  // 表示
  open: boolean;
  onClose: () => void;
  
  // コンテンツ
  title?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  
  // サイズ
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  
  // 動作
  closeOnOverlayClick?: boolean;
  closeOnEscape?: boolean;
  preventScroll?: boolean;
  
  // スタイル
  centered?: boolean;
  className?: string;
}
```

### 6.4 Popover / Tooltip

```typescript
interface PopoverProps {
  // トリガー
  trigger: 'hover' | 'click' | 'focus';
  
  // コンテンツ
  content: React.ReactNode;
  children: React.ReactElement;
  
  // 位置
  placement?: 'top' | 'right' | 'bottom' | 'left' | 'auto';
  offset?: [number, number];
  
  // 動作
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  
  // スタイル
  arrow?: boolean;
  maxWidth?: number | string;
}
```

## 7. 専用コンポーネント

### 7.1 SVGEditor（SVGエディター）

```typescript
interface SVGEditorProps {
  // データ
  value?: string;
  defaultValue?: string;
  onChange?: (svg: string) => void;
  
  // ツール
  tools?: Tool[];
  defaultTool?: ToolType;
  
  // キャンバス
  width?: number;
  height?: number;
  gridSize?: number;
  snapToGrid?: boolean;
  
  // パーツライブラリ
  showPartLibrary?: boolean;
  partCategories?: PartCategory[];
  onPartSelect?: (part: SVGPart) => void;
  
  // UIオプション
  showToolbar?: boolean;
  showSidebar?: boolean;
  showRuler?: boolean;
  showGrid?: boolean;
}
```

### 7.2 MeasurementTable（寸法表）

```typescript
interface MeasurementTableProps {
  // データ
  measurements: Measurement[];
  onChange?: (measurements: Measurement[]) => void;
  
  // グレーディング
  enableGrading?: boolean;
  sizes?: string[];
  baseSize?: string;
  
  // 許容差
  tolerances?: Tolerance[];
  defaultTolerance?: number;
  
  // UIオプション
  editable?: boolean;
  showDiagram?: boolean;
  highlightOnHover?: boolean;
  
  // バリデーション
  validationRules?: ValidationRule[];
  onValidationError?: (errors: ValidationError[]) => void;
}
```

### 7.3 ColorPicker（カラーピッカー）

```typescript
interface ColorPickerProps {
  // 値
  value?: string;
  defaultValue?: string;
  onChange?: (color: string) => void;
  
  // フォーマット
  format?: 'hex' | 'rgb' | 'hsl';
  alpha?: boolean;
  
  // プリセット
  presets?: string[];
  showPresets?: boolean;
  
  // UI
  showInput?: boolean;
  size?: 'sm' | 'md' | 'lg';
  placement?: 'top' | 'bottom' | 'left' | 'right';
}
```

### 7.4 FileUpload（ファイルアップロード）

```typescript
interface FileUploadProps {
  // 設定
  accept?: string;
  multiple?: boolean;
  maxSize?: number; // bytes
  maxFiles?: number;
  
  // アップロード
  action?: string;
  headers?: Record<string, string>;
  data?: Record<string, any>;
  
  // UI
  droppable?: boolean;
  showFileList?: boolean;
  listType?: 'text' | 'picture' | 'picture-card';
  
  // イベント
  onChange?: (files: UploadFile[]) => void;
  onUpload?: (file: File) => Promise<UploadResponse>;
  onRemove?: (file: UploadFile) => void | boolean;
  
  // カスタマイズ
  renderUploadButton?: () => React.ReactNode;
  renderFileItem?: (file: UploadFile) => React.ReactNode;
}
```

## 8. ユーティリティコンポーネント

### 8.1 Loading（ローディング）

```typescript
interface LoadingProps {
  // タイプ
  type?: 'spinner' | 'dots' | 'bar' | 'skeleton';
  
  // サイズ
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  
  // テキスト
  text?: string;
  textPosition?: 'top' | 'bottom' | 'left' | 'right';
  
  // オーバーレイ
  overlay?: boolean;
  blur?: boolean;
}
```

### 8.2 Skeleton（スケルトン）

```typescript
interface SkeletonProps {
  // タイプ
  variant?: 'text' | 'circular' | 'rectangular';
  
  // サイズ
  width?: number | string;
  height?: number | string;
  
  // アニメーション
  animation?: 'pulse' | 'wave' | false;
  
  // 複数行
  count?: number;
  spacing?: number;
}
```

### 8.3 Divider（区切り線）

```typescript
interface DividerProps {
  // 方向
  orientation?: 'horizontal' | 'vertical';
  
  // スタイル
  variant?: 'solid' | 'dashed' | 'dotted';
  
  // テキスト
  children?: React.ReactNode;
  textAlign?: 'left' | 'center' | 'right';
  
  // 間隔
  spacing?: number | string;
}
```

## 9. テーマ設定

### 9.1 テーマプロバイダー

```typescript
interface Theme {
  colors: {
    primary: ColorScale;
    gray: ColorScale;
    success: string;
    warning: string;
    error: string;
    info: string;
  };
  
  typography: {
    fontFamily: {
      sans: string;
      mono: string;
    };
    fontSize: {
      xs: string;
      sm: string;
      base: string;
      lg: string;
      xl: string;
      '2xl': string;
      '3xl': string;
    };
    fontWeight: {
      light: number;
      normal: number;
      medium: number;
      semibold: number;
      bold: number;
    };
    lineHeight: {
      tight: number;
      normal: number;
      relaxed: number;
    };
  };
  
  spacing: {
    [key: number]: string;
  };
  
  borderRadius: {
    none: string;
    sm: string;
    base: string;
    md: string;
    lg: string;
    xl: string;
    full: string;
  };
  
  shadows: {
    none: string;
    sm: string;
    base: string;
    md: string;
    lg: string;
    xl: string;
  };
  
  transitions: {
    fast: string;
    base: string;
    slow: string;
  };
  
  breakpoints: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
}
```

### 9.2 ダークモード対応

```typescript
interface DarkModeConfig {
  // モード
  mode: 'light' | 'dark' | 'auto';
  
  // トグル方法
  storageKey?: string;
  defaultMode?: 'light' | 'dark';
  
  // カスタマイズ
  darkTheme?: Partial<Theme>;
  
  // トランジション
  transition?: boolean;
  transitionDuration?: number;
}
```

## 10. アクセシビリティ

### 10.1 ARIA属性

全てのコンポーネントは適切なARIA属性をサポート：

- `aria-label`
- `aria-labelledby`
- `aria-describedby`
- `aria-required`
- `aria-invalid`
- `aria-disabled`
- `aria-expanded`
- `aria-selected`
- `aria-checked`
- `role`

### 10.2 キーボードナビゲーション

全てのインタラクティブコンポーネントはキーボード操作をサポート：

- Tabキーでのフォーカス移動
- Enter/Spaceキーでのアクション実行
- 矢印キーでのナビゲーション
- Escapeキーでのキャンセル

## 11. パフォーマンス最適化

### 11.1 レンダリング最適化

- React.memoを使用したメモ化
- useMemo/useCallbackの適切な使用
- 仮想スクロールの実装（大量データの場合）
- 遅延ローディング

### 11.2 バンドルサイズ最適化

- Tree shaking対応
- コンポーネント単位でのインポート
- CSS-in-JSの最適化
- アイコンのSVGスプライト化