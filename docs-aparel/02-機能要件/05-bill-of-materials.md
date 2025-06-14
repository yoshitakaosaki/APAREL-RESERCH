# 機能要件仕様書 - 05.BOM（部材表）管理機能

## 1. 機能概要

BOM（Bill of Materials）管理機能は、製品を構成するすべての材料・部材を一元管理し、調達から原価計算まで幅広く活用される中核機能です。正確性と効率性を両立し、サプライチェーン全体の最適化に貢献します。

## 2. 機能要件

### 2.1 部材マスタ管理機能

#### 素材マスタ
- **基本情報管理**:
  - 素材名（多言語対応）
  - 素材コード（自動採番/手動）
  - カテゴリー分類（階層構造）
  - サプライヤー情報
- **技術仕様**:
  - 組成（%表示、合計100%検証）
  - 目付（g/m²、oz/yd²）
  - 幅（有効幅、総幅）
  - 織り・編み構造
  - 仕上げ加工情報
- **品質情報**:
  - 物性データ（強度、伸度等）
  - 堅牢度データ
  - 認証情報（OEKO-TEX等）
  - 検査証明書添付

#### 付属品マスタ
- **分類管理**:
  - ボタン、ファスナー、芯地等
  - ブランド別管理
  - サイズ・色展開
- **仕様詳細**:
  - 寸法・規格
  - 材質・構成
  - 表面処理
  - 取付方法
- **調達情報**:
  - MOQ（最小発注数量）
  - リードタイム
  - 単価情報
  - 代替品設定

### 2.2 BOM構築機能

#### 階層構造管理
- **マルチレベルBOM**:
  - 製品→パーツ→素材の階層
  - アセンブリ単位の管理
  - サブアセンブリ対応
  - 無限階層サポート
- **部位別割当**:
  - ビジュアルマッピング
  - ドラッグ&ドロップ割当
  - 一括割当機能
  - コピー&ペースト対応

#### 数量計算機能
- **使用量算出**:
  - パターン効率計算
  - ロス率設定（5-15%）
  - サイズ別数量計算
  - カラー別集計
- **自動計算**:
  - 要尺計算
  - 付属品個数計算
  - 予備率適用
  - 端数処理ルール

### 2.3 カラー・サイズ展開管理

#### カラーマトリックス
- **カラー別BOM**:
  - 基本色設定
  - カラー別素材指定
  - 配色パターン管理
  - カラーグループ化
- **ビジュアル表示**:
  - カラーチップ表示
  - Pantone連携
  - LAB値管理
  - 画像サンプル添付

#### サイズ展開対応
- **サイズ別数量**:
  - グレーディング対応
  - サイズ別要尺
  - 比率計算
  - 効率最適化
- **生産数量連動**:
  - 発注数量入力
  - 自動集計
  - ロット管理
  - 在庫引当

### 2.4 サプライヤー管理

#### ベンダー情報
- **基本情報**:
  - 企業情報
  - 連絡先（複数登録）
  - 取引条件
  - 評価履歴
- **供給能力**:
  - 生産キャパシティ
  - 認証情報
  - 品質レベル
  - 納期実績

#### 発注管理連携
- **見積機能**:
  - 複数見積比較
  - 価格履歴
  - 為替対応
  - 条件交渉メモ
- **発注書生成**:
  - 自動生成
  - 承認ワークフロー
  - EDI連携
  - 進捗追跡

### 2.5 原価計算機能

#### コスト集計
- **材料費計算**:
  - 素材費自動集計
  - 付属品費集計
  - ロス込み計算
  - 通貨換算
- **詳細分析**:
  - 部位別コスト
  - ABC分析
  - 原価構成比
  - 目標原価比較

#### シミュレーション
- **What-if分析**:
  - 素材変更シミュレーション
  - 数量変更影響
  - 為替変動対応
  - 代替品検討

## 3. UI/UX要件

### 3.1 画面構成
- **リスト/グリッド表示**:
  - 階層ツリー表示
  - 詳細グリッド
  - カード表示
  - カスタムビュー
- **フィルター・検索**:
  - 多条件検索
  - お気に入り登録
  - 履歴参照
  - スマートサジェスト

### 3.2 操作性
- **ドラッグ&ドロップ**:
  - 素材割当
  - 順序変更
  - グループ化
  - 一括操作
- **インライン編集**:
  - セル直接編集
  - 一括更新
  - 数式入力
  - コピー&ペースト

### 3.3 ビジュアル表現
- **グラフィカル表示**:
  - 構成ツリー図
  - 円グラフ（構成比）
  - 積み上げグラフ
  - ヒートマップ
- **画像統合**:
  - 素材画像表示
  - ズーム機能
  - 比較表示
  - スライドショー

## 4. 連携要件

### 4.1 セクション間連携
- **カバーページ（01）**:
  - 基本情報参照
  - 素材サマリー表示
- **フラット図（02-03）**:
  - 部位マッピング
  - ビジュアル確認
- **カラーウェイ（06）**:
  - 色展開同期
  - 配色確認
- **生地詳細（17）**:
  - 詳細仕様参照
  - 双方向更新
- **原価計算（19）**:
  - コストデータ提供
  - 自動転記

### 4.2 外部システム連携
- **ERP連携**:
  - マスタ同期
  - 在庫参照
  - 発注連携
- **PLM連携**:
  - 素材ライブラリ
  - 承認履歴
  - 変更管理
- **EDI対応**:
  - 標準フォーマット
  - 自動送受信
  - エラー処理

## 5. バリデーション要件

### 5.1 データ整合性
- **必須チェック**:
  - 主要素材指定
  - 数量入力
  - 単位統一
  - 合計検証
- **論理チェック**:
  - 組成合計100%
  - 数量妥当性
  - 重複チェック
  - 依存関係確認

### 5.2 業務ルール
- **承認要件**:
  - 金額閾値
  - 新規サプライヤー
  - 仕様変更
  - 代替品使用
- **警告表示**:
  - MOQ未達
  - 納期遅延リスク
  - 在庫不足
  - 価格変動

## 6. 性能要件

- **データ処理**:
  - 1000品目まで：1秒以内表示
  - 10000品目まで：5秒以内表示
  - リアルタイム集計
- **同時アクセス**:
  - 100ユーザー同時編集
  - 楽観的ロック
  - 競合解決機能
- **レスポンス**:
  - 検索：200ms以内
  - 更新：500ms以内
  - 集計：1秒以内

## 7. データ管理

### 7.1 履歴管理
- **変更履歴**:
  - 全項目の変更追跡
  - 変更理由記録
  - 承認履歴
  - ロールバック機能
- **バージョン管理**:
  - BOMバージョン
  - 有効期限設定
  - 比較機能
  - マージ機能

### 7.2 インポート/エクスポート
- **対応形式**:
  - Excel（テンプレート提供）
  - CSV/TSV
  - XML/JSON
  - PDF（レポート）
- **マッピング機能**:
  - カラム対応設定
  - 変換ルール
  - バリデーション
  - エラーレポート

## 8. セキュリティ要件

- **アクセス制御**:
  - 素材別権限
  - サプライヤー別制限
  - 価格情報マスキング
  - 承認権限階層
- **監査証跡**:
  - 全操作ログ
  - ログイン履歴
  - エクスポート記録
  - 異常検知アラート