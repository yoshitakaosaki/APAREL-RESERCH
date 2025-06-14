# 機能要件仕様書 - 17.生地詳細機能

## 1. 機能概要

生地詳細機能は、製品に使用される生地の技術仕様、品質特性、調達情報を包括的に管理し、素材選定から品質管理まで一貫したデータ管理を実現する機能です。素材の特性を最大限に活かした製品開発を支援します。

## 2. 機能要件

### 2.1 生地マスタ管理

#### 基本情報登録
- **識別情報**:
  - 生地名（商品名/一般名）
  - 品番/品名
  - メーカー/ミル
  - 開発シーズン/年度
- **分類情報**:
  - 素材カテゴリー
  - 用途分類
  - 機能分類
  - 価格帯
- **画像管理**:
  - 生地見本写真
  - 拡大画像
  - 表裏画像
  - カラーバリエーション

#### 技術仕様管理
- **素材組成**:
  - 繊維種類/混率（%）
  - 繊度（デニール/番手）
  - 撚り仕様
  - 原料産地
- **構造仕様**:
  - 織り/編み組織
  - 密度（タテ×ヨコ）
  - 目付（g/m²、oz/yd²）
  - 厚み（mm）
  - 幅（有効幅/総幅）

### 2.2 品質特性管理

#### 物性データ
- **基本物性**:
  - 引張強度（N）
  - 破裂強度（kPa）
  - 引裂強度（N）
  - 摩耗強度（回）
- **寸法安定性**:
  - 洗濯収縮率（%）
  - 熱収縮率
  - 伸長回復率
  - 型崩れ性
- **快適性能**:
  - 通気性（cc/cm²/s）
  - 吸水性/速乾性
  - 保温性（clo値）
  - 透湿性（g/m²/24h）

#### 堅牢度データ
- **色堅牢度**:
  - 洗濯堅牢度（変退色/汚染）
  - 摩擦堅牢度（乾/湿）
  - 汗堅牢度（酸/アルカリ）
  - 光堅牢度（級）
- **機能堅牢度**:
  - 撥水耐久性
  - 防汚性能
  - 抗菌性維持
  - UVカット性能

### 2.3 加工仕様管理

#### 仕上げ加工
- **機能加工**:
  - 撥水/防水加工
  - 防汚/防臭加工
  - 形態安定加工
  - ストレッチ加工
- **風合い加工**:
  - 柔軟加工
  - 防しわ加工
  - 起毛/シャーリング
  - エンボス/プリーツ
- **特殊加工**:
  - コーティング
  - ラミネート
  - ボンディング
  - 機能性付与

#### 染色仕様
- **染色方法**:
  - 先染め/後染め
  - 染料種類
  - 染色工程
  - 色止め処理
- **プリント対応**:
  - 適正プリント手法
  - インク相性
  - 前処理要否
  - 発色特性

### 2.4 調達情報管理

#### サプライヤー情報
- **取引先詳細**:
  - メーカー/商社
  - 工場所在地
  - 連絡先情報
  - 取引実績
- **供給条件**:
  - MOQ（最小発注量）
  - リードタイム
  - 在庫状況
  - 生産能力

#### 価格・条件管理
- **価格情報**:
  - 単価（通貨別）
  - 数量割引
  - 季節変動
  - 為替考慮
- **取引条件**:
  - 支払条件
  - 納品形態
  - 品質保証
  - 返品条件

### 2.5 用途・制限管理

#### 推奨用途
- **アイテム適性**:
  - 推奨アイテム
  - 適合デザイン
  - 不適合用途
  - 注意事項
- **加工適性**:
  - 縫製上の注意
  - プレス条件
  - 接着相性
  - 特殊ミシン要否

#### 使用制限
- **技術的制限**:
  - 温度制限
  - 化学薬品耐性
  - 機械的制約
  - 経年変化
- **法的制限**:
  - 有害物質規制
  - 環境規制対応
  - 認証取得状況
  - 輸出入制限

## 3. UI/UX要件

### 3.1 検索・フィルター
- **多条件検索**:
  - 素材組成
  - 物性範囲
  - 価格帯
  - 在庫有無
- **ビジュアル検索**:
  - 色検索
  - 風合い検索
  - パターン検索
  - 類似検索

### 3.2 比較機能
- **並列比較**:
  - 最大6点比較
  - 項目選択表示
  - 差分ハイライト
  - レーダーチャート
- **評価支援**:
  - スコアリング
  - 重み付け設定
  - ランキング表示
  - 推奨提示

## 4. 連携要件

### 4.1 BOM連携
- **自動転記**:
  - 素材情報提供
  - 数量計算支援
  - 価格連動
  - 在庫確認

### 4.2 品質連携
- **ケアラベル（15）**:
  - 洗濯表示根拠
  - 制限事項反映
- **試験データ**:
  - 結果自動取込
  - 判定支援

## 5. バリデーション要件

### 5.1 データ完全性
- **必須項目**:
  - 組成情報
  - 基本物性
  - 供給者情報
  - 価格情報

### 5.2 整合性確認
- **技術的妥当性**:
  - 物性値範囲
  - 組成合計100%
  - 試験法準拠
  - 単位統一

## 6. 性能要件

- **データベース**:
  - 50,000生地管理
  - 高速検索（1秒以内）
  - 画像表示（2秒以内）
- **分析処理**:
  - 比較演算：即時
  - 統計処理：5秒以内

## 7. 出力機能

### 7.1 技術資料
- **生地仕様書**:
  - 詳細スペック
  - 試験成績書
  - 取扱注意書
  - 証明書類

### 7.2 調達資料
- **発注関連**:
  - 見積依頼書
  - 発注仕様書
  - 品質要求書
  - 検査基準書