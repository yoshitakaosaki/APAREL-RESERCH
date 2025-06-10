# AIエージェント実装 技術スタック詳細仕様書

## 1. 概要

本ドキュメントは、テックパック生成アプリケーションのAIエージェント実装で使用する各技術の詳細、適用目的、適用範囲を包括的に説明します。

## 2. コア技術スタック

### 2.1 プログラミング言語

#### Python 3.11+

**適用目的**
- AI/ML ライブラリの豊富なエコシステム活用
- 非同期処理による高性能実装
- 型ヒントによる開発効率向上

**適用範囲**
- 全AIエージェントの実装
- MLモデルの統合
- API サーバー実装
- バッチ処理タスク

**選定理由**
```python
# Python 3.11+ の新機能活用例
from typing import TypeAlias, Self
from dataclasses import dataclass
import asyncio

# 型エイリアスによる可読性向上
ResponseData: TypeAlias = dict[str, Any]

# 構造的パターンマッチング（3.10+）
def process_task(task_type: str) -> str:
    match task_type:
        case "term_extraction":
            return "nlp_pipeline"
        case "svg_generation":
            return "cv_pipeline"
        case _:
            return "default_pipeline"

# 高速化された例外処理（3.11+）
try:
    result = await process_with_timeout()
except TimeoutError:
    # 3.11では例外処理が最大10%高速化
    handle_timeout()
```

### 2.2 Webフレームワーク

#### FastAPI 0.109.0+

**適用目的**
- 高性能な非同期APIサーバー構築
- 自動API文書生成
- 型安全性の確保
- WebSocket対応

**適用範囲**
- REST API エンドポイント
- WebSocket通信
- 非同期タスク管理API
- ヘルスチェックエンドポイント

**実装例**
```python
from fastapi import FastAPI, WebSocket, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

app = FastAPI(title="AI Agent API", version="1.0.0")

class TaskRequest(BaseModel):
    agent_type: str = Field(..., description="エージェントタイプ")
    priority: str = Field(default="normal", pattern="^(low|normal|high)$")
    configuration: dict = Field(..., description="タスク設定")
    
    class Config:
        schema_extra = {
            "example": {
                "agent_type": "term_collector",
                "priority": "high",
                "configuration": {
                    "source": {"type": "document", "url": "https://..."},
                    "languages": ["ja", "en"]
                }
            }
        }

@app.post("/api/v1/ai/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks
):
    """AIタスクを作成し、非同期で実行"""
    task_id = generate_task_id()
    
    # バックグラウンドタスクとして実行
    background_tasks.add_task(
        process_ai_task,
        task_id,
        request
    )
    
    return TaskResponse(
        task_id=task_id,
        status="queued",
        estimated_time=estimate_processing_time(request)
    )

@app.websocket("/ws/tasks/{task_id}")
async def task_progress(websocket: WebSocket, task_id: str):
    """タスク進捗のリアルタイム配信"""
    await websocket.accept()
    
    try:
        while True:
            progress = await get_task_progress(task_id)
            await websocket.send_json({
                "task_id": task_id,
                "progress": progress.percentage,
                "status": progress.status,
                "current_step": progress.current_step
            })
            
            if progress.is_completed:
                break
                
            await asyncio.sleep(1)
    finally:
        await websocket.close()
```

### 2.3 非同期タスクキュー

#### Celery 5.3.4+

**適用目的**
- 長時間実行タスクの非同期処理
- 分散タスク実行
- タスクの優先度管理
- リトライ機構

**適用範囲**
- AI推論タスク
- ドキュメント処理
- バッチ処理
- 定期実行タスク

**実装例**
```python
from celery import Celery, Task
from celery.signals import task_prerun, task_postrun
from kombu import Exchange, Queue
import time

# Celeryアプリケーション設定
celery_app = Celery('ai_agents')

celery_app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/1',
    
    # タスクルーティング
    task_routes={
        'agents.nlp.*': {'queue': 'nlp_queue'},
        'agents.cv.*': {'queue': 'cv_queue'},
        'agents.ml.*': {'queue': 'ml_queue'}
    },
    
    # 優先度付きキュー
    task_queues=(
        Queue('nlp_queue', Exchange('nlp'), routing_key='nlp',
              queue_arguments={'x-max-priority': 10}),
        Queue('cv_queue', Exchange('cv'), routing_key='cv',
              queue_arguments={'x-max-priority': 10}),
    ),
    
    # パフォーマンス設定
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=1000,
    task_acks_late=True,
    
    # タイムアウト設定
    task_time_limit=3600,  # ハードリミット: 1時間
    task_soft_time_limit=3300,  # ソフトリミット: 55分
)

# カスタムタスククラス
class MLTask(Task):
    """機械学習タスクの基底クラス"""
    
    _model = None
    
    @property
    def model(self):
        """モデルの遅延ロード"""
        if self._model is None:
            self._model = self.load_model()
        return self._model
    
    def load_model(self):
        """モデルロード（サブクラスで実装）"""
        raise NotImplementedError

@celery_app.task(
    bind=True,
    base=MLTask,
    name='agents.nlp.extract_terms',
    max_retries=3,
    default_retry_delay=60
)
def extract_terms_task(self, document_url: str, config: dict):
    """用語抽出タスク"""
    try:
        # ドキュメント取得
        document = fetch_document(document_url)
        
        # NLP処理
        terms = self.model.extract_terms(
            document,
            languages=config.get('languages', ['ja']),
            confidence_threshold=config.get('confidence', 0.7)
        )
        
        return {
            'status': 'success',
            'terms': terms,
            'processing_time': time.time() - self.request.time_start
        }
        
    except SoftTimeLimitExceeded:
        # ソフトタイムアウト - グレースフルシャットダウン
        return {
            'status': 'timeout',
            'partial_results': self.get_partial_results()
        }
    except Exception as exc:
        # リトライ
        raise self.retry(exc=exc)
```

### 2.4 データベース

#### PostgreSQL 15+ with pgvector

**適用目的**
- 構造化データの永続化
- ベクトル類似検索
- トランザクション管理
- 全文検索

**適用範囲**
- タスクメタデータ
- 用語集データベース
- ベクトル埋め込み保存
- 監査ログ

**実装例**
```sql
-- pgvector拡張の有効化
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- 類似テキスト検索

-- 用語テーブル（ベクトル検索対応）
CREATE TABLE terms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- JSONB for flexible schema
    translations JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    -- Vector embedding for semantic search
    embedding vector(768),  -- BERT-based embeddings
    
    -- Full-text search
    search_vector tsvector GENERATED ALWAYS AS (
        to_tsvector('japanese', 
            COALESCE(translations->>'ja'->>'name', '') || ' ' ||
            COALESCE(translations->>'ja'->>'description', '')
        )
    ) STORED
);

-- インデックス
CREATE INDEX idx_terms_embedding ON terms 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

CREATE INDEX idx_terms_search ON terms 
USING gin(search_vector);

CREATE INDEX idx_terms_metadata ON terms 
USING gin(metadata jsonb_path_ops);

-- 類似用語検索関数
CREATE OR REPLACE FUNCTION find_similar_terms(
    query_embedding vector(768),
    limit_count int DEFAULT 10,
    threshold float DEFAULT 0.8
)
RETURNS TABLE(
    id UUID,
    code VARCHAR,
    similarity float,
    translations JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.code,
        1 - (t.embedding <=> query_embedding) as similarity,
        t.translations
    FROM terms t
    WHERE 1 - (t.embedding <=> query_embedding) > threshold
    ORDER BY t.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;
```

#### SQLAlchemy 2.0.25

**適用目的**
- ORM によるデータベース抽象化
- 非同期データベース操作
- マイグレーション管理
- コネクションプーリング

**実装例**
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, DateTime, JSON
from pgvector.sqlalchemy import Vector
import uuid

# 非同期エンジン設定
engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/aiagents",
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

class Term(Base):
    __tablename__ = "terms"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String(50), unique=True, nullable=False)
    translations = Column(JSON, nullable=False, default=dict)
    metadata = Column(JSON, default=dict)
    embedding = Column(Vector(768))  # pgvector型
    created_at = Column(DateTime, server_default=func.now())
    
    # ハイブリッド検索メソッド
    @classmethod
    async def hybrid_search(
        cls,
        session: AsyncSession,
        query_text: str,
        query_embedding: list,
        alpha: float = 0.5  # テキスト検索とベクトル検索の重み
    ):
        """テキスト検索とベクトル検索を組み合わせた検索"""
        
        # ベクトル類似度検索
        vector_similarity = func.greatest(
            0,
            1 - cls.embedding.cosine_distance(query_embedding)
        )
        
        # テキスト類似度検索
        text_similarity = func.similarity(
            cls.translations['ja']['name'].astext,
            query_text
        )
        
        # ハイブリッドスコア
        hybrid_score = (
            alpha * text_similarity + 
            (1 - alpha) * vector_similarity
        )
        
        results = await session.execute(
            select(cls, hybrid_score.label('score'))
            .filter(
                or_(
                    text_similarity > 0.3,
                    vector_similarity > 0.7
                )
            )
            .order_by(hybrid_score.desc())
            .limit(20)
        )
        
        return results.all()
```

### 2.5 キャッシュ/メッセージブローカー

#### Redis 7+

**適用目的**
- 高速キャッシュ
- セッション管理
- パブリッシュ/サブスクライブ
- 分散ロック

**適用範囲**
- API レスポンスキャッシュ
- タスクキュー（Celery バックエンド）
- リアルタイム通信
- レート制限

**実装例**
```python
import redis.asyncio as redis
from typing import Optional, Any
import json
import asyncio
from datetime import timedelta

class RedisCache:
    """Redis キャッシュマネージャー"""
    
    def __init__(self, url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(url, decode_responses=True)
        
    async def get_or_set(
        self,
        key: str,
        factory_fn,
        ttl: Optional[int] = 3600
    ) -> Any:
        """キャッシュから取得、なければ生成して保存"""
        
        # キャッシュチェック
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        
        # 分散ロックで重複実行防止
        lock_key = f"lock:{key}"
        lock = await self.redis.set(
            lock_key, "1", 
            nx=True,  # 存在しない場合のみセット
            ex=30     # 30秒でロック解放
        )
        
        if not lock:
            # 他のプロセスが処理中 - 待機
            await asyncio.sleep(0.1)
            return await self.get_or_set(key, factory_fn, ttl)
        
        try:
            # 値を生成
            value = await factory_fn()
            
            # キャッシュに保存
            await self.redis.setex(
                key,
                ttl,
                json.dumps(value)
            )
            
            return value
        finally:
            # ロック解放
            await self.redis.delete(lock_key)
    
    async def invalidate_pattern(self, pattern: str):
        """パターンマッチでキャッシュ無効化"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(
                cursor, 
                match=pattern,
                count=100
            )
            
            if keys:
                await self.redis.delete(*keys)
            
            if cursor == 0:
                break

# Pub/Sub実装
class RedisPubSub:
    """Redis Pub/Sub マネージャー"""
    
    def __init__(self, url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(url)
        self.pubsub = self.redis.pubsub()
        
    async def publish_task_update(
        self, 
        task_id: str, 
        update: dict
    ):
        """タスク更新を配信"""
        channel = f"task:{task_id}"
        message = json.dumps({
            "timestamp": time.time(),
            "update": update
        })
        
        await self.redis.publish(channel, message)
    
    async def subscribe_task_updates(
        self, 
        task_id: str,
        handler
    ):
        """タスク更新を購読"""
        channel = f"task:{task_id}"
        await self.pubsub.subscribe(channel)
        
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    data = json.loads(message['data'])
                    await handler(data)
        finally:
            await self.pubsub.unsubscribe(channel)
```

## 3. AI/機械学習フレームワーク

### 3.1 深層学習フレームワーク

#### PyTorch 2.1.2

**適用目的**
- ニューラルネットワークモデルの実装
- カスタムモデルの訓練
- 推論の最適化
- GPU活用

**適用範囲**
- LLMの推論
- 画像認識モデル
- カスタムML モデル
- ファインチューニング

**実装例**
```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist

class OptimizedModelInference:
    """最適化されたモデル推論"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # モデルロード
        self.model = self.load_model(model_path)
        
        # 最適化
        self.model.eval()
        if self.device.type == "cuda":
            self.model = self.model.half()  # FP16推論
            self.model = torch.jit.script(self.model)  # JITコンパイル
            
        # バッチ処理用の設定
        self.max_batch_size = 32
        self.use_amp = True
        
    def load_model(self, path: str):
        """モデルロード（チェックポイント対応）"""
        checkpoint = torch.load(path, map_location=self.device)
        
        model = YourModelClass()  # モデル定義
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    @torch.no_grad()
    def batch_inference(self, inputs: list) -> list:
        """バッチ推論with最適化"""
        results = []
        
        # Dynamic batching
        for i in range(0, len(inputs), self.max_batch_size):
            batch = inputs[i:i + self.max_batch_size]
            batch_tensor = self.preprocess_batch(batch)
            
            # Mixed precision推論
            if self.use_amp and self.device.type == "cuda":
                with autocast():
                    outputs = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
            
            # 後処理
            batch_results = self.postprocess_outputs(outputs)
            results.extend(batch_results)
        
        return results
    
    def optimize_for_deployment(self):
        """本番環境向け最適化"""
        # TorchScript変換
        example_input = torch.randn(1, 3, 224, 224).to(self.device)
        traced_model = torch.jit.trace(self.model, example_input)
        
        # 量子化（INT8）
        quantized_model = torch.quantization.quantize_dynamic(
            traced_model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # ONNXエクスポート（オプション）
        torch.onnx.export(
            self.model,
            example_input,
            "model.onnx",
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return quantized_model
```

#### Transformers 4.36.2

**適用目的**
- 事前学習済みNLPモデルの活用
- トークナイザーの統一管理
- モデルハブへのアクセス
- ファインチューニング

**適用範囲**
- BERT/GPT系モデルの使用
- 多言語NLP処理
- 埋め込み生成
- テキスト生成

**実装例**
```python
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import numpy as np

class MultilingualNLPProcessor:
    """多言語NLP処理"""
    
    def __init__(self):
        # 日本語BERT
        self.ja_tokenizer = AutoTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese-v3"
        )
        self.ja_model = AutoModelForTokenClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-v3"
        )
        
        # 多言語モデル
        self.multilingual_pipeline = pipeline(
            "token-classification",
            model="xlm-roberta-large-finetuned-conll03-english",
            aggregation_strategy="simple"
        )
        
    def extract_fashion_terms(self, text: str, language: str = "ja"):
        """ファッション用語抽出"""
        
        if language == "ja":
            # 日本語処理
            inputs = self.ja_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            outputs = self.ja_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # トークンをエンティティに変換
            entities = self._convert_to_entities(
                inputs.input_ids[0],
                predictions[0],
                self.ja_tokenizer
            )
            
        else:
            # 多言語処理
            entities = self.multilingual_pipeline(text)
        
        # ファッション関連フィルタリング
        fashion_entities = [
            e for e in entities 
            if self._is_fashion_related(e['word'], language)
        ]
        
        return fashion_entities
    
    def fine_tune_for_fashion(self, training_data: list):
        """ファッション領域へのファインチューニング"""
        
        # データセット準備
        dataset = Dataset.from_list(training_data)
        
        # トークナイズ
        def tokenize_function(examples):
            return self.ja_tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # 訓練設定
        training_args = TrainingArguments(
            output_dir="./fashion-bert",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,  # Mixed precision
            gradient_checkpointing=True,  # メモリ節約
            push_to_hub=False
        )
        
        # トレーナー初期化
        trainer = Trainer(
            model=self.ja_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.ja_tokenizer,
            data_collator=self._data_collator
        )
        
        # 訓練実行
        trainer.train()
        
        return trainer.model
```

### 3.2 コンピュータビジョン

#### OpenCV 4.9.0.80

**適用目的**
- 画像前処理
- 基本的な画像解析
- 形状検出
- 色抽出

**適用範囲**
- 画像からのパーツ検出
- 色分析
- サイズ測定
- 品質チェック

**実装例**
```python
import cv2
import numpy as np
from typing import List, Tuple, Dict

class FashionImageAnalyzer:
    """ファッション画像解析"""
    
    def analyze_garment_image(self, image_path: str) -> Dict:
        """衣服画像の総合解析"""
        
        # 画像読み込み
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 解析結果
        results = {
            "colors": self.extract_dominant_colors(img_rgb),
            "edges": self.detect_edges(img),
            "contours": self.find_garment_contours(img),
            "measurements": self.estimate_measurements(img),
            "quality_score": self.assess_image_quality(img)
        }
        
        return results
    
    def extract_dominant_colors(
        self, 
        image: np.ndarray, 
        n_colors: int = 5
    ) -> List[Dict]:
        """主要色の抽出"""
        
        # 画像をリシェイプ
        pixels = image.reshape(-1, 3)
        
        # K-means クラスタリング
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels.astype(np.float32),
            n_colors,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # 各クラスタの割合を計算
        unique_labels, counts = np.unique(labels, return_counts=True)
        percentages = counts / counts.sum()
        
        # 結果をフォーマット
        colors = []
        for i, (center, percentage) in enumerate(zip(centers, percentages)):
            colors.append({
                "rgb": center.astype(int).tolist(),
                "hex": self.rgb_to_hex(center),
                "percentage": float(percentage),
                "name": self.get_color_name(center)
            })
        
        return sorted(colors, key=lambda x: x['percentage'], reverse=True)
    
    def detect_edges(self, image: np.ndarray) -> np.ndarray:
        """エッジ検出（デザイン要素抽出用）"""
        
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラー
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出
        edges = cv2.Canny(blurred, 50, 150)
        
        # モルフォロジー処理で細かいノイズ除去
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_garment_contours(self, image: np.ndarray) -> List[np.ndarray]:
        """衣服の輪郭検出"""
        
        # 前処理
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 輪郭検出
        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 面積でフィルタリング（ノイズ除去）
        min_area = image.shape[0] * image.shape[1] * 0.01
        significant_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > min_area
        ]
        
        # 輪郭を簡略化
        simplified_contours = []
        for cnt in significant_contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            simplified_contours.append(approx)
        
        return simplified_contours
```

#### Ultralytics YOLOv8

**適用目的**
- 物体検出
- インスタンスセグメンテーション
- 衣服パーツの認識
- リアルタイム処理

**適用範囲**
- ボタン、ポケット等のパーツ検出
- 衣服の種類分類
- 欠陥検出
- 位置特定

**実装例**
```python
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np

class FashionObjectDetector:
    """ファッションアイテム検出"""
    
    def __init__(self, model_path: str = "yolov8n.pt"):
        # カスタムモデルまたは事前学習済みモデル
        self.model = YOLO(model_path)
        
        # ファッション関連クラス
        self.fashion_classes = {
            'button': 0,
            'pocket': 1,
            'collar': 2,
            'zipper': 3,
            'logo': 4,
            'stitch': 5
        }
        
    def detect_fashion_parts(
        self, 
        image_path: str,
        confidence: float = 0.7
    ) -> List[Dict]:
        """ファッションパーツの検出"""
        
        # 推論実行
        results = self.model(
            image_path,
            conf=confidence,
            iou=0.5,
            imgsz=640,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 結果を構造化
        detections = []
        for r in results:
            for box in r.boxes:
                detection = {
                    'class': self.model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                    'area': float((box.xyxy[0][2] - box.xyxy[0][0]) * 
                                 (box.xyxy[0][3] - box.xyxy[0][1]))
                }
                
                # セグメンテーションマスクがある場合
                if hasattr(r, 'masks') and r.masks is not None:
                    detection['mask'] = r.masks.data[0].cpu().numpy()
                
                detections.append(detection)
        
        return detections
    
    def train_custom_model(self, dataset_path: str):
        """カスタムモデルの訓練"""
        
        # YOLOv8訓練設定
        self.model.train(
            data=f"{dataset_path}/dataset.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            device='cuda',
            workers=8,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            box=0.05,
            cls=0.5,
            dfl=1.5,
            patience=50,
            save=True,
            cache=True,
            amp=True  # Mixed precision
        )
        
        # 検証
        metrics = self.model.val()
        
        return metrics
```

### 3.3 自然言語処理

#### spaCy 3.7.2

**適用目的**
- 高速なNLP処理
- 多言語対応
- カスタムNERモデル
- 依存構造解析

**適用範囲**
- テキスト前処理
- 固有表現抽出
- 品詞タグ付け
- 文書類似度計算

**実装例**
```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json

class SpacyNLPProcessor:
    """spaCyベースのNLP処理"""
    
    def __init__(self):
        # 言語別モデル
        self.nlp_ja = spacy.load("ja_core_news_lg")
        self.nlp_en = spacy.load("en_core_web_lg")
        
        # カスタムNERコンポーネント追加
        self._add_fashion_ner()
        
    def _add_fashion_ner(self):
        """ファッション用語NERの追加"""
        
        # カスタムエンティティラベル
        fashion_labels = [
            "GARMENT_TYPE",    # 衣服タイプ
            "FABRIC",          # 生地
            "COLOR",           # 色
            "PATTERN",         # 柄
            "DETAIL",          # ディテール
            "BRAND"            # ブランド
        ]
        
        # NERパイプラインに追加
        ner = self.nlp_ja.get_pipe("ner")
        for label in fashion_labels:
            ner.add_label(label)
    
    def process_fashion_document(
        self, 
        text: str, 
        language: str = "ja"
    ) -> Dict:
        """ファッション文書の解析"""
        
        # 言語別処理
        nlp = self.nlp_ja if language == "ja" else self.nlp_en
        doc = nlp(text)
        
        # エンティティ抽出
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": getattr(ent, "confidence", 1.0)
            })
        
        # 重要な名詞句抽出
        noun_chunks = []
        for chunk in doc.noun_chunks:
            if self._is_fashion_relevant(chunk.text):
                noun_chunks.append({
                    "text": chunk.text,
                    "root": chunk.root.text,
                    "dep": chunk.root.dep_
                })
        
        # 文書ベクトル（類似度計算用）
        doc_vector = doc.vector.tolist()
        
        return {
            "entities": entities,
            "noun_chunks": noun_chunks,
            "doc_vector": doc_vector,
            "token_count": len(doc),
            "sentences": [sent.text for sent in doc.sents]
        }
    
    def train_custom_ner(self, training_data: List[Tuple[str, Dict]]):
        """カスタムNERモデルの訓練"""
        
        # 訓練データをspaCy形式に変換
        examples = []
        for text, annotations in training_data:
            doc = self.nlp_ja.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # 訓練設定
        nlp = self.nlp_ja
        optimizer = nlp.initialize()
        
        # 訓練ループ
        for epoch in range(30):
            losses = {}
            
            # バッチ処理
            for batch in spacy.util.minibatch(examples, size=16):
                nlp.update(batch, sgd=optimizer, losses=losses)
            
            print(f"Epoch {epoch}: {losses}")
        
        # モデル保存
        nlp.to_disk("./custom_fashion_ner")
        
        return nlp
```

#### MeCab 1.0.8

**適用目的**
- 日本語形態素解析
- 読み仮名抽出
- 品詞分解
- 専門用語抽出

**適用範囲**
- 日本語テキストの前処理
- カタカナ語抽出
- 複合語認識
- 用語正規化

**実装例**
```python
import MeCab
import re
from collections import Counter

class JapaneseMorphAnalyzer:
    """日本語形態素解析"""
    
    def __init__(self, dict_path: str = None):
        # カスタム辞書対応
        if dict_path:
            self.tagger = MeCab.Tagger(f"-d {dict_path}")
        else:
            self.tagger = MeCab.Tagger()
            
        # ファッション用語辞書
        self.fashion_terms = set([
            "ボタン", "ファスナー", "ポケット", "襟", "袖",
            "ステッチ", "プリーツ", "ギャザー", "ダーツ"
        ])
        
    def extract_fashion_terms(self, text: str) -> List[Dict]:
        """ファッション専門用語の抽出"""
        
        # 形態素解析
        node = self.tagger.parseToNode(text)
        
        terms = []
        compound_noun = []
        
        while node:
            features = node.feature.split(',')
            surface = node.surface
            pos = features[0]
            
            # 名詞の場合
            if pos == '名詞':
                compound_noun.append(surface)
                
                # ファッション用語チェック
                if surface in self.fashion_terms:
                    terms.append({
                        'term': surface,
                        'pos': pos,
                        'reading': features[7] if len(features) > 7 else '',
                        'type': 'known_fashion_term'
                    })
            else:
                # 複合名詞の処理
                if len(compound_noun) > 1:
                    compound = ''.join(compound_noun)
                    if self._is_fashion_compound(compound):
                        terms.append({
                            'term': compound,
                            'pos': '複合名詞',
                            'type': 'compound_fashion_term'
                        })
                compound_noun = []
            
            node = node.next
        
        return terms
    
    def extract_katakana_terms(self, text: str) -> List[str]:
        """カタカナ専門用語の抽出"""
        
        # カタカナ連続パターン
        katakana_pattern = re.compile(r'[ァ-ヴー]+')
        katakana_terms = katakana_pattern.findall(text)
        
        # 長さでフィルタリング（2文字以上）
        significant_terms = [
            term for term in katakana_terms 
            if len(term) >= 2
        ]
        
        return significant_terms
    
    def normalize_terms(self, terms: List[str]) -> Dict[str, List[str]]:
        """用語の正規化とグループ化"""
        
        normalized = {}
        
        for term in terms:
            # 基本形を取得
            node = self.tagger.parseToNode(term)
            base_forms = []
            
            while node:
                features = node.feature.split(',')
                if len(features) > 6 and features[6] != '*':
                    base_forms.append(features[6])
                else:
                    base_forms.append(node.surface)
                node = node.next
            
            base_form = ''.join(base_forms)
            
            # グループ化
            if base_form not in normalized:
                normalized[base_form] = []
            normalized[base_form].append(term)
        
        return normalized
```

## 4. ストレージとファイル処理

### 4.1 オブジェクトストレージ

#### AWS S3 (boto3 1.34.25)

**適用目的**
- 大容量ファイルの保存
- 静的アセット配信
- バックアップ
- アーカイブ

**適用範囲**
- 画像・PDFファイル
- 生成されたSVG
- エクスポートデータ
- ログアーカイブ

**実装例**
```python
import boto3
from botocore.exceptions import ClientError
import asyncio
from typing import Optional, List
import mimetypes

class S3StorageManager:
    """S3ストレージマネージャー"""
    
    def __init__(self, bucket_name: str, region: str = 'ap-northeast-1'):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client('s3', region_name=region)
        self.s3_resource = boto3.resource('s3', region_name=region)
        
    async def upload_file_async(
        self, 
        file_path: str, 
        s3_key: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """非同期ファイルアップロード"""
        
        # コンテンツタイプを推測
        content_type, _ = mimetypes.guess_type(file_path)
        
        # アップロード設定
        extra_args = {
            'ContentType': content_type or 'application/octet-stream',
            'ServerSideEncryption': 'AES256'
        }
        
        if metadata:
            extra_args['Metadata'] = metadata
        
        # マルチパートアップロード（大きなファイル用）
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB以上
            return await self._multipart_upload(file_path, s3_key, extra_args)
        
        # 通常のアップロード
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self.s3_client.upload_file,
            file_path,
            self.bucket_name,
            s3_key,
            extra_args
        )
        
        # 署名付きURL生成
        url = self.generate_presigned_url(s3_key, expiration=3600)
        
        return url
    
    async def _multipart_upload(
        self, 
        file_path: str, 
        s3_key: str, 
        extra_args: Dict
    ):
        """マルチパートアップロード"""
        
        # TransferConfigでマルチパート設定
        from boto3.s3.transfer import TransferConfig
        
        config = TransferConfig(
            multipart_threshold=1024 * 25,  # 25MB
            max_concurrency=10,
            multipart_chunksize=1024 * 25,
            use_threads=True
        )
        
        # プログレスコールバック
        def upload_callback(bytes_transferred):
            print(f"Uploaded {bytes_transferred} bytes")
        
        # アップロード実行
        self.s3_client.upload_file(
            file_path,
            self.bucket_name,
            s3_key,
            ExtraArgs=extra_args,
            Config=config,
            Callback=upload_callback
        )
        
        return f"s3://{self.bucket_name}/{s3_key}"
    
    def setup_lifecycle_policy(self):
        """ライフサイクルポリシーの設定"""
        
        lifecycle_policy = {
            'Rules': [
                {
                    'ID': 'ArchiveOldFiles',
                    'Status': 'Enabled',
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                },
                {
                    'ID': 'DeleteTempFiles',
                    'Status': 'Enabled',
                    'Prefix': 'temp/',
                    'Expiration': {
                        'Days': 7
                    }
                }
            ]
        }
        
        self.s3_client.put_bucket_lifecycle_configuration(
            Bucket=self.bucket_name,
            LifecycleConfiguration=lifecycle_policy
        )
```

### 4.2 ファイル処理

#### Pillow 10.2.0

**適用目的**
- 画像形式変換
- リサイズ・最適化
- 基本的な画像処理
- サムネイル生成

**適用範囲**
- アップロード画像の処理
- プレビュー生成
- 画像メタデータ抽出
- 形式統一化

**実装例**
```python
from PIL import Image, ImageOps, ImageDraw
import io
from typing import Tuple, Optional

class ImageProcessor:
    """画像処理ユーティリティ"""
    
    def __init__(self):
        self.supported_formats = {'JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF'}
        self.max_size = (4096, 4096)
        
    def process_upload_image(
        self, 
        image_data: bytes,
        target_format: str = 'WEBP'
    ) -> Tuple[bytes, Dict]:
        """アップロード画像の処理"""
        
        # 画像を開く
        img = Image.open(io.BytesIO(image_data))
        
        # メタデータ抽出
        metadata = {
            'original_format': img.format,
            'original_size': img.size,
            'mode': img.mode,
            'has_transparency': img.mode in ('RGBA', 'LA')
        }
        
        # EXIF データ保持
        exif = img.info.get('exif', b'')
        
        # 画像の正規化
        img = self._normalize_image(img)
        
        # リサイズ（必要な場合）
        if img.size[0] > self.max_size[0] or img.size[1] > self.max_size[1]:
            img.thumbnail(self.max_size, Image.Resampling.LANCZOS)
        
        # フォーマット変換
        output = io.BytesIO()
        save_kwargs = {
            'format': target_format,
            'optimize': True,
            'quality': 85 if target_format in ['JPEG', 'WEBP'] else None
        }
        
        if exif and target_format == 'JPEG':
            save_kwargs['exif'] = exif
            
        img.save(output, **save_kwargs)
        
        return output.getvalue(), metadata
    
    def generate_thumbnail(
        self, 
        image_data: bytes, 
        size: Tuple[int, int] = (256, 256)
    ) -> bytes:
        """サムネイル生成"""
        
        img = Image.open(io.BytesIO(image_data))
        
        # アスペクト比を維持してリサイズ
        img.thumbnail(size, Image.Resampling.LANCZOS)
        
        # 正方形にする場合（パディング追加）
        if img.size[0] != img.size[1]:
            # 新しいキャンバス作成
            new_img = Image.new('RGBA', size, (255, 255, 255, 0))
            
            # 中央に配置
            paste_x = (size[0] - img.size[0]) // 2
            paste_y = (size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            img = new_img
        
        # WebP形式で保存
        output = io.BytesIO()
        img.save(output, format='WEBP', quality=80)
        
        return output.getvalue()
    
    def extract_colors_for_svg(
        self, 
        image_data: bytes,
        num_colors: int = 8
    ) -> List[str]:
        """SVG生成用の色抽出"""
        
        img = Image.open(io.BytesIO(image_data))
        
        # パレットモードに変換
        img = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
        
        # パレットから色を抽出
        palette = img.getpalette()
        colors = []
        
        for i in range(num_colors):
            r = palette[i * 3]
            g = palette[i * 3 + 1]
            b = palette[i * 3 + 2]
            
            # HEX形式に変換
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            colors.append(hex_color)
        
        return colors
```

## 5. 開発・運用ツール

### 5.1 コンテナ化

#### Docker

**適用目的**
- 環境の標準化
- 依存関係の管理
- マイクロサービス化
- 開発環境の統一

**適用範囲**
- 全サービスのコンテナ化
- 開発環境構築
- CI/CD パイプライン
- 本番環境デプロイ

**実装例**
```dockerfile
# マルチステージビルド Dockerfile
FROM python:3.11-slim as builder

# ビルド依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# 依存関係のインストール（キャッシュ活用）
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# 本番イメージ
FROM python:3.11-slim

# ランタイム依存関係
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    mecab \
    mecab-ipadic-utf8 \
    && rm -rf /var/lib/apt/lists/*

# ユーザー作成
RUN useradd -m -u 1000 aiagent

WORKDIR /app

# ビルドステージから依存関係をコピー
COPY --from=builder /root/.local /home/aiagent/.local
COPY --from=builder /build /app

# 権限設定
RUN chown -R aiagent:aiagent /app

USER aiagent

# パスの設定
ENV PATH=/home/aiagent/.local/bin:$PATH
ENV PYTHONPATH=/app

# ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# 起動コマンド
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Kubernetes

**適用目的**
- コンテナオーケストレーション
- 自動スケーリング
- 負荷分散
- 自己修復

**適用範囲**
- 本番環境の管理
- マイクロサービス連携
- リソース管理
- デプロイメント自動化

**実装例**
```yaml
# Kubernetes デプロイメント設定
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-agent-api
  namespace: ai-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-agent-api
  template:
    metadata:
      labels:
        app: ai-agent-api
    spec:
      containers:
      - name: api
        image: ai-agents/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      
      # GPUを使用するコンテナ
      - name: ml-worker
        image: ai-agents/ml-worker:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /models
      
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-agent-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-agent-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: pending_tasks
      target:
        type: AverageValue
        averageValue: "30"
```

### 5.2 モニタリング・ロギング

#### Prometheus + Grafana

**適用目的**
- メトリクス収集
- 可視化
- アラート管理
- パフォーマンス分析

**適用範囲**
- システムメトリクス
- アプリケーションメトリクス
- カスタムメトリクス
- SLA監視

**実装例**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
from functools import wraps

# メトリクス定義
task_counter = Counter(
    'ai_agent_tasks_total',
    'Total number of AI agent tasks',
    ['agent_type', 'status']
)

task_duration = Histogram(
    'ai_agent_task_duration_seconds',
    'Task processing duration',
    ['agent_type'],
    buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120, 300, 600)
)

active_tasks = Gauge(
    'ai_agent_active_tasks',
    'Number of active tasks',
    ['agent_type']
)

model_inference_time = Histogram(
    'ai_model_inference_seconds',
    'Model inference time',
    ['model_name', 'model_version']
)

# デコレーターでメトリクス収集
def track_metrics(agent_type: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # アクティブタスク増加
            active_tasks.labels(agent_type=agent_type).inc()
            
            # 処理時間計測
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                # 成功カウント
                task_counter.labels(
                    agent_type=agent_type,
                    status='success'
                ).inc()
                return result
                
            except Exception as e:
                # 失敗カウント
                task_counter.labels(
                    agent_type=agent_type,
                    status='error'
                ).inc()
                raise
                
            finally:
                # 処理時間記録
                duration = time.time() - start_time
                task_duration.labels(agent_type=agent_type).observe(duration)
                
                # アクティブタスク減少
                active_tasks.labels(agent_type=agent_type).dec()
        
        return wrapper
    return decorator

# 使用例
@track_metrics("term_collector")
async def process_term_extraction(document: str):
    # 処理実装
    pass

# Prometheusエンドポイント起動
start_http_server(9090)
```

#### Structured Logging (structlog)

**適用目的**
- 構造化ログ出力
- コンテキスト情報付加
- ログ相関
- 効率的な検索

**適用範囲**
- アプリケーションログ
- エラーログ
- 監査ログ
- デバッグ情報

**実装例**
```python
import structlog
from structlog.processors import JSONRenderer, TimeStamper
import logging

# structlog設定
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.FUNC_NAME,
            ]
        ),
        JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# ロガー作成
logger = structlog.get_logger()

# コンテキスト付きロギング
class AIAgentLogger:
    def __init__(self, agent_id: str, task_id: str):
        self.logger = logger.bind(
            agent_id=agent_id,
            task_id=task_id
        )
    
    def log_task_start(self, task_type: str, config: dict):
        self.logger.info(
            "task_started",
            task_type=task_type,
            config=config,
            event_type="task_lifecycle"
        )
    
    def log_model_inference(self, model: str, input_size: int, duration: float):
        self.logger.info(
            "model_inference_completed",
            model=model,
            input_size=input_size,
            duration_ms=duration * 1000,
            event_type="ml_inference"
        )
    
    def log_error(self, error: Exception, context: dict = None):
        self.logger.error(
            "task_error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            event_type="error",
            exc_info=True
        )

# 使用例
agent_logger = AIAgentLogger("agent_001", "task_123")
agent_logger.log_task_start("term_extraction", {"language": "ja"})
```

## 6. セキュリティツール

### 6.1 認証・認可

#### JWT (PyJWT)

**適用目的**
- トークンベース認証
- ステートレス認証
- API アクセス制御
- セッション管理

**適用範囲**
- API認証
- サービス間通信
- ユーザーセッション
- 一時的なアクセストークン

**実装例**
```python
import jwt
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict
import secrets

class JWTManager:
    """JWT管理"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        
    def create_access_token(
        self,
        subject: str,
        expires_delta: Optional[timedelta] = None,
        additional_claims: Optional[Dict] = None
    ) -> str:
        """アクセストークン生成"""
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=15)
        
        payload = {
            "sub": subject,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": secrets.token_urlsafe(16),  # JWT ID
            "type": "access"
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict:
        """トークン検証"""
        
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # 追加の検証
            if payload.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise Exception("Token has expired")
        except jwt.JWTError:
            raise Exception("Invalid token")
    
    def create_service_token(self, service_name: str) -> str:
        """サービス間通信用トークン"""
        
        payload = {
            "sub": f"service:{service_name}",
            "exp": datetime.now(timezone.utc) + timedelta(hours=24),
            "iat": datetime.now(timezone.utc),
            "type": "service",
            "permissions": self._get_service_permissions(service_name)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
```

### 6.2 暗号化

#### cryptography

**適用目的**
- データ暗号化
- キー管理
- ハッシュ化
- デジタル署名

**適用範囲**
- 機密データの保護
- APIキーの暗号化
- ファイル暗号化
- 通信の暗号化

**実装例**
```python
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class EncryptionManager:
    """暗号化管理"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.key = base64.urlsafe_b64encode(master_key.encode()[:32].ljust(32))
        else:
            self.key = Fernet.generate_key()
        
        self.cipher = Fernet(self.key)
    
    def encrypt_data(self, data: str) -> str:
        """データ暗号化"""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """データ復号化"""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """パスワードからキー導出"""
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_file(self, file_path: str, output_path: str):
        """ファイル暗号化"""
        
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        encrypted_data = self.cipher.encrypt(file_data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
    
    def generate_api_key(self) -> tuple[str, str]:
        """APIキー生成（平文と暗号化版）"""
        
        # ランダムAPIキー生成
        api_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        # 暗号化して保存
        encrypted_key = self.encrypt_data(api_key)
        
        return api_key, encrypted_key
```

## 7. まとめ

本仕様書で定義した技術スタックは、以下の要件を満たすよう選定されています：

1. **スケーラビリティ**: マイクロサービスアーキテクチャと非同期処理による水平スケーリング
2. **パフォーマンス**: 最適化されたML推論とキャッシング戦略
3. **信頼性**: 分散システムとフォールバック機構
4. **保守性**: コンテナ化と構造化ログによる運用効率化
5. **セキュリティ**: 多層防御とゼロトラスト原則

各技術は特定の目的に最適化されており、相互に連携して完全なAIエージェントシステムを構成します。