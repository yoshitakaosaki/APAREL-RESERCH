# AIエージェントシステム Python実装設計書

## 1. 概要

本ドキュメントは、テックパック用語集とSVGパーツの自動収集・生成を行うAIエージェントシステムのPython実装設計を定義します。

### 1.1 システム構成

```mermaid
graph TB
    subgraph "Python AIエージェントシステム"
        A[エージェントオーケストレーター]
        B[用語収集エージェント]
        C[画像収集エージェント]
        D[SVG生成エージェント]
        E[品質保証エージェント]
        F[MCP統合エージェント]
    end
    
    subgraph "外部システム"
        G[(PostgreSQL)]
        H[Redis Queue]
        I[S3/Blob Storage]
        J[外部API/MCP]
    end
    
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    
    B --> G
    C --> I
    D --> G
    E --> G
    F --> J
    
    A --> H
```

### 1.2 技術スタック

- **言語**: Python 3.11+
- **フレームワーク**: FastAPI (Web API), Celery (タスクキュー)
- **データベース**: PostgreSQL 15+ with pgvector
- **キャッシュ/キュー**: Redis 7+
- **ストレージ**: AWS S3 / Azure Blob Storage
- **コンテナ**: Docker, Kubernetes

## 2. プロジェクト構造

```
techpack-ai-agents/
├── agents/
│   ├── __init__.py
│   ├── base/
│   │   ├── __init__.py
│   │   ├── agent.py              # 基底エージェントクラス
│   │   ├── message.py            # メッセージング
│   │   └── exceptions.py         # カスタム例外
│   ├── collectors/
│   │   ├── __init__.py
│   │   ├── term_collector.py     # 用語収集エージェント
│   │   ├── image_scraper.py      # 画像収集エージェント
│   │   └── document_parser.py    # ドキュメント解析
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── svg_generator.py      # SVG生成エージェント
│   │   ├── parametrizer.py       # パラメータ化
│   │   └── style_unifier.py      # スタイル統一
│   ├── validators/
│   │   ├── __init__.py
│   │   ├── quality_assurance.py  # 品質保証
│   │   ├── categorizer.py        # 分類エージェント
│   │   └── translator.py         # 翻訳エージェント
│   └── integrations/
│       ├── __init__.py
│       ├── mcp_client.py          # MCP統合
│       └── external_apis.py       # 外部API連携
├── core/
│   ├── __init__.py
│   ├── config.py                  # 設定管理
│   ├── database.py                # DB接続
│   ├── models.py                  # SQLAlchemyモデル
│   ├── schemas.py                 # Pydanticスキーマ
│   └── utils.py                   # ユーティリティ
├── ml/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── object_detection.py   # 物体検出
│   │   ├── segmentation.py       # セグメンテーション
│   │   ├── vectorization.py      # ベクトル化
│   │   └── nlp.py                # 自然言語処理
│   └── pipelines/
│       ├── __init__.py
│       ├── image_to_svg.py        # 画像→SVGパイプライン
│       └── text_extraction.py     # テキスト抽出パイプライン
├── orchestration/
│   ├── __init__.py
│   ├── orchestrator.py            # オーケストレーター
│   ├── scheduler.py               # スケジューラー
│   └── workflow.py                # ワークフロー定義
├── api/
│   ├── __init__.py
│   ├── main.py                    # FastAPI アプリ
│   ├── routes/
│   │   ├── agents.py
│   │   ├── tasks.py
│   │   └── monitoring.py
│   └── dependencies.py
├── tasks/
│   ├── __init__.py
│   ├── celery_app.py              # Celery設定
│   └── worker.py                  # ワーカータスク
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.worker
│   └── docker-compose.yml
├── kubernetes/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── configmap.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## 3. 主要パッケージ・ライブラリ

### 3.1 コアフレームワーク

```python
# requirements.txt

# Web Framework
fastapi==0.109.0
uvicorn==0.27.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Task Queue
celery==5.3.4
redis==5.0.1
flower==2.0.1  # Celery monitoring

# Database
sqlalchemy==2.0.25
alembic==1.13.1
psycopg2-binary==2.9.9
pgvector==0.2.4  # PostgreSQL vector extension

# Async Support
aiohttp==3.9.1
aiofiles==23.2.1
httpx==0.26.0
```

### 3.2 AI/ML パッケージ

```python
# AI/ML Libraries

# Deep Learning Frameworks
torch==2.1.2
torchvision==0.16.2
transformers==4.36.2  # Hugging Face transformers

# Computer Vision
opencv-python==4.9.0.80
Pillow==10.2.0
scikit-image==0.22.0
albumentations==1.3.1  # Image augmentation

# Object Detection & Segmentation
ultralytics==8.1.0  # YOLOv8
segment-anything==1.0  # SAM
detectron2==0.6  # Facebook's detection library

# SVG Processing
svgwrite==1.4.3
svgpathtools==1.6.1
cairosvg==2.7.1
lxml==5.1.0

# NLP
spacy==3.7.2
nltk==3.8.1
mecab-python3==1.0.8  # Japanese morphological analysis
sudachipy==0.6.8  # Japanese tokenizer
langdetect==1.0.9

# OCR
pytesseract==0.3.10
easyocr==1.7.1

# Vector Databases
chromadb==0.4.22
faiss-cpu==1.7.4
```

### 3.3 ユーティリティ・統合

```python
# Utilities & Integration

# Web Scraping
beautifulsoup4==4.12.3
selenium==4.17.2
playwright==1.41.1
scrapy==2.11.0

# File Processing
python-multipart==0.0.6
python-magic==0.4.27
PyPDF2==3.0.1
pdfplumber==0.10.3
openpyxl==3.1.2

# Cloud Storage
boto3==1.34.25  # AWS
azure-storage-blob==12.19.0
google-cloud-storage==2.13.0

# Monitoring & Logging
prometheus-client==0.19.0
structlog==24.1.0
sentry-sdk==1.40.0

# Testing
pytest==7.4.4
pytest-asyncio==0.23.3
pytest-cov==4.1.0
faker==22.2.0

# Development
black==23.12.1
ruff==0.1.14
mypy==1.8.0
pre-commit==3.6.0
```

## 4. エージェント実装詳細

### 4.1 基底エージェントクラス

```python
# agents/base/agent.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import structlog
from pydantic import BaseModel

from core.config import settings
from core.database import get_db
from .message import AgentMessage, MessageType
from .exceptions import AgentError

# 既存認証基盤のインポート
from authentication.jwt_validator import JWTValidator
from utils.errors import APIError, ErrorCode


class AgentStatus(BaseModel):
    """エージェントステータス"""
    agent_id: str
    agent_type: str
    status: str  # idle, working, paused, error
    current_task: Optional[str] = None
    queue_size: int = 0
    metrics: Dict[str, Any] = {}
    last_heartbeat: datetime
    user_context: Optional[Dict[str, Any]] = None  # ユーザー認証情報


class BaseAgent(ABC):
    """全エージェントの基底クラス（認証基盤統合）"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        self.logger = structlog.get_logger().bind(agent_id=agent_id)
        
        # JWT検証器の初期化
        self.jwt_validator = JWTValidator(
            jwks_endpoint=settings.JWKS_ENDPOINT,
            cache_ttl=3600
        )
        self.status = AgentStatus(
            agent_id=agent_id,
            agent_type=self.__class__.__name__,
            status="idle",
            last_heartbeat=datetime.utcnow()
        )
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """タスク処理のメイン実装"""
        pass
    
    @abstractmethod
    async def validate_input(self, task: Dict[str, Any]) -> bool:
        """入力検証"""
        pass
    
    async def start(self):
        """エージェント起動"""
        self._running = True
        self.logger.info("Agent started")
        
        # メッセージ処理ループ
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=settings.AGENT_HEARTBEAT_INTERVAL
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                await self._heartbeat()
            except Exception as e:
                self.logger.error("Error in agent loop", error=str(e))
                await self._handle_error(e)
    
    async def stop(self):
        """エージェント停止"""
        self._running = False
        self.logger.info("Agent stopped")
    
    async def send_message(self, message: AgentMessage):
        """メッセージ送信"""
        await self._message_queue.put(message)
    
    async def _handle_message(self, message: AgentMessage):
        """メッセージ処理"""
        self.logger.info("Handling message", message_type=message.type)
        
        try:
            if message.type == MessageType.TASK:
                await self._process_task(message.payload)
            elif message.type == MessageType.CONTROL:
                await self._handle_control(message.payload)
            elif message.type == MessageType.QUERY:
                await self._handle_query(message.payload)
        except Exception as e:
            self.logger.error("Message handling error", error=str(e))
            raise
    
    async def _process_task(self, task: Dict[str, Any]):
        """タスク処理"""
        self.status.status = "working"
        self.status.current_task = task.get("task_id")
        
        try:
            # 入力検証
            if not await self.validate_input(task):
                raise AgentError("Invalid input")
            
            # メイン処理
            result = await self.process(task)
            
            # 結果保存
            await self._save_result(task["task_id"], result)
            
            self.logger.info("Task completed", task_id=task["task_id"])
            
        finally:
            self.status.status = "idle"
            self.status.current_task = None
    
    async def _heartbeat(self):
        """ハートビート送信"""
        self.status.last_heartbeat = datetime.utcnow()
        # Redis にステータス保存
        await self._update_status()
    
    async def _update_status(self):
        """ステータス更新"""
        # Implementation for status update in Redis
        pass
    
    async def _save_result(self, task_id: str, result: Dict[str, Any]):
        """結果保存"""
        # Implementation for saving results to PostgreSQL
        pass
```

### 4.2 用語収集エージェント

```python
# agents/collectors/term_collector.py

import asyncio
from typing import Dict, List, Any, Optional
import spacy
import nltk
from transformers import pipeline
import MeCab
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from agents.base import BaseAgent
from core.models import Term, TermTranslation
from ml.nlp import TermExtractor, ContextAnalyzer


class TermCollectorAgent(BaseAgent):
    """用語収集エージェント"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # NLPモデル初期化
        self.nlp_models = {
            'ja': spacy.load('ja_core_news_lg'),
            'en': spacy.load('en_core_web_lg')
        }
        
        # 専門用語抽出器
        self.term_extractor = TermExtractor(
            confidence_threshold=config.get('confidence_threshold', 0.7)
        )
        
        # コンテキスト分析器
        self.context_analyzer = ContextAnalyzer()
        
        # BERT-based NER for fashion terms
        self.ner_pipeline = pipeline(
            "ner",
            model="cl-tohoku/bert-base-japanese-char-v2",
            aggregation_strategy="simple"
        )
        
        # MeCab for Japanese morphological analysis
        self.mecab = MeCab.Tagger('-Owakati')
        
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """用語収集処理"""
        source = task['source']
        options = task.get('options', {})
        
        # テキスト抽出
        text = await self._extract_text(source)
        
        # 言語検出
        language = self._detect_language(text)
        
        # NLP処理
        doc = self.nlp_models[language](text)
        
        # 専門用語候補抽出
        term_candidates = await self._extract_term_candidates(
            doc, language, options
        )
        
        # コンテキスト分析
        enriched_terms = await self._analyze_context(
            term_candidates, doc, language
        )
        
        # カテゴリ分類
        categorized_terms = await self._categorize_terms(enriched_terms)
        
        # 既存用語との照合
        final_terms = await self._validate_terms(categorized_terms)
        
        return {
            'extracted_terms': len(final_terms),
            'new_terms': sum(1 for t in final_terms if t['is_new']),
            'updated_terms': sum(1 for t in final_terms if not t['is_new']),
            'terms': final_terms,
            'metadata': {
                'source': source,
                'language': language,
                'processing_time': task.get('processing_time')
            }
        }
    
    async def _extract_term_candidates(
        self, 
        doc: Any, 
        language: str, 
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """専門用語候補抽出"""
        candidates = []
        
        # 名詞句抽出
        noun_phrases = self._extract_noun_phrases(doc, language)
        
        # TF-IDF によるキーワード抽出
        keywords = self._extract_keywords_tfidf(doc.text, language)
        
        # パターンマッチング
        pattern_terms = self._extract_by_patterns(doc.text, language)
        
        # BERT NER
        if language == 'ja':
            ner_results = self.ner_pipeline(doc.text)
            fashion_entities = [
                ent for ent in ner_results 
                if self._is_fashion_related(ent['word'])
            ]
            candidates.extend([
                {
                    'term': ent['word'],
                    'confidence': ent['score'],
                    'method': 'ner',
                    'context': self._get_context(doc.text, ent['start'], ent['end'])
                }
                for ent in fashion_entities
            ])
        
        # 統合と重複除去
        candidates = self._merge_candidates(
            noun_phrases + keywords + pattern_terms + candidates
        )
        
        return candidates
    
    async def _analyze_context(
        self,
        candidates: List[Dict[str, Any]],
        doc: Any,
        language: str
    ) -> List[Dict[str, Any]]:
        """コンテキスト分析と用語エンリッチメント"""
        enriched = []
        
        for candidate in candidates:
            term = candidate['term']
            
            # 文脈での使用例収集
            usage_examples = self._collect_usage_examples(doc, term)
            
            # 関連語抽出
            related_terms = self._extract_related_terms(doc, term)
            
            # 定義文の検出
            definition = self._extract_definition(doc, term)
            
            # カテゴリヒント抽出
            category_hints = self._extract_category_hints(doc, term)
            
            enriched_candidate = {
                **candidate,
                'usage_examples': usage_examples[:5],  # 最大5例
                'related_terms': related_terms[:10],    # 最大10語
                'definition': definition,
                'category_hints': category_hints,
                'language': language
            }
            
            enriched.append(enriched_candidate)
        
        return enriched
    
    def _extract_noun_phrases(self, doc: Any, language: str) -> List[Dict[str, Any]]:
        """名詞句抽出"""
        noun_phrases = []
        
        if language == 'ja':
            # 日本語の複合名詞抽出
            for sent in doc.sents:
                noun_chunk = []
                for token in sent:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        noun_chunk.append(token.text)
                    elif noun_chunk:
                        if len(noun_chunk) > 1:
                            phrase = ''.join(noun_chunk)
                            noun_phrases.append({
                                'term': phrase,
                                'confidence': 0.8,
                                'method': 'noun_phrase'
                            })
                        noun_chunk = []
        else:
            # 英語の名詞句抽出
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) > 1:
                    noun_phrases.append({
                        'term': chunk.text,
                        'confidence': 0.8,
                        'method': 'noun_phrase'
                    })
        
        return noun_phrases
    
    def _is_fashion_related(self, term: str) -> bool:
        """ファッション関連用語判定"""
        # ファッション関連キーワードリスト
        fashion_keywords = {
            'ja': ['襟', 'ボタン', 'ポケット', '袖', '裾', 'ステッチ', 
                   '生地', 'ファスナー', 'カラー', 'デザイン'],
            'en': ['collar', 'button', 'pocket', 'sleeve', 'hem', 'stitch',
                   'fabric', 'zipper', 'color', 'design']
        }
        
        # キーワードマッチング
        for lang, keywords in fashion_keywords.items():
            for keyword in keywords:
                if keyword in term.lower():
                    return True
        
        return False
```

### 4.3 SVG生成エージェント

```python
# agents/generators/svg_generator.py

import asyncio
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import cv2
import torch
from PIL import Image
import svgwrite
from pathlib import Path

from agents.base import BaseAgent
from ml.models import ObjectDetector, Segmenter, Vectorizer
from ml.pipelines.image_to_svg import ImageToSVGPipeline


class SVGGeneratorAgent(BaseAgent):
    """SVG生成エージェント"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # モデル初期化
        self.object_detector = ObjectDetector(
            model_path=config['models']['object_detection']
        )
        self.segmenter = Segmenter(
            model_path=config['models']['segmentation']
        )
        self.vectorizer = Vectorizer(
            model_path=config['models']['vectorization']
        )
        
        # パイプライン初期化
        self.pipeline = ImageToSVGPipeline(
            detector=self.object_detector,
            segmenter=self.segmenter,
            vectorizer=self.vectorizer
        )
        
        # GPU使用可能か確認
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.logger.info(f"Using device: {self.device}")
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """SVG生成処理"""
        input_data = task['input']
        part_type = task.get('part_type', 'auto')
        options = task.get('options', {})
        
        # 画像読み込み
        image = await self._load_image(input_data['image_url'])
        
        # 前処理
        preprocessed = await self._preprocess_image(image, options)
        
        # パーツタイプ自動判定
        if part_type == 'auto':
            part_type = await self._detect_part_type(preprocessed)
        
        # SVG生成戦略選択
        strategy = self._select_strategy(part_type, options)
        
        # SVG生成
        svg_data = await self._generate_svg(
            preprocessed, 
            part_type, 
            strategy, 
            options
        )
        
        # 後処理とパラメータ化
        final_svg = await self._postprocess_svg(svg_data, options)
        
        # 品質評価
        quality_metrics = await self._evaluate_quality(final_svg, image)
        
        return {
            'svg_data': final_svg['content'],
            'parameters': final_svg['parameters'],
            'metadata': {
                'part_type': part_type,
                'strategy': strategy,
                'quality_metrics': quality_metrics,
                'processing_time': task.get('processing_time')
            }
        }
    
    async def _preprocess_image(
        self, 
        image: np.ndarray, 
        options: Dict[str, Any]
    ) -> np.ndarray:
        """画像前処理"""
        # 背景除去
        if options.get('remove_background', True):
            image = await self._remove_background(image)
        
        # ノイズ除去
        if options.get('denoise', True):
            image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # エッジ強調
        if options.get('enhance_edges', True):
            edges = cv2.Canny(image, 50, 150)
            image = cv2.addWeighted(image, 0.7, edges[:,:,np.newaxis], 0.3, 0)
        
        # 正規化
        image = self._normalize_image(image)
        
        return image
    
    async def _generate_svg(
        self,
        image: np.ndarray,
        part_type: str,
        strategy: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """SVG生成メイン処理"""
        
        if strategy == 'geometric':
            return await self._geometric_vectorization(image, part_type)
        elif strategy == 'ml_based':
            return await self._ml_vectorization(image, part_type)
        elif strategy == 'hybrid':
            return await self._hybrid_vectorization(image, part_type)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    async def _ml_vectorization(
        self,
        image: np.ndarray,
        part_type: str
    ) -> Dict[str, Any]:
        """機械学習ベースのベクトル化"""
        # 物体検出
        detections = await self.object_detector.detect(image, part_type)
        
        # セグメンテーション
        masks = await self.segmenter.segment(image, detections)
        
        # 各セグメントをベクトル化
        svg_elements = []
        for i, (detection, mask) in enumerate(zip(detections, masks)):
            # マスク領域切り出し
            roi = self._extract_roi(image, mask)
            
            # ニューラルベクトル化
            vector_data = await self.vectorizer.vectorize(roi)
            
            # SVG要素に変換
            svg_element = self._vector_to_svg(vector_data, detection)
            svg_elements.append(svg_element)
        
        # SVG統合
        final_svg = self._merge_svg_elements(svg_elements)
        
        return {
            'paths': final_svg['paths'],
            'attributes': final_svg['attributes'],
            'bounding_box': self._calculate_bounding_box(final_svg['paths'])
        }
    
    async def _postprocess_svg(
        self,
        svg_data: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """SVG後処理とパラメータ化"""
        # パス最適化
        if options.get('optimize_paths', True):
            svg_data['paths'] = self._optimize_paths(svg_data['paths'])
        
        # パス簡略化
        if options.get('simplify_paths', True):
            svg_data['paths'] = self._simplify_paths(
                svg_data['paths'],
                tolerance=options.get('simplification_tolerance', 0.5)
            )
        
        # パラメータ抽出
        parameters = self._extract_parameters(svg_data)
        
        # SVG文字列生成
        svg_content = self._generate_svg_string(svg_data, parameters)
        
        return {
            'content': svg_content,
            'parameters': parameters,
            'metadata': svg_data.get('metadata', {})
        }
    
    def _extract_parameters(self, svg_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """パラメータ抽出"""
        parameters = []
        
        # サイズパラメータ
        bbox = svg_data['bounding_box']
        parameters.append({
            'id': 'width',
            'name': {'ja': '幅', 'en': 'Width'},
            'type': 'number',
            'value': bbox['width'],
            'constraints': {
                'min': bbox['width'] * 0.5,
                'max': bbox['width'] * 2.0,
                'unit': 'mm'
            }
        })
        
        parameters.append({
            'id': 'height',
            'name': {'ja': '高さ', 'en': 'Height'},
            'type': 'number',
            'value': bbox['height'],
            'constraints': {
                'min': bbox['height'] * 0.5,
                'max': bbox['height'] * 2.0,
                'unit': 'mm'
            }
        })
        
        # 色パラメータ
        colors = self._extract_colors(svg_data)
        for i, color in enumerate(colors):
            parameters.append({
                'id': f'color_{i}',
                'name': {'ja': f'色{i+1}', 'en': f'Color {i+1}'},
                'type': 'color',
                'value': color,
                'group': 'colors'
            })
        
        return parameters
```

### 4.4 MCP統合エージェント

```python
# agents/integrations/mcp_client.py

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import jwt

from agents.base import BaseAgent
from core.config import settings


class MCPProvider:
    """MCP プロバイダー定義"""
    
    def __init__(self, config: Dict[str, Any]):
        self.id = config['id']
        self.name = config['name']
        self.type = config['type']
        self.endpoint = config['endpoint']
        self.auth_config = config['authentication']
        self.rate_limit = config.get('rate_limit', {})
        self._token_cache = {}
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def authenticate(self) -> str:
        """認証処理"""
        if self.auth_config['type'] == 'oauth2':
            return await self._oauth2_auth()
        elif self.auth_config['type'] == 'api_key':
            return self.auth_config['credentials']['api_key']
        else:
            raise ValueError(f"Unsupported auth type: {self.auth_config['type']}")
    
    async def _oauth2_auth(self) -> str:
        """OAuth2認証"""
        # トークンキャッシュ確認
        cached_token = self._token_cache.get('access_token')
        if cached_token and cached_token['expires_at'] > datetime.utcnow():
            return cached_token['token']
        
        # 新規トークン取得
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.auth_config['token_endpoint'],
                data={
                    'grant_type': 'client_credentials',
                    'client_id': self.auth_config['credentials']['client_id'],
                    'client_secret': self.auth_config['credentials']['client_secret'],
                    'scope': self.auth_config.get('scope', '')
                }
            ) as response:
                token_data = await response.json()
                
                # キャッシュ保存
                self._token_cache['access_token'] = {
                    'token': token_data['access_token'],
                    'expires_at': datetime.utcnow() + timedelta(
                        seconds=token_data['expires_in']
                    )
                }
                
                return token_data['access_token']


class MCPIntegrationAgent(BaseAgent):
    """MCP統合エージェント"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # プロバイダー初期化
        self.providers: Dict[str, MCPProvider] = {}
        for provider_config in config['providers']:
            provider = MCPProvider(provider_config)
            self.providers[provider.id] = provider
        
        # レート制限管理
        self.rate_limiters = {}
        
        # キャッシュ設定
        self.cache_ttl = config.get('cache_ttl', 3600)
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """MCP統合処理"""
        task_type = task['type']
        
        if task_type == 'search':
            return await self._process_search(task)
        elif task_type == 'sync':
            return await self._process_sync(task)
        elif task_type == 'validate':
            return await self._process_validate(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _process_search(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """外部検索処理"""
        query = task['query']
        target_providers = task.get('providers', list(self.providers.keys()))
        options = task.get('options', {})
        
        # 並列検索実行
        search_tasks = []
        for provider_id in target_providers:
            if provider_id in self.providers:
                provider = self.providers[provider_id]
                search_task = self._search_provider(provider, query, options)
                search_tasks.append(search_task)
        
        # 結果収集
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 結果統合
        merged_results = self._merge_search_results(results, query)
        
        return {
            'query': query,
            'total_results': len(merged_results),
            'results': merged_results,
            'providers_status': self._get_providers_status(results)
        }
    
    async def _search_provider(
        self,
        provider: MCPProvider,
        query: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """プロバイダー検索"""
        # レート制限チェック
        await self._check_rate_limit(provider.id)
        
        # 認証
        auth_token = await provider.authenticate()
        
        # リクエスト作成
        headers = {
            'Authorization': f'Bearer {auth_token}',
            'Content-Type': 'application/json'
        }
        
        params = {
            'q': query,
            'limit': options.get('limit', 50),
            'language': options.get('language', 'ja')
        }
        
        # API呼び出し
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{provider.endpoint}/search",
                headers=headers,
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'provider_id': provider.id,
                        'status': 'success',
                        'data': data
                    }
                else:
                    return {
                        'provider_id': provider.id,
                        'status': 'error',
                        'error': f"HTTP {response.status}"
                    }
    
    def _merge_search_results(
        self,
        results: List[Any],
        query: str
    ) -> List[Dict[str, Any]]:
        """検索結果統合"""
        merged = []
        seen_terms = set()
        
        for result in results:
            if isinstance(result, dict) and result.get('status') == 'success':
                provider_data = result['data']
                
                for item in provider_data.get('items', []):
                    # 重複チェック
                    term_key = f"{item.get('name')}_{item.get('language')}"
                    if term_key not in seen_terms:
                        seen_terms.add(term_key)
                        
                        merged.append({
                            'source': result['provider_id'],
                            'confidence': item.get('confidence', 1.0),
                            'data': item
                        })
        
        # スコアでソート
        merged.sort(key=lambda x: x['confidence'], reverse=True)
        
        return merged
```

## 5. データベーススキーマ (PostgreSQL)

```sql
-- AI タスクテーブル
CREATE TABLE ai_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority VARCHAR(10) DEFAULT 'normal',
    input_data JSONB NOT NULL,
    configuration JSONB,
    result JSONB,
    error JSONB,
    progress JSONB DEFAULT '{"percentage": 0}',
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- タスクログテーブル
CREATE TABLE ai_task_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES ai_tasks(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    level VARCHAR(10) NOT NULL,
    message TEXT,
    details JSONB
);

-- エージェントステータステーブル
CREATE TABLE agent_status (
    agent_id VARCHAR(100) PRIMARY KEY,
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    current_task UUID REFERENCES ai_tasks(id),
    metrics JSONB,
    last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 抽出された用語の一時テーブル
CREATE TABLE extracted_terms_temp (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES ai_tasks(id) ON DELETE CASCADE,
    term_text VARCHAR(500) NOT NULL,
    language VARCHAR(10) NOT NULL,
    confidence FLOAT NOT NULL,
    method VARCHAR(50),
    context TEXT,
    usage_examples TEXT[],
    related_terms TEXT[],
    definition TEXT,
    category_hints JSONB,
    source_location TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 生成されたSVGの一時テーブル
CREATE TABLE generated_svg_temp (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES ai_tasks(id) ON DELETE CASCADE,
    part_type VARCHAR(50) NOT NULL,
    svg_content TEXT NOT NULL,
    parameters JSONB,
    quality_metrics JSONB,
    generation_method VARCHAR(50),
    source_image_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- MCP プロバイダー設定
CREATE TABLE mcp_providers (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(200) NOT NULL,
    type VARCHAR(50) NOT NULL,
    endpoint TEXT NOT NULL,
    auth_config JSONB NOT NULL,
    rate_limit JSONB,
    mapping_config JSONB,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- インデックス
CREATE INDEX idx_ai_tasks_status ON ai_tasks(status);
CREATE INDEX idx_ai_tasks_agent_type ON ai_tasks(agent_type);
CREATE INDEX idx_ai_tasks_created_at ON ai_tasks(created_at DESC);
CREATE INDEX idx_extracted_terms_task_id ON extracted_terms_temp(task_id);
CREATE INDEX idx_generated_svg_task_id ON generated_svg_temp(task_id);

-- pgvector 拡張（ベクトル検索用）
CREATE EXTENSION IF NOT EXISTS vector;

-- 用語ベクトルテーブル
CREATE TABLE term_vectors (
    term_id UUID REFERENCES terms(id) ON DELETE CASCADE,
    language VARCHAR(10) NOT NULL,
    vector vector(768),  -- BERT embeddings
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (term_id, language)
);

-- ベクトル検索インデックス
CREATE INDEX ON term_vectors USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);
```

## 6. 非同期タスク管理 (Celery)

```python
# tasks/celery_app.py

from celery import Celery
from celery.schedules import crontab
from kombu import Exchange, Queue

from core.config import settings

# Celeryアプリ初期化
celery_app = Celery(
    'techpack_ai_agents',
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Celery設定
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # タスクルーティング
    task_routes={
        'tasks.collectors.*': {'queue': 'collectors'},
        'tasks.generators.*': {'queue': 'generators'},
        'tasks.validators.*': {'queue': 'validators'},
        'tasks.integrations.*': {'queue': 'integrations'}
    },
    
    # キュー定義
    task_queues=(
        Queue('collectors', Exchange('collectors'), routing_key='collectors'),
        Queue('generators', Exchange('generators'), routing_key='generators'),
        Queue('validators', Exchange('validators'), routing_key='validators'),
        Queue('integrations', Exchange('integrations'), routing_key='integrations'),
        Queue('default', Exchange('default'), routing_key='default')
    ),
    
    # 定期タスク
    beat_schedule={
        'sync-external-terms': {
            'task': 'tasks.integrations.sync_external_terms',
            'schedule': crontab(hour=2, minute=0),  # 毎日午前2時
            'options': {'queue': 'integrations'}
        },
        'cleanup-temp-data': {
            'task': 'tasks.maintenance.cleanup_temp_data',
            'schedule': crontab(hour=3, minute=0),  # 毎日午前3時
            'options': {'queue': 'default'}
        },
        'update-ai-models': {
            'task': 'tasks.maintenance.check_model_updates',
            'schedule': crontab(hour=4, minute=0, day_of_week=1),  # 毎週月曜午前4時
            'options': {'queue': 'default'}
        }
    },
    
    # パフォーマンス設定
    worker_prefetch_multiplier=2,
    worker_max_tasks_per_child=1000,
    task_time_limit=3600,  # 1時間
    task_soft_time_limit=3300,  # 55分
    
    # 結果バックエンド設定
    result_backend_transport_options={
        'master_name': 'mymaster',
        'visibility_timeout': 3600
    },
    result_expires=86400,  # 24時間
)
```

```python
# tasks/worker.py

from celery import Task
from typing import Dict, Any
import asyncio
import structlog

from tasks.celery_app import celery_app
from agents.collectors import TermCollectorAgent
from agents.generators import SVGGeneratorAgent
from core.database import get_db_session


logger = structlog.get_logger()


class AsyncTask(Task):
    """非同期タスクのベースクラス"""
    
    def run(self, *args, **kwargs):
        """同期ラッパー"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.async_run(*args, **kwargs))
    
    async def async_run(self, *args, **kwargs):
        """非同期実装"""
        raise NotImplementedError


@celery_app.task(base=AsyncTask, bind=True, name='tasks.collectors.collect_terms')
class CollectTermsTask(AsyncTask):
    """用語収集タスク"""
    
    async def async_run(self, task_id: str, input_data: Dict[str, Any]):
        logger.info("Starting term collection task", task_id=task_id)
        
        async with get_db_session() as db:
            try:
                # タスクステータス更新
                await self._update_task_status(db, task_id, 'processing')
                
                # エージェント初期化
                agent = TermCollectorAgent(
                    agent_id=f"collector_{task_id}",
                    config=input_data.get('configuration', {})
                )
                
                # 処理実行
                result = await agent.process(input_data)
                
                # 結果保存
                await self._save_results(db, task_id, result)
                
                # 完了ステータス更新
                await self._update_task_status(db, task_id, 'completed')
                
                logger.info("Term collection completed", task_id=task_id)
                return result
                
            except Exception as e:
                logger.error("Term collection failed", task_id=task_id, error=str(e))
                await self._update_task_status(db, task_id, 'failed', error=str(e))
                raise


@celery_app.task(base=AsyncTask, bind=True, name='tasks.generators.generate_svg')
class GenerateSVGTask(AsyncTask):
    """SVG生成タスク"""
    
    async def async_run(self, task_id: str, input_data: Dict[str, Any]):
        logger.info("Starting SVG generation task", task_id=task_id)
        
        async with get_db_session() as db:
            try:
                # タスクステータス更新
                await self._update_task_status(db, task_id, 'processing')
                
                # エージェント初期化
                agent = SVGGeneratorAgent(
                    agent_id=f"generator_{task_id}",
                    config=input_data.get('configuration', {})
                )
                
                # 処理実行
                result = await agent.process(input_data)
                
                # 結果保存
                await self._save_results(db, task_id, result)
                
                # 完了ステータス更新
                await self._update_task_status(db, task_id, 'completed')
                
                logger.info("SVG generation completed", task_id=task_id)
                return result
                
            except Exception as e:
                logger.error("SVG generation failed", task_id=task_id, error=str(e))
                await self._update_task_status(db, task_id, 'failed', error=str(e))
                raise
```

## 7. Docker設定

```dockerfile
# docker/Dockerfile

FROM python:3.11-slim

# システム依存関係
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 \
    tesseract-ocr \
    tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

# Python環境設定
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# 依存関係インストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SpaCyモデルダウンロード
RUN python -m spacy download ja_core_news_lg
RUN python -m spacy download en_core_web_lg

# アプリケーションコピー
COPY . .

# エントリーポイント
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml

version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: techpack_ai
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/techpack_ai
      REDIS_URL: redis://redis:6379/0
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    volumes:
      - ./ml/models:/app/ml/models
  
  worker_collectors:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    command: celery -A tasks.celery_app worker -Q collectors -l info
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/techpack_ai
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./ml/models:/app/ml/models
  
  worker_generators:
    build:
      context: .
      dockerfile: docker/Dockerfile.worker
    command: celery -A tasks.celery_app worker -Q generators -l info
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/techpack_ai
      REDIS_URL: redis://redis:6379/0
      CUDA_VISIBLE_DEVICES: "0"  # GPU使用時
    depends_on:
      - postgres
      - redis
    volumes:
      - ./ml/models:/app/ml/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  beat:
    build:
      context: .
      dockerfile: docker/Dockerfile
    command: celery -A tasks.celery_app beat -l info
    environment:
      DATABASE_URL: postgresql://postgres:postgres@postgres:5432/techpack_ai
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - postgres
      - redis
  
  flower:
    build:
      context: .
      dockerfile: docker/Dockerfile
    command: celery -A tasks.celery_app flower
    environment:
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/0
    ports:
      - "5555:5555"
    depends_on:
      - redis

volumes:
  postgres_data:
```

## 8. 設定管理

```python
# core/config.py

from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os


class Settings(BaseSettings):
    # アプリケーション設定
    APP_NAME: str = "TechPack AI Agents"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # データベース
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    
    # Redis
    REDIS_URL: str
    REDIS_POOL_SIZE: int = 10
    
    # ストレージ
    STORAGE_TYPE: str = "s3"  # s3, azure, local
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    
    # AI/MLモデル
    MODELS_PATH: str = "/app/ml/models"
    MODEL_CONFIGS: Dict[str, Any] = {
        "object_detection": {
            "name": "yolov8-fashion",
            "path": "yolov8/best.pt",
            "confidence_threshold": 0.7
        },
        "segmentation": {
            "name": "sam-fashion",
            "path": "sam/model.pth",
            "device": "auto"
        },
        "vectorization": {
            "name": "deepsvg-custom",
            "path": "deepsvg/checkpoint.pth",
            "max_paths": 50
        },
        "nlp": {
            "japanese": "cl-tohoku/bert-base-japanese-char-v2",
            "english": "bert-base-uncased"
        }
    }
    
    # エージェント設定
    AGENT_HEARTBEAT_INTERVAL: int = 30  # 秒
    AGENT_TASK_TIMEOUT: int = 3600  # 1時間
    AGENT_MAX_RETRIES: int = 3
    
    # MCP設定
    MCP_PROVIDERS: List[Dict[str, Any]] = []
    MCP_CACHE_TTL: int = 3600
    MCP_REQUEST_TIMEOUT: int = 30
    
    # セキュリティ
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ロギング
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # モニタリング
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

## 9. 使用例（認証基盤統合）

### 9.1 認証付き用語収集タスクの実行

```python
# 用語収集タスクの作成（認証付き）
import httpx
from datetime import datetime

# BFF-Web経由でアクセストークンを取得済みと仮定
access_token = "eyJhbGciOiJSUzI1NiIs..."

api_url = "http://localhost:8000/api/v1/ai/tasks"

# 認証ヘッダー付きリクエスト
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

task_data = {
    "agent_type": "term_collector",
    "priority": "high",
    "configuration": {
        "source": {
            "type": "document",
            "url": "https://example.com/fashion_catalog.pdf"
        },
        "extraction": {
            "languages": ["ja", "en"],
            "confidence_threshold": 0.75,
            "categories": ["garment_parts"],
            "extract_images": True
        }
    }
}

# 認証付きリクエスト送信
response = httpx.post(api_url, json=task_data, headers=headers)

# スコープ不足の場合
if response.status_code == 403:
    print("エラー: 'terms:extract' スコープが必要です")
elif response.status_code == 401:
    print("エラー: トークンが無効または期限切れです")
elif response.status_code == 200:
    task_id = response.json()["data"]["task_id"]
    print(f"タスク作成成功: {task_id}")

# タスク状態の確認（認証付き）
status_response = httpx.get(f"{api_url}/{task_id}", headers=headers)
print(status_response.json())
```

### 9.2 SVG生成タスクの実行

```python
# SVG生成タスクの作成
task_data = {
    "agent_type": "svg_generator",
    "priority": "normal",
    "configuration": {
        "input": {
            "image_url": "https://example.com/button_photo.jpg",
            "part_type": "button"
        },
        "options": {
            "remove_background": True,
            "optimize_paths": True,
            "generate_parameters": True
        }
    }
}

response = httpx.post(api_url, json=task_data)
```

## 10. パフォーマンス最適化

### 10.1 並列処理
- Celeryワーカーの水平スケーリング
- 非同期I/O (asyncio) の活用
- バッチ処理の実装

### 10.2 キャッシング
- Redis によるクエリ結果キャッシュ
- モデル推論結果のキャッシュ
- 外部API レスポンスのキャッシュ

### 10.3 リソース管理
- GPU メモリの効率的な利用
- モデルの遅延ロード
- メモリリークの防止

## 11. モニタリング・ログ

### 11.1 メトリクス
- Prometheus によるメトリクス収集
- Grafana ダッシュボード
- カスタムメトリクス定義

### 11.2 ログ管理
- structlog による構造化ログ
- ELK スタックとの統合
- エラートラッキング (Sentry)

### 11.3 ヘルスチェック
- エージェント生存監視
- タスクキューの監視
- モデル可用性チェック