# LLMデプロイメント・運用ガイド

## 1. 概要

本ドキュメントは、テックパック生成アプリケーションで使用するLLMのデプロイメント方法、運用手順、および最適化ガイドラインを提供します。

## 2. デプロイメント構成

### 2.1 ハイブリッド構成

```yaml
deployment_architecture:
  on_premise:
    models:
      - llama-3.1-8b
      - llama-3.1-70b
      - mistral-7b
      - bert-base-japanese
    infrastructure:
      gpu_nodes:
        - type: NVIDIA A100 40GB
          count: 4
          purpose: "Llama 70B推論"
        - type: NVIDIA RTX 4090
          count: 8
          purpose: "Llama 8B/Mistral推論"
      cpu_nodes:
        - type: AMD EPYC 7763
          count: 2
          purpose: "BERT/軽量モデル"
  
  cloud_api:
    providers:
      anthropic:
        models:
          - claude-3-haiku-20240307
          - claude-3.5-sonnet-20241022
        region: us-east-1
      
      openai:
        models:
          - gpt-3.5-turbo-0125
          - gpt-4-turbo-preview
        region: us-east-1
      
      google:
        models:
          - gemini-1.5-pro-latest
        region: asia-northeast1
```

### 2.2 ローカルLLMデプロイメント

#### vLLM による高速推論

```python
# vllm_server.py
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import torch

app = FastAPI()

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    
class InferenceResponse(BaseModel):
    text: str
    model: str
    usage: dict

# モデル初期化
llm_8b = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=8192,
    dtype="float16",
    enforce_eager=False,  # CUDAグラフ最適化
    enable_lora=True,     # LoRAサポート
    max_lora_rank=64
)

llm_70b = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,  # 4GPU並列
    gpu_memory_utilization=0.95,
    max_model_len=4096,
    dtype="float16",
    enforce_eager=False
)

@app.post("/v1/completions/llama-8b", response_model=InferenceResponse)
async def generate_8b(request: InferenceRequest):
    try:
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop
        )
        
        outputs = llm_8b.generate([request.prompt], sampling_params)
        output = outputs[0]
        
        return InferenceResponse(
            text=output.outputs[0].text,
            model="llama-3.1-8b",
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
                "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Kubernetes デプロイメント
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama-8b
spec:
  replicas: 2
  selector:
    matchLabels:
      app: vllm-llama-8b
  template:
    metadata:
      labels:
        app: vllm-llama-8b
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["python", "-m", "vllm.entrypoints.openai.api_server"]
        args:
          - "--model=meta-llama/Llama-3.1-8B-Instruct"
          - "--tensor-parallel-size=1"
          - "--gpu-memory-utilization=0.9"
          - "--max-model-len=8192"
          - "--port=8000"
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "32Gi"
            cpu: "8"
        env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
      nodeSelector:
        gpu-type: rtx4090
"""
```

#### Triton Inference Server による最適化

```python
# triton_config.pbtxt
name: "llama_8b_ensemble"
platform: "ensemble"
max_batch_size: 16
input [
  {
    name: "text"
    data_type: TYPE_STRING
    dims: [1]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_STRING
    dims: [1]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "tokenizer"
      model_version: -1
      input_map {
        key: "text"
        value: "text"
      }
      output_map {
        key: "input_ids"
        value: "input_ids"
      }
    },
    {
      model_name: "llama_8b_tensorrt"
      model_version: -1
      input_map {
        key: "input_ids"
        value: "input_ids"
      }
      output_map {
        key: "logits"
        value: "logits"
      }
    },
    {
      model_name: "postprocessor"
      model_version: -1
      input_map {
        key: "logits"
        value: "logits"
      }
      output_map {
        key: "output"
        value: "output"
      }
    }
  ]
}

# TensorRT最適化スクリプト
import tensorrt as trt
import torch
from transformers import AutoModelForCausalLM

def optimize_model_with_tensorrt(model_path: str, output_path: str):
    """モデルをTensorRTで最適化"""
    
    # PyTorchモデルロード
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    
    # ONNXエクスポート
    dummy_input = torch.randint(0, 50000, (1, 512))
    torch.onnx.export(
        model,
        dummy_input,
        f"{output_path}/model.onnx",
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size', 1: 'sequence'}
        },
        opset_version=16
    )
    
    # TensorRTエンジン構築
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # 最適化設定
    config.max_workspace_size = 16 * (1 << 30)  # 16GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16精度
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    
    # プロファイル設定（動的バッチサイズ）
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "input_ids",
        min=(1, 1),
        opt=(8, 512),
        max=(16, 2048)
    )
    config.add_optimization_profile(profile)
    
    # エンジンビルド
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    with open(f"{output_path}/model.onnx", 'rb') as f:
        parser.parse(f.read())
    
    engine = builder.build_engine(network, config)
    
    # シリアライズ
    with open(f"{output_path}/model.trt", 'wb') as f:
        f.write(engine.serialize())
```

### 2.3 API統合レイヤー

```python
# llm_gateway.py
from typing import Dict, Any, Optional
import asyncio
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from collections import defaultdict
import redis
import pickle

class LLMGateway:
    """統一的なLLMアクセスインターフェース"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            decode_responses=False
        )
        self.rate_limiters = defaultdict(RateLimiter)
        self.circuit_breakers = defaultdict(CircuitBreaker)
        
    async def generate(
        self,
        model: str,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """統一的な生成API"""
        
        # キャッシュチェック
        cache_key = self._generate_cache_key(model, prompt, kwargs)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            return cached_result
        
        # レート制限チェック
        if not await self._check_rate_limit(model):
            raise RateLimitException(f"Rate limit exceeded for {model}")
        
        # サーキットブレーカーチェック
        if not self.circuit_breakers[model].is_closed():
            raise ServiceUnavailableException(f"Service {model} is unavailable")
        
        try:
            # モデル別ルーティング
            if model.startswith('llama') or model.startswith('mistral'):
                result = await self._call_local_model(model, prompt, **kwargs)
            elif model.startswith('claude'):
                result = await self._call_claude(model, prompt, **kwargs)
            elif model.startswith('gpt'):
                result = await self._call_openai(model, prompt, **kwargs)
            elif model.startswith('gemini'):
                result = await self._call_gemini(model, prompt, **kwargs)
            else:
                raise ValueError(f"Unknown model: {model}")
            
            # 成功をサーキットブレーカーに記録
            self.circuit_breakers[model].record_success()
            
            # キャッシュ保存
            self._save_cache(cache_key, result, ttl=3600)
            
            return result
            
        except Exception as e:
            # 失敗をサーキットブレーカーに記録
            self.circuit_breakers[model].record_failure()
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _call_claude(
        self, 
        model: str, 
        prompt: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Claude API呼び出し"""
        
        headers = {
            "x-api-key": self.config['anthropic']['api_key'],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get('max_tokens', 1024),
            "temperature": kwargs.get('temperature', 0.7),
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                return {
                    'text': result['content'][0]['text'],
                    'model': model,
                    'usage': result['usage'],
                    'provider': 'anthropic',
                    'latency': response.headers.get('X-Response-Time', 0)
                }
```

## 3. パフォーマンス最適化

### 3.1 モデル量子化

```python
# quantization.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import bitsandbytes as bnb

class ModelQuantizer:
    """モデル量子化ユーティリティ"""
    
    @staticmethod
    def quantize_gptq(
        model_path: str,
        output_path: str,
        bits: int = 4
    ):
        """GPTQ量子化（4bit/8bit）"""
        
        # 量子化設定
        quantize_config = BaseQuantizeConfig(
            bits=bits,
            group_size=128,
            desc_act=False,
            model_file_base_name="model"
        )
        
        # モデルロード
        model = AutoGPTQForCausalLM.from_pretrained(
            model_path,
            quantize_config=quantize_config,
            device_map="auto"
        )
        
        # キャリブレーションデータ
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        calibration_data = [
            "アパレル業界における技術仕様書の作成方法について説明してください。",
            "ボタンの種類と特徴を日本語と英語で列挙してください。",
            # ... more calibration samples
        ]
        
        # 量子化実行
        model.quantize(calibration_data, use_triton=True)
        
        # 保存
        model.save_quantized(output_path)
        tokenizer.save_pretrained(output_path)
    
    @staticmethod
    def quantize_bnb(
        model_path: str,
        load_in_4bit: bool = True
    ):
        """BitsAndBytes量子化（動的）"""
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        return model
```

### 3.2 バッチ処理最適化

```python
# batch_optimizer.py
import asyncio
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import time

@dataclass
class BatchRequest:
    id: str
    prompt: str
    model: str
    kwargs: Dict[str, Any]
    timestamp: float
    future: asyncio.Future

class BatchProcessor:
    """動的バッチ処理最適化"""
    
    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_time: float = 0.1  # 100ms
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queues = defaultdict(list)
        self.processing = False
        
    async def add_request(
        self,
        request: BatchRequest
    ) -> Any:
        """リクエストをバッチキューに追加"""
        
        queue = self.queues[request.model]
        queue.append(request)
        
        # バッチ処理開始条件
        if len(queue) >= self.max_batch_size:
            asyncio.create_task(self._process_batch(request.model))
        elif not self.processing:
            asyncio.create_task(self._wait_and_process(request.model))
        
        return await request.future
    
    async def _wait_and_process(self, model: str):
        """タイムアウト待機後にバッチ処理"""
        await asyncio.sleep(self.max_wait_time)
        await self._process_batch(model)
    
    async def _process_batch(self, model: str):
        """バッチ処理実行"""
        self.processing = True
        queue = self.queues[model]
        
        if not queue:
            self.processing = False
            return
        
        # バッチサイズ分取得
        batch = queue[:self.max_batch_size]
        self.queues[model] = queue[self.max_batch_size:]
        
        try:
            # プロンプトをバッチ化
            prompts = [req.prompt for req in batch]
            
            # パディング処理
            padded_prompts = self._pad_prompts(prompts)
            
            # バッチ推論実行
            results = await self._run_batch_inference(
                model, 
                padded_prompts,
                batch[0].kwargs  # 同じkwargsを仮定
            )
            
            # 結果を各リクエストに配布
            for i, req in enumerate(batch):
                req.future.set_result(results[i])
                
        except Exception as e:
            # エラーを全リクエストに伝播
            for req in batch:
                req.future.set_exception(e)
        finally:
            self.processing = False
            
            # 残りのキューを処理
            if self.queues[model]:
                asyncio.create_task(self._process_batch(model))
```

### 3.3 分散推論

```python
# distributed_inference.py
import ray
from ray import serve
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

@serve.deployment(
    num_replicas=4,
    ray_actor_options={
        "num_gpus": 1,
        "num_cpus": 4,
        "memory": 32 * 1024 * 1024 * 1024  # 32GB
    }
)
class DistributedLLMEndpoint:
    """Ray Serveによる分散推論エンドポイント"""
    
    def __init__(self, model_path: str):
        # 分散初期化
        if dist.is_available():
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        
        # モデルロード
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=f"cuda:{self.rank}",
            torch_dtype=torch.float16
        )
        
        # DDP設定
        if dist.is_available():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.rank]
            )
    
    async def __call__(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """推論実行"""
        prompt = request['prompt']
        kwargs = request.get('kwargs', {})
        
        # トークナイズ
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(f"cuda:{self.rank}")
        
        # 推論
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get('max_tokens', 512),
                temperature=kwargs.get('temperature', 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # デコード
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return {
            'text': generated_text,
            'model': self.model_name,
            'usage': self._calculate_usage(inputs, outputs)
        }

# Ray Serveデプロイメント
serve.run(
    DistributedLLMEndpoint.bind(
        model_path="meta-llama/Llama-3.1-70B-Instruct"
    ),
    route_prefix="/llm",
    host="0.0.0.0",
    port=8000
)
```

## 4. コスト最適化戦略

### 4.1 動的モデル選択

```python
class CostOptimizer:
    """コストベースのモデル選択最適化"""
    
    def __init__(self, budget_tracker: BudgetTracker):
        self.budget_tracker = budget_tracker
        self.model_costs = {
            # ローカルモデル（インフラコストのみ）
            'llama-3.1-8b': {'input': 0.0001, 'output': 0.0001},
            'llama-3.1-70b': {'input': 0.001, 'output': 0.001},
            
            # API モデル（トークンあたりコスト）
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'claude-3.5-sonnet': {'input': 0.003, 'output': 0.015},
            'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
            'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        }
        
    def select_model_within_budget(
        self,
        task_requirements: Dict[str, Any],
        remaining_budget: float
    ) -> str:
        """予算内で最適なモデルを選択"""
        
        # タスク要件から必要な品質レベルを判定
        required_quality = task_requirements.get('quality', 'medium')
        estimated_tokens = task_requirements.get('estimated_tokens', 1000)
        
        # 品質要件を満たすモデル候補
        candidates = self._get_quality_matching_models(required_quality)
        
        # コスト計算
        model_costs = []
        for model in candidates:
            cost = self._estimate_cost(model, estimated_tokens)
            if cost <= remaining_budget:
                model_costs.append((model, cost))
        
        # コスト効率が最も良いモデルを選択
        if model_costs:
            return min(model_costs, key=lambda x: x[1])[0]
        else:
            # 予算内に収まるモデルがない場合は最も安いモデル
            return 'llama-3.1-8b'
```

### 4.2 キャッシング戦略

```python
class SemanticCache:
    """セマンティックキャッシュによるコスト削減"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_store = faiss.IndexFlatIP(384)  # 内積
        self.cache_store = {}
        self.similarity_threshold = 0.95
        
    async def get_or_generate(
        self,
        prompt: str,
        model: str,
        generator_func
    ) -> Dict[str, Any]:
        """類似プロンプトのキャッシュを確認または生成"""
        
        # プロンプトのベクトル化
        embedding = self.embedding_model.encode([prompt])[0]
        
        # 類似検索
        if self.vector_store.ntotal > 0:
            distances, indices = self.vector_store.search(
                embedding.reshape(1, -1), 
                k=1
            )
            
            if distances[0][0] > self.similarity_threshold:
                # キャッシュヒット
                cache_key = list(self.cache_store.keys())[indices[0][0]]
                cached_result = self.cache_store[cache_key]
                
                # キャッシュメタデータを追加
                cached_result['cache_hit'] = True
                cached_result['similarity_score'] = float(distances[0][0])
                
                return cached_result
        
        # キャッシュミス - 新規生成
        result = await generator_func(prompt, model)
        
        # キャッシュに追加
        self.vector_store.add(embedding.reshape(1, -1))
        self.cache_store[prompt] = result
        
        result['cache_hit'] = False
        return result
```

## 5. モニタリングとアラート

### 5.1 メトリクス収集

```python
# metrics_collector.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

# Prometheusメトリクス定義
llm_requests_total = Counter(
    'llm_requests_total', 
    'Total LLM requests',
    ['model', 'status']
)

llm_request_duration = Histogram(
    'llm_request_duration_seconds',
    'LLM request duration',
    ['model']
)

llm_token_usage = Counter(
    'llm_token_usage_total',
    'Total tokens used',
    ['model', 'type']  # type: input/output
)

llm_cost_total = Counter(
    'llm_cost_total_dollars',
    'Total cost in dollars',
    ['model']
)

llm_error_rate = Gauge(
    'llm_error_rate',
    'Error rate by model',
    ['model']
)

class LLMMetricsCollector:
    """LLMメトリクス収集"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        start_http_server(9090)  # Prometheusエンドポイント
        
    def record_request(
        self,
        model: str,
        duration: float,
        status: str,
        tokens: Dict[str, int],
        cost: float
    ):
        """リクエストメトリクスを記録"""
        
        # カウンター更新
        llm_requests_total.labels(model=model, status=status).inc()
        
        # レイテンシ記録
        llm_request_duration.labels(model=model).observe(duration)
        
        # トークン使用量
        llm_token_usage.labels(model=model, type='input').inc(
            tokens.get('input_tokens', 0)
        )
        llm_token_usage.labels(model=model, type='output').inc(
            tokens.get('output_tokens', 0)
        )
        
        # コスト追跡
        llm_cost_total.labels(model=model).inc(cost)
        
        # ログ出力
        self.logger.info(
            f"LLM Request: model={model}, status={status}, "
            f"duration={duration:.2f}s, cost=${cost:.4f}"
        )
```

### 5.2 アラート設定

```yaml
# prometheus_alerts.yml
groups:
  - name: llm_alerts
    interval: 30s
    rules:
      # 高エラー率
      - alert: HighLLMErrorRate
        expr: |
          rate(llm_requests_total{status="error"}[5m]) 
          / rate(llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.model }}"
      
      # 高レイテンシ
      - alert: HighLLMLatency
        expr: |
          histogram_quantile(0.95, 
            rate(llm_request_duration_seconds_bucket[5m])
          ) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM latency detected"
          description: "95th percentile latency is {{ $value }}s for {{ $labels.model }}"
      
      # コスト急増
      - alert: LLMCostSpike
        expr: |
          rate(llm_cost_total_dollars[1h]) > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "LLM cost spike detected"
          description: "Hourly cost rate is ${{ $value }} for {{ $labels.model }}"
      
      # モデル利用不可
      - alert: LLMModelUnavailable
        expr: |
          up{job="llm_health"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "LLM model unavailable"
          description: "Model {{ $labels.model }} has been down for 2 minutes"
```

## 6. トラブルシューティング

### 6.1 一般的な問題と解決方法

```python
class LLMTroubleshooter:
    """LLM関連の問題診断と自動修復"""
    
    def diagnose_issue(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """問題を診断して解決策を提案"""
        
        diagnosis = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'possible_causes': [],
            'recommended_actions': []
        }
        
        # OOMエラー
        if "out of memory" in str(error).lower():
            diagnosis['possible_causes'].extend([
                "GPU memory exhausted",
                "Batch size too large",
                "Model too large for available memory"
            ])
            diagnosis['recommended_actions'].extend([
                "Reduce batch size",
                "Use gradient checkpointing",
                "Switch to smaller model or quantized version",
                "Clear GPU cache: torch.cuda.empty_cache()"
            ])
        
        # レート制限
        elif "rate limit" in str(error).lower():
            diagnosis['possible_causes'].append("API rate limit exceeded")
            diagnosis['recommended_actions'].extend([
                "Implement exponential backoff",
                "Use request batching",
                "Upgrade API tier",
                "Distribute requests across multiple API keys"
            ])
        
        # タイムアウト
        elif "timeout" in str(error).lower():
            diagnosis['possible_causes'].extend([
                "Model server overloaded",
                "Network issues",
                "Request too complex"
            ])
            diagnosis['recommended_actions'].extend([
                "Increase timeout duration",
                "Retry with exponential backoff",
                "Simplify prompt",
                "Check model server health"
            ])
        
        return diagnosis
    
    async def auto_remediate(self, diagnosis: Dict[str, Any]) -> bool:
        """可能な場合は自動修復を試みる"""
        
        error_type = diagnosis['error_type']
        
        if error_type == "OutOfMemoryError":
            # GPUキャッシュクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return True
                
        elif error_type == "RateLimitError":
            # 代替モデルに切り替え
            context = diagnosis['context']
            if context.get('model') == 'gpt-4-turbo':
                context['fallback_model'] = 'gpt-3.5-turbo'
                return True
                
        return False
```

## 7. セキュリティ考慮事項

### 7.1 APIキー管理

```python
# secure_key_manager.py
from cryptography.fernet import Fernet
import os
from typing import Dict, Optional
import json

class SecureKeyManager:
    """APIキーの安全な管理"""
    
    def __init__(self, key_file: str = ".keys.enc"):
        self.key_file = key_file
        self.master_key = self._get_or_create_master_key()
        self.cipher = Fernet(self.master_key)
        
    def _get_or_create_master_key(self) -> bytes:
        """マスターキーの取得または生成"""
        key_env = os.environ.get('MASTER_KEY')
        
        if key_env:
            return key_env.encode()
        else:
            # HSM/KMSから取得（本番環境）
            # ここではデモ用に環境変数を使用
            raise ValueError("MASTER_KEY not found in environment")
    
    def store_api_key(self, provider: str, key: str):
        """APIキーを暗号化して保存"""
        keys = self._load_keys()
        keys[provider] = self.cipher.encrypt(key.encode()).decode()
        self._save_keys(keys)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """APIキーを復号化して取得"""
        keys = self._load_keys()
        encrypted_key = keys.get(provider)
        
        if encrypted_key:
            return self.cipher.decrypt(encrypted_key.encode()).decode()
        return None
    
    def rotate_keys(self):
        """キーローテーション"""
        old_keys = self._load_keys()
        new_master_key = Fernet.generate_key()
        new_cipher = Fernet(new_master_key)
        
        # 全キーを再暗号化
        new_keys = {}
        for provider, encrypted_key in old_keys.items():
            decrypted = self.cipher.decrypt(encrypted_key.encode())
            new_keys[provider] = new_cipher.encrypt(decrypted).decode()
        
        # 新しいマスターキーとキーを保存
        self.master_key = new_master_key
        self.cipher = new_cipher
        self._save_keys(new_keys)
```

### 7.2 プロンプトインジェクション対策

```python
class PromptSanitizer:
    """プロンプトインジェクション対策"""
    
    def __init__(self):
        self.forbidden_patterns = [
            r"ignore previous instructions",
            r"disregard all prior",
            r"system prompt",
            r"你是.*助手",  # 中国語のインジェクション
            r"あなたは.*アシスタント",  # 日本語のインジェクション
        ]
        
    def sanitize_prompt(self, prompt: str) -> str:
        """危険なパターンを検出・除去"""
        
        # 小文字変換して検査
        lower_prompt = prompt.lower()
        
        for pattern in self.forbidden_patterns:
            if re.search(pattern, lower_prompt, re.IGNORECASE):
                raise SecurityException(
                    f"Potential prompt injection detected: {pattern}"
                )
        
        # 特殊文字のエスケープ
        sanitized = html.escape(prompt)
        
        # 長さ制限
        max_length = 10000
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized
    
    def validate_response(self, response: str) -> bool:
        """レスポンスの安全性検証"""
        
        # 機密情報の漏洩チェック
        sensitive_patterns = [
            r"api[_-]?key",
            r"password",
            r"secret",
            r"token",
            r"\b[A-Za-z0-9]{40}\b",  # APIキーパターン
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return False
        
        return True
```