# MetaClaw Remote Training Backend - 设计文档

## 1. 目标

在 MetaClaw 现有后端选择机制（`sdk_backend.py`：tinker / mint）基础上，新增 `remote` 后端，支持将 RL 训练任务发送到自建 GPU 服务器执行。

**核心原则**：
- 扩展，不替换。Tinker / MinT 原有链路完全保留
- 用户通过 `config.yaml` 切换后端，零代码改动
- 数据安全：训练数据和模型权重留在自有服务器

## 2. 架构

```
┌─────────────────────────────────────────────────┐
│              MetaClaw (Mac 本地)                  │
│                                                   │
│  config.yaml: backend=remote                      │
│       ↓                                           │
│  sdk_backend.py → resolve → RemoteBackend         │
│       ↓                                           │
│  trainer.py / api_server.py / data_formatter.py   │
│  通过抽象接口调用，不直接 import tinker             │
│       ↓                                           │
│  remote_backend/client.py                         │
│  (HTTP 客户端，实现与 tinker 相同的接口)            │
└──────────────────────┬────────────────────────────┘
                       │ HTTPS (Bearer Token 认证)
                       ▼
┌─────────────────────────────────────────────────┐
│         远端 GPU 服务器 (8×A100-80GB)             │
│                                                   │
│  metaclaw-training-server (FastAPI)               │
│  ├── 推理引擎: vLLM / SGLang                      │
│  ├── 训练引擎: PyTorch + PEFT/LoRA               │
│  ├── Checkpoint 管理: 本地文件系统                 │
│  └── 权重热切换: LoRA merge → 重载推理引擎         │
└─────────────────────────────────────────────────┘
```

## 3. Tinker SDK 接口全量梳理

### 3.1 trainer.py 中的调用

```python
# setup()
import tinker
service_client = tinker.ServiceClient()
training_client = await service_client.create_lora_training_client_async(
    base_model=config.model_name,   # e.g. "Qwen/Qwen3-4B"
    rank=config.lora_rank,          # e.g. 32
)
# 可选：从 checkpoint 恢复
await training_client.load_state_async(ckpt_path)  # str

# 获取初始推理客户端
sampling_client = await training_client.save_weights_and_get_sampling_client_async()

# _train_on_batch()
import tinker
await training_client.forward_backward_async(data_D, loss_fn=config.loss_fn)
# data_D: list[tinker.Datum]
# loss_fn: str ("importance_sampling" | "ppo" | "cispo")

await training_client.optim_step_async(
    tinker.AdamParams(learning_rate=config.learning_rate)
)

sampling_client = await training_client.save_weights_and_get_sampling_client_async(
    name="openclaw_lora"
)
# 返回 SamplingClient 对象

result = await training_client.save_state_async(name=ckpt_name)
# result.path: str (checkpoint 路径)
```

### 3.2 data_formatter.py 中的调用

```python
import tinker
from tinker import TensorData

# sample_to_datum()
model_input = tinker.ModelInput.from_ints(all_tokens[:-1])  # list[int] → ModelInput

datum = tinker.Datum(
    model_input=model_input,
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens, dtype=torch.long)),
        "logprobs":      TensorData.from_torch(torch.tensor(logprobs, dtype=torch.float32)),
        "advantages":    TensorData.from_torch(torch.tensor(advantages, dtype=torch.float32)),
    },
)
```

### 3.3 api_server.py 中的调用（推理）

```python
import tinker

# _forward_to_tinker()
chunk = tinker.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
model_input = tinker.ModelInput(chunks=[chunk])

sampling_params = tinker.SamplingParams(
    temperature=0.7,
    max_tokens=2048,
    top_k=50,
    top_p=0.95,
    stop=["..."],  # optional
)

response = await sampling_client.sample_async(
    prompt=model_input,
    num_samples=1,
    sampling_params=sampling_params,
    include_prompt_logprobs=False,
    topk_prompt_logprobs=0,
)

# 返回结构
seq = response.sequences[0]
seq.tokens      # list[int] — 生成的 token ids
seq.logprobs    # list[float] — 每个 token 的 log prob
seq.stop_reason # str — 停止原因
```

### 3.4 完整类型接口汇总

```
tinker.ServiceClient
  └── create_lora_training_client_async(base_model: str, rank: int) → TrainingClient

tinker.TrainingClient (training_client)
  ├── forward_backward_async(datums: list[Datum], loss_fn: str)
  ├── optim_step_async(params: AdamParams)
  ├── save_weights_and_get_sampling_client_async(name: str = "") → SamplingClient
  ├── save_state_async(name: str) → SaveResult (has .path: str)
  └── load_state_async(path: str)

tinker.SamplingClient (sampling_client)
  └── sample_async(prompt: ModelInput, num_samples: int,
                   sampling_params: SamplingParams,
                   include_prompt_logprobs: bool,
                   topk_prompt_logprobs: int) → SampleResponse

tinker.SampleResponse
  └── sequences: list[Sequence]

tinker.Sequence
  ├── tokens: list[int]
  ├── logprobs: list[float]
  └── stop_reason: str

tinker.Datum(model_input: ModelInput, loss_fn_inputs: dict[str, TensorData])
tinker.ModelInput
  ├── from_ints(tokens: list[int]) → ModelInput        (用于训练)
  └── ModelInput(chunks=[EncodedTextChunk])             (用于推理)
tinker.EncodedTextChunk(tokens: list[int], type: str)
tinker.TensorData
  └── from_torch(tensor: torch.Tensor) → TensorData
tinker.SamplingParams(temperature, max_tokens, top_k, top_p, stop)
tinker.AdamParams(learning_rate: float)
```

## 4. 改动清单

### 4.1 MetaClaw 侧改动（Mac 本地，6 个文件）

#### 4.1.1 `config.py` — 新增 remote 配置字段

```python
# 新增字段
backend: str = "auto"              # "auto" | "tinker" | "mint" | "remote"
remote_url: str = ""               # e.g. "http://115.190.60.96:8000"
remote_api_key: str = ""           # Bearer token 认证
remote_timeout_s: float = 600.0    # 单次请求超时
```

#### 4.1.2 `config_store.py` — 新增 rl.backend / rl.remote 配置桥接

`_DEFAULTS` 新增：
```python
"rl": {
    ...
    "backend": "auto",
    "remote_url": "",
    "remote_api_key": "",
    "remote_timeout_s": 600,
}
```

`to_metaclaw_config()` 新增字段映射。

#### 4.1.3 `sdk_backend.py` — 扩展后端选择

```python
_VALID_BACKENDS = {"auto", "tinker", "mint", "remote"}

# 新增 remote 后端检测
def _has_remote_signal(config) -> bool:
    return bool(getattr(config, "remote_url", ""))

def infer_backend_key(config) -> str:
    configured = configured_backend_name(config)
    if configured in {"tinker", "mint", "remote"}:
        return configured
    if _has_remote_signal(config):
        return "remote"
    if _has_mint_signal(config) and _module_available("mint"):
        return "mint"
    return "tinker"
```

`resolve_sdk_backend()` 返回 `SDKBackend` 时，remote 后端导入 `metaclaw.remote_backend` 模块。

#### 4.1.4 `trainer.py` — 通过 backend 抽象替代直接 import tinker

**关键改动**：不再硬编码 `import tinker`，改为通过 `sdk_backend` 动态获取。

```python
# 原来
import tinker
service_client = tinker.ServiceClient()

# 改后
from .sdk_backend import resolve_sdk_backend
backend = resolve_sdk_backend(self.config)
sdk = backend.module  # tinker / mint / remote_backend
service_client = sdk.ServiceClient()
```

同理，`tinker.AdamParams` → `sdk.AdamParams`

#### 4.1.5 `data_formatter.py` — 延迟导入 + 后端感知

将 `import tinker` 改为接收 backend module 参数，或使用全局注册：

```python
# 方案 A：函数参数传入
def sample_to_datum(sample, advantage, kl_penalty_coef=0.0, sdk=None):
    if sdk is None:
        import tinker as sdk
    model_input = sdk.ModelInput.from_ints(...)
    ...

# 方案 B：模块级注册（更简洁）
_sdk = None
def set_sdk(module):
    global _sdk
    _sdk = module

def _get_sdk():
    global _sdk
    if _sdk is None:
        import tinker
        _sdk = tinker
    return _sdk
```

推荐方案 B，改动最小。trainer.py 在 setup() 中调用 `data_formatter.set_sdk(backend.module)` 即可。

#### 4.1.6 `api_server.py` — _forward_to_tinker 改为后端感知

将 `import tinker` 替换为使用注入的 backend module：

```python
# 构造函数新增参数
def __init__(self, ..., sdk_module=None):
    self._sdk = sdk_module

# _forward_to_tinker 中
sdk = self._sdk or __import__("tinker")
chunk = sdk.EncodedTextChunk(tokens=list(prompt_ids), type="encoded_text")
model_input = sdk.ModelInput(chunks=[chunk])
sampling_params = sdk.SamplingParams(**sp_kwargs)
```

### 4.2 新增模块：`metaclaw/remote_backend/`

```
metaclaw/remote_backend/
├── __init__.py          # 导出 ServiceClient, Datum, ModelInput, etc.
├── client.py            # HTTP 客户端，实现所有接口
├── types.py             # 数据类型 (Datum, ModelInput, TensorData, etc.)
└── serialization.py     # Datum 序列化/反序列化 (tensor → bytes)
```

#### 核心类实现（client.py）

```python
import httpx

class ServiceClient:
    def __init__(self, base_url=None, api_key=None):
        self.base_url = base_url or os.environ.get("REMOTE_TRAINING_URL", "")
        self.api_key = api_key or os.environ.get("REMOTE_TRAINING_API_KEY", "")

    async def create_lora_training_client_async(self, base_model, rank):
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/v1/lora/create", json={
                "base_model": base_model,
                "rank": rank,
            }, headers=self._auth_headers(), timeout=300)
            resp.raise_for_status()
            session_id = resp.json()["session_id"]
        return TrainingClient(self.base_url, self.api_key, session_id)

class TrainingClient:
    def __init__(self, base_url, api_key, session_id):
        ...

    async def forward_backward_async(self, datums, loss_fn):
        payload = serialize_datums(datums)  # tensor → numpy bytes → base64
        await self._post("/v1/train/forward_backward", {
            "session_id": self.session_id,
            "datums": payload,
            "loss_fn": loss_fn,
        })

    async def optim_step_async(self, adam_params):
        await self._post("/v1/train/optim_step", {
            "session_id": self.session_id,
            "learning_rate": adam_params.learning_rate,
        })

    async def save_weights_and_get_sampling_client_async(self, name=""):
        resp = await self._post("/v1/weights/save", {
            "session_id": self.session_id,
            "name": name,
        })
        return SamplingClient(self.base_url, self.api_key, resp["sampling_endpoint"])

    async def save_state_async(self, name):
        resp = await self._post("/v1/checkpoint/save", {
            "session_id": self.session_id,
            "name": name,
        })
        return SaveResult(path=resp["path"])

    async def load_state_async(self, path):
        await self._post("/v1/checkpoint/load", {
            "session_id": self.session_id,
            "path": path,
        })

class SamplingClient:
    async def sample_async(self, prompt, num_samples, sampling_params, **kwargs):
        resp = await self._post("/v1/sample", {
            "tokens": prompt.to_token_list(),
            "num_samples": num_samples,
            "temperature": sampling_params.temperature,
            "max_tokens": sampling_params.max_tokens,
            "top_k": sampling_params.top_k,
            "top_p": sampling_params.top_p,
            "stop": sampling_params.stop,
        })
        return SampleResponse.from_dict(resp)
```

#### 数据序列化（serialization.py）

Datum 中的 TensorData 内含 PyTorch tensor，需要序列化传输：

```python
import base64
import numpy as np

def serialize_tensor_data(td: TensorData) -> dict:
    """TensorData → {dtype, shape, data_b64}"""
    arr = td.to_numpy()  # 或 td.tensor.numpy()
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode(),
    }

def serialize_datum(datum: Datum) -> dict:
    return {
        "model_input_tokens": datum.model_input.to_token_list(),
        "loss_fn_inputs": {
            k: serialize_tensor_data(v)
            for k, v in datum.loss_fn_inputs.items()
        },
    }
```

### 4.3 远端服务：`metaclaw-training-server/`

部署在 GPU 服务器上的 FastAPI 服务。

```
metaclaw-training-server/
├── server.py            # FastAPI 主入口
├── training_engine.py   # PEFT/LoRA 训练逻辑
├── inference_engine.py  # vLLM/SGLang 推理逻辑
├── checkpoint.py        # Checkpoint 管理
├── auth.py              # Bearer Token 认证中间件
├── config.py            # 服务端配置
└── requirements.txt     # 依赖
```

#### API 端点设计

```
POST /v1/lora/create
  Request:  { "base_model": "Qwen/Qwen3-4B", "rank": 32 }
  Response: { "session_id": "uuid", "status": "ready" }
  行为: 加载 base model → 创建 PEFT LoRA adapter → 启动 vLLM 推理引擎

POST /v1/train/forward_backward
  Request:  { "session_id": "...", "datums": [...], "loss_fn": "importance_sampling" }
  Response: { "status": "ok", "loss": 0.123 }
  行为: 反序列化 datums → 前向传播 → 计算 loss → 反向传播 → 梯度累积

POST /v1/train/optim_step
  Request:  { "session_id": "...", "learning_rate": 1e-4 }
  Response: { "status": "ok", "grad_norm": 0.456 }
  行为: optimizer.step() → optimizer.zero_grad()

POST /v1/weights/save
  Request:  { "session_id": "...", "name": "openclaw_lora" }
  Response: { "status": "ok", "sampling_endpoint": "..." }
  行为: 保存 LoRA 权重 → 合并到 base model → 重载 vLLM 推理引擎

POST /v1/checkpoint/save
  Request:  { "session_id": "...", "name": "step_0005" }
  Response: { "path": "/data/checkpoints/step_0005" }

POST /v1/checkpoint/load
  Request:  { "session_id": "...", "path": "/data/checkpoints/step_0005" }
  Response: { "status": "ok" }

POST /v1/sample
  Request:  { "tokens": [1, 2, 3, ...], "num_samples": 1,
              "temperature": 0.7, "max_tokens": 2048, ... }
  Response: { "sequences": [{ "tokens": [...], "logprobs": [...],
              "stop_reason": "stop" }] }
  行为: vLLM 推理

GET /v1/health
  Response: { "status": "healthy", "gpu_count": 8, "model": "Qwen/Qwen3-4B" }
```

#### 训练引擎核心逻辑（training_engine.py）

```python
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

class TrainingSession:
    def __init__(self, base_model: str, rank: int):
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(self.model, lora_config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4
        )
        self.model.train()

    def forward_backward(self, datums, loss_fn):
        """MetaClaw 的 GRPO loss 实现"""
        total_loss = 0
        for datum in datums:
            input_ids = datum["model_input_tokens"]
            target_tokens = datum["loss_fn_inputs"]["target_tokens"]
            advantages = datum["loss_fn_inputs"]["advantages"]
            old_logprobs = datum["loss_fn_inputs"]["logprobs"]

            # Forward
            outputs = self.model(
                input_ids=torch.tensor([input_ids], device=self.model.device)
            )
            logits = outputs.logits[0]  # (T, vocab)

            # 计算 new log probs
            new_logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = new_logprobs.gather(
                1, torch.tensor(target_tokens, device=self.model.device).unsqueeze(1)
            ).squeeze(1)

            # Importance sampling loss (GRPO)
            if loss_fn == "importance_sampling":
                ratio = torch.exp(token_logprobs - torch.tensor(old_logprobs, device=self.model.device))
                loss = -(ratio * torch.tensor(advantages, device=self.model.device)).mean()
            # ... 其他 loss_fn

            loss.backward()
            total_loss += loss.item()
        return total_loss / len(datums)

    def optim_step(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad()
```

#### 推理引擎热切换（inference_engine.py）

```python
from vllm import LLM, SamplingParams as VLLMSamplingParams

class InferenceEngine:
    def __init__(self, model_path: str):
        self.llm = LLM(model=model_path, tensor_parallel_size=4)
        # 训练用 4 卡，推理用 4 卡（或共享）

    def reload(self, new_model_path: str):
        """权重更新后重载"""
        del self.llm
        torch.cuda.empty_cache()
        self.llm = LLM(model=new_model_path, tensor_parallel_size=4)

    def sample(self, token_ids, sampling_params):
        # vLLM 原生支持 token ids 输入
        outputs = self.llm.generate(
            prompt_token_ids=[token_ids],
            sampling_params=VLLMSamplingParams(
                temperature=sampling_params["temperature"],
                max_tokens=sampling_params["max_tokens"],
                top_k=sampling_params["top_k"],
                top_p=sampling_params["top_p"],
                logprobs=1,  # 返回 logprobs
            ),
        )
        ...
```

## 5. 用户配置方式

```yaml
# ~/.metaclaw/config.yaml

mode: rl                          # 切换到 rl 模式

rl:
  enabled: true
  backend: remote                 # ← 关键：选择 remote 后端
  remote_url: http://115.190.60.96:8000
  remote_api_key: your-secret-key
  remote_timeout_s: 600
  model: Qwen/Qwen3-4B
  lora_rank: 32
  batch_size: 4
  prm_model: gpt-5.4
  prm_api_key: sk-xxx
  prm_url: https://vibe.deepminer.ai/v1
```

切回 Tinker 只需：
```yaml
rl:
  backend: tinker                 # 或 auto
  tinker_api_key: tk-xxx
```

## 6. 数据序列化方案

### 选项比较

| 方案 | 优点 | 缺点 |
|------|------|------|
| JSON + base64 numpy | 简单，调试方便 | 体积大，序列化慢 |
| MessagePack + raw bytes | 紧凑，快速 | 需要额外依赖 |
| Protobuf | 最紧凑 | 定义复杂 |

**推荐：JSON + base64 numpy**，因为：
1. Datum 中的 tensor 通常很小（单条序列 < 64K tokens × 4 bytes ≈ 256KB）
2. 一个 batch 通常 4 条 ≈ 1MB，网络传输不是瓶颈
3. 调试友好，可直接 curl 测试

后续如果遇到瓶颈，再切 MessagePack。

## 7. 安全

1. **Bearer Token 认证**：所有 API 调用携带 `Authorization: Bearer <token>`
2. **SSH 隧道（可选）**：`ssh -L 8000:localhost:8000 -p 65000 root@115.190.60.96`，本地走 localhost
3. **HTTPS（可选）**：用 Let's Encrypt 或自签证书
4. **最小化暴露**：训练服务只监听需要的端口

推荐初期用 **SSH 隧道 + Bearer Token**，简单安全。

## 8. GPU 资源分配（8×A100-80GB）

| 用途 | GPU | 显存 |
|------|-----|------|
| 训练（PEFT/LoRA） | GPU 0-3 | ~40GB（Qwen3-4B bf16 + LoRA） |
| 推理（vLLM） | GPU 4-7 | ~20GB（Qwen3-4B bf16） |

如果训练更大模型（如 Qwen3-72B），可以调整分配。

## 9. 代码仓库 & 文件组织

所有改动提交到 fork 仓库：**https://github.com/OctoClaws/MetaClaw**

```
OctoClaws/MetaClaw/
├── metaclaw/
│   ├── config.py              # [改] 新增 remote 配置字段
│   ├── config_store.py        # [改] 新增 rl.backend/remote 配置桥接
│   ├── sdk_backend.py         # [改] 扩展 remote 后端选择
│   ├── trainer.py             # [改] 用 backend 抽象替代 import tinker
│   ├── data_formatter.py      # [改] 可注入 SDK 模块
│   ├── api_server.py          # [改] 可注入 SDK 模块
│   └── remote_backend/        # [新] remote 后端客户端
│       ├── __init__.py
│       ├── client.py          # HTTP 客户端（实现 Tinker 接口）
│       ├── types.py           # 数据类型
│       └── serialization.py   # tensor 序列化
├── metaclaw_training_server/  # [新] 远端训练服务
│   ├── __init__.py
│   ├── server.py              # FastAPI 主入口
│   ├── training_engine.py     # PEFT/LoRA 训练引擎
│   ├── inference_engine.py    # vLLM/SGLang 推理引擎
│   ├── checkpoint.py          # Checkpoint 管理
│   ├── auth.py                # Bearer Token 认证
│   └── config.py              # 服务端配置
├── docs/
│   └── remote-training.md     # [新] 远端训练使用文档
├── scripts/
│   └── start_training_server.sh  # [新] 一键启动远端服务
└── requirements-server.txt    # [新] 远端服务依赖
```

**关键原则**：
- 远端服务 `metaclaw_training_server/` 也放在同一仓库，方便统一管理和推广
- 用户只需 clone 一个仓库，在 GPU 服务器上跑 server，在本地配置 client
- README 中新增 "Remote GPU Training" 章节，作为亮点功能宣传

## 10. 实现顺序

### Phase 1：远端服务（GPU 服务器）
1. 搭建 FastAPI 服务框架 + 认证中间件
2. 实现推理引擎（vLLM）— `/v1/sample`
3. 实现训练引擎（PEFT/LoRA）— `/v1/lora/create`, `/v1/train/*`
4. 实现权重热切换 — `/v1/weights/save`
5. 实现 Checkpoint 管理 — `/v1/checkpoint/*`
6. 端到端测试

### Phase 2：MetaClaw 客户端改动
1. `config.py` + `config_store.py` 新增配置字段
2. `remote_backend/` 模块：实现所有 Tinker 接口的 HTTP 客户端
3. `sdk_backend.py` 扩展 remote 后端选择
4. `trainer.py` 改用 backend 抽象（替代直接 import tinker）
5. `data_formatter.py` 改用可注入的 SDK 模块
6. `api_server.py` 改用可注入的 SDK 模块
7. 集成测试

### Phase 3：文档 & 推广
1. `docs/remote-training.md` 使用文档
2. README 新增 Remote GPU Training 章节
3. 一键部署脚本
4. 提交 PR 到上游（可选）

### Phase 4：优化
1. 权重热切换优化（避免重载整个 vLLM）
2. 数据传输优化（如果 JSON+base64 成为瓶颈）
3. 多会话支持（多个 agent 同时训练）
4. 监控 dashboard

## 10. 风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| vLLM 重载慢（每步训练后） | 训练循环延迟 | 用 LoRA adapter 热切换替代全量重载 |
| 公网传输大 tensor | 延迟增加 | SSH 隧道 / 压缩 / 小 batch |
| GPU OOM | 训练失败 | gradient checkpointing + 合理分卡 |
| MetaClaw 版本更新覆盖改动 | 功能丢失 | 改动最小化 + 提交 PR 到上游 |
| loss_fn 实现与 Tinker 不一致 | 训练效果差异 | 参考 tinker-cookbook 源码对齐 |
