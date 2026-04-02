# Remote GPU Training

Train MetaClaw on your own GPU servers instead of Tinker cloud. Keep training data and model weights on your infrastructure.

## Overview

```
MetaClaw (local)  ── HTTP ──►  Training Server (your GPU server)
   config.yaml:                   ├── Training: PEFT/LoRA
   backend: remote                ├── Inference: vLLM or HuggingFace
                                  └── Checkpoints: local filesystem
```

The remote backend is a **drop-in alternative** to Tinker. Switch with one config change — no code modifications needed.

## Quick Start

### 1. Deploy the Training Server

On your GPU server:

```bash
# Clone the repo
git clone https://github.com/OctoClaws/MetaClaw.git
cd MetaClaw

# Install dependencies (most should already be present on GPU servers)
pip install -r metaclaw_training_server/requirements.txt

# Start the server
export METACLAW_API_KEY="your-secret-key"
bash scripts/start_training_server.sh --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/v1/health
# {"status":"healthy","gpu_count":8,"model":"","step":0,...}
```

### 2. Configure MetaClaw

In `~/.metaclaw/config.yaml`:

```yaml
mode: rl

rl:
  enabled: true
  backend: remote
  remote_url: http://your-gpu-server:8000
  remote_api_key: your-secret-key
  model: Qwen/Qwen3-4B    # or local path on GPU server
  lora_rank: 32
  batch_size: 4
```

That's it. MetaClaw will automatically route all training and inference to your server.

### 3. Switch Back to Tinker

```yaml
rl:
  backend: tinker    # or "auto"
  tinker_api_key: tk-xxx
```

## Server Configuration

### Command Line Options

```bash
python3 -m metaclaw_training_server.server \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "your-secret-key" \
  --checkpoint-dir /data/metaclaw_checkpoints \
  --training-devices 0,1 \
  --inference-devices 2,3 \
  --inference-backend vllm \
  --log-level INFO
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `METACLAW_API_KEY` | _(empty)_ | Bearer token for authentication |
| `METACLAW_PORT` | `8000` | Server port |
| `METACLAW_CHECKPOINT_DIR` | `/mnt/data/metaclaw_checkpoints` | Where to save checkpoints |
| `METACLAW_TRAINING_DEVICES` | `0` | Comma-separated GPU ids for training |
| `METACLAW_INFERENCE_DEVICES` | `1` | Comma-separated GPU ids for inference |
| `METACLAW_INFERENCE_BACKEND` | `hf` | `vllm` or `hf` (HuggingFace generate) |

### GPU Allocation

For a typical 8×A100-80GB setup with Qwen3-4B:

| Use | GPUs | VRAM |
|-----|------|------|
| Training (PEFT/LoRA) | GPU 0 | ~10 GB |
| Inference (vLLM) | GPU 1 | ~10 GB |
| Free | GPU 2–7 | Available |

For larger models (e.g. 72B), scale up the device allocation:

```bash
--training-devices 0,1,2,3 --inference-devices 4,5,6,7
```

### Inference Backend

**`hf` (HuggingFace generate)** — Simple, no extra setup. Uses the training model directly for inference. Recommended for getting started.

**`vllm`** — Fast inference with LoRA hot-swap. After each training step, the new LoRA adapter is loaded dynamically without restarting vLLM. Recommended for production.

## API Reference

All endpoints require `Authorization: Bearer <api_key>` header (except `/v1/health`).

### GET /v1/health

Health check. Always open (no auth required).

**Response:**
```json
{
  "status": "healthy",
  "gpu_count": 8,
  "model": "Qwen/Qwen3-4B",
  "step": 42,
  "inference_backend": "vllm",
  "version": "0.1.0"
}
```

### POST /v1/lora/create

Create a new LoRA training session. Loads the base model and initializes the LoRA adapter.

**Request:**
```json
{
  "base_model": "Qwen/Qwen3-4B",
  "rank": 32
}
```

`base_model` can be a HuggingFace model id or a local path on the GPU server.

**Response:**
```json
{
  "session_id": "a1b2c3d4",
  "status": "ready",
  "model": "Qwen/Qwen3-4B",
  "rank": 32,
  "trainable_params": "328.60M / 3890.12M"
}
```

### POST /v1/train/forward_backward

Forward + backward pass on a batch of training datums. Does not update weights — call `optim_step` separately.

**Request:**
```json
{
  "session_id": "a1b2c3d4",
  "datums": [
    {
      "model_input_tokens": [1, 2, 3, ...],
      "target_tokens": [2, 3, 4, ...],
      "logprobs": [0.0, 0.0, -1.5, ...],
      "advantages": [0.0, 0.0, 0.3, ...]
    }
  ],
  "loss_fn": "importance_sampling"
}
```

Supported `loss_fn` values:
- `importance_sampling` — GRPO-style (default)
- `ppo` — Clipped PPO
- `cispo` — Conservative importance sampling

**Response:**
```json
{
  "status": "ok",
  "loss": 0.123,
  "total_tokens": 256,
  "num_datums": 4
}
```

### POST /v1/train/optim_step

Run optimizer step (AdamW) and zero gradients.

**Request:**
```json
{
  "session_id": "a1b2c3d4",
  "learning_rate": 1e-4
}
```

**Response:**
```json
{
  "status": "ok",
  "step": 1,
  "grad_norm": 0.456,
  "avg_loss": 0.123,
  "learning_rate": 0.0001
}
```

### POST /v1/weights/save

Save LoRA adapter weights. If vLLM is the inference backend, the new adapter is hot-swapped automatically.

**Request:**
```json
{
  "session_id": "a1b2c3d4",
  "name": "step_0010"
}
```

**Response:**
```json
{
  "status": "ok",
  "adapter_path": "/data/metaclaw_checkpoints/adapters/step_0010",
  "step": 10
}
```

### POST /v1/checkpoint/save

Save full training state (adapter weights + optimizer state + step count).

**Request:**
```json
{
  "session_id": "a1b2c3d4",
  "name": "checkpoint_v1"
}
```

**Response:**
```json
{
  "status": "ok",
  "path": "/data/metaclaw_checkpoints/checkpoints/checkpoint_v1"
}
```

### POST /v1/checkpoint/load

Resume training from a saved checkpoint.

**Request:**
```json
{
  "session_id": "a1b2c3d4",
  "path": "/data/metaclaw_checkpoints/checkpoints/checkpoint_v1"
}
```

### POST /v1/sample

Generate tokens with the current model (base + LoRA adapter).

**Request:**
```json
{
  "tokens": [1, 2, 3, 4, 5],
  "num_samples": 1,
  "temperature": 0.7,
  "max_tokens": 2048,
  "top_k": 50,
  "top_p": 0.95,
  "stop": ["\n"]
}
```

**Response:**
```json
{
  "sequences": [
    {
      "tokens": [6, 7, 8, 9, 10],
      "logprobs": [-1.2, -0.8, -1.5, -0.3, -2.1],
      "stop_reason": "stop"
    }
  ]
}
```

## Security

### Authentication

Set `METACLAW_API_KEY` on the server. All requests (except `/v1/health`) require:

```
Authorization: Bearer your-secret-key
```

### SSH Tunnel (Recommended for Public Networks)

If your GPU server is on a public network, use an SSH tunnel instead of exposing the port directly:

```bash
# On your local machine
ssh -L 8000:localhost:8000 user@gpu-server -N

# Then configure MetaClaw to use localhost
rl:
  remote_url: http://localhost:8000
```

### HTTPS (Optional)

For production, put the server behind a reverse proxy (nginx/caddy) with TLS:

```
MetaClaw → HTTPS → nginx → HTTP → training server
```

## Troubleshooting

### Connection refused

Check that the server is running and the port is accessible:

```bash
# On GPU server
curl http://localhost:8000/v1/health

# From local machine (if using SSH tunnel)
curl http://localhost:8000/v1/health
```

### Model not found (HuggingFace)

If the GPU server can't access HuggingFace, use a local model path:

```yaml
rl:
  model: /mnt/data/models/Qwen3-4B   # local path on GPU server
```

### GPU out of memory

Reduce model size, LoRA rank, or adjust GPU allocation:

```bash
# Use more GPUs for training
--training-devices 0,1,2,3

# Reduce LoRA rank
rl:
  lora_rank: 16
```

### Slow inference

Switch from HuggingFace generate to vLLM:

```bash
--inference-backend vllm --inference-devices 1,2
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  MetaClaw (local)                      │
│                                                        │
│  ┌─────────────┐    ┌──────────────┐                  │
│  │ trainer.py   │───►│ sdk_backend  │                  │
│  │              │    │ .py          │                  │
│  │ data_        │    │ backend=     │                  │
│  │ formatter.py │    │  remote      │                  │
│  └──────┬───────┘    └──────┬───────┘                  │
│         │                   │                          │
│  ┌──────▼───────────────────▼───────┐                  │
│  │     remote_backend/client.py     │                  │
│  │  ServiceClient / TrainingClient  │                  │
│  │  SamplingClient                  │                  │
│  └──────────────┬───────────────────┘                  │
└─────────────────┼────────────────────────────────────┘
                  │ HTTP (Bearer Token)
┌─────────────────▼────────────────────────────────────┐
│              GPU Server                                │
│                                                        │
│  ┌────────────────────────────────────────┐            │
│  │     metaclaw_training_server           │            │
│  │  ┌──────────────┐  ┌───────────────┐  │            │
│  │  │ training_    │  │ inference_    │  │            │
│  │  │ engine.py    │  │ engine.py     │  │            │
│  │  │ (PEFT/LoRA)  │  │ (vLLM/HF)    │  │            │
│  │  │ GPU 0        │  │ GPU 1        │  │            │
│  │  └──────────────┘  └───────────────┘  │            │
│  └────────────────────────────────────────┘            │
└────────────────────────────────────────────────────────┘
```

## Backend Comparison

| Feature | Tinker | Remote |
|---------|--------|--------|
| Setup | API key only | Deploy server on GPU |
| Data location | Tinker cloud | Your server |
| GPU cost | Pay-per-use | Your hardware |
| Supported models | Tinker catalog | Any HuggingFace model |
| Inference | Tinker sampling | vLLM / HuggingFace |
| LoRA hot-swap | Built-in | vLLM native |
| Checkpoints | Tinker storage | Local filesystem |
| Network | Internet required | LAN / SSH tunnel |
