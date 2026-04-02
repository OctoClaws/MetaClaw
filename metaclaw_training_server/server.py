"""FastAPI server — MetaClaw Remote Training Service.

Exposes Tinker-compatible APIs backed by PEFT/LoRA + vLLM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .auth import BearerAuthMiddleware
from .config import ServerConfig
from .inference_engine import InferenceEngine
from .serialization import deserialize_batch
from .training_engine import TrainingSession

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Request / Response models                                                    #
# --------------------------------------------------------------------------- #


class LoRACreateRequest(BaseModel):
    base_model: str
    rank: int = 32


class LoRACreateResponse(BaseModel):
    session_id: str
    status: str
    model: str
    rank: int
    trainable_params: str = ""


class ForwardBackwardRequest(BaseModel):
    session_id: str
    datums: list[dict[str, Any]]
    loss_fn: str = "importance_sampling"


class ForwardBackwardResponse(BaseModel):
    status: str
    loss: float
    total_tokens: int
    num_datums: int


class OptimStepRequest(BaseModel):
    session_id: str
    learning_rate: float = 1e-4


class OptimStepResponse(BaseModel):
    status: str
    step: int
    grad_norm: float
    avg_loss: float
    learning_rate: float


class WeightsSaveRequest(BaseModel):
    session_id: str
    name: str = ""


class WeightsSaveResponse(BaseModel):
    status: str
    adapter_path: str
    step: int


class CheckpointSaveRequest(BaseModel):
    session_id: str
    name: str


class CheckpointSaveResponse(BaseModel):
    status: str
    path: str


class CheckpointLoadRequest(BaseModel):
    session_id: str
    path: str


class SampleRequest(BaseModel):
    tokens: list[int]
    num_samples: int = 1
    temperature: float = 0.7
    max_tokens: int = 2048
    top_k: int = 50
    top_p: float = 0.95
    stop: Optional[list[str]] = None


class HealthResponse(BaseModel):
    status: str
    gpu_count: int
    model: str = ""
    step: int = 0
    inference_backend: str = ""
    version: str = "0.1.0"


# --------------------------------------------------------------------------- #
# Application                                                                  #
# --------------------------------------------------------------------------- #


def _truncate_messages(
    messages: list[dict], max_tokens: int, tokenizer
) -> list[dict]:
    """Truncate messages to fit within max_tokens.

    Strategy: keep system prompt short, keep last N user/assistant turns.
    System prompt is truncated first (it's usually the longest).
    """
    if not messages or tokenizer is None:
        return messages

    # Estimate total tokens
    def _extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content) if content else ""

    total_text = "".join(_extract_text(m.get("content")) for m in messages)
    estimated_tokens = len(tokenizer.encode(total_text, add_special_tokens=False))

    if estimated_tokens <= max_tokens:
        return messages

    logger.warning(
        "[Server] Prompt too long (%d tokens > %d max), truncating",
        estimated_tokens, max_tokens,
    )

    result = []
    # Truncate system prompt to ~1000 chars max
    for msg in messages:
        if msg.get("role") == "system":
            content = _extract_text(msg.get("content"))
            if len(content) > 1000:
                content = content[:500] + "\n...(truncated)...\n" + content[-500:]
            result.append({"role": "system", "content": content})
        else:
            result.append(dict(msg))

    # If still too long, keep only last few turns
    if len(result) > 5:
        # Keep system + last 4 messages
        system_msgs = [m for m in result if m.get("role") == "system"]
        other_msgs = [m for m in result if m.get("role") != "system"]
        result = system_msgs + other_msgs[-4:]

    return result


def create_app(config: Optional[ServerConfig] = None) -> FastAPI:
    """Create the FastAPI application."""
    if config is None:
        config = ServerConfig.from_env()

    app = FastAPI(
        title="MetaClaw Training Server",
        version="0.1.0",
        description="Self-hosted GPU training backend for MetaClaw",
    )

    # Auth middleware
    if config.api_key:
        app.add_middleware(BearerAuthMiddleware, api_key=config.api_key)

    # State
    app.state.config = config
    app.state.session: Optional[TrainingSession] = None
    app.state.inference: Optional[InferenceEngine] = None
    app.state.executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------ #
    # Helper: run sync training ops in thread pool                        #
    # ------------------------------------------------------------------ #

    async def _run_in_thread(fn, *args, **kwargs):
        import functools
        loop = asyncio.get_event_loop()
        func = functools.partial(fn, *args, **kwargs)
        return await loop.run_in_executor(app.state.executor, func)

    # ------------------------------------------------------------------ #
    # Endpoints                                                            #
    # ------------------------------------------------------------------ #

    @app.get("/v1/health", response_model=HealthResponse)
    async def health():
        session = app.state.session
        inference = app.state.inference
        return HealthResponse(
            status="healthy",
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            model=session.base_model_name if session else "",
            step=session.step_count if session else 0,
            inference_backend=(
                "vllm" if inference and inference.available else "hf"
            ),
        )

    @app.post("/v1/lora/create", response_model=LoRACreateResponse)
    async def lora_create(req: LoRACreateRequest):
        """Create a new LoRA training session."""
        cfg = app.state.config
        session_id = str(uuid.uuid4())[:8]

        logger.info(
            "[Server] Creating LoRA session: model=%s rank=%d",
            req.base_model,
            req.rank,
        )

        # Update config with request params
        cfg.base_model = req.base_model
        cfg.lora_rank = req.rank

        # Create training session
        session = TrainingSession(
            base_model=req.base_model,
            rank=req.rank,
            config=cfg,
        )
        await _run_in_thread(session.initialize)

        app.state.session = session

        # Initialize inference engine (best-effort, run in thread to avoid blocking)
        if cfg.inference_backend == "vllm":
            inference = InferenceEngine(cfg)
            try:
                await _run_in_thread(inference.initialize_sync, req.base_model)
                app.state.inference = inference
            except Exception as e:
                logger.warning(
                    "[Server] vLLM init failed, using HF fallback: %s", e
                )
                app.state.inference = None

        trainable, total = session.model.get_nb_trainable_parameters()
        return LoRACreateResponse(
            session_id=session_id,
            status="ready",
            model=req.base_model,
            rank=req.rank,
            trainable_params=f"{trainable / 1e6:.2f}M / {total / 1e6:.2f}M",
        )

    @app.post("/v1/train/forward_backward", response_model=ForwardBackwardResponse)
    async def forward_backward(req: ForwardBackwardRequest):
        """Forward + backward pass on a batch of datums."""
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")

        device = session._device
        datums = deserialize_batch(req.datums, device)

        result = await _run_in_thread(
            session.forward_backward, datums, req.loss_fn
        )

        return ForwardBackwardResponse(
            status="ok",
            loss=result["loss"],
            total_tokens=result["total_tokens"],
            num_datums=result["num_datums"],
        )

    @app.post("/v1/train/optim_step", response_model=OptimStepResponse)
    async def optim_step(req: OptimStepRequest):
        """Optimizer step + zero gradients."""
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")

        result = await _run_in_thread(session.optim_step, req.learning_rate)

        return OptimStepResponse(
            status="ok",
            step=result["step"],
            grad_norm=result["grad_norm"],
            avg_loss=result["avg_loss"],
            learning_rate=result["learning_rate"],
        )

    @app.post("/v1/weights/save", response_model=WeightsSaveResponse)
    async def weights_save(req: WeightsSaveRequest):
        """Save LoRA adapter weights and update inference engine."""
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")

        adapter_path = await _run_in_thread(session.save_adapter, req.name)

        # Hot-swap adapter in inference engine
        inference = app.state.inference
        if inference and inference.available:
            inference.update_adapter(adapter_path)

        return WeightsSaveResponse(
            status="ok",
            adapter_path=adapter_path,
            step=session.step_count,
        )

    @app.post("/v1/checkpoint/save", response_model=CheckpointSaveResponse)
    async def checkpoint_save(req: CheckpointSaveRequest):
        """Save full training checkpoint."""
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")

        path = await _run_in_thread(session.save_checkpoint, req.name)
        return CheckpointSaveResponse(status="ok", path=path)

    @app.post("/v1/checkpoint/load")
    async def checkpoint_load(req: CheckpointLoadRequest):
        """Load training state from checkpoint."""
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")

        await _run_in_thread(session.load_checkpoint, req.path)
        return {"status": "ok", "step": session.step_count}

    @app.post("/v1/sample")
    async def sample(req: SampleRequest):
        """Generate tokens using inference engine or training model."""
        inference = app.state.inference
        session = app.state.session

        # Try vLLM first
        if inference and inference.available:
            try:
                result = await _run_in_thread(
                    inference.sample,
                    req.tokens,
                    req.num_samples,
                    req.temperature,
                    req.max_tokens,
                    req.top_k,
                    req.top_p,
                    req.stop,
                )
                return result
            except Exception as e:
                logger.warning(
                    "[Server] vLLM inference failed, falling back to HF: %s", e
                )

        # Fallback to HF generate from training session
        if session is not None:
            # Encode stop strings to token ids
            stop_ids = None
            if req.stop and session.tokenizer:
                stop_ids = []
                for s in req.stop:
                    ids = session.tokenizer.encode(s, add_special_tokens=False)
                    if ids:
                        stop_ids.extend(ids)

            result = await _run_in_thread(
                session.generate,
                req.tokens,
                req.max_tokens,
                req.temperature,
                req.top_k,
                req.top_p,
                stop_ids,
            )
            return {"sequences": [result]}

        raise HTTPException(
            status_code=503, detail="No inference engine available"
        )

    # ------------------------------------------------------------------ #
    # OpenAI-compatible chat completions (text in → text out)             #
    # ------------------------------------------------------------------ #

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """OpenAI-compatible chat completions endpoint.

        Handles tokenization internally so the client doesn't need
        a local tokenizer. Used by MetaClaw's remote backend.
        """
        session = app.state.session
        if session is None:
            raise HTTPException(status_code=503, detail="No active training session")
        if session.tokenizer is None:
            raise HTTPException(status_code=503, detail="No tokenizer available")

        body = await request.json()
        messages = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens") or 4096)
        # Ensure enough room for thinking + response (Qwen3 uses <think> blocks)
        if max_tokens < 1024:
            max_tokens = 1024
        top_k = int(body.get("top_k", 50))
        top_p = float(body.get("top_p", 0.95))
        stop = body.get("stop")
        model_name = body.get("model", "metaclaw")

        # Truncate overly long messages to fit model's effective context
        # Small models (< 8B) degrade significantly with very long prompts
        server_cfg = app.state.config
        max_prompt_tokens = min(
            getattr(session, '_max_prompt_tokens', 4096),
            server_cfg.vllm_max_model_len - max_tokens - 512  # leave room for response
        )

        # Truncate system prompt if total is too long
        truncated_messages = _truncate_messages(messages, max_prompt_tokens, session.tokenizer)

        # Apply chat template → token ids
        try:
            prompt_text = session.tokenizer.apply_chat_template(
                truncated_messages,
                tools=body.get("tools"),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = session.tokenizer.encode(prompt_text, add_special_tokens=False)
        except Exception as e:
            logger.error("[Server] chat template failed: %s", e)
            raise HTTPException(status_code=400, detail=f"Chat template error: {e}")

        # Encode stop strings to token ids
        stop_ids = None
        if stop:
            stop_ids = []
            for s in (stop if isinstance(stop, list) else [stop]):
                ids = session.tokenizer.encode(s, add_special_tokens=False)
                if ids:
                    stop_ids.extend(ids)

        # Generate
        result = await _run_in_thread(
            session.generate,
            prompt_ids,
            max_tokens,
            temperature,
            top_k,
            top_p,
            stop_ids,
        )

        # Decode response tokens → text
        response_text = session.tokenizer.decode(
            result["tokens"], skip_special_tokens=True
        )

        # Separate <think>...</think> reasoning from content
        reasoning_content = ""
        content = response_text
        import re
        think_match = re.search(r"<think>(.*?)</think>(.*)", response_text, re.DOTALL)
        if think_match:
            reasoning_content = think_match.group(1).strip()
            content = think_match.group(2).strip()
        elif response_text.startswith("<think>"):
            # Incomplete think block (model hit max_tokens mid-thinking)
            reasoning_content = response_text.replace("<think>", "").strip()
            content = ""

        # Build logprobs in OpenAI format
        lp_content = [
            {"token": "", "logprob": float(lp), "top_logprobs": []}
            for lp in result.get("logprobs", [])
        ]

        assistant_message = {"role": "assistant", "content": content}
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content

        return {
            "id": f"chatcmpl-remote-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": assistant_message,
                "finish_reason": result.get("stop_reason", "stop"),
                "logprobs": {"content": lp_content} if lp_content else None,
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(result["tokens"]),
                "total_tokens": len(prompt_ids) + len(result["tokens"]),
            },
        }

    return app


# --------------------------------------------------------------------------- #
# CLI entry point                                                              #
# --------------------------------------------------------------------------- #


def main():
    """Start the training server."""
    import argparse

    parser = argparse.ArgumentParser(description="MetaClaw Training Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="", help="Bearer token for auth")
    parser.add_argument(
        "--checkpoint-dir",
        default="/mnt/data/metaclaw_checkpoints",
    )
    parser.add_argument(
        "--training-devices",
        default="0",
        help="Comma-separated GPU ids for training",
    )
    parser.add_argument(
        "--inference-devices",
        default="1",
        help="Comma-separated GPU ids for inference",
    )
    parser.add_argument(
        "--inference-backend",
        default="vllm",
        choices=["vllm", "hf"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    config = ServerConfig(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        checkpoint_dir=args.checkpoint_dir,
        training_devices=[int(d) for d in args.training_devices.split(",")],
        inference_devices=[int(d) for d in args.inference_devices.split(",")],
        inference_backend=args.inference_backend,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
