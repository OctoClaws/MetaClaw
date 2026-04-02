"""Server-side configuration for the remote training service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    # ------------------------------------------------------------------ #
    # Server                                                               #
    # ------------------------------------------------------------------ #
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""  # Bearer token; empty = no auth

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    base_model: str = ""  # set at runtime via /v1/lora/create
    lora_rank: int = 32
    lora_alpha_ratio: float = 2.0  # alpha = rank * ratio
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    torch_dtype: str = "bfloat16"  # "float16" | "bfloat16" | "float32"

    # ------------------------------------------------------------------ #
    # GPU allocation                                                       #
    # ------------------------------------------------------------------ #
    training_devices: list[int] = field(default_factory=lambda: [0])
    inference_devices: list[int] = field(default_factory=lambda: [1])

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    ppo_clip_eps: float = 0.2  # PPO clipping epsilon

    # ------------------------------------------------------------------ #
    # Inference (vLLM)                                                     #
    # ------------------------------------------------------------------ #
    inference_backend: str = "vllm"  # "vllm" | "hf" (huggingface generate)
    vllm_max_model_len: int = 20000
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_lora_rank: int = 64

    # ------------------------------------------------------------------ #
    # Checkpoint                                                           #
    # ------------------------------------------------------------------ #
    checkpoint_dir: str = "/mnt/data/metaclaw_checkpoints"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        cfg = cls()
        cfg.api_key = os.environ.get("METACLAW_API_KEY", cfg.api_key)
        cfg.port = int(os.environ.get("METACLAW_PORT", cfg.port))
        cfg.checkpoint_dir = os.environ.get(
            "METACLAW_CHECKPOINT_DIR", cfg.checkpoint_dir
        )
        return cfg
