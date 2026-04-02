"""vLLM-based inference engine with LoRA hot-swap support.

Provides fast inference using vLLM, with the ability to dynamically
swap LoRA adapters after each training step without restarting.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .config import ServerConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """vLLM inference engine with dynamic LoRA adapter support.

    Usage:
        engine = InferenceEngine(config)
        await engine.initialize("Qwen/Qwen3-4B")
        result = engine.sample(token_ids, params)
        engine.update_adapter("/path/to/new/adapter")  # hot-swap
    """

    def __init__(self, config: ServerConfig):
        self.config = config
        self._llm = None
        self._adapter_path: Optional[str] = None
        self._adapter_version: int = 0
        self._base_model: str = ""
        self._tokenizer = None

    def initialize_sync(self, base_model: str):
        """Initialize vLLM with base model and LoRA support.

        This is a synchronous method (vLLM init is blocking).
        Call from a thread pool to avoid blocking the event loop.
        """
        self._base_model = base_model

        # Set CUDA_VISIBLE_DEVICES for inference
        devices = ",".join(str(d) for d in self.config.inference_devices)
        logger.info(
            "[InferenceEngine] Initializing vLLM on GPU %s with %s ...",
            devices,
            base_model,
        )

        try:
            from vllm import LLM

            self._llm = LLM(
                model=base_model,
                tensor_parallel_size=len(self.config.inference_devices),
                gpu_memory_utilization=self.config.vllm_gpu_memory_utilization,
                max_model_len=self.config.vllm_max_model_len,
                enable_lora=True,
                max_lora_rank=self.config.vllm_max_lora_rank,
                trust_remote_code=True,
            )

            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )

            logger.info("[InferenceEngine] vLLM ready with LoRA support")

        except ImportError:
            logger.warning(
                "[InferenceEngine] vLLM not available, inference will fall "
                "back to training engine's HF generate"
            )
            self._llm = None

        except Exception as e:
            logger.error(
                "[InferenceEngine] vLLM init failed: %s — falling back to HF",
                e,
                exc_info=True,
            )
            self._llm = None

    @property
    def available(self) -> bool:
        return self._llm is not None

    def update_adapter(self, adapter_path: str):
        """Register a new LoRA adapter version for subsequent requests."""
        self._adapter_version += 1
        self._adapter_path = adapter_path
        logger.info(
            "[InferenceEngine] Adapter updated: v%d → %s",
            self._adapter_version,
            adapter_path,
        )

    def sample(
        self,
        token_ids: list[int],
        num_samples: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_k: int = 50,
        top_p: float = 0.95,
        stop: Optional[list[str]] = None,
    ) -> dict:
        """Generate tokens with optional LoRA adapter.

        Returns:
            {
                "sequences": [
                    {"tokens": [...], "logprobs": [...], "stop_reason": "..."}
                ]
            }
        """
        if self._llm is None:
            raise RuntimeError("vLLM not initialized")

        from vllm import SamplingParams as VLLMSamplingParams

        params = VLLMSamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            stop=stop or [],
            logprobs=1,  # return per-token logprobs
        )

        # Build LoRA request if adapter is available
        lora_request = None
        if self._adapter_path and os.path.isdir(self._adapter_path):
            try:
                from vllm.lora.request import LoRARequest

                lora_request = LoRARequest(
                    lora_name=f"metaclaw_v{self._adapter_version}",
                    lora_int_id=self._adapter_version,
                    lora_path=self._adapter_path,
                )
            except ImportError:
                logger.warning(
                    "[InferenceEngine] LoRARequest not available in this vLLM version"
                )

        outputs = self._llm.generate(
            prompt_token_ids=[token_ids],
            sampling_params=params,
            lora_request=lora_request,
        )

        sequences = []
        for output in outputs:
            for completion in output.outputs:
                tokens = list(completion.token_ids)

                # Extract logprobs
                lp_list = []
                if completion.logprobs:
                    for lp_dict in completion.logprobs:
                        if lp_dict:
                            # Get the logprob of the chosen token
                            vals = list(lp_dict.values())
                            lp_list.append(
                                vals[0].logprob if vals else 0.0
                            )
                        else:
                            lp_list.append(0.0)

                sequences.append(
                    {
                        "tokens": tokens,
                        "logprobs": lp_list,
                        "stop_reason": completion.finish_reason or "stop",
                    }
                )

        return {"sequences": sequences}

    @property
    def tokenizer(self):
        return self._tokenizer
