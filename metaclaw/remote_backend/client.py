"""HTTP client implementing Tinker-compatible interfaces for remote training.

Provides ServiceClient, TrainingClient, and SamplingClient that
communicate with metaclaw_training_server over HTTP.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import httpx

from .serialization import serialize_batch
from .types import (
    AdamParams,
    ModelInput,
    SampleResponse,
    SamplingParams,
    SaveResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0  # 10 minutes for training ops


class RemoteError(Exception):
    """Error from remote training server."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Remote error {status_code}: {detail}")


class SamplingClient:
    """Tinker-compatible SamplingClient that calls remote server."""

    def __init__(self, base_url: str, api_key: str, timeout: float):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def sample_async(
        self,
        prompt: ModelInput,
        num_samples: int = 1,
        sampling_params: Optional[SamplingParams] = None,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
        **kwargs,
    ) -> SampleResponse:
        """Generate tokens from remote server."""
        sp = sampling_params or SamplingParams()
        payload = {
            "tokens": prompt.to_token_list(),
            "num_samples": num_samples,
            "temperature": sp.temperature,
            "max_tokens": sp.max_tokens,
            "top_k": sp.top_k,
            "top_p": sp.top_p,
            "stop": sp.stop or [],
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/sample",
                json=payload,
                headers=self._headers(),
            )
            if resp.status_code != 200:
                raise RemoteError(resp.status_code, resp.text)
            return SampleResponse.from_dict(resp.json())


class TrainingClient:
    """Tinker-compatible TrainingClient that calls remote server."""

    def __init__(
        self, base_url: str, api_key: str, session_id: str, timeout: float
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._session_id = session_id
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _post(self, path: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}{path}",
                json=payload,
                headers=self._headers(),
            )
            if resp.status_code != 200:
                raise RemoteError(resp.status_code, resp.text)
            return resp.json()

    async def forward_backward_async(self, datums: list, loss_fn: str = "importance_sampling"):
        """Send datums to remote server for forward + backward pass."""
        serialized = serialize_batch(datums)
        result = await self._post("/v1/train/forward_backward", {
            "session_id": self._session_id,
            "datums": serialized,
            "loss_fn": loss_fn,
        })
        logger.info(
            "[RemoteBackend] forward_backward: loss=%.4f datums=%d",
            result.get("loss", 0),
            result.get("num_datums", 0),
        )

    async def optim_step_async(self, params: AdamParams):
        """Send optimizer step command to remote server."""
        result = await self._post("/v1/train/optim_step", {
            "session_id": self._session_id,
            "learning_rate": params.learning_rate,
        })
        logger.info(
            "[RemoteBackend] optim_step: step=%d grad_norm=%.4f",
            result.get("step", 0),
            result.get("grad_norm", 0),
        )

    async def save_weights_and_get_sampling_client_async(
        self, name: str = ""
    ) -> SamplingClient:
        """Save weights on remote server and return a SamplingClient."""
        result = await self._post("/v1/weights/save", {
            "session_id": self._session_id,
            "name": name,
        })
        logger.info(
            "[RemoteBackend] weights saved: %s (step %d)",
            result.get("adapter_path", ""),
            result.get("step", 0),
        )
        return SamplingClient(self._base_url, self._api_key, self._timeout)

    async def save_state_async(self, name: str = "") -> SaveResult:
        """Save full checkpoint on remote server."""
        result = await self._post("/v1/checkpoint/save", {
            "session_id": self._session_id,
            "name": name,
        })
        return SaveResult(path=result.get("path", ""))

    async def load_state_async(self, path: str):
        """Load checkpoint on remote server."""
        await self._post("/v1/checkpoint/load", {
            "session_id": self._session_id,
            "path": path,
        })
        logger.info("[RemoteBackend] checkpoint loaded: %s", path)


class ServiceClient:
    """Tinker-compatible ServiceClient for remote backend.

    Usage:
        client = ServiceClient(base_url="http://gpu-server:8000", api_key="xxx")
        training_client = await client.create_lora_training_client_async(
            base_model="Qwen/Qwen3-4B", rank=32
        )
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._base_url = (
            base_url
            or os.environ.get("REMOTE_TRAINING_URL", "")
        ).rstrip("/")
        self._api_key = api_key or os.environ.get(
            "REMOTE_TRAINING_API_KEY", ""
        )
        self._timeout = timeout

        if not self._base_url:
            raise ValueError(
                "Remote training URL not configured. "
                "Set remote_url in config.yaml or REMOTE_TRAINING_URL env var."
            )

    async def create_lora_training_client_async(
        self, base_model: str, rank: int = 32
    ) -> TrainingClient:
        """Create a LoRA training session on the remote server."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/lora/create",
                json={"base_model": base_model, "rank": rank},
                headers=headers,
            )
            if resp.status_code != 200:
                raise RemoteError(resp.status_code, resp.text)

            data = resp.json()
            session_id = data["session_id"]
            logger.info(
                "[RemoteBackend] Training session created: id=%s model=%s rank=%d params=%s",
                session_id,
                base_model,
                rank,
                data.get("trainable_params", ""),
            )

        return TrainingClient(
            self._base_url, self._api_key, session_id, self._timeout
        )
