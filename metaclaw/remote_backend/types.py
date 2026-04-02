"""Tinker-compatible data types for the remote backend.

These classes mirror the Tinker SDK interfaces so that MetaClaw's
trainer.py, data_formatter.py, and api_server.py can use them
as drop-in replacements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AdamParams:
    """Mirror of tinker.AdamParams."""

    learning_rate: float = 1e-4


@dataclass
class TensorData:
    """Mirror of tinker.TensorData — wraps a tensor for serialization."""

    _data: Any = None  # torch.Tensor internally

    @classmethod
    def from_torch(cls, tensor) -> "TensorData":
        td = cls()
        td._data = tensor
        return td

    def to_list(self) -> list:
        """Convert to plain Python list for JSON serialization."""
        if self._data is None:
            return []
        return self._data.tolist()

    @property
    def dtype_str(self) -> str:
        if self._data is None:
            return "float32"
        import torch

        dtype_map = {
            torch.float32: "float32",
            torch.float16: "float16",
            torch.bfloat16: "bfloat16",
            torch.int64: "int64",
            torch.int32: "int32",
        }
        return dtype_map.get(self._data.dtype, "float32")


@dataclass
class ModelInput:
    """Mirror of tinker.ModelInput."""

    _tokens: list[int] = field(default_factory=list)
    chunks: Optional[list] = None  # for EncodedTextChunk compatibility

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        mi = cls()
        mi._tokens = list(tokens)
        return mi

    def to_token_list(self) -> list[int]:
        """Extract token ids."""
        if self._tokens:
            return self._tokens
        # Fallback: extract from chunks
        if self.chunks:
            tokens = []
            for chunk in self.chunks:
                if hasattr(chunk, "tokens"):
                    tokens.extend(chunk.tokens)
                elif isinstance(chunk, dict):
                    tokens.extend(chunk.get("tokens", []))
            return tokens
        return []


@dataclass
class EncodedTextChunk:
    """Mirror of tinker.EncodedTextChunk."""

    tokens: list[int] = field(default_factory=list)
    type: str = "encoded_text"


@dataclass
class SamplingParams:
    """Mirror of tinker.SamplingParams."""

    temperature: float = 0.7
    max_tokens: int = 2048
    top_k: int = 50
    top_p: float = 0.95
    stop: Optional[list[str]] = None


@dataclass
class Datum:
    """Mirror of tinker.Datum."""

    model_input: Optional[ModelInput] = None
    loss_fn_inputs: dict[str, TensorData] = field(default_factory=dict)


@dataclass
class Sequence:
    """A single generated sequence from sampling."""

    tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    stop_reason: str = "stop"


@dataclass
class SampleResponse:
    """Response from sample_async."""

    sequences: list[Sequence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleResponse":
        seqs = []
        for s in data.get("sequences", []):
            seqs.append(
                Sequence(
                    tokens=s.get("tokens", []),
                    logprobs=s.get("logprobs", []),
                    stop_reason=s.get("stop_reason", "stop"),
                )
            )
        return cls(sequences=seqs)


@dataclass
class SaveResult:
    """Result of save_state_async."""

    path: str = ""
