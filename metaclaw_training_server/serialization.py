"""Datum (de)serialization for remote training.

Wire format (JSON-friendly, sufficient for typical sequence lengths < 64K):
  {
      "model_input_tokens": [1, 2, 3, ...],
      "target_tokens": [2, 3, 4, ...],
      "logprobs": [0.0, 0.0, -1.5, ...],
      "advantages": [0.0, 0.0, 0.3, ...]
  }

For large batches, numpy binary + base64 can be used (future optimization).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DeserializedDatum:
    """Server-side representation of a training datum."""

    model_input_tokens: list[int]
    target_tokens: torch.Tensor  # (T,) int64
    logprobs: torch.Tensor  # (T,) float32
    advantages: torch.Tensor  # (T,) float32


def deserialize_datum(raw: dict[str, Any], device: torch.device) -> DeserializedDatum:
    """Convert wire-format dict to DeserializedDatum with tensors on device."""
    return DeserializedDatum(
        model_input_tokens=raw["model_input_tokens"],
        target_tokens=torch.tensor(
            raw["target_tokens"], dtype=torch.long, device=device
        ),
        logprobs=torch.tensor(
            raw["logprobs"], dtype=torch.float32, device=device
        ),
        advantages=torch.tensor(
            raw["advantages"], dtype=torch.float32, device=device
        ),
    )


def deserialize_batch(
    raw_datums: list[dict[str, Any]], device: torch.device
) -> list[DeserializedDatum]:
    """Deserialize a batch of datums."""
    return [deserialize_datum(d, device) for d in raw_datums]
