"""Datum serialization for sending to remote training server.

Converts Datum objects (containing ModelInput + TensorData) into
JSON-serializable dicts for HTTP transport.
"""

from __future__ import annotations

from typing import Any

from .types import Datum


def serialize_datum(datum: Datum) -> dict[str, Any]:
    """Convert a Datum to a wire-format dict.

    Wire format:
        {
            "model_input_tokens": [1, 2, 3, ...],
            "target_tokens": [2, 3, 4, ...],
            "logprobs": [0.0, 0.0, -1.5, ...],
            "advantages": [0.0, 0.0, 0.3, ...]
        }
    """
    result: dict[str, Any] = {}

    # Model input tokens
    if datum.model_input is not None:
        result["model_input_tokens"] = datum.model_input.to_token_list()
    else:
        result["model_input_tokens"] = []

    # Loss function inputs (TensorData → list)
    for key, td in datum.loss_fn_inputs.items():
        result[key] = td.to_list()

    return result


def serialize_batch(datums: list[Datum]) -> list[dict[str, Any]]:
    """Serialize a list of Datums for HTTP transport."""
    return [serialize_datum(d) for d in datums]
