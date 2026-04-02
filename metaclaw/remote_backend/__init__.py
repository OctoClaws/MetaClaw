"""Remote training backend for MetaClaw.

Drop-in replacement for the Tinker SDK, routing all training and
inference operations to a self-hosted GPU server over HTTP.

Usage in MetaClaw (via sdk_backend.py):
    backend = resolve_sdk_backend(config)  # returns remote backend
    sdk = backend.module
    service_client = sdk.ServiceClient(base_url=..., api_key=...)
"""

from .client import SamplingClient, ServiceClient, TrainingClient
from .types import (
    AdamParams,
    Datum,
    EncodedTextChunk,
    ModelInput,
    SampleResponse,
    SamplingParams,
    SaveResult,
    Sequence,
    TensorData,
)

__all__ = [
    "ServiceClient",
    "TrainingClient",
    "SamplingClient",
    "AdamParams",
    "Datum",
    "EncodedTextChunk",
    "ModelInput",
    "SampleResponse",
    "SamplingParams",
    "SaveResult",
    "Sequence",
    "TensorData",
]
