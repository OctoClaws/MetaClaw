#!/bin/bash
# Start MetaClaw Remote Training Server
#
# Usage:
#   ./start_training_server.sh [--api-key KEY] [--port 8000]
#
# Environment variables:
#   METACLAW_API_KEY       - Bearer token for authentication
#   METACLAW_PORT          - Server port (default: 8000)
#   METACLAW_CHECKPOINT_DIR - Checkpoint directory

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
PORT="${METACLAW_PORT:-8000}"
API_KEY="${METACLAW_API_KEY:-}"
CHECKPOINT_DIR="${METACLAW_CHECKPOINT_DIR:-/mnt/data/metaclaw_checkpoints}"
TRAINING_DEVICES="${METACLAW_TRAINING_DEVICES:-0}"
INFERENCE_DEVICES="${METACLAW_INFERENCE_DEVICES:-1}"
INFERENCE_BACKEND="${METACLAW_INFERENCE_BACKEND:-hf}"  # start with hf, switch to vllm when stable
LOG_LEVEL="${METACLAW_LOG_LEVEL:-INFO}"

# Parse command line args (override env vars)
while [[ $# -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift 2 ;;
        --api-key) API_KEY="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --training-devices) TRAINING_DEVICES="$2"; shift 2 ;;
        --inference-devices) INFERENCE_DEVICES="$2"; shift 2 ;;
        --inference-backend) INFERENCE_BACKEND="$2"; shift 2 ;;
        --log-level) LOG_LEVEL="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo "============================================"
echo "  MetaClaw Remote Training Server"
echo "============================================"
echo "  Port:              $PORT"
echo "  API Key:           ${API_KEY:+(set)}"
echo "  Checkpoint Dir:    $CHECKPOINT_DIR"
echo "  Training GPUs:     $TRAINING_DEVICES"
echo "  Inference GPUs:    $INFERENCE_DEVICES"
echo "  Inference Backend: $INFERENCE_BACKEND"
echo "============================================"

cd "$PROJECT_DIR"

python3 -m metaclaw_training_server.server \
    --host 0.0.0.0 \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --training-devices "$TRAINING_DEVICES" \
    --inference-devices "$INFERENCE_DEVICES" \
    --inference-backend "$INFERENCE_BACKEND" \
    --log-level "$LOG_LEVEL"
