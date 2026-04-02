"""PEFT/LoRA training engine.

Implements the same loss functions as Tinker:
  - importance_sampling (GRPO-style)
  - ppo (clipped PPO)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ServerConfig
from .serialization import DeserializedDatum

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class TrainingSession:
    """Manages a single LoRA training session.

    Lifecycle:
        create() → forward_backward() → optim_step() → save_weights() → repeat
    """

    def __init__(
        self,
        base_model: str,
        rank: int,
        config: ServerConfig,
    ):
        self.base_model_name = base_model
        self.rank = rank
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.step_count = 0
        self._device = torch.device(
            f"cuda:{config.training_devices[0]}"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._accumulated_loss = 0.0
        self._accumulated_count = 0

    def initialize(self):
        """Load model + create LoRA adapter + optimizer."""
        logger.info(
            "[TrainingEngine] Loading %s on %s ...", self.base_model_name, self._device
        )

        dtype = _DTYPE_MAP.get(self.config.torch_dtype, torch.bfloat16)

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=dtype,
            device_map={"": self._device},
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=int(self.rank * self.config.lora_alpha_ratio),
            lora_dropout=0.0,
            target_modules=self.config.lora_target_modules,
        )
        self.model = get_peft_model(self.model, lora_config)

        if self.config.gradient_checkpointing:
            # Enable gradient checkpointing for memory efficiency
            self.model.gradient_checkpointing_enable()
            # Ensure input embeddings require gradients (needed for gradient checkpointing + LoRA)
            if hasattr(self.model, "enable_input_embeddings_gradients"):
                self.model.enable_input_embeddings_gradients()
            else:
                for param in self.model.get_input_embeddings().parameters():
                    param.requires_grad = True

        # Optimizer (AdamW, lr set per optim_step call)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-4,  # placeholder, updated in optim_step
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "[TrainingEngine] Model ready: %s | LoRA rank=%d | "
            "trainable=%.2fM / total=%.2fM (%.2f%%)",
            self.base_model_name,
            self.rank,
            trainable / 1e6,
            total / 1e6,
            100.0 * trainable / total,
        )

    def forward_backward(
        self,
        datums: list[DeserializedDatum],
        loss_fn: str = "importance_sampling",
    ) -> dict:
        """Forward + backward pass on a batch of datums.

        Does NOT call optimizer.step() — that's a separate call.

        Returns metrics dict with loss info.
        """
        if self.model is None:
            raise RuntimeError("Training session not initialized")

        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        for datum in datums:
            loss = self._compute_loss(datum, loss_fn)
            loss.backward()
            total_loss += loss.item()
            # Count response tokens (where advantages != 0)
            resp_count = (datum.advantages != 0).sum().item()
            total_tokens += max(resp_count, 1)

        self._accumulated_loss += total_loss
        self._accumulated_count += len(datums)

        return {
            "loss": total_loss / max(len(datums), 1),
            "total_tokens": total_tokens,
            "num_datums": len(datums),
        }

    def _compute_loss(
        self, datum: DeserializedDatum, loss_fn: str
    ) -> torch.Tensor:
        """Compute RL loss for a single datum."""
        input_ids = torch.tensor(
            [datum.model_input_tokens], dtype=torch.long, device=self._device
        )

        # Forward pass
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[0]  # (T, vocab_size)

        # Compute new log probs at target positions
        new_log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = new_log_probs.gather(
            1, datum.target_tokens.unsqueeze(1)
        ).squeeze(1)  # (T,)

        # Old log probs from datum
        old_log_probs = datum.logprobs  # (T,)

        # Advantages (0 for prompt positions, nonzero for response)
        advantages = datum.advantages  # (T,)

        # Response mask: positions where advantages are nonzero
        mask = (advantages != 0).float()
        num_response = mask.sum().clamp(min=1.0)

        if loss_fn == "importance_sampling":
            # GRPO-style: ratio * advantage
            ratio = torch.exp(token_log_probs - old_log_probs)
            # Clamp ratio for numerical stability
            ratio = ratio.clamp(0.01, 100.0)
            loss = -(ratio * advantages).sum() / num_response

        elif loss_fn == "ppo":
            # Clipped PPO
            ratio = torch.exp(token_log_probs - old_log_probs)
            ratio = ratio.clamp(0.01, 100.0)
            eps = self.config.ppo_clip_eps
            clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
            surr1 = ratio * advantages
            surr2 = clipped * advantages
            loss = -torch.min(surr1, surr2).sum() / num_response

        elif loss_fn == "cispo":
            # Conservative importance sampling: only clip upward ratios
            ratio = torch.exp(token_log_probs - old_log_probs)
            ratio = ratio.clamp(0.01, 100.0)
            eps = self.config.ppo_clip_eps
            # For positive advantages, clip upper bound; for negative, clip lower
            clipped = torch.where(
                advantages >= 0,
                torch.clamp(ratio, max=1.0 + eps),
                torch.clamp(ratio, min=1.0 - eps),
            )
            loss = -(clipped * advantages).sum() / num_response

        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        return loss

    def optim_step(self, learning_rate: float) -> dict:
        """Run optimizer step and zero gradients."""
        if self.optimizer is None:
            raise RuntimeError("Training session not initialized")

        # Update learning rate
        for pg in self.optimizer.param_groups:
            pg["lr"] = learning_rate

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        ).item()

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step_count += 1

        avg_loss = (
            self._accumulated_loss / max(self._accumulated_count, 1)
        )
        self._accumulated_loss = 0.0
        self._accumulated_count = 0

        return {
            "step": self.step_count,
            "grad_norm": grad_norm,
            "avg_loss": avg_loss,
            "learning_rate": learning_rate,
        }

    def save_adapter(self, name: str = "") -> str:
        """Save LoRA adapter weights to disk. Returns the save path."""
        save_name = name or f"adapter_step_{self.step_count:04d}"
        save_path = os.path.join(self.config.checkpoint_dir, "adapters", save_name)
        os.makedirs(save_path, exist_ok=True)

        self.model.save_pretrained(save_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)

        logger.info("[TrainingEngine] Adapter saved: %s", save_path)
        return save_path

    def save_checkpoint(self, name: str) -> str:
        """Save full training state (adapter + optimizer + step count)."""
        ckpt_path = os.path.join(self.config.checkpoint_dir, "checkpoints", name)
        os.makedirs(ckpt_path, exist_ok=True)

        # Save adapter
        self.model.save_pretrained(ckpt_path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(ckpt_path)

        # Save optimizer state + step count
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": self.step_count,
            },
            os.path.join(ckpt_path, "training_state.pt"),
        )

        logger.info("[TrainingEngine] Checkpoint saved: %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str):
        """Load training state from checkpoint."""
        from peft import PeftModel

        # Load adapter weights
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(path, adapter_name="default")
        else:
            logger.warning(
                "[TrainingEngine] load_adapter not available, "
                "attempting manual weight load"
            )
            state_dict_path = os.path.join(path, "adapter_model.safetensors")
            if not os.path.exists(state_dict_path):
                state_dict_path = os.path.join(path, "adapter_model.bin")
            if os.path.exists(state_dict_path):
                from safetensors.torch import load_file

                state_dict = load_file(state_dict_path)
                self.model.load_state_dict(state_dict, strict=False)

        # Load optimizer state + step count
        training_state_path = os.path.join(path, "training_state.pt")
        if os.path.exists(training_state_path):
            state = torch.load(
                training_state_path, map_location=self._device, weights_only=True
            )
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.step_count = state["step_count"]
            logger.info(
                "[TrainingEngine] Checkpoint loaded: %s (step %d)",
                path,
                self.step_count,
            )
        else:
            logger.warning(
                "[TrainingEngine] No training_state.pt found in %s", path
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: list[int],
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        stop_token_ids: Optional[list[int]] = None,
    ) -> dict:
        """Generate tokens using the current model (fallback HF inference).

        Returns dict with keys: tokens, logprobs, stop_reason.
        """
        if self.model is None:
            raise RuntimeError("Training session not initialized")

        self.model.eval()

        input_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=self._device
        )

        # Determine stop tokens
        eos_ids = set()
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)
        if stop_token_ids:
            eos_ids.update(stop_token_ids)

        generated_tokens: list[int] = []
        generated_logprobs: list[float] = []
        stop_reason = "stop"

        current_ids = input_tensor

        for _ in range(max_tokens):
            outputs = self.model(input_ids=current_ids)
            next_logits = outputs.logits[0, -1, :]  # (vocab,)

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature
            else:
                # Greedy
                pass

            # Log probabilities
            log_probs = torch.log_softmax(next_logits, dim=-1)

            if temperature > 0:
                # Top-k filtering
                if top_k > 0:
                    top_k_vals, top_k_idx = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    filter_mask = torch.full_like(next_logits, float("-inf"))
                    filter_mask.scatter_(0, top_k_idx, top_k_vals)
                    next_logits = filter_mask

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                    cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cum_probs > top_p
                    remove_mask[1:] = remove_mask[:-1].clone()
                    remove_mask[0] = False
                    sorted_logits[remove_mask] = float("-inf")
                    next_logits = sorted_logits.scatter(0, sorted_idx, sorted_logits)

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
            else:
                next_token = next_logits.argmax().item()

            token_logprob = log_probs[next_token].item()

            generated_tokens.append(next_token)
            generated_logprobs.append(token_logprob)

            if next_token in eos_ids:
                stop_reason = "stop"
                break

            # Append to input for next iteration
            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token]], device=self._device)],
                dim=1,
            )
        else:
            stop_reason = "length"

        return {
            "tokens": generated_tokens,
            "logprobs": generated_logprobs,
            "stop_reason": stop_reason,
        }
