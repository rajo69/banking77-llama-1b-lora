"""
src/2_train_intel.py
====================
Core training script: fine-tunes Llama-3.2-1B-Instruct on the processed
banking77 dataset using standard HuggingFace (transformers + peft + trl).

Unsloth requires a CUDA/XPU GPU and cannot run on Intel Iris Xe iGPU on
Windows. This script uses the standard stack which runs on CPU.

Key features:
  - LoRA adapters via peft (r=16, alpha=16), ~4M trainable params
  - TimeBasedCheckpointCallback: saves a checkpoint every 1 hour
  - SFTTrainer (trl 0.24+) with TensorBoard logging
  - try/finally guarantees the final LoRA adapter is always saved

Run:
    python src/2_train_intel.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

# TrainerCallback imported at module level so TimeBasedCheckpointCallback
# is a proper static subclass (no __bases__ assignment at runtime).
from transformers import TrainerCallback

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(LOG_DIR, "2_train_intel.log"), mode="w"),
    ],
)
logger = logging.getLogger("train_intel")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
DATA_PATH       = "data/processed_banking77.jsonl"
MODEL_NAME      = "unsloth/Llama-3.2-1B-Instruct"
FINAL_MODEL_DIR = "models/final_lora"
CHECKPOINT_DIR  = "checkpoints"
HOURLY_CKPT_DIR = "checkpoints/hourly_backup"

for _dir in [FINAL_MODEL_DIR, HOURLY_CKPT_DIR, CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters (tuned for CPU: small batch, gradient accumulation)
# ---------------------------------------------------------------------------
MAX_SEQ_LENGTH   = 256   # Keep short for CPU speed
LORA_RANK        = 16
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
BATCH_SIZE       = 1     # CPU: 1 sample at a time
GRAD_ACCUM_STEPS = 16    # Effective batch = 16
LEARNING_RATE    = 2e-4
NUM_EPOCHS       = 3
WARMUP_RATIO     = 0.03
WEIGHT_DECAY     = 0.01
CHECKPOINT_EVERY_SECONDS = 3600  # 1 hour


# =============================================================================
# TimeBasedCheckpointCallback: proper TrainerCallback subclass
# =============================================================================

class TimeBasedCheckpointCallback(TrainerCallback):
    """
    Saves a timestamped checkpoint every `interval_seconds` of wall-clock
    training time.

    Must call `callback.attach_trainer(trainer)` after SFTTrainer is
    initialized so that `on_step_end` can call `trainer.save_model()`.
    """

    def __init__(
        self,
        save_dir: str = HOURLY_CKPT_DIR,
        interval_seconds: float = CHECKPOINT_EVERY_SECONDS,
    ) -> None:
        self.save_dir         = save_dir
        self.interval_seconds = interval_seconds
        self._last_save_time  = time.time()
        self._save_count      = 0
        self._trainer         = None

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """Called by HF Trainer after every optimizer step."""
        now     = time.time()
        elapsed = now - self._last_save_time

        if elapsed >= self.interval_seconds:
            self._save_count += 1
            timestamp  = time.strftime("%Y%m%d_%H%M%S")
            ckpt_path  = os.path.join(self.save_dir, f"ckpt_{timestamp}")
            os.makedirs(ckpt_path, exist_ok=True)

            logger.info(
                "[TimeBasedCheckpointCallback] %.0fs elapsed — "
                "saving hourly checkpoint #%d to: %s",
                elapsed, self._save_count, ckpt_path,
            )

            if self._trainer is not None:
                try:
                    self._trainer.save_model(ckpt_path)
                    logger.info(
                        "[TimeBasedCheckpointCallback] Checkpoint #%d saved.",
                        self._save_count,
                    )
                except Exception as e:
                    logger.error(
                        "[TimeBasedCheckpointCallback] Save failed: %s", e
                    )
            else:
                logger.warning(
                    "[TimeBasedCheckpointCallback] Trainer not attached; "
                    "skipping save. Call attach_trainer() first."
                )

            self._last_save_time = now
            # Let HF handle its own save state; we only add the timed save
            control.should_save = False

    def attach_trainer(self, trainer) -> None:
        self._trainer = trainer
        logger.info(
            "[TimeBasedCheckpointCallback] Trainer attached. "
            "Hourly saves -> %s", self.save_dir
        )


# =============================================================================
# Alpaca prompt formatting
# =============================================================================

ALPACA_PROMPT = """\
Below is an instruction that describes a task, paired with an input. \
Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def format_alpaca_prompt(sample: dict) -> dict:
    return {
        "text": ALPACA_PROMPT.format(
            instruction=sample["instruction"],
            input=sample["input"],
            output=sample["output"],
        )
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("Llama-3.2-1B-Instruct Fine-tuning — CPU mode")
    logger.info("Stack: transformers + peft + trl (no Unsloth)")
    logger.info("=" * 60)

    if not os.path.exists(DATA_PATH):
        logger.error(
            "Dataset not found at '%s'. Run `python src/1_prepare_data.py` first.",
            DATA_PATH,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Imports (deferred so logging is active before any heavy import)
    # ------------------------------------------------------------------
    logger.info("Importing torch / transformers / peft / trl...")
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset

    logger.info("Imports OK. torch: %s", torch.__version__)
    logger.info("Running on: %s", "CPU" if not torch.cuda.is_available() else "CUDA")

    # ------------------------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer loaded.")

    # ------------------------------------------------------------------
    # Load base model (float32, no quantization, CPU compatible)
    # ------------------------------------------------------------------
    logger.info("Loading base model: %s (float32, CPU)...", MODEL_NAME)
    logger.info("This may take a few minutes on first run (downloading ~2.5 GB).")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False          # Required for gradient checkpointing
    model.enable_input_require_grads()      # Required for LoRA + gradient checkpointing
    logger.info("Base model loaded.")

    # ------------------------------------------------------------------
    # LoRA configuration: passed directly to SFTTrainer
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    logger.info("LoRA config: r=%d, alpha=%d, dropout=%.2f", LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

    # ------------------------------------------------------------------
    # Load and format dataset
    # ------------------------------------------------------------------
    logger.info("Loading dataset from: %s", DATA_PATH)
    raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    logger.info("Dataset loaded: %d samples.", len(raw_dataset))

    formatted_dataset = raw_dataset.map(
        format_alpaca_prompt,
        remove_columns=raw_dataset.column_names,
        desc="Formatting prompts",
    )
    logger.info("Dataset formatted.")

    # ------------------------------------------------------------------
    # Training configuration
    # ------------------------------------------------------------------
    sft_config = SFTConfig(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,        # Saves ~30% RAM at cost of compute
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        logging_dir=LOG_DIR,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        report_to=["tensorboard"],
        dataloader_num_workers=0,           # Required on Windows
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        seed=42,
        remove_unused_columns=True,
    )

    # ------------------------------------------------------------------
    # Hourly checkpoint callback
    # ------------------------------------------------------------------
    time_ckpt_cb = TimeBasedCheckpointCallback(
        save_dir=HOURLY_CKPT_DIR,
        interval_seconds=CHECKPOINT_EVERY_SECONDS,
    )

    # ------------------------------------------------------------------
    # SFTTrainer
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[time_ckpt_cb],
    )

    # Attach trainer so the callback can call save_model()
    time_ckpt_cb.attach_trainer(trainer)

    # Log trainable parameter count
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in trainer.model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.2f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("  Epochs          : %d", NUM_EPOCHS)
    logger.info("  Effective batch : %d", BATCH_SIZE * GRAD_ACCUM_STEPS)
    logger.info("  Max seq length  : %d", MAX_SEQ_LENGTH)
    logger.info("  TensorBoard     : tensorboard --logdir=%s", LOG_DIR)
    logger.info("=" * 60)

    training_ok = False
    try:
        result = trainer.train()
        training_ok = True
        metrics = result.metrics
        logger.info("Training complete. Metrics: %s", metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — saving current state before exit...")
    except Exception as e:
        logger.error("Training error: %s", e, exc_info=True)
    finally:
        logger.info("Saving final LoRA adapter to: %s", FINAL_MODEL_DIR)
        try:
            trainer.save_model(FINAL_MODEL_DIR)
            tokenizer.save_pretrained(FINAL_MODEL_DIR)
            logger.info("Final model saved to '%s'.", FINAL_MODEL_DIR)
        except Exception as e:
            logger.error("Failed to save final model: %s", e, exc_info=True)

        if training_ok:
            logger.info("Done. Run inference: python src/3_resume_and_infer.py")


if __name__ == "__main__":
    main()
