# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fine-tuning pipeline for `unsloth/Llama-3.2-1B-Instruct` on the `mteb/banking77` intent classification dataset. Targets Intel Iris Xe integrated GPU (CPU/IPEX fallback). Managed with `uv`.

## Environment Setup

```bash
chmod +x setup_env.sh && ./setup_env.sh
source .venv/Scripts/activate   # Windows (Git Bash)
# source .venv/bin/activate     # Linux/macOS
```

The setup script clones Unsloth from GitHub (into `./unsloth/`) and installs the `[intel-gpu-torch290]` extra. If that fails, it falls back to standard Unsloth. IPEX installation is attempted but non-fatal.

## Pipeline Execution Order

```bash
python src/1_prepare_data.py       # Download & format dataset → data/processed_banking77.jsonl
python src/2_train_intel.py        # Train LoRA adapter → models/final_lora/
python src/3_resume_and_infer.py   # Auto-resume + inference test
```

Monitor training in a separate terminal:
```bash
tensorboard --logdir=logs/
```

## Important: Unsloth Cannot Run on This Hardware

Unsloth explicitly raises `NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.` when no CUDA/XPU device is found. Intel Iris Xe iGPU is not exposed as an XPU device on Windows (IPEX only supports Linux). The training script uses the standard HF stack (`transformers + peft + trl`) on CPU instead.

`3_resume_and_infer.py` loads LoRA adapters via `peft.PeftModel` + `merge_and_unload()` (no Unsloth).

## Architecture

### Data Flow
`mteb/banking77` (HuggingFace) → Alpaca instruction format → `data/processed_banking77.jsonl` → SFTTrainer

The Alpaca prompt template used across all scripts:
```
### Instruction:
Classify the following banking customer query into one of 77 intent categories...
### Input:
{customer query}
### Response:
{intent_label_name}
```
Integer labels (0 to 76) are resolved to string names via `LABEL_NAMES` list in `1_prepare_data.py`. This same list must remain consistent if the data script is ever modified.

### Training Architecture (`2_train_intel.py`, CPU)
- **Base model**: loaded in float32 via `AutoModelForCausalLM` (no quantization on CPU)
- **Adapters**: LoRA `r=16, alpha=16` via `peft.LoraConfig`, passed directly to `SFTTrainer(peft_config=...)`
- **`TimeBasedCheckpointCallback(TrainerCallback)`**: saves to `checkpoints/hourly_backup/ckpt_YYYYMMDD_HHMMSS/` every 1800 seconds (30 min). Requires `callback.attach_trainer(trainer)` after `SFTTrainer` init.
- **fp16/bf16**: both off; the Intel Iris Xe path trains in fp32
- **Optimizer**: `adamw_torch` (not `paged_adamw_32bit` or `adamw_8bit`, which require CUDA)

### Colab Notebook (`colab/banking77_finetune.ipynb`)
Self-contained notebook for Google Colab T4 GPU. Key differences from the local CPU scripts:
- Uses **Unsloth** (`FastLanguageModel`) since CUDA is available on Colab
- `load_in_4bit=True`, `max_seq_length=512`, `batch_size=8`, `packing=True`
- **Optimizer**: `adamw_8bit` (bitsandbytes, saves VRAM)
- **Checkpoint interval**: 1800s (30 min), saved to Google Drive at `MyDrive/banking77_finetune/`
- Cell 9 is the resume cell: run it instead of cells 4 to 6 after a disconnect

### Recovery Logic (`3_resume_and_infer.py`)
Model discovery priority: `models/final_lora` → `checkpoints/hourly_backup/` (newest by name) → `checkpoints/checkpoint-*` (highest step) → base model fallback. All paths are checked with `adapter_config.json` or `config.json` presence.

## Key Constraints

- **`uv` only**: do not use `pip` directly; all installs go through `uv pip install`
- **No fp16/bf16**: keep `fp16=False, bf16=False` in `SFTConfig`
- **`dataloader_num_workers=0`**: required to avoid multiprocessing issues on Windows
- **Unsloth source install**: the `unsloth/` directory in the repo root is a git clone, not a PyPI package; changes to it affect the installed package
