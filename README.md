# unsloth_code

> Fast, memory-efficient LLM fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) — LoRA & QLoRA training, inference, and export in one place.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/powered%20by-Unsloth-green.svg)](https://github.com/unslothai/unsloth)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#license)

This repository contains scripts and configuration for fine-tuning open large language models
(Llama, Mistral, Qwen, Gemma, Phi, gpt-oss, and 500+ others) using **Unsloth**. Unsloth rewrites
the model's compute path with custom Triton kernels, so you can train **up to 2× faster** while
using **up to 70% less VRAM** — with **no loss in accuracy**.

---

## Table of Contents

- [Why Unsloth](#why-unsloth)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Training](#training)
- [Inference](#inference)
- [Saving & Exporting](#saving--exporting)
- [Hyperparameters](#hyperparameters)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why Unsloth

| Capability            | Standard fine-tuning | With Unsloth                          |
| --------------------- | -------------------- | ------------------------------------- |
| Training speed        | 1×                   | up to **2× faster**                   |
| VRAM usage            | 1×                   | up to **70% less**                    |
| Accuracy              | baseline             | **no degradation**                    |
| Minimum GPU           | 16 GB+               | works from **~3 GB VRAM**             |
| Context length        | baseline             | up to **4× longer** (gradient ckpt)   |

LoRA and QLoRA freeze the base model and train only small low-rank adapter matrices
(~1% of the parameters). **QLoRA** additionally quantizes the frozen base model to 4-bit,
cutting memory by another ~75% — making single-GPU fine-tuning of 8B–14B models practical
on consumer hardware.

## Features

- **QLoRA / LoRA / full fine-tuning** with a single configuration switch.
- **4-bit, 8-bit, and 16-bit** training paths.
- Built on Hugging Face **`transformers`** + **`trl`** — no exotic APIs to learn.
- **2× faster native inference** via `FastLanguageModel.for_inference()`.
- One-line export to **GGUF** (Ollama, llama.cpp, LM Studio), **merged 16-bit** (vLLM),
  and the **Hugging Face Hub**.
- Supports text, vision/multimodal, embedding, TTS, and RL (GRPO/DPO) workflows.

## Requirements

- **OS:** Linux, WSL2, or Windows.
- **GPU:** NVIDIA GPU with CUDA (compute capability 7.0+ — e.g. RTX 20/30/40/50 series,
  A100, H100). AMD and Intel GPUs are supported via Unsloth's dedicated install guides.
- **Python:** 3.10 or newer (3.13 recommended).
- **VRAM:** ~3 GB for small models; ~8 GB comfortably fine-tunes an 8B model in 4-bit.

## Installation

The simplest path on Linux / WSL:

```bash
pip install unsloth
```

Recommended (isolated environment with [`uv`](https://github.com/astral-sh/uv)):

```bash
# Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

```powershell
# Windows (PowerShell)
winget install -e --id Python.Python.3.13
winget install -e --id astral-sh.uv
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

To pull the latest fixes without touching your other dependencies:

```bash
pip install --upgrade --no-cache-dir --no-deps unsloth unsloth_zoo
```

> See the official [installation guide](https://unsloth.ai/docs/get-started/install)
> for Docker, Blackwell (RTX 50xx), AMD, and Intel instructions.

## Project Structure

A recommended layout for this repository:

```text
unsloth_code/
├── README.md
├── requirements.txt        # pinned dependencies
├── configs/
│   └── llama3_lora.yaml    # model + training hyperparameters
├── data/                   # local datasets (git-ignored)
├── src/
│   ├── train.py            # fine-tuning entry point
│   ├── inference.py        # load adapter + generate
│   └── export.py           # GGUF / merged / Hub export
└── outputs/                # checkpoints & adapters (git-ignored)
```

## Quickstart

```bash
# 1. Install dependencies
pip install unsloth

# 2. Run training (script below)
python src/train.py

# 3. Run inference with the trained adapter
python src/inference.py
```

## Training

`src/train.py` — a complete, runnable QLoRA fine-tune on the cleaned Alpaca dataset:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

max_seq_length = 2048

# 1. Load a 4-bit quantized model + tokenizer.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype          = None,        # None -> auto-detect (float16 / bfloat16)
    load_in_4bit   = True,        # 4-bit QLoRA. Set False for 16-bit LoRA.
)

# 2. Attach LoRA adapters — only ~1% of parameters are trained.
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0,        # 0 is optimized in Unsloth
    bias                       = "none",   # "none" is optimized in Unsloth
    use_gradient_checkpointing = "unsloth",# 4x longer context, less VRAM
    random_state               = 3407,
)

# 3. Format the dataset into a single "text" column.
alpaca_prompt = """Below is an instruction that describes a task, paired with an input \
that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # required so generation knows when to stop

def formatting_prompts_func(examples):
    texts = []
    for instruction, inp, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(alpaca_prompt.format(instruction, inp, output) + EOS_TOKEN)
    return {"text": texts}

dataset = load_dataset("yahma/alpaca-cleaned", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# 4. Configure and run the trainer.
trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,      # effective batch size = 8
        warmup_steps                = 5,
        max_steps                   = 60,    # use num_train_epochs for a full run
        learning_rate               = 2e-4,
        fp16                        = not is_bfloat16_supported(),
        bf16                        = is_bfloat16_supported(),
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        output_dir                  = "outputs",
        report_to                   = "none", # set to "wandb" to log to W&B
    ),
)

trainer.train()

# 5. Save the LoRA adapter (small — typically ~100 MB).
model.save_pretrained("outputs/lora_model")
tokenizer.save_pretrained("outputs/lora_model")
```

Swap `model_name` for any model in the [Unsloth catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog),
for example `unsloth/Qwen3-8B`, `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`, or
`unsloth/gemma-3-4b-it`.

## Inference

`src/inference.py` — load the trained adapter and generate:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "outputs/lora_model",   # the saved adapter folder
    max_seq_length = 2048,
    dtype          = None,
    load_in_4bit   = True,
)

# Enable Unsloth's 2x faster inference path.
FastLanguageModel.for_inference(model)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input \
that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
    [alpaca_prompt.format("Continue the Fibonacci sequence.", "1, 1, 2, 3, 5, 8", "")],
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

## Saving & Exporting

`src/export.py` — convert the fine-tuned model to a deployable format:

```python
# LoRA adapter only — smallest artifact, requires the base model at load time.
model.save_pretrained("outputs/lora_model")
tokenizer.save_pretrained("outputs/lora_model")

# Merged 16-bit weights — best for vLLM / Hugging Face deployment.
model.save_pretrained_merged("outputs/model-16bit", tokenizer, save_method="merged_16bit")

# Merged 4-bit weights — smaller footprint.
model.save_pretrained_merged("outputs/model-4bit", tokenizer, save_method="merged_4bit")

# GGUF — for Ollama, llama.cpp, and LM Studio.
model.save_pretrained_gguf("outputs/model-gguf", tokenizer, quantization_method="q4_k_m")

# Push directly to the Hugging Face Hub (use a token with write access).
model.push_to_hub_merged(
    "your-username/your-model",
    tokenizer,
    save_method = "merged_16bit",
    token       = "hf_xxx",   # better: read from the HF_TOKEN environment variable
)
```

> Never hard-code secrets. Read tokens from the environment instead:
> `import os; token = os.environ["HF_TOKEN"]`.

## Hyperparameters

| Parameter                     | Default | Notes                                                        |
| ----------------------------- | ------- | ------------------------------------------------------------ |
| `r` (LoRA rank)               | 16      | 8–128. Higher = more capacity and more VRAM.                 |
| `lora_alpha`                  | 16      | A common rule of thumb is `lora_alpha = r` (or `2 * r`).     |
| `learning_rate`               | 2e-4    | Try `1e-4`, `5e-5`, or `2e-5` for steadier convergence.      |
| `per_device_train_batch_size` | 2       | Raise for better GPU use; watch for padding overhead.        |
| `gradient_accumulation_steps` | 4       | Increases effective batch size without extra VRAM.           |
| `max_steps` / `num_train_epochs` | 60   | Use `num_train_epochs = 1–3` for full runs to avoid overfit. |
| `max_seq_length`              | 2048    | Unsloth supports 4× longer context than standard training.   |

A training loss settling around **0.5–1.0** is usually healthy. A loss approaching 0
often signals overfitting — validate on held-out data.

## Troubleshooting

| Symptom                                    | Fix                                                                       |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `CUDA out of memory`                       | Lower `per_device_train_batch_size`, reduce `max_seq_length`, or use 4-bit.|
| `torch`/CUDA version mismatch on install   | Install with `uv pip install unsloth --torch-backend=auto`.                |
| Model generates without stopping           | Ensure `EOS_TOKEN` is appended to every training example.                  |
| Slow or no loss decrease                   | Adjust `learning_rate`, check dataset formatting, train for more steps.    |
| `bitsandbytes` errors on Windows           | Follow the official [Windows install guide](https://unsloth.ai/docs/get-started/install/windows-installation). |

## Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) — Daniel Han, Michael Han, and the Unsloth team.
- [Hugging Face](https://huggingface.co/) — `transformers`, `datasets`, and `trl`.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — GGUF export and local inference.

```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title  = {Unsloth},
  url    = {https://github.com/unslothai/unsloth},
  year   = {2023}
}
```

## License

This repository is released under the MIT License — see `LICENSE` for details.
The Unsloth library itself is dual-licensed under Apache-2.0 and AGPL-3.0.
