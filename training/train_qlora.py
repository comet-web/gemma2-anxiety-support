"""QLoRA supervised fine-tuning of Gemma-2 for CBT dialogue (TRL SFTTrainer + PEFT).

- Base model loaded in 4-bit NF4 with double quantisation (bitsandbytes); LoRA adapters on all
  attention and MLP projections.
- Loss on counselor turns only (`assistant_only_loss`), via a Gemma-2 chat template with
  {% generation %} markers. The rendered text is identical to the stock Gemma-2 template.
- Eval loss / perplexity is measured before training (= the base model, since LoRA B is
  initialised to zero) and after training, on the same held-out conversations.

Usage: python training/train_qlora.py --config configs/train_qlora.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_template import GEMMA2_TRAIN_TEMPLATE  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_qlora.yaml"))
    ap.add_argument("--max-train-samples", type=int, default=None, help="override config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples

    assert torch.cuda.is_available(), "QLoRA training needs a CUDA GPU (use the Colab notebook)."
    bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)} | compute dtype: {compute_dtype}")

    data_dir = ROOT / cfg["data_dir"]
    ds = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"), "eval": str(data_dir / "eval.jsonl")})
    ds = ds.remove_columns([c for c in ds["train"].column_names if c != "messages"])
    train_ds, eval_ds = ds["train"], ds["eval"]
    if cfg.get("max_train_samples"):
        train_ds = train_ds.shuffle(seed=cfg["seed"]).select(range(min(cfg["max_train_samples"], len(train_ds))))
    if cfg.get("max_eval_samples"):
        eval_ds = eval_ds.select(range(min(cfg["max_eval_samples"], len(eval_ds))))
    print(f"train={len(train_ds)} eval={len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    tokenizer.chat_template = GEMMA2_TRAIN_TEMPLATE
    tokenizer.padding_side = "right"

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb,
        torch_dtype=compute_dtype,
        attn_implementation="eager",  # recommended for Gemma-2 training (attention logit soft-capping)
        device_map={"": 0},
    )

    peft_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    out_dir = ROOT / cfg["output_dir"]
    sft_args = SFTConfig(
        output_dir=str(out_dir),
        max_length=cfg["max_length"],
        assistant_only_loss=cfg["assistant_only_loss"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        seed=cfg["seed"],
        report_to=cfg["report_to"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    # Sanity check: the assistant mask must cover only counselor tokens.
    sample = trainer.train_dataset[0]
    if "assistant_masks" in sample:
        frac = sum(sample["assistant_masks"]) / len(sample["assistant_masks"])
        print(f"assistant-token fraction in sample 0: {frac:.2%}")
        assert 0.05 < frac < 0.95, "assistant mask looks wrong; check the chat template"

    t0 = time.time()
    base_metrics = trainer.evaluate()
    print("base (pre-training) eval:", base_metrics)
    train_result = trainer.train()
    ft_metrics = trainer.evaluate()
    print("fine-tuned eval:", ft_metrics)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    results = {
        "model_id": cfg["model_id"],
        "gpu": torch.cuda.get_device_name(0),
        "train_conversations": len(train_ds),
        "eval_conversations": len(eval_ds),
        "train_runtime_min": round((time.time() - t0) / 60, 1),
        "train_loss": train_result.training_loss,
        "base_eval_loss": base_metrics["eval_loss"],
        "base_eval_perplexity": math.exp(base_metrics["eval_loss"]),
        "finetuned_eval_loss": ft_metrics["eval_loss"],
        "finetuned_eval_perplexity": math.exp(ft_metrics["eval_loss"]),
        "config": cfg,
    }
    results_dir = ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
