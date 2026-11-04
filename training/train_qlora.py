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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_template import GEMMA2_TRAIN_TEMPLATE  # noqa: E402

SMOKE_MODEL_ID = "trl-internal-testing/tiny-Gemma2ForCausalLM"


def check_torchao() -> None:
    """Colab preinstalls an old torchao; recent PEFT raises on any torchao < 0.16 while wrapping
    the model, even though torchao isn't used here. Fail early with the fix instead."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        v = version("torchao")
    except PackageNotFoundError:
        return
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) < (0, 16):
        raise SystemExit(f"torchao {v} is installed but PEFT needs >= 0.16 (or no torchao). "
                         "Run:  pip uninstall -y torchao   (it is not used by this project)")


class StopOnNonFiniteLoss(TrainerCallback):
    """Gemma-2 can overflow in fp16 (e.g. on a T4, which has no bf16). Fail fast instead of
    training on NaNs for hours."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(float(loss)):
            raise RuntimeError(
                f"Training loss became {loss} at step {state.global_step}. This is usually fp16 overflow "
                "with Gemma-2: switch the Colab runtime to an L4 or A100 (bf16), or lower learning_rate."
            )


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "train_qlora.yaml"))
    ap.add_argument("--max-train-samples", type=int, default=None, help="override config")
    ap.add_argument("--smoke", action="store_true",
                    help="2-step run on a tiny random Gemma-2 (CPU ok, no 4-bit) to check the pipeline end to end")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.smoke:
        cfg.update(model_id=SMOKE_MODEL_ID, output_dir="outputs/smoke", max_train_samples=8, max_eval_samples=4,
                   max_length=512, load_in_4bit=False, optim="adamw_torch", gradient_checkpointing=False,
                   per_device_train_batch_size=2, gradient_accumulation_steps=1, logging_steps=1,
                   eval_steps=1, save_steps=1)

    check_torchao()
    on_gpu = torch.cuda.is_available()
    if not on_gpu and not args.smoke:
        raise SystemExit("QLoRA training needs a CUDA GPU (use the Colab notebook), or pass --smoke.")
    # Native bf16 needs compute capability >= 8.0 (Ampere+). torch.cuda.is_bf16_supported() also
    # counts emulation and returns True on a T4, which would make training slow and unstable.
    bf16 = on_gpu and torch.cuda.get_device_capability()[0] >= 8
    fp16 = on_gpu and not bf16
    compute_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    print(f"device: {torch.cuda.get_device_name(0) if on_gpu else 'cpu'} | compute dtype: {compute_dtype}")

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

    bnb = None
    if cfg["load_in_4bit"]:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=bnb,
        dtype=compute_dtype,          # transformers v5: `dtype` replaces `torch_dtype`
        attn_implementation="eager",  # recommended for Gemma-2 training (attention logit soft-capping)
        device_map={"": 0} if on_gpu else None,
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
        # transformers v5 removed `warmup_ratio`; a float in (0, 1) passed to `warmup_steps` is a ratio.
        warmup_steps=cfg["warmup_ratio"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        optim=cfg["optim"],
        weight_decay=cfg["weight_decay"],
        max_grad_norm=cfg["max_grad_norm"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        use_cpu=not on_gpu,
        max_steps=2 if args.smoke else -1,
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
        callbacks=[StopOnNonFiniteLoss()],
    )
    trainer.model.print_trainable_parameters()

    # Sanity check that the loss really covers counselor turns only. Depending on the TRL version the
    # mask is exposed as `assistant_masks` or already applied to `labels` (-100 = ignored).
    sample = trainer.train_dataset[0]
    ids = sample["input_ids"]
    if "assistant_masks" in sample:
        trained = [bool(m) for m in sample["assistant_masks"]]
    elif "labels" in sample:
        trained = [lab != -100 for lab in sample["labels"]]
    else:
        raise RuntimeError(f"cannot find the loss mask in the processed dataset (columns: {list(sample)})")
    frac = sum(trained) / len(trained)
    trained_text = tokenizer.decode([t for t, keep in zip(ids, trained) if keep])
    ignored_text = tokenizer.decode([t for t, keep in zip(ids, trained) if not keep])
    print(f"loss-token fraction in sample 0: {frac:.2%}")
    print("  trained on :", trained_text[:160].replace("\n", " / "))
    print("  ignored    :", ignored_text[:160].replace("\n", " / "))
    assert 0.05 < frac < 0.95, "loss mask looks wrong; check the chat template"
    assert "<start_of_turn>user" not in trained_text, "user turns are being trained on"

    t0 = time.time()
    base_metrics = trainer.evaluate()
    print("base (pre-training) eval:", base_metrics)
    train_result = trainer.train()
    ft_metrics = trainer.evaluate()
    print("fine-tuned eval:", ft_metrics)

    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    # The tokenizer is deliberately not saved next to the adapter: it carries the training-only
    # template (generation markers). vLLM and inference use the base model's stock tokenizer.
    for name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json", "chat_template.jinja"):
        (adapter_dir / name).unlink(missing_ok=True)

    results = {
        "model_id": cfg["model_id"],
        "gpu": torch.cuda.get_device_name(0) if on_gpu else "cpu",
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
    # Smoke runs must never overwrite the real metrics file.
    results_dir = out_dir if args.smoke else ROOT / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "training_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
