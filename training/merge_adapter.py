"""Optional: merge the LoRA adapter into the base weights (16-bit) and/or push to the HF Hub.

vLLM can serve the adapter directly (`--enable-lora --lora-modules gemma-cbt=...`), which is
what docker-compose does. Merging is useful to serve a single model without the LoRA
overhead, or to share it.

Usage:
  python training/merge_adapter.py --adapter outputs/gemma2-cbt-qlora/adapter --out outputs/gemma2-cbt-merged
  python training/merge_adapter.py --adapter ... --push-adapter your-hf-user/gemma2-2b-cbt-lora
"""

from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-2-2b-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", default=None, help="directory for merged weights")
    ap.add_argument("--push-adapter", default=None, help="HF repo id to upload the adapter to")
    ap.add_argument("--push-merged", default=None, help="HF repo id to upload the merged model to")
    args = ap.parse_args()

    # native bf16 only on compute capability >= 8 (is_bf16_supported() also counts emulation, e.g. on T4)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, device_map="auto")
    # Keep the base model's stock chat template for inference.
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = PeftModel.from_pretrained(base, args.adapter)

    if args.push_adapter:
        model.push_to_hub(args.push_adapter, private=True)
        print(f"adapter pushed to {args.push_adapter}")

    if args.out or args.push_merged:
        merged = model.merge_and_unload()
        if args.out:
            merged.save_pretrained(args.out, safe_serialization=True)
            tokenizer.save_pretrained(args.out)
            print(f"merged model saved to {args.out}")
        if args.push_merged:
            merged.push_to_hub(args.push_merged, private=True)
            tokenizer.push_to_hub(args.push_merged, private=True)
            print(f"merged model pushed to {args.push_merged}")


if __name__ == "__main__":
    main()
