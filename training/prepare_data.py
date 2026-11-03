"""Convert CACTUS (CBT counseling dialogues) into Gemma-2 chat-format SFT data.

Source: LangAGI-Lab/cactus on the Hugging Face Hub (31,577 synthetic CBT dialogues, each with
the client's automatic thought, distortion labels and the CBT technique used).

Processing:
- split the `dialogue` string into turns ("Counselor:" -> assistant, "Client:" -> user)
- strip the client's name from counselor turns (prevents the model inventing names)
- drop leading counselor turns (Gemma-2 conversations must start with a user turn)
- merge same-speaker runs, drop a trailing user turn (must end on a counselor turn)
- fold the CBT system prompt into the first user turn (Gemma-2 has no system role;
  inference does the same, see src/anxiety_support/llm_client.py)
- keep anxiety-focused cases by default (`--focus anxiety`, ~13k of 31.6k)
- deterministic train / eval / test split (94 / 3 / 3)

Usage: python training/prepare_data.py --out data/processed [--limit 2000]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxiety_support.prompts import COUNSELOR_SYSTEM_PROMPT  # noqa: E402

ANXIETY_RE = re.compile(r"anxi|worr|panic|nervous|fear|afraid|scared", re.IGNORECASE)
SPEAKER_RE = re.compile(r"^(Counselor|Client)\s*:\s*(.*)$")
ROLE = {"Counselor": "assistant", "Client": "user"}


def parse_dialogue(dialogue: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for line in dialogue.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            turns.append({"role": ROLE[m.group(1)], "content": m.group(2).strip()})
        elif turns:  # continuation of the previous turn
            turns[-1]["content"] += " " + line
    return turns


def presenting_problem(intake_form: str) -> str:
    m = re.search(r"Presenting Problem\s*(.*?)(\n\s*\n|\n3\.)", intake_form or "", re.S)
    return m.group(1) if m else ""


def is_anxiety_case(row: dict) -> bool:
    return bool(ANXIETY_RE.search(presenting_problem(row.get("intake_form", ""))))


def client_first_name(intake_form: str) -> str | None:
    m = re.search(r"Name:\s*\n?\s*([A-Z][a-zA-Z'-]+)", intake_form or "")
    return m.group(1) if m else None


def scrub_name(text: str, name: str | None) -> str:
    if not name:
        return text
    text = re.sub(rf",\s*{re.escape(name)}\b", "", text)          # "Thanks, Brooke." -> "Thanks."
    text = re.sub(rf"\b{re.escape(name)},\s*", "", text)          # "Brooke, how ..." -> "how ..."
    text = re.sub(rf"\b{re.escape(name)}\b", "you", text)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


def to_messages(row: dict, system_prompt: str = COUNSELOR_SYSTEM_PROMPT) -> list[dict[str, str]] | None:
    name = client_first_name(row.get("intake_form", ""))
    turns = parse_dialogue(row["dialogue"])
    for t in turns:
        if t["role"] == "assistant":
            t["content"] = scrub_name(t["content"], name)

    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    merged: list[dict[str, str]] = []
    for t in turns:
        if not t["content"]:
            continue
        if merged and merged[-1]["role"] == t["role"]:
            merged[-1]["content"] += " " + t["content"]
        else:
            merged.append(dict(t))
    while merged and merged[-1]["role"] != "assistant":
        merged.pop()
    if len(merged) < 4:
        return None
    merged[0]["content"] = f"{system_prompt}\n\n{merged[0]['content']}"
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="LangAGI-Lab/cactus")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows (debugging)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focus", choices=["anxiety", "all"], default="anxiety",
                    help="anxiety = keep dialogues whose intake 'Presenting Problem' mentions anxiety/worry/panic/fear")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    records, skipped, filtered = [], 0, 0
    for row in ds:
        if args.focus == "anxiety" and not is_anxiety_case(row):
            filtered += 1
            continue
        msgs = to_messages(row)
        if msgs is None:
            skipped += 1
            continue
        records.append({
            "messages": msgs,
            "meta": {"cbt_technique": row["cbt_technique"], "patterns": row["patterns"],
                     "attitude": row["attitude"], "thought": row["thought"]},
        })

    random.Random(args.seed).shuffle(records)
    n = len(records)
    n_eval = n_test = max(1, int(0.03 * n))
    splits = {"test": records[:n_test], "eval": records[n_test:n_test + n_eval], "train": records[n_test + n_eval:]}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    turns = [len(r["messages"]) for r in records]
    chars = [sum(len(m["content"]) for m in r["messages"]) for r in records]
    stats = {
        "source": args.dataset,
        "rows_in": len(ds),
        "conversations_out": n,
        "focus": args.focus,
        "filtered_out_by_focus": filtered,
        "skipped_malformed": skipped,
        "splits": {k: len(v) for k, v in splits.items()},
        "turns_per_conversation_mean": round(sum(turns) / n, 1),
        "chars_per_conversation_mean": round(sum(chars) / n),
        "top_cbt_techniques": Counter(r["meta"]["cbt_technique"] for r in records).most_common(8),
        "attitude": dict(Counter(r["meta"]["attitude"] for r in records)),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
