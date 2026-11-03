"""Response-quality evaluation: base Gemma-2 vs the QLoRA CBT adapter on held-out CACTUS conversations.

For each sampled test conversation we cut at a random counselor turn, give both models the same
context (system prompt folded into the first user turn), and compare their next counselor turn.

Automatic metrics (always):
  ref_similarity     cosine similarity to the real counselor turn (all-MiniLM-L6-v2 embeddings)
  words              mean reply length
  ends_with_question share of replies that end in a question (guided discovery)
  multi_question     share of replies asking more than one question (CBT: one question at a time)

LLM judge (optional, --judge-model): pairwise preference on a CBT rubric (CTRS-inspired:
empathy, guided discovery, focus on cognitions, appropriateness). Each pair is judged in both
orders to cancel position bias; only consistent verdicts count as wins.

Usage (both models served by one vLLM with --enable-lora):
  python evaluation/eval_responses.py --base-model google/gemma-2-2b-it --ft-model gemma-cbt -n 100 \
      [--judge-model <model> --judge-base-url <url> --judge-api-key <key>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anxiety_support.llm_client import OpenAICompatLLM  # noqa: E402

JUDGE_PROMPT = """You are an expert CBT supervisor rating counselor replies (Cognitive Therapy Rating Scale style).
Given the conversation so far, decide which next counselor reply is better overall, considering:
1. Empathy and warmth
2. Guided discovery: asks one focused, open question rather than lecturing
3. Focus on cognitions: helps the client notice, examine or reframe thoughts
4. Appropriateness: concise, on-topic, no diagnosis or medication advice

Conversation:
{context}

Reply A: {a}

Reply B: {b}

Return JSON only."""


class Verdict(BaseModel):
    better: str  # "A", "B" or "tie"
    reason: str = ""


def sample_cases(n: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "data" / "processed" / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    cases = []
    for r in rows[:n]:
        msgs = r["messages"]
        cut_points = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i >= 3]
        k = rng.choice(cut_points)
        cases.append({"context": msgs[:k], "reference": msgs[k]["content"], "technique": r["meta"]["cbt_technique"]})
    return cases


def render(context: list[dict], last: int = 6) -> str:
    who = {"user": "Client", "assistant": "Counselor"}
    tail = context[-last:]
    return "\n".join(f"{who[m['role']]}: {m['content']}" for m in tail)


def auto_metrics(replies: list[str], refs: list[str], embed) -> dict:
    a, b = embed(replies), embed(refs)
    sims = (a * b).sum(axis=1)
    return {
        "ref_similarity": round(float(sims.mean()), 4),
        "words": round(float(np.mean([len(r.split()) for r in replies])), 1),
        "ends_with_question": round(float(np.mean([r.rstrip().endswith("?") for r in replies])), 3),
        "multi_question": round(float(np.mean([r.count("?") > 1 for r in replies])), 3),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--ft-model", default="gemma-cbt")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-api-key", default="EMPTY")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    from fastembed import TextEmbedding

    emb_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

    def embed(texts):
        v = np.array(list(emb_model.embed(texts)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    llm = OpenAICompatLLM(args.llm_base_url)
    cases = sample_cases(args.n, args.seed)
    sem = asyncio.Semaphore(args.concurrency)

    async def gen(model, ctx):
        async with sem:
            return await llm.chat(model, ctx, max_tokens=200, temperature=0.0)

    base = await asyncio.gather(*(gen(args.base_model, c["context"]) for c in cases))
    ft = await asyncio.gather(*(gen(args.ft_model, c["context"]) for c in cases))
    refs = [c["reference"] for c in cases]

    report = {
        "n": len(cases),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": args.base_model,
        "ft_model": args.ft_model,
        "auto": {"base": auto_metrics(base, refs, embed), "finetuned": auto_metrics(ft, refs, embed),
                 "reference": auto_metrics(refs, refs, embed)},
    }

    if args.judge_model:
        judge = OpenAICompatLLM(args.judge_base_url or args.llm_base_url, args.judge_api_key)

        async def judge_pair(ctx, a, b):
            async with sem:
                v = await judge.structured(args.judge_model, [{"role": "user", "content": JUDGE_PROMPT.format(context=render(ctx), a=a, b=b)}], Verdict, max_tokens=200)
            return (v.better.strip().upper() if v else "TIE")

        fwd = await asyncio.gather(*(judge_pair(c["context"], f, b) for c, f, b in zip(cases, ft, base)))   # A = fine-tuned
        rev = await asyncio.gather(*(judge_pair(c["context"], b, f) for c, f, b in zip(cases, ft, base)))   # A = base
        wins = losses = ties = 0
        for x, y in zip(fwd, rev):
            if x == "A" and y == "B":
                wins += 1
            elif x == "B" and y == "A":
                losses += 1
            else:
                ties += 1  # tie or inconsistent across orderings
        report["judge"] = {"model": args.judge_model, "finetuned_wins": wins, "base_wins": losses,
                           "ties_or_inconsistent": ties, "finetuned_win_rate": round(wins / len(cases), 3)}

    report["examples"] = [{"context_tail": render(c["context"], 2), "reference": c["reference"], "base": b, "finetuned": f}
                          for c, b, f in list(zip(cases, base, ft))[:5]]

    out = ROOT / "evaluation" / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "responses_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"# Response quality: `{args.base_model}` vs `{args.ft_model}` (n={len(cases)})", "",
          f"_Generated {report['generated_at']}_", "",
          "| Metric | Base | Fine-tuned | Real counselor |", "|---|---|---|---|"]
    for k in ["ref_similarity", "words", "ends_with_question", "multi_question"]:
        md.append(f"| {k} | {report['auto']['base'][k]} | {report['auto']['finetuned'][k]} | {report['auto']['reference'][k]} |")
    if "judge" in report:
        j = report["judge"]
        md += ["", f"LLM judge (`{j['model']}`, both orderings): fine-tuned wins {j['finetuned_wins']}, "
               f"base wins {j['base_wins']}, tie/inconsistent {j['ties_or_inconsistent']}."]
    (out / "responses_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    asyncio.run(main())
