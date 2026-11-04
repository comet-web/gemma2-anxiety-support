# Project walkthrough

How the system works internally. It follows one message through the runtime and one conversation through
training, then records the implementation details that are easy to get wrong. For setup and results, see
the [README](../README.md).

## 1. One message through the runtime

Example: a user sends *"My placement interview is tomorrow and I'm sure I'll mess it up"*.

| Step | Where | What happens |
|---|---|---|
| 1 | `api/main.py` `POST /v1/chat` | Validates the request (1–4000 chars), loads or creates the session, starts a latency timer. |
| 2 | `safety/rails.py` `GuardedPipeline.handle` | Appends the message to the session transcript, stores the session and a trace dict in a `ContextVar` so NeMo actions can reach them, calls `LLMRails.generate_async`. |
| 3 | Input rail `moderate user input` (`guardrails/rails.co`) | Runs `check_user_input`: crisis keyword regex, then Llama Guard 3 on the last turns (role `User`). S11 → crisis reply, S6 → medication redirect, other unsafe → refusal, then `stop`. |
| 4 | Dialog rails, embedding match | The message is embedded (all-MiniLM-L6-v2 via FastEmbed) and compared with the Colang example sentences. Best score ≥ 0.45 → that intent's flow (crisis / medication / diagnosis / off-topic / jailbreak) and a fixed reply. Otherwise → fallback intent `share feelings`. |
| 5 | Flow `cbt conversation` | `check_topic`: one short structured LLM call (on topic yes/no), run only on fall-through messages. Off topic → steer back. On topic → `cbt_agent_respond`. |
| 6 | `agent/cbt_agent.py` | Planner (base Gemma-2, JSON-schema decoding) picks a phase and a tool, e.g. `identify_thought` + `detect_distortions("I'm sure I'll mess it up")` → *fortune-telling*. Up to `AGENT_MAX_STEPS` tool calls, then the responder (CBT LoRA adapter) writes the reply with the phase, thought record and tool results as hidden notes. |
| 7 | Output rail `moderate bot output` | Llama Guard 3 on the reply (role `Agent`). Unsafe → reply withheld. Fixed replies written by us skip this check. |
| 8 | `api/main.py` | Returns the reply, `route`, both moderation verdicts, agent steps, thought record and latency. |

Every response carries its `route`, so each decision can be audited and counted.

## 2. One conversation through training

1. **Load** `LangAGI-Lab/cactus` (31,577 dialogues, one `dialogue` string each: `Counselor: …\nClient: …`).
2. **Filter** to anxiety cases: the intake form's *Presenting Problem* mentions anxiety, worry, panic or fear → 13,101.
3. **Parse** speaker lines into `user` / `assistant` turns, joining continuation lines.
4. **Clean**: remove the client's first name from counselor turns, drop leading counselor turns, merge
   same-speaker runs, drop a trailing client turn, require ≥ 4 turns.
5. **Fold** the CBT system prompt into the first user turn (Gemma-2 has no system role).
6. **Split** with a fixed seed into train / eval / test (94/3/3).
7. **Render** with `training/chat_template.py`: the stock Gemma-2 layout plus `{% generation %}` markers around
   assistant text, so the loss mask covers counselor tokens and `<end_of_turn>`.
8. **Train** with TRL `SFTTrainer`: 4-bit NF4 base, LoRA r=16/α=32 on q, k, v, o, gate, up and down
   projections, `assistant_only_loss`, gradient checkpointing, paged 8-bit AdamW.
9. **Measure** eval loss before training (LoRA B = 0, so this is the base model) and after, on the same eval set.

## 3. Design decisions

- **Steering is a cascade.** The keyword net and embedding match are free and deterministic. The LLM topic check
  costs one short call and runs only when they don't match. Llama Guard covers general harm categories that
  example sentences can't.
- **Crisis is a destination, not a block.** Three independent detectors all route to the same supportive reply
  with region-specific helplines (`CRISIS_REGION`). Over-escalating hyperbole is accepted as the safer error.
- **Llama Guard prompt built in code.** The 1B and 8B checkpoints behave the same, the taxonomy is explicit, and
  the parser only needs two lines.
- **Fail closed.** If moderation is unreachable, the user gets a safe fallback reply.
- **Bounded agent with structured outputs.** Planner and extraction calls are JSON-schema constrained. Any
  failure falls back to "reply without a tool", and the tool loop has a hard cap.
- **One vLLM process, two roles.** The base model (planner, topic check) and the CBT adapter (responder) share
  weights through multi-LoRA serving.

## 4. Implementation notes

Details found while building and running the project. Most fail silently or give confusing errors.

- **NeMo similarity scores are not cosine.** `embeddings_only_similarity_threshold` is compared with
  `1 − √(2 − 2·cos)/2` (Annoy-compatible). A threshold of 0.40 means cosine ≈ 0.28. The 0.45 used here
  corresponds to cosine 0.40, tuned on the dev split only.
- **`torch.cuda.is_bf16_supported()` is True on a T4** because it counts emulation. Precision is chosen from
  compute capability (≥ 8 → bf16, otherwise fp16).
- **TRL casts LoRA weights of 4-bit models to bf16** regardless of hardware. With fp16 autocast the gradient
  scaler then fails (`… not implemented for 'BFloat16'`), so trainable weights are cast back to fp32 when bf16
  isn't used (standard mixed precision).
- **TRL 1.x stores the loss mask in `labels`**, not in an `assistant_masks` column. The training script reads
  either, decodes trained vs ignored tokens, and asserts no user text is trained on.
- **Transformers v5**: `warmup_ratio` was removed (a float `warmup_steps` is a ratio), `torch_dtype` became
  `dtype`, and `apply_chat_template` returns a dict.
- **Old `torchao`** (preinstalled on Colab) makes recent PEFT raise while wrapping the model. It isn't used here,
  so the notebook uninstalls it and the script checks for it at start-up.
- **The OpenAI Python SDK uses its own HTTP stack**, so request-mocking libraries don't intercept it. The HTTP
  tests use a real localhost stub server instead.
- **The training-only chat template is not saved next to the adapter.** Inference uses the base tokenizer's
  stock template.

## 5. Verification

| Check | How |
|---|---|
| Unit and integration tests | `pytest -q`: 49 tests, including the real NeMo runtime (stubbed models) and a localhost stub vLLM server |
| Steering regression | `python evaluation/eval_safety.py --target rails-offline --split test` (runs in CI) |
| Training pipeline | `python training/train_qlora.py --smoke`: 2 steps on a tiny random Gemma-2, about 1 minute on CPU |
| Docker image | built in CI on every push |
| GPU stages | `notebooks/colab_train_and_eval.ipynb` |

## 6. Extending

- **Another region's helplines:** add an entry to `RESOURCES` in `safety/crisis.py` and set `CRISIS_REGION`.
- **A new steering intent:** add `define user …` examples, a `define bot …` reply and a flow to
  `guardrails/rails.co`, then add labelled prompts to `evaluation/data/safety_dev.jsonl` and re-run the eval.
- **A new agent tool:** implement it in `agent/tools.py`, add its name to `ToolName` and the planner prompt,
  and handle it in `CBTAgent.run_tool`.
- **A bigger model:** change `model_id` in `configs/train_qlora.yaml` and the served model in `docker-compose.yml`.
