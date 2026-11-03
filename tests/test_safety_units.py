import pytest

from anxiety_support.safety.crisis import crisis_message, detect_crisis
from anxiety_support.safety.llama_guard import CATEGORIES, LlamaGuard, build_prompt, parse_output


# ------------------------------------------------------------------ Llama Guard
def test_prompt_contains_taxonomy_and_target_role():
    p = build_prompt([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], role="Agent")
    assert "S11: Self-Harm." in p
    assert all(code in p for code in CATEGORIES)
    assert "User: hi" in p and "Agent: hello" in p
    assert "ONLY THE LAST Agent message" in p
    assert not p.startswith("<|begin_of_text|>")  # vLLM adds BOS itself
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


@pytest.mark.parametrize(
    "raw,safe,cats",
    [
        ("safe", True, []),
        ("\n\nsafe\n", True, []),
        ("unsafe\nS11", False, ["S11"]),
        ("unsafe\nS1, s10", False, ["S1", "S10"]),
        ("unsafe", False, []),
    ],
)
def test_parse_output(raw, safe, cats):
    r = parse_output(raw)
    assert r.safe is safe and r.categories == cats and r.error is None


def test_parse_garbage_is_unsafe_with_error():
    r = parse_output("I think this is fine")
    assert not r.safe and r.error


async def test_guard_fail_closed_on_network_error(fake_llm):
    guard = LlamaGuard(fake_llm(completion=ConnectionError("down")), "guard", fail_closed=True)
    r = await guard.check_input("hello")
    assert not r.safe and "unavailable" in r.error


async def test_guard_fail_open_when_configured(fake_llm):
    guard = LlamaGuard(fake_llm(completion=ConnectionError("down")), "guard", fail_closed=False)
    assert (await guard.check_input("hello")).safe


async def test_guard_self_harm_flag(fake_llm):
    guard = LlamaGuard(fake_llm(completion="unsafe\nS11"), "guard")
    r = await guard.check_input("...")
    assert r.self_harm and r.to_dict()["category_names"] == ["Self-Harm"]


# ------------------------------------------------------------------ crisis net
@pytest.mark.parametrize("text", [
    "I want to kill myself",
    "i've been thinking about suicide",
    "I don't want to live anymore",
    "they'd be better off without me",
    "I keep cutting my arms",
    "I want to end my life",
])
def test_crisis_positive(text):
    assert detect_crisis(text)


@pytest.mark.parametrize("text", [
    "I'm anxious about my exam",
    "this deadline is killing me",
    "I cut my hair yesterday",
    "my phone battery died",
])
def test_crisis_negative(text):
    assert not detect_crisis(text)


def test_crisis_message_has_helplines():
    assert "14416" in crisis_message("IN")
    assert "988" in crisis_message("US")
    assert "findahelpline" in crisis_message("XX")
