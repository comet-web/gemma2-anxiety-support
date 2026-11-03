from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

from anxiety_support.llm_client import fold_system_prompt, merge_consecutive, parse_json_model
from anxiety_support.prompts import COUNSELOR_SYSTEM_PROMPT
from chat_template import GEMMA2_TRAIN_TEMPLATE
from prepare_data import client_first_name, is_anxiety_case, parse_dialogue, scrub_name, to_messages


# ------------------------------------------------------------------ CACTUS conversion
def test_parse_dialogue_handles_continuation_lines(cactus_row):
    turns = parse_dialogue(cactus_row["dialogue"])
    assert turns[0]["role"] == "assistant"
    assert turns[3]["content"] == "Some of the animals didn't recognize me. It felt like a punch to the gut."


def test_name_scrub(cactus_row):
    assert client_first_name(cactus_row["intake_form"]) == "Brooke"
    assert scrub_name("Brooke, that sounds painful.", "Brooke") == "That sounds painful."
    assert scrub_name("Let's look at it together, Brooke.", "Brooke") == "Let's look at it together."


def test_to_messages_gemma_constraints(cactus_row):
    msgs = to_messages(cactus_row)
    roles = [m["role"] for m in msgs]
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert all(a != b for a, b in zip(roles, roles[1:])), "roles must alternate"
    assert msgs[0]["content"].startswith(COUNSELOR_SYSTEM_PROMPT)
    assert not any("Brooke" in m["content"] for m in msgs if m["role"] == "assistant")
    assert msgs[-1]["content"].startswith("Let's look at the evidence")  # trailing client turn dropped


def test_anxiety_focus_filter(cactus_row):
    assert is_anxiety_case(cactus_row)
    assert not is_anxiety_case({**cactus_row, "intake_form": "2. Presenting Problem\nI am angry at my brother.\n\n3. x"})


# ------------------------------------------------------------------ chat formatting
def test_fold_system_prompt():
    out = fold_system_prompt([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
    assert out == [{"role": "user", "content": "SYS\n\nhi"}]
    out = fold_system_prompt([{"role": "system", "content": "SYS"}, {"role": "assistant", "content": "a"}])
    assert out[0] == {"role": "user", "content": "SYS"}


def test_merge_consecutive():
    out = merge_consecutive([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}])
    assert out == [{"role": "user", "content": "a\n\nb"}]


def test_parse_json_model_tolerates_fences():
    from anxiety_support.safety.topic import TopicCheck

    assert parse_json_model('```json\n{"on_topic": false}\n```', TopicCheck).on_topic is False
    assert parse_json_model("not json", TopicCheck) is None


class _GenerationExt(Extension):
    """Minimal stand-in for transformers' {% generation %} support: renders the block as-is."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


def test_train_template_matches_stock_gemma2_layout():
    env = ImmutableSandboxedEnvironment(extensions=[_GenerationExt], trim_blocks=True, lstrip_blocks=True)
    msgs = [{"role": "user", "content": "hi "}, {"role": "assistant", "content": " hello"}, {"role": "user", "content": "x"}]
    out = env.from_string(GEMMA2_TRAIN_TEMPLATE).render(messages=msgs, bos_token="<bos>", add_generation_prompt=True)
    assert out == (
        "<bos><start_of_turn>user\nhi<end_of_turn>\n"
        "<start_of_turn>model\nhello<end_of_turn>\n"
        "<start_of_turn>user\nx<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
