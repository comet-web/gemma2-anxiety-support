from anxiety_support.agent.cbt_agent import CBTAgent
from anxiety_support.agent.state import Phase, Session, SessionStore, ThoughtRecord


def _session(text="I failed one test so I'm stupid at everything"):
    return Session(id="s", messages=[{"role": "user", "content": text}])


async def test_plan_tool_then_respond(fake_llm):
    llm = fake_llm(structured_queue=[
        {"phase": "examine_distortion", "tool": "detect_distortions", "tool_input": "I'm stupid at everything"},
        {"distortions": [{"label": "overgeneralization", "evidence": "everything"},
                         {"label": "not-a-real-label", "evidence": "x"}]},
        {"phase": "examine_distortion", "tool": "respond"},
    ], chat_reply="It sounds like one result is speaking for everything. What evidence is there?")
    agent = CBTAgent(llm, "planner", "responder", max_steps=3)
    s = _session()
    res = await agent.respond(s)

    assert res.phase == "examine_distortion" and s.phase == Phase.examine_distortion
    assert [st.tool for st in res.steps] == ["detect_distortions"]
    assert "overgeneralization" in res.steps[0].observation
    assert s.record.distortions == ["overgeneralization"]  # unknown label filtered out
    chat_call = [c for c in llm.calls if c[0] == "chat"][-1]
    assert chat_call[1] == "responder"
    assert "overgeneralization" in chat_call[2][0]["content"]  # observation reaches the responder


async def test_planner_failure_falls_back_to_plain_reply(fake_llm):
    llm = fake_llm(structured_queue=[None], chat_reply="I'm here with you.")
    res = await CBTAgent(llm, "p", "r").respond(_session())
    assert res.reply == "I'm here with you." and res.steps == []


async def test_max_steps_bounds_tool_loop(fake_llm):
    plan = {"phase": "coping_now", "tool": "coping_exercise", "tool_input": "box_breathing"}
    llm = fake_llm(structured_queue=[plan] * 10)
    res = await CBTAgent(llm, "p", "r", max_steps=2).respond(_session("My heart is racing right now"))
    assert len(res.steps) == 2 and "Box breathing" in res.steps[0].observation


async def test_thought_record_tool_merges(fake_llm):
    llm = fake_llm(structured_queue=[
        {"phase": "explore_situation", "tool": "update_thought_record"},
        {"situation": "exam result", "emotion": "shame", "emotion_intensity": 80},
        {"phase": "explore_situation", "tool": "respond"},
    ])
    s = _session()
    s.record.automatic_thought = "I'm stupid"
    await CBTAgent(llm, "p", "r").respond(s)
    assert s.record.situation == "exam result" and s.record.emotion_intensity == 80
    assert s.record.automatic_thought == "I'm stupid"  # not overwritten by null


def test_thought_record_summary():
    assert ThoughtRecord().summary() == "(empty)"
    assert "emotion=fear" in ThoughtRecord(emotion="fear").summary()


def test_session_store_ttl():
    store = SessionStore(ttl_seconds=0)
    s = store.get_or_create(None)
    s.updated_at -= 10
    store.get_or_create("other")
    assert store.get(s.id) is None
