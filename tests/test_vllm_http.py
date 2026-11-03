"""HTTP-level tests against a stub vLLM OpenAI-compatible server running on localhost.

A real socket server is used (rather than request mocking) because the OpenAI SDK ships its
own HTTP stack, and this checks the exact JSON we send to vLLM.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from anxiety_support.agent.cbt_agent import Plan
from anxiety_support.llm_client import OpenAICompatLLM
from anxiety_support.safety.llama_guard import LlamaGuard


@pytest.fixture
def stub_vllm():
    state = {"requests": [], "completion_text": "safe", "chat_content": "ok"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append((self.path, body))
            if self.path.endswith("/chat/completions"):
                resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": state["chat_content"]}}]}
            else:
                resp = {"id": "x", "object": "text_completion", "created": 0, "model": body["model"],
                        "choices": [{"index": 0, "text": state["completion_text"], "finish_reason": "stop"}]}
            data = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["base_url"] = f"http://127.0.0.1:{server.server_address[1]}/v1"
    yield state
    server.shutdown()


async def test_llama_guard_uses_completions_endpoint(stub_vllm):
    stub_vllm["completion_text"] = "unsafe\nS11"
    guard = LlamaGuard(OpenAICompatLLM(stub_vllm["base_url"]), "meta-llama/Llama-Guard-3-1B")
    res = await guard.check_input("I want to disappear")
    path, body = stub_vllm["requests"][-1]
    assert path == "/v1/completions"
    assert body["model"] == "meta-llama/Llama-Guard-3-1B" and body["temperature"] == 0.0
    assert "User: I want to disappear" in body["prompt"]
    assert res.self_harm


async def test_structured_sends_json_schema_and_folds_system(stub_vllm):
    stub_vllm["chat_content"] = '{"phase": "rapport", "tool": "respond"}'
    llm = OpenAICompatLLM(stub_vllm["base_url"])
    plan = await llm.structured("planner", [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}], Plan)
    path, body = stub_vllm["requests"][-1]
    assert path == "/v1/chat/completions"
    assert body["response_format"]["type"] == "json_schema"
    assert "respond" in json.dumps(body["response_format"]["json_schema"]["schema"])
    assert body["messages"] == [{"role": "user", "content": "SYS\n\nhi"}]  # Gemma-2: no system role
    assert plan.tool == "respond"
