"""Exercise native E-Mem requests through the real OpenAI serializer, offline."""

import importlib.util
import json
from pathlib import Path

import httpx
import openai
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "methods/e-mem/src/agent/base.py"
SPEC = importlib.util.spec_from_file_location("emem_transport_base", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}},
    "required": ["query"],
}}}]
TOOL_MESSAGE = {"role": "assistant", "content": None, "tool_calls": [{
    "id": "call_test", "type": "function", "function": {
        "name": "lookup", "arguments": '{"query":"original question"}',
    },
}]}


class Agent(MODULE.BaseAgent):
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        assert tool_name == "lookup"
        assert arguments == {"query": "original question"}
        return "original tool result"


@pytest.fixture
def client_factory():
    clients = []

    def make(messages, status=200):
        requests = []
        responses = iter(messages)

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(status, json={
                "id": "response_test", "object": "chat.completion", "created": 0,
                "model": "Qwen/Qwen3.5-9B", "choices": [{
                    "index": 0, "message": next(responses), "finish_reason": "stop",
                }],
            })

        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        agent = Agent({"api_key": "offline-test", "base_url": "https://test.invalid/v1",
                       "model": "Qwen/Qwen3.5-9B", "http_client": client, "max_retries": 0},
                      system_prompt="original system prompt")
        return agent, requests

    yield make
    for client in clients:
        client.close()


@pytest.mark.parametrize("tools", [None, []])
def test_no_tools_omits_field_and_preserves_generation(client_factory, tools):
    agent, requests = client_factory([{"role": "assistant", "content": "original answer"}])
    assert agent.generate_response("original question", tools=tools, max_tokens=713,
                                   temperature=0.2, repetition_penalty=1.1) == "original answer"
    assert requests == [{"model": "Qwen/Qwen3.5-9B", "messages": [
        {"role": "system", "content": "original system prompt"},
        {"role": "user", "content": "original question"},
    ], "max_tokens": 713, "temperature": 0.2, "repetition_penalty": 1.1}]
    assert agent.messgages == [{"role": "system", "content": "original system prompt"}]


@pytest.mark.parametrize("max_rounds, second_has_tools", [(5, True), (1, False)])
def test_nonempty_tools_and_native_round_behavior_preserved(client_factory, max_rounds, second_has_tools):
    agent, requests = client_factory([TOOL_MESSAGE, {"role": "assistant", "content": "answer"}])
    assert agent.generate_response("original question", tools=TOOLS, max_tool_rounds=max_rounds) == "answer"
    assert len(requests) == 2
    assert requests[0]["tools"] == TOOLS
    assert ("tools" in requests[1]) is second_has_tools
    if second_has_tools:
        assert requests[1]["tools"] == TOOLS
    assert requests[1]["messages"][-1] == {
        "role": "tool", "tool_call_id": "call_test", "name": "lookup", "content": "original tool result",
    }


def test_api_error_still_propagates(client_factory):
    agent, requests = client_factory([{"role": "assistant", "content": "rejected"}], status=400)
    with pytest.raises(openai.BadRequestError):
        agent.generate_response("original question")
    assert len(requests) == 1
