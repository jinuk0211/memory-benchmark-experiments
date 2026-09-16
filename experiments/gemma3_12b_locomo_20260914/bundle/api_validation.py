"""Validate exact context lengths and local OpenAI-compatible API behavior."""
import json
from pathlib import Path

from full_context_runner import build_context
from openai import OpenAI
import requests


ROOT = Path("/workspace/gemma3_locomo_51017220")
DATA = ROOT / "bundle/MemoryData/datasets/LoCoMo/locomo10.json"


def main() -> None:
    dataset = json.loads(DATA.read_text(encoding="utf-8"))
    rows = []
    newline = chr(10)
    for sample in dataset:
        context = build_context(sample)
        question = sample["qa"][0]["question"]
        prompt = f"Memory notes:{newline}{context}{newline}{newline}Question: {question}{newline}Answer:"
        response = requests.post(
            "http://127.0.0.1:8000/tokenize",
            json={"content": prompt, "add_special": False, "with_pieces": False},
            timeout=120,
        )
        response.raise_for_status()
        rows.append((sample["sample_id"], len(response.json()["tokens"])))
    print("TOKEN_COUNTS", sorted(rows, key=lambda item: item[1]))
    print("MAX_EXACT_RAW", max(value for _, value in rows))

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY", timeout=600)
    plain = client.chat.completions.create(
        model="gemma-3-12b-it",
        messages=[{"role": "user", "content": "Reply with exactly READY"}],
        temperature=0,
        max_tokens=8,
    )
    print("CHAT", repr(plain.choices[0].message.content), plain.usage)
    tooled = client.chat.completions.create(
        model="gemma-3-12b-it",
        messages=[
            {
                "role": "user",
                "content": "Remember that my favorite color is blue. Use the save_memory tool.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save one memory",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                        "required": ["text"],
                    },
                },
            }
        ],
        tool_choice="required",
        temperature=0,
        max_tokens=64,
    )
    print("TOOLS", tooled.choices[0].message.model_dump_json())


if __name__ == "__main__":
    main()