"""Short same-model concurrency test and live HiGMem smoke check."""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from openai import OpenAI
from run_vast import Journal, MeteredLLM, StrictFPHM, answer_question
import prompts

root = Path(sys.argv[1] if len(sys.argv) > 1 else "smoke_run")
root.mkdir(exist_ok=True)
args = SimpleNamespace(model="Qwen/Qwen3.5-9B", api_base="http://127.0.0.1:18080/v1")
client = OpenAI(base_url=args.api_base, api_key="EMPTY", timeout=300)
prompt = prompts.CREATE_TURN_NOTE_ISOLATED_PROMPT.format(
    speaker="Alice", timestamp="2024-01-10", content="I bought a blue Toyota yesterday. I use it to drive to my teaching job.")

def request(index):
    response = client.chat.completions.create(model=args.model,
        messages=[{"role": "system", "content": "Return JSON."},
                  {"role": "user", "content": prompt + f"\nRequest number: {index}"}],
        response_format={"type": "json_object"}, temperature=0, max_tokens=512,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    json.loads(response.choices[0].message.content)
    assert response.choices[0].finish_reason == "stop"
    return response.usage.model_dump()

request(-1)
results = []
for concurrency in (1, 8, 16):
    started = time.perf_counter()
    with ThreadPoolExecutor(concurrency) as pool:
        usage = list(pool.map(request, range(16)))
    elapsed = time.perf_counter()-started
    output_tokens = sum(row["completion_tokens"] for row in usage)
    result = dict(concurrency=concurrency, requests=16, seconds=elapsed,
                  output_tokens=output_tokens, output_tokens_per_second=output_tokens/elapsed)
    results.append(result)
    print(json.dumps(result), flush=True)
(root / "benchmark.json").write_text(json.dumps(results, indent=2))
usage = Journal(root / "usage.jsonl")
predictions = Journal(root / "predictions.jsonl")
controller = MeteredLLM(args, usage, "synthetic", "construction")
system = StrictFPHM(SimpleNamespace(llm=controller), "synthetic", use_character_profile=False,
                    use_event_metadata_mode=True, ablation_no_link=True, k_event_affiliation=10,
                    log_dir=str(root / "logs"))
system.add_turn("D1:1", "I bought a blue Toyota yesterday.", "Alice", "2024-01-10")
system.add_turn("D1:2", "I drive it to my teaching job.", "Alice", "2024-01-10")
system.build_indices()
qa = dict(question="What brand of car did Alice buy?", answer="Toyota", category=4, evidence=["D1:1"])
answer_question(args, system, "synthetic", 0, qa, usage, predictions)
system.executor.shutdown()
controller.client.close()
from official_locomo_evaluation import eval_question_answering
rows = [json.loads(line) for line in (root / "predictions.jsonl").read_text().splitlines()]
scores, _, _ = eval_question_answering(rows)
assert scores[-1] == 1.0, rows[-1]
print("LIVE HIGMEM SMOKE COMPLETE", flush=True)
