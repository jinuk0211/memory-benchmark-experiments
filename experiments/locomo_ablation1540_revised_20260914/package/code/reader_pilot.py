"""Compare two frozen reader prompts on 300 archived, unchanged contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "vendor"))
sys.path.insert(0, str(ROOT / "vendor/source"))
import refine as core  # noqa: E402


def read(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.inputs / "contexts.json")
    prompts = read(args.inputs / "reader_prompts.json")
    if len(rows) != 300 or len({row["id"] for row in rows}) != 300:
        raise ValueError("Expected 300 unique frozen questions")
    if set(prompts) != {"legacy", "grounded"}:
        raise ValueError("Expected exactly two frozen reader candidates")
    for row in rows:
        if "gold" in row or "answer" in row or core.digest(row["context"]) != row["context_sha256"]:
            raise ValueError("Context integrity or answer separation failed")
    contract = {
        "population": "previously_exposed_dev300", "context_sha256": sha(args.inputs / "contexts.json"),
        "prompts": prompts, "seed": 20260907, "output_cap": 96,
        "script_sha256": sha(Path(__file__)),
        "runtime_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "vendor").rglob("*.py"))},
        "environment": read(ROOT / "environment.json"),
        "selection_rule": "Highest dev300 Ours overall official F1; ties prefer legacy",
    }
    path = args.out / "protocol.json"
    if path.exists() and read(path) != contract:
        raise ValueError("Pilot contract changed; use a new output directory")
    core.save(path, contract)
    from transfer_runtime import Runtime
    rt = Runtime(SimpleNamespace(stage="evaluate", out=args.out / "runtime", model="Qwen/Qwen3.5-9B",
                 embed_model="sentence-transformers/all-MiniLM-L6-v2", embed_batch_size=64, seed=20260907),
                 contract["environment"])
    for row in rows:
        if rt.ntok(row["context"]) != row["read_tokens"] or row["read_tokens"] > 2048:
            raise ValueError("Frozen evidence-token count changed")
    users = [f'Conversation memory:\n{row["context"]}\n\nQuestion: {row["question"]}\nAnswer:' for row in rows]
    for name in ("legacy", "grounded"):
        print(f"PILOT_START {name} n={len(rows)}", flush=True)
        answers = rt.generate(prompts[name], users, max_tokens=96)
        if len(answers) != len(rows):
            raise ValueError("Missing native answers")
        result = []
        for row, user, answer in zip(rows, users, answers):
            key = core.digest([rt.model_meta, rt.args.seed, prompts[name], user, 96, False])
            native = read(rt.cache / "generations" / f"{key}.json")
            if native["text"] != answer or not isinstance(answer, str):
                raise ValueError("Native response integrity failed")
            chat = rt.tok.apply_chat_template(
                [{"role": "system", "content": prompts[name]}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if (native["finish_reason"] not in {"stop", "length"}
                    or type(native["output_tokens"]) is not int
                    or not 0 <= native["output_tokens"] <= 96
                    or type(native["input_tokens"]) is not int
                    or native["input_tokens"] != rt.ntok(chat)):
                raise ValueError("Native stop/token receipt gate failed")
            result.append({**row, "reader": name, "prediction": answer, "generation_cache_key": key,
                           "finish_reason": native["finish_reason"], "input_tokens": native["input_tokens"],
                           "output_tokens": native["output_tokens"]})
        core.save(args.out / f"{name}.json", result)
        print(f"PILOT_COMPLETE {name}", flush=True)
    core.save(args.out / "COMPLETE.json", {"readers": list(prompts), "n": len(rows)})


if __name__ == "__main__":
    main()
