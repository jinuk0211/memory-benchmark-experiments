"""Read-only progress snapshot for the running experiment."""
import json
import pickle
import time
from pathlib import Path

root = Path("vast_run")
counts = {}
for path in sorted(root.glob("*.pkl")):
    with path.open("rb") as stream:
        state = pickle.load(stream)
    counts[path.stem] = dict(turns=state["count"], complete=state["complete"], events=len(state["events"]))
usage_path = root / "usage.jsonl"
usage = []
if usage_path.exists():
    for line in usage_path.read_text().splitlines():
        try:
            usage.append(json.loads(line))
        except json.JSONDecodeError:
            pass
predictions_path = root / "predictions.jsonl"
predictions = predictions_path.read_text().splitlines() if predictions_path.exists() else []
recent = [row for row in usage if row["started"] > time.time()-120 and row.get("usage")]
result = dict(checkpoints=counts, turns_checkpointed=sum(row["turns"] for row in counts.values()),
              questions_completed=len(predictions), requests=len(usage),
              failed_requests=sum("error" in row for row in usage),
              truncated=sum(row.get("finish_reason") not in (None, "stop") for row in usage),
              prompt_tokens=sum(row.get("usage", {}).get("prompt_tokens", 0) for row in usage if row.get("usage")),
              completion_tokens=sum(row.get("usage", {}).get("completion_tokens", 0) for row in usage if row.get("usage")),
              recent_output_tokens_per_second=sum(row["usage"]["completion_tokens"] for row in recent)/120)
print(json.dumps(result, indent=2))
