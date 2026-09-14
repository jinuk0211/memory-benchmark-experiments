"""Validate a whole Mem0 action plan before applying any storage mutation."""
import json
import os
from pathlib import Path
import time
import uuid


def plan_errors(plan, mapping):
    if not isinstance(plan, dict) or not isinstance(plan.get("memory"), list):
        return ["Response must be an object containing the complete memory list."]
    errors, deleted = [], set()
    for index, action in enumerate(plan["memory"]):
        if not isinstance(action, dict):
            errors.append(f"memory[{index}] must be an object.")
            continue
        event = action.get("event")
        if not isinstance(event, str) or event not in {"ADD", "UPDATE", "DELETE", "NONE"}:
            errors.append(f"memory[{index}].event is missing or invalid; choose ADD, UPDATE, DELETE, or NONE.")
            continue
        if event != "NONE" and (not isinstance(action.get("text"), str) or not action["text"].strip()):
            errors.append(f"memory[{index}].text must be a nonempty string.")
        if event in {"UPDATE", "DELETE"}:
            key = str(action.get("id"))
            if key not in mapping:
                errors.append(f"memory[{index}].id {key!r} is not an existing memory ID.")
            if key in deleted:
                errors.append(f"memory[{index}].id {key!r} was already deleted earlier in this plan.")
            if event == "DELETE":
                deleted.add(key)
    return errors


def _journal(record):
    print("[mem0 plan validation] " + json.dumps(record, ensure_ascii=False), flush=True)
    auxiliary = os.getenv("METER_AUXILIARY_JOURNAL")
    if auxiliary:
        path = Path(auxiliary).with_name("mem0_action_repairs.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def validated_plan(plan, mapping, prompt, generate, clean):
    """Keep valid initial plans unchanged; explicitly repair at most twice."""
    group = uuid.uuid4().hex
    for attempt in range(3):
        errors = plan_errors(plan, mapping)
        if not errors:
            if attempt:
                _journal({"group": group, "attempt": attempt, "time": time.time(),
                          "accepted": True, "plan": plan})
            for action in plan["memory"]:
                if action["event"] in {"UPDATE", "DELETE"}:
                    action["id"] = str(action["id"])
            return plan
        _journal({"group": group, "attempt": attempt, "time": time.time(),
                  "accepted": False, "errors": errors, "plan": plan})
        if attempt == 2:
            raise ValueError("Mem0 action plan remains invalid after two corrections: " + "; ".join(errors))
        feedback = (
            "The complete plan failed validation. No actions have been applied. "
            + " ".join(errors)
            + " Valid existing memory IDs: " + json.dumps(list(mapping))
            + ". Return the complete corrected JSON memory action plan, preserving all facts. "
              "Every entry needs an explicit event chosen from ADD, UPDATE, DELETE, NONE. "
              "Use only existing IDs for UPDATE or DELETE. Do not omit entries to hide errors."
        )
        corrected = generate(messages=[{"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
            {"role": "user", "content": feedback}], response_format={"type": "json_object"})
        try:
            plan = json.loads(clean(corrected))
        except (ValueError, TypeError):
            plan = {"invalid_response": corrected}
        if isinstance(plan, list):
            plan = {"memory": plan}
        elif isinstance(plan, dict) and "memory" not in plan and "memories" in plan:
            plan = dict(plan, memory=plan["memories"])
