"""Strict repair variant: validate evolution before applying any memory changes.

The initial native request is unchanged. Exact tag duplicates are rejected, not
silently removed or capped; this extra strictness is not an upstream guarantee.
"""

import json
import logging
import os
from collections import Counter
from collections.abc import Callable


def _validate_response(response: str, indices: list[int]) -> None:
    if not isinstance(response, str):
        raise ValueError("Evolution response must be JSON text")  # noqa: TRY004 - invalid model output
    data = json.loads(response)
    fields = {"should_evolve", "actions", "suggested_connections", "tags_to_update",
              "new_context_neighborhood", "new_tags_neighborhood"}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError("Evolution response must contain exactly the six native fields")
    if type(data["should_evolve"]) is not bool:
        raise ValueError("should_evolve must be a boolean")
    actions = data["actions"]
    if not isinstance(actions, list) or any(x not in ("strengthen", "update_neighbor") for x in actions):
        raise ValueError("actions must contain only strengthen or update_neighbor")
    strengthen = data["should_evolve"] and "strengthen" in actions
    update_neighbor = data["should_evolve"] and "update_neighbor" in actions
    connections = data["suggested_connections"]
    if not isinstance(connections, list) or any(type(x) is not int for x in connections):
        raise ValueError("suggested_connections must contain integer indices")
    if strengthen and any(x not in indices for x in connections):
        raise ValueError("suggested_connections must contain provided global neighbor indices")
    contexts, neighbors = data["new_context_neighborhood"], data["new_tags_neighborhood"]
    if (not isinstance(contexts, list) or len(contexts) != len(indices)
            or any(not isinstance(x, str) for x in contexts)):
        raise ValueError("new_context_neighborhood must have one string per neighbor in input order")
    if not isinstance(neighbors, list) or len(neighbors) != len(indices):
        raise ValueError("new_tags_neighborhood must have one tag array per neighbor in input order")
    for tags in [data["tags_to_update"], *neighbors]:
        if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
            raise ValueError("Every tag array must contain strings only")
    consumed_tags = ([("tags_to_update", data["tags_to_update"])] if strengthen else [])
    if update_neighbor:
        consumed_tags += [(f"new_tags_neighborhood[{index}]", tags) for index, tags in enumerate(neighbors)]
    for field, tags in consumed_tags:
        duplicates = [tag for tag, count in Counter(tags).items() if count > 1]
        if duplicates:
            raise ValueError(f"{field} repeats exact tags {json.dumps(duplicates)}; emit each only once")


def _length_failure(error: Exception) -> bool:
    body = getattr(error, "body", None)
    return (getattr(error, "status_code", None) == 502 and isinstance(body, dict)
            and body.get("type") == "comparison_incomplete_output"
            and body.get("finish_reasons") == ["length"])


def evolution_completion(get_completion: Callable[..., str], indices: list[int],
                         prompt: str, **kwargs) -> str:
    """Keep the first request intact; allow two metered semantic corrections."""
    if os.getenv("BASELINE_STRICT_COMPARISON") != "1":
        return get_completion(prompt, **kwargs)
    ordered_indices = [int(index) for index in indices]
    current_prompt = prompt
    for attempt in range(3):
        try:
            response = get_completion(current_prompt, **kwargs)
        except Exception as error:
            if not _length_failure(error):
                raise
            failure = error
            reason = "The previous generation reached the context limit without finishing."
        else:
            try:
                _validate_response(response, ordered_indices)
            except ValueError as error:
                failure = error
                reason = str(error)
                if os.getenv("AMEM_LOG_INVALID_EVOLUTION") == "1":
                    logging.getLogger(__name__).warning("A-MEM rejected evolution JSON: %s", response)
            else:
                return response
        if attempt == 2:
            raise failure
        logging.getLogger(__name__).warning("A-MEM evolution correction %s/2: %s", attempt + 1, reason)
        current_prompt = (prompt + "\n\nCorrection required: " + reason
            + f"\nReturn the complete JSON object. The {len(ordered_indices)} input neighbors, in order, "
            + f"have global indices {ordered_indices}. Both neighborhood arrays must have exactly "
            + f"{len(ordered_indices)} entries in that order. Keep all relevant information. "
            + "Within each tag list, use each exact tag only once; different neighbors may share tags. "
            + "Do not repeat tag sequences. Follow the original schema and finish the JSON object.")
