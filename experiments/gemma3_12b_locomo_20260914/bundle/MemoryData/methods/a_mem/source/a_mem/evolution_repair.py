"""Strict repair variant: validate evolution before applying any memory changes.

The initial native request is unchanged. Exact tag duplicates are rejected, not
silently removed or capped; this extra strictness is not an upstream guarantee.
"""

import json
import logging
import os
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
    connections = data["suggested_connections"]
    if not isinstance(connections, list) or any(type(x) is not int or x not in indices for x in connections):
        raise ValueError("suggested_connections must contain provided global neighbor indices")
    contexts, neighbors = data["new_context_neighborhood"], data["new_tags_neighborhood"]
    if not isinstance(contexts, list) or any(not isinstance(x, str) for x in contexts):
        raise ValueError("new_context_neighborhood must be a string list")
    if not isinstance(neighbors, list):
        raise ValueError("new_tags_neighborhood must be a list")
    for tags in [data["tags_to_update"], *neighbors]:
        if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
            raise ValueError("Every tag array must contain strings only")


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
            else:
                return response
        if attempt == 2:
            raise failure
        logging.getLogger(__name__).warning("A-MEM evolution correction %s/2: %s", attempt + 1, reason)
        current_prompt = (prompt + "\n\nCorrection required: " + reason
            + f"\nReturn the complete JSON object. The {len(ordered_indices)} input neighbors, in order, "
            + f"have global indices {ordered_indices}. Both neighborhood arrays "
            + "may omit unchanged trailing neighbors. Keep all relevant information. "
            + "Within each tag list, use each exact tag only once; different neighbors may share tags. "
            + "Do not repeat tag sequences. Follow the original schema and finish the JSON object.")
