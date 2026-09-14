"""Candidate-only JSONL hook; importing this file does not modify any runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

JOURNAL_NAME = "simplemem_location_normalization.jsonl"


def record_location_normalization(attempt_dir: Path, event: dict[str, Any]) -> None:
    """Append original response and location conversion to this attempt's journal.

    The attempt directory must already exist. Failure propagates before the
    candidate parser returns a normalized entry; no silent unaudited conversion.
    """
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with (Path(attempt_dir) / JOURNAL_NAME).open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())
