"""HiGMem's existing hierarchy and retrieval with the controlled embedding model."""
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from native_six import EMBEDDING_MODEL


class ProxyEncoder:
    """Replace only the embedding encoder used by native cosine retrieval.

    Native HiGMem passes plain document/query text to encode without a task
    instruction. Preserve that preprocessing and record this choice in the run.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("HiGMem native encoder expects a list of strings")
        vectors = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL, input=batch, dimensions=1024,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in ordered] != list(range(len(batch))):
                raise ValueError("Embedding response coverage mismatch")
            if any(len(item.embedding) != 1024 for item in ordered):
                raise ValueError("Embedding dimension mismatch")
            vectors.extend(item.embedding for item in ordered)
        return np.asarray(vectors, dtype=np.float32).reshape((-1, 1024))


def create_system(llm: Any, embedding_client: Any, qid: str, log_dir: Path) -> Any:
    """Use the existing reviewed HiGMem variant; never load its old MiniLM model."""
    import memory_layer
    from run_vast import StrictFPHM

    original_factory = memory_layer._get_sentence_transformer
    encoder = ProxyEncoder(embedding_client)

    def controlled_factory(model_name: str, device: str | None = None) -> ProxyEncoder:
        if model_name != "all-MiniLM-L6-v2" or device is not None:
            raise ValueError("Unexpected HiGMem encoder initialization")
        return encoder

    memory_layer._get_sentence_transformer = controlled_factory
    try:
        return StrictFPHM(
            SimpleNamespace(llm=llm), qid, use_character_profile=False,
            use_event_metadata_mode=True, ablation_no_link=True,
            k_event_affiliation=10, log_dir=str(log_dir),
        )
    finally:
        memory_layer._get_sentence_transformer = original_factory


def source_turns(sessions: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    """Provide every original turn in original order, including assistant turns."""
    return [(f"session{si}_turn{ti}", turn["content"], turn["role"], session["date"])
            for si, session in enumerate(sessions)
            for ti, turn in enumerate(session["turns"])]


def retrieve_candidates(system: Any, question: str, question_date: str) -> tuple[list[str], dict]:
    """Keep native query rewriting, event/turn selection and chronological order."""
    import prompts

    query_text = f"Question date: {question_date}\nQuestion: {question}"
    query = system._get_llm_json_response(
        prompts.QUERY_REWRITING_PROMPT.format(original_query=query_text),
        {"name": "response", "schema": {"type": "object", "properties": {
            "keyword_query": {"type": "string"},
            "profile_retrieval_keys": {"type": "array", "items": {"type": "string"}}},
            "required": ["keyword_query", "profile_retrieval_keys"]}},
        caller="generate_keyword_query",
    )
    if not query["keyword_query"]:
        raise ValueError("HiGMem produced an empty search query")
    context, trace = system.retrieve_for_query(
        query_text, query["keyword_query"], [], 0, 10, 10, return_trace=True,
    )
    if trace["mode"] != "full":
        raise ValueError("Unexpected HiGMem retrieval variant")
    turns = [system.turn_notes[tid] for tid in trace["relevant_turn_ids"]
             if tid in system.turn_notes]
    try:
        turns.sort(key=lambda turn: turn.timestamp)
    except (TypeError, ValueError):
        turns.sort(key=lambda turn: turn.id)
    parts = [
        f"--- Turn Start ---\nTimestamp: {turn.timestamp}\nSpeaker: {turn.speaker}\n"
        f"Content: {turn.content}\nContext Summary: {turn.context}\n--- Turn End ---"
        for turn in turns
    ]
    if "\n\n".join(parts) != context:
        raise ValueError("HiGMem native context formatting changed")
    return parts, trace