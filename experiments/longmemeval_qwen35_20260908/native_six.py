"""Source-only ingestion and native retrieval for the six MemoryData adapters.

Run one history in one process. AgentWrapper owns native initialization and
persistence; the common reader is called only after retrieve_candidates().
"""
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SIX_METHODS = ("e_mem", "simplemem", "mem0", "langmem", "lightmem_direct", "a_mem")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def source_chunks(sessions: list[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    """Match the existing LME loader's 4096-character, complete-turn chunks."""
    for session in sessions:
        header = f'Session {session["session_id"]} ({session["date"]})\n'
        lines, size = [], 0
        for turn in session["turns"]:
            line = f'{turn["role"]}: {turn["content"]}'
            if lines and size + len(line) > 4096:
                yield session["date"], header + "\n".join(lines)
                lines, size = [], 0
            lines.append(line)
            size += len(line)
        if lines:
            yield session["date"], header + "\n".join(lines)



def process_environment(state: Path, llm_url: str, embedding_url: str) -> dict[str, str]:
    """Overrides required in each isolated history subprocess, before imports."""
    from controlled_reader import MODEL

    return {
        "OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": llm_url,
        "EMBEDDING_BASE_URL": embedding_url,
        "SIMPLEMEM_API_KEY": "EMPTY", "SIMPLEMEM_EMBEDDING_API_KEY": "EMPTY",
        "LIGHTMEM_MODEL": MODEL, "LIGHTMEM_BASE_URL": llm_url,
        "LIGHTMEM_EMBEDDING_MODEL": EMBEDDING_MODEL,
        "LIGHTMEM_EMBEDDING_DIMENSION": "1024",
        "MEM0_DIR": str(state / "mem0_home"),
        "BASELINE_STRICT_COMPARISON": "1",
    }
def build_config(
    method: str, state: Path, tokenizer: Path, llm_url: str, embedding_url: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse native settings and explicitly select the controlled environment."""
    from scripts.run_six_baselines import build_agent_config
    from controlled_reader import MODEL

    if method not in SIX_METHODS:
        raise ValueError(f"Unknown method: {method}")
    native_method = "lightmem" if method == "lightmem_direct" else method
    args = SimpleNamespace(model=MODEL, base_url=llm_url,
                           embedding_base_url=embedding_url,
                           embedding_model=EMBEDDING_MODEL, embedding_dim=1024)
    config = build_agent_config(native_method, args)
    config.update(
        artifact_root=str(state.parent), output_dir=str(state.parent / "outputs"),
        tokenizer_model=str(tokenizer), tokenizer_encoding=None,
        e_mem_tokenizer_model=str(tokenizer), e_mem_memory_model=MODEL,
        lightmem_ingest_mode="direct", lightmem_messages_use="hybrid",
        record_llm_io=True,
    )
    dataset = dict(
        dataset="LongMemEval", sub_dataset="longmemeval_s_official",
        context_max_length=10000000, generation_max_length=96, chunk_size=4096,
    )
    return config, dataset


def ingest_chunk(agent: Any, method: str, date: str, text: str, context_id: str) -> None:
    """Present full source chunks to native builders, with original session time."""
    if method in ("e_mem", "langmem"):
        agent.benchmark_memory.add_chunk(text)
    elif method == "simplemem":
        agent.simplemem.add_chunk(text, timestamp=date)
    elif method == "lightmem_direct":
        agent.lightmem.add_chunk(text, timestamp=date)
    elif method == "a_mem":
        stamp = datetime.strptime(date, "%Y/%m/%d (%a) %H:%M").strftime("%Y%m%d%H%M")
        agent.a_mem.add_chunk(text, timestamp=stamp)
    elif method == "mem0":
        agent._handle_mem0_agent(text, memorizing=True, query_id=None, context_id=context_id)
    else:
        raise ValueError(f"Unknown method: {method}")


def a_mem_candidates(adapter: Any, query: str) -> list[str]:
    """Collect native formatted notes, including links, without changing selection.

    retrieve_items() excludes linked neighbors and therefore is not equivalent.
    This temporary instance formatter records exactly the native formatted units.
    The enclosing process performs only one retrieval at a time.
    """
    parts: list[str] = []
    original = adapter._format_memory_note
    had_instance_override = "_format_memory_note" in vars(adapter)
    prior_override = vars(adapter).get("_format_memory_note")

    def record_note(note: Any) -> str:
        text = original(note)
        parts.append(text)
        return text

    adapter._format_memory_note = record_note
    try:
        joined, _ = adapter.retrieve_with_source_groups(query)
    finally:
        if had_instance_override:
            adapter._format_memory_note = prior_override
        else:
            del adapter._format_memory_note
    if "\n".join(parts) != joined:
        raise ValueError("A-MEM native formatting changed; refusing mismatched context units")
    return parts


def retrieve_candidates(
    agent: Any, method: str, question: str, question_date: str, context_id: str,
) -> list[str]:
    """Retrieve with the native algorithm; no target answer/type/evidence input."""
    query = f"Question date: {question_date}\nQuestion: {question}"
    if method in ("e_mem", "langmem"):
        contexts = agent.benchmark_memory.retrieve(query)
    elif method == "simplemem":
        contexts = [entry["text"] for entry in agent.simplemem.retrieve_entries(query)
                    if entry.get("text")]
    elif method == "lightmem_direct":
        contexts = agent._flatten_text_items(agent.lightmem.retrieve(query))
    elif method == "a_mem":
        contexts = a_mem_candidates(agent.a_mem, query)
    elif method == "mem0":
        result = agent.memory.search(
            query=query, user_id=f"context_{context_id}_{agent.sub_dataset}",
            limit=agent.retrieve_num,
        )
        contexts = []
        for entry in agent._normalize_mem0_search_results(result):
            text = entry.get("memory") or entry.get("text") or entry.get("content") or ""
            if isinstance(text, str) and text.strip():
                contexts.append(text)
    else:
        raise ValueError(f"Unknown method: {method}")
    if not isinstance(contexts, list) or not all(isinstance(text, str) for text in contexts):
        raise TypeError("Native retrieval must return a list of text units")
    return contexts