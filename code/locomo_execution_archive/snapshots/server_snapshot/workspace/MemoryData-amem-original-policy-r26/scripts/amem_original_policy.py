"""Opt-in original A-MEM request/failure semantics with response-linked outcomes.

Upstream: WujiangXu/A-mem 0c8039f28fdcc08189a23c07a3437d9d2482f9c2.
The comparison still supplies Qwen FP16, MiniLM and temperature zero; this is
not a claim to reproduce upstream's default model or sampling distribution.
"""
import copy
import hashlib
import json
import os
import threading
from pathlib import Path

POLICY = 'amem_original_failure_v1'
UPSTREAM_SCHEMA_SHA256 = '5223745074b0dab699e769148f74449565ecebf332167c905605b7bfda861520'
REMOVED_LINES = (
    '                                Include each distinct tag only once. Do not repeat tags or contexts.\n',
    '                                Emit each action at most once and only reference input neighbor indices.\n',
)
_JOURNAL_LOCK = threading.Lock()


def request_stage(payload: dict) -> str | None:
    """Recognize only the two native schema/prompt pairs, not QA or other calls."""
    messages = payload.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or messages[0] != {'role': 'system', 'content': 'You must respond with a JSON object.'}
            or messages[1].get('role') != 'user' or not isinstance(messages[1].get('content'), str)):
        return None
    prompt = messages[1]['content'].lstrip()
    schema = payload.get('response_format')
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if (digest == UPSTREAM_SCHEMA_SHA256 and prompt.startswith(
            'You are an AI memory evolution agent responsible for managing and evolving a knowledge base.')
            and not any(line.strip() in prompt for line in REMOVED_LINES)):
        return 'evolution'
    string = {'type': 'string'}
    metadata = {'type': 'json_schema', 'json_schema': {'name': 'response', 'strict': True, 'schema': {
        'type': 'object', 'properties': {'keywords': {'type': 'array', 'items': string},
        'context': string, 'tags': {'type': 'array', 'items': string}},
        'required': ['keywords', 'context', 'tags'], 'additionalProperties': False}}}
    if schema == metadata and prompt.startswith('Generate a structured analysis of the following content by:'):
        return 'metadata'
    return None


def original_completion(get_completion, indices, prompt, **kwargs):
    """Remove exactly the adapter additions; preserve facts, order and duplicates."""
    lines = prompt.splitlines(keepends=True)
    if any(lines.count(line) != 1 for line in REMOVED_LINES):
        raise ValueError('Original policy requires exactly the two known adapter instruction lines')
    restored = ''.join(line for line in lines if line not in REMOVED_LINES)
    options = copy.deepcopy(kwargs)
    schema = options['response_format']
    properties = schema['json_schema']['schema']['properties']
    expected = {'actions.maxItems': 2, 'actions.items.enum': ['strengthen', 'update_neighbor'],
                'suggested_connections.maxItems': len(indices), 'new_context_neighborhood.minItems': len(indices),
                'new_context_neighborhood.maxItems': len(indices), 'new_tags_neighborhood.minItems': len(indices),
                'new_tags_neighborhood.maxItems': len(indices)}
    for path, value in expected.items():
        parts, container = path.split('.'), properties
        for part in parts[:-1]:
            container = container[part]
        if container.pop(parts[-1]) != value:
            raise ValueError('Unexpected adapter schema constraint')
    if hashlib.sha256(json.dumps(schema, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != UPSTREAM_SCHEMA_SHA256:
        raise ValueError('Original evolution schema SHA256 mismatch')
    options['max_tokens'] = 1000  # Native generation budget, never input truncation.
    return get_completion(restored, **options)


def capture_response(controller, response) -> None:
    """Bind a returned native response to its current operation, without tokens."""
    from utils.request_metering import current_operation_snapshot

    snapshot = current_operation_snapshot()
    if (snapshot is None or snapshot.phase != 'memory_add' or not snapshot.sample_id
            or not os.getenv('METER_RUN_ID') or os.getenv('METER_METHOD') != 'a_mem'
            or not os.getenv('AMEM_SEMANTIC_JOURNAL_DIR')):
        raise RuntimeError('Original failure policy requires attributed semantic journaling')
    if (not isinstance(response.id, str) or not response.id or len(response.choices) != 1
            or response.model != 'Qwen/Qwen3.5-9B'
            or response.choices[0].finish_reason not in ('stop', 'length')
            or not isinstance(response.choices[0].message.content, str)):
        raise ValueError('Original failure policy requires one identifiable raw stop/length response')
    controller._amem_original_response = {
        'schema_version': 1, 'policy': POLICY, 'run_id': os.environ['METER_RUN_ID'], 'method': 'a_mem',
        'operation_id': snapshot.operation_id, 'phase': snapshot.phase, 'sample_id': snapshot.sample_id,
        'question_id': snapshot.question_id, 'response_id': response.id, 'response_model': response.model,
        'finish_reason': response.choices[0].finish_reason,
        'content_sha256': hashlib.sha256(response.choices[0].message.content.encode()).hexdigest()}


def record_outcome(controller, stage: str, outcome: str, **counts) -> None:
    """One fsynced semantic row per response; one file per process avoids interleaving."""
    metadata = getattr(controller, '_amem_original_response', None)
    if not isinstance(metadata, dict):
        raise RuntimeError('No unconsumed native response attribution for semantic outcome')  # noqa: TRY004 - missing runtime state
    row = {**metadata, 'stage': stage, 'outcome': outcome, **counts}
    folder = Path(os.environ['AMEM_SEMANTIC_JOURNAL_DIR'])
    folder.mkdir(parents=True, exist_ok=True)
    with _JOURNAL_LOCK, (folder / f'{os.getpid()}.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    controller._amem_original_response = None


def apply_evolution(memory, note, data, indices):
    """Original process_memory L828-857 semantics; no added validation or repair."""
    should_evolve = data['should_evolve']
    applied = context_fallbacks = 0
    strengthened = False
    if should_evolve:
        for action in data['actions']:
            if action == 'strengthen':
                note.links.extend(data['suggested_connections'])
                note.tags = data['tags_to_update']
                strengthened = True
            elif action == 'update_neighbor':
                contexts, tags = data['new_context_neighborhood'], data['new_tags_neighborhood']
                notes, ids = list(memory.memories.values()), list(memory.memories.keys())
                for i in range(min(len(indices), len(tags))):
                    neighbor = notes[indices[i]]
                    context = contexts[i] if i < len(contexts) else neighbor.context
                    context_fallbacks += i >= len(contexts)
                    neighbor.tags, neighbor.context = tags[i], context
                    memory.memories[ids[indices[i]]] = neighbor
                    applied += 1
    record_outcome(memory.llm_controller.llm, 'evolution',
                   'evolution_applied' if strengthened or applied else 'evolution_noop',
                   neighbor_count=len(indices), applied_neighbor_count=applied,
                   context_fallback_count=context_fallbacks)
    return should_evolve, note
