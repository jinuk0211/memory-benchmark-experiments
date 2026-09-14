"""A recorded location-list compatibility overlay; upstream files remain unchanged."""
import ast
import json
import os
from pathlib import Path
import threading
import types


def install(builder, upstream: Path, audit: Path):
    lock = threading.Lock()
    audit.touch(exist_ok=False)

    def normalize(value, response, dialogue_ids):
        if not isinstance(value, list):
            return value
        if not all(isinstance(item, str) for item in value):
            raise ValueError('location array must contain strings only')
        normalized = ', '.join(value)
        record = {'policy': 'location-list-str-join-v1', 'dialogue_ids': dialogue_ids,
                  'original_location': value, 'normalized_location': normalized,
                  'raw_response': response}
        with lock, audit.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return normalized

    tree = ast.parse(upstream.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MemoryBuilder')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_parse_llm_response')
    changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'MemoryEntry':
            for keyword in node.keywords:
                if keyword.arg == 'location':
                    keyword.value = ast.Call(func=ast.Name(id='_normalize_location', ctx=ast.Load()),
                        args=[keyword.value, ast.Name(id='response', ctx=ast.Load()),
                              ast.Name(id='dialogue_ids', ctx=ast.Load())], keywords=[])
                    changed += 1
    if changed != 1:
        raise ValueError('Expected exactly one pinned location parse expression')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(builder._parse_llm_response.__func__.__globals__)
    namespace['_normalize_location'] = normalize
    exec(compile(module, str(upstream) + ':location-list-overlay-v1', 'exec'), namespace)
    builder._parse_llm_response = types.MethodType(namespace['_parse_llm_response'], builder)
