"""A recorded location-list compatibility overlay; upstream files remain unchanged."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import threading
import types


def install(builder, upstream: Path, audit: Path):
    lock = threading.Lock()
    audit.touch(exist_ok=False)
    syntax_audit = audit.with_name('json_syntax_compat_audit.jsonl')
    syntax_audit.touch(exist_ok=False)
    spec = importlib.util.spec_from_file_location('simplemem_json_syntax_v4', Path(__file__).with_name('json_syntax_compat.py'))
    syntax = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(syntax)

    def extract_array(response, dialogue_ids):
        try:
            original = builder.llm_client.extract_json(response)
        except ValueError:
            original = None
        if isinstance(original, list):
            return original
        repaired, changes = syntax.repair_locations(response)
        # The native extractor searches objects before arrays. Require the entire first
        # balanced array and never accept its first object as a complete memory batch.
        start = repaired.find('[')
        if start < 0:
            raise ValueError('No complete memory array')
        data, end = json.JSONDecoder().raw_decode(repaired, start)
        if not isinstance(data, list) or repaired[end:].strip() not in ('', '```'):
            raise ValueError('Complete memory JSON array could not be recovered losslessly')
        record = {'policy': 'complete-array-and-bare-location-strings-v1',
                  'dialogue_ids': dialogue_ids, 'raw_response': response,
                  'repaired_response': repaired, 'location_repairs': changes,
                  'memory_entries': len(data)}
        with lock, syntax_audit.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return data

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
    extract_changed = 0
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'extract_json':
            node.func = ast.Name(id='_extract_memory_array', ctx=ast.Load())
            node.args.append(ast.Name(id='dialogue_ids', ctx=ast.Load()))
            extract_changed += 1
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'MemoryEntry':
            for keyword in node.keywords:
                if keyword.arg == 'location':
                    keyword.value = ast.Call(func=ast.Name(id='_normalize_location', ctx=ast.Load()),
                        args=[keyword.value, ast.Name(id='response', ctx=ast.Load()),
                              ast.Name(id='dialogue_ids', ctx=ast.Load())], keywords=[])
                    changed += 1
    if changed != 1 or extract_changed != 1:
        raise ValueError('Expected exactly one pinned location parse expression')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(builder._parse_llm_response.__func__.__globals__)
    namespace['_normalize_location'] = normalize
    namespace['_extract_memory_array'] = extract_array
    exec(compile(module, str(upstream) + ':location-list-overlay-v1', 'exec'), namespace)
    builder._parse_llm_response = types.MethodType(namespace['_parse_llm_response'], builder)
