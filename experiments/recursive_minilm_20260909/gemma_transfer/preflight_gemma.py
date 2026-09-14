"""CPU-only cached-tokenizer validation of complete initial writer sessions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
from guard_inputs import validate
from transfer_data import load_longmemeval
from transfer_runtime import MODEL, validate_model_config
from chat_tokenizer_compat import ListChatTokenizer
from transformers import AutoTokenizer


def main() -> None:
    data = ROOT / 'data/longmemeval18.json'
    lock = validate(data, 'longmemeval', None)
    environment = json.loads((ROOT / 'environment.json').read_text())
    metadata = environment['models'][MODEL]
    validate_model_config(metadata)
    tok = ListChatTokenizer(AutoTokenizer.from_pretrained(metadata['path'], local_files_only=True))
    counts = []
    for sample in load_longmemeval(data):
        for session in core.session_data(sample):
            user = f"Session date: {session['date']}\nORIGINAL dialogue:\n" + '\n'.join(t['text'] for t in session['turns']) + '\n\nMemory facts:'
            text = tok.apply_chat_template([{'role': 'system', 'content': core.WRITER}, {'role': 'user', 'content': user}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            tokens = len(tok.encode(text, add_special_tokens=False))
            if tokens + 512 > 65536:
                raise ValueError('Complete initial writer session exceeds frozen Gemma capacity')
            counts.append(tokens)
    receipt = {'model': metadata, 'input_lock': lock, 'initial_writer_sessions': len(counts),
               'max_actual_prompt_tokens': max(counts), 'max_answer_tokens': 512, 'writer_capacity': 65536,
               'all_initial_sessions_fit': True, 'gpu_loaded': False,
               'later_generated_prompts_checked_by_runtime_without_truncation': True}
    (ROOT / 'GEMMA_CPU_PREFLIGHT.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
