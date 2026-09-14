"""Exercise the frozen generation and likelihood code on synthetic source text."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import refine as core
from evidence_utility import answer_token_request, reader_user
from transfer_runtime import Scorer


def check_prompt_serialization(scorer, user, answer):
    """Verify generation and NLL share exact non-thinking prefix token IDs."""
    from vllm import SamplingParams
    messages = [{'role': 'system', 'content': core.READER}, {'role': 'user', 'content': user}]
    rendered = scorer.tok.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    expected = list(scorer.tok.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, enable_thinking=False))
    if scorer.tok.encode(rendered, add_special_tokens=False) != expected:
        raise ValueError('Chat template text and token IDs differ')
    request, start, target = answer_token_request(scorer.tok, user, answer)
    if request['prompt_token_ids'][:start] != expected or request['prompt_token_ids'][start:] != target:
        raise ValueError('NLL serialization differs from generation prefix')
    outputs = scorer.llm.generate([rendered, {'prompt_token_ids': expected}],
        SamplingParams(temperature=0, max_tokens=1), use_tqdm=False)
    if len(outputs) != 2 or any(list(output.prompt_token_ids) != expected for output in outputs):
        raise ValueError('Installed vLLM changed serialized prompt token IDs')
    return {'matched': True, 'prompt_tokens': len(expected), 'token_ids_sha256': core.digest(expected)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('GPU smoke requires a fresh output directory; cached outputs are not a compatibility test')
    environment = json.loads(args.environment.read_text())
    scorer = Scorer(args, environment, args.out)
    question = 'What is the code Mira chose?'
    answer = 'silver maple'
    context = '(Session date: 2024-01-02) D1:1 Mira: The code I chose is silver maple.'
    jobs = [{'user': reader_user(question, text), 'answer': answer}
            for text in ('', context)]
    serialization = check_prompt_serialization(scorer, jobs[1]['user'], answer)
    scorer.score(jobs)
    for job in jobs:
        score = job['score']
        if score['answer_tokens'] != len(scorer.tok.encode(answer, add_special_tokens=False)):
            raise ValueError('Answer likelihood token coverage differs from frozen scoring')
    controls = {'full': {**jobs[1], 'source_ids': ['D1:1'], 'context_tokens': scorer.ntok(context)}}
    generated = scorer.generate(controls)
    if not generated['full']['text'].strip():
        raise ValueError('Synthetic reader returned an empty generation')
    report = {
        'status': 'gpu_smoke_passed', 'purpose': 'Compatibility only; no benchmark outcomes',
        'model': args.model, 'model_meta': environment['models'][args.model],
        'packages': {name: importlib.metadata.version(name)
                     for name in ('vllm', 'torch', 'transformers', 'sentence-transformers')},
        'language_model_only': True, 'enable_thinking': False,
        'source_scorer_sha256': hashlib.sha256((ROOT / 'source/run_evidence_utility.py').read_bytes()).hexdigest(),
        'prompt_serialization': serialization,
        'likelihoods': [job['score'] for job in jobs], 'generation': generated,
    }
    core.save(args.out / 'gpu_smoke.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()