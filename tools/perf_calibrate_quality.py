#!/usr/bin/env python3
"""Optional offline preparation tool; not a dependency of the stdlib HTTP runner.

Requires transformers in the operator's preparation environment. Never downloads
models or executes checkpoint Python. Reference template counts are not proof of
what a remote engine renders; compare them with retained HTTP usage.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/perf'))
from w34perf.quality import _checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.snapshot), local_files_only=True,
                                              trust_remote_code=False)
    template = args.snapshot / 'chat_template.jinja'
    if template.exists():
        tokenizer.chat_template = template.read_text()
    if not isinstance(tokenizer.chat_template, str):
        parser.error('a single explicit chat template is required')
    checks = []
    for check, prompt in _checks(json.loads(args.fixture.read_text())):
        ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise ValueError('tokenizer did not return a flat token ID list')
        checks.append({'check': check['id'], 'reference_template_prompt_tokens': len(ids)})
    report = {'model_revision': args.snapshot.name,
              'tokenizer_sha256': hashlib.sha256((args.snapshot / 'tokenizer.json').read_bytes()).hexdigest(),
              'chat_template_sha256': hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'method': 'AutoTokenizer.apply_chat_template; return_dict=False; add_generation_prompt=True; enable_thinking=False; not server attestation; native renderers may differ',
              'checks': checks}
    with args.out.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
