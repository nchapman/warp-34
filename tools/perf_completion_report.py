#!/usr/bin/env python3
"""Derive compact, explicitly non-certifying completion tables from raw bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks/perf'))
from w34perf.resources import summarize_resources
from w34perf.evidence import quality_status


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(roots):
    runs, startups = [], []
    for root in roots:
        if not root.is_dir():
            startups.append({'path': str(root), 'status': 'not_run'})
            continue
        for path in sorted(root.rglob('qualification.json')):
            value = json.loads(path.read_text())
            startups.append({'path': str(path), 'sha256': sha(path),
                             'engine_id': value.get('engine', {}).get('id'),
                             **{k: value.get(k) for k in ('status', 'startup_s', 'elapsed_s',
                                'error', 'quality_status', 'measurement_status', 'measurement_error')}})
        for path in sorted(root.rglob('manifest.json')):
            manifest = json.loads(path.read_text())
            directory = path.parent
            summary = directory / 'summary.json'
            rows = json.loads(summary.read_text()) if summary.exists() else []
            cases = {}
            for row in rows:
                key = (row['engine_id'], row['case_id'])
                cases.setdefault(key, []).append(row)
            groups = []
            for (engine, case), trials in sorted(cases.items()):
                valid = all(t.get('token_accounting_complete') and t.get('completion_tokens') is not None
                            and t.get('duration_s', 0) > 0 for t in trials)
                groups.append({'engine': engine, 'case': case, 'trials': len(trials),
                    'requests': sum(t['requests'] for t in trials),
                    'success': sum(t['successful_requests'] for t in trials),
                    'failed': sum(t['failed_requests'] for t in trials),
                    'output_tokens_s': (sum(t['completion_tokens'] for t in trials) /
                                        sum(t['duration_s'] for t in trials)) if valid else None,
                    'peak_prompt_tokens': max((t.get('peak_prompt_tokens') or 0 for t in trials), default=0) or None,
                    'publication_blockers': sorted({b for t in trials for b in t.get('publication_blockers', [])})})
            selected = set(manifest.get('selected_engines', []))
            deployments = [{k: v for k, v in e.items() if k not in ('request_overrides',)}
                for e in manifest.get('config', {}).get('engines', []) if e.get('id') in selected]
            quality = {e['id']: quality_status(manifest.get('evidence', []), directory, e)
                       for e in manifest.get('config', {}).get('engines', []) if e.get('id') in selected}
            if not (directory / 'qualification.json').exists() and not (directory.parent / 'qualification.json').exists():
                for engine_id in sorted(selected):
                    startups.append({'path': str(directory), 'engine_id': engine_id,
                                     'status': 'startup_receipt_not_retained',
                                     'quality_status': quality[engine_id],
                                     'measurement_status': manifest.get('status')})
            hashes = {p.name: sha(p) for p in (path, summary, directory / 'coverage.json',
                       directory / 'SHA256SUMS.json', directory / 'telemetry.jsonl') if p.exists()}
            runs.append({'path': str(directory), 'status': manifest.get('status'), 'input_sha256': hashes,
                         'deployments': deployments, 'cases': groups, 'quality': quality,
                         'resources': summarize_resources(directory / 'telemetry.jsonl')})
    report = {'schema_version': 1, 'publication_ready': False,
              'limitations': ['Qualification measurements; not a winner ranking.',
                'Different weight/KV/cohort/renderer/GPU settings are not controlled equivalents.',
                'Repetitions within one server lifetime are not independent restart trials.',
                'Baseline quality passes do not establish long-context quality.',
                'Resource peaks cover whole runs including warmup and priming.'],
              'startup_outcomes': startups, 'runs': runs}
    lines = ['# Completion campaign measurements', '', '**Qualification evidence, not a winner ranking or publication certification.**', '',
             '| Deployment | Case | Trials | Successful / requested | Output tok/s |',
             '|---|---|---:|---:|---:|']
    for run in runs:
        for row in run['cases']:
            rate = 'unknown' if row['output_tokens_s'] is None else f"{row['output_tokens_s']:.2f}"
            lines.append(f"| {row['engine']} | {row['case']} | {row['trials']} | {row['success']} / {row['requests']} | {rate} |")
    lines += ['', '## Startup / qualification outcomes', '']
    for outcome in startups:
        lines.append(f"- `{outcome.get('engine_id', outcome['path'])}`: startup **{outcome.get('status')}**, quality **{outcome.get('quality_status', 'not run')}**, measurements **{outcome.get('measurement_status', 'not run')}**.")
        if outcome.get('error'):
            lines.append('  - ' + str(outcome['error']).replace('\n', ' '))
    lines += ['', '## Interpretation', ''] + ['- ' + v for v in report['limitations']]
    lines += ['', 'Exact deployment settings, per-case blockers, resource summaries and input hashes are in the accompanying JSON.',
              'Raw manifests, request events, quality evidence, startup logs and telemetry remain in the referenced result directories.', '']
    return report, '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='new output prefix (.json and .md)')
    args = parser.parse_args()
    report, markdown = generate(args.roots)
    paths = [args.out.with_suffix('.json'), args.out.with_suffix('.md')]
    if any(p.exists() for p in paths):
        parser.error('derived output already exists; select a new prefix')
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    with paths[1].open('x') as output:
        output.write(markdown)
    print(f"{len(report['runs'])} runs, {len(report['startup_outcomes'])} startup outcomes; publication_ready=False")


if __name__ == '__main__':
    main()
