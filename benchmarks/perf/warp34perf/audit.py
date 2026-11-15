"""Read-only inventory of historical bundles, including incomplete evidence.

Counts are as recorded in summaries, not independent verification of raw requests.
An audit never certifies publication readiness or merges different deployments.
"""
import hashlib
import json
from pathlib import Path


def _object(value, label, errors):
    if isinstance(value, dict):
        return value
    errors.append(f'{label} must be an object')
    return {}


def _objects(value, label, errors):
    if not isinstance(value, list):
        errors.append(f'{label} must be an array')
        return []
    valid = []
    for index, item in enumerate(value):
        if isinstance(item, dict):
            valid.append(item)
        else:
            errors.append(f'{label}[{index}] must be an object')
    return valid


def _manifest_parts(manifest, errors):
    # Validate only the nested structures consumed by inventory/comparison.
    config = dict(_object(manifest.get('config', {}), 'config', errors))
    provenance = _object(manifest.get('provenance', {}), 'provenance', errors)
    for field in ('engines', 'cases'):
        config[field] = _objects(config.get(field, []), f'config.{field}', errors)
    return config, provenance


def audit(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"not a results directory: {root}")
    bundles = []
    totals = dict(requests=0, successful_requests=0, failed_requests=0)
    for manifest in sorted(root.rglob('manifest.json')):
        directory = manifest.parent
        bundle = {'path': directory.relative_to(root).as_posix(),
                  'sha256': {}, 'errors': [], 'cells': [], 'coverage': []}
        data = {}
        for name in ('manifest.json', 'summary.json', 'coverage.json'):
            path = directory / name
            if not path.exists():
                bundle['errors'].append(f'missing {name}')
                continue
            raw = path.read_bytes()
            bundle['sha256'][name] = hashlib.sha256(raw).hexdigest()
            try:
                data[name] = json.loads(raw)
            except (ValueError, UnicodeError) as exc:
                bundle['errors'].append(f'{name}: {exc}')
        meta = data.get('manifest.json', {})
        if isinstance(meta, dict):
            bundle['status'] = meta.get('status', 'unknown')
            bundle['created_utc'] = meta.get('created_utc')
            config, _ = _manifest_parts(meta, bundle['errors'])
            bundle['deployments'] = config['engines']
        else:
            bundle['errors'].append('manifest.json must be an object')
        bundle['coverage'] = data.get('coverage.json', [])
        _objects(bundle['coverage'], 'coverage.json', bundle['errors'])
        rows = data.get('summary.json', [])
        if not isinstance(rows, list):
            bundle['errors'].append('summary.json must be an array')
            rows = []
        cells = {}
        for index, row in enumerate(rows):
            try:
                if not isinstance(row, dict):
                    raise ValueError('trial must be an object')
                key = (row['engine_id'], row['case_id'])
                if not all(isinstance(v, str) for v in key):
                    raise ValueError('engine/case identifiers must be strings')
                rep = row['repetition']
                counts = {k: row[k] for k in totals}
                if any(type(v) is not int or v < 0 for v in [rep, *counts.values()]):
                    raise ValueError('repetitions/counts must be nonnegative integers')
                blockers = row.get('publication_blockers', [])
                if not isinstance(blockers, list) or any(not isinstance(v, str) for v in blockers):
                    raise ValueError('publication_blockers must be strings')
            except (KeyError, TypeError, ValueError) as exc:
                bundle['errors'].append(f'summary row {index}: {exc}')
                continue
            cell = cells.setdefault(key, {'engine_id': key[0], 'case_id': key[1],
                                          'repetitions': [], 'publication_blockers': [],
                                          **dict.fromkeys(totals, 0)})
            if rep in cell['repetitions']:
                bundle['errors'].append(f'duplicate trial {key} repetition {rep}')
            cell['repetitions'].append(rep)
            cell['publication_blockers'] = sorted(set(cell['publication_blockers'] + blockers))
            for field, count in counts.items():
                cell[field] += count
                totals[field] += count
        bundle['cells'] = sorted(cells.values(), key=lambda c: (c['engine_id'], c['case_id']))
        for cell in bundle['cells']:
            cell['repetitions'].sort()
        bundles.append(bundle)
    return {'schema_version': 1, 'publication_certified': False,
            'count_semantics': 'Recorded summary counts; historical bundles are not independent trials across bundles.',
            'totals': totals, 'bundles': bundles}
