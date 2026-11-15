"""Attach explicit operator-supplied evidence without inferring its authority."""
import hashlib
import json
from .config import digest
from pathlib import Path
import shutil


def attach(paths, out):
    """Copy evidence byte-for-byte; callers must supply sanitized, non-secret logs.

    Hashes identify bytes, not trusted signatures or proof of loaded remote weights.
    Numeric prefixes avoid basename collisions without retaining absolute paths.
    """
    sources = [Path(p) for p in paths]
    for source in sources:
        if not source.is_file():
            raise ValueError(f'evidence is not a file: {source}')
    if not sources:
        return []
    directory = Path(out) / 'evidence'
    directory.mkdir(exist_ok=False)
    records = []
    for index, source in enumerate(sources):
        destination = directory / f'{index:03d}-{source.name}'
        with source.open('rb') as inp, destination.open('xb') as output:
            shutil.copyfileobj(inp, output)
        digest = hashlib.sha256()
        with destination.open('rb') as inp:
            for block in iter(lambda: inp.read(1024 * 1024), b''):
                digest.update(block)
        records.append({'path': destination.relative_to(out).as_posix(),
                        'sha256': digest.hexdigest(), 'bytes': destination.stat().st_size,
                        'trust': 'operator-supplied; attachment does not attest loaded weights or quality'})
    return records


def quality_status(attachments, out, engine):
    """Recognize attached results for this exact profile, without remote attestation."""
    statuses = []
    for attachment in attachments:
        path = Path(out) / attachment['path']
        if path.suffix != '.json':
            continue
        try:
            value = json.loads(path.read_text())
        except (ValueError, UnicodeError):
            continue
        if (not isinstance(value, dict) or value.get('model_profile_sha256') != digest(engine)
                or value.get('schema_version') != 1 or not isinstance(value.get('checks'), list)
                or not value['checks']):
            continue
        checks = value['checks']
        passed = (value.get('status') == 'passed' and
                  all(isinstance(c, dict) and c.get('status') == 'passed' for c in checks))
        statuses.append('passed' if passed else 'failed')
    # Never allow an attached short pass to conceal another attached failure.
    return 'failed' if 'failed' in statuses else 'passed' if statuses else 'missing'
