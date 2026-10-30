"""Audit real sampler NVTX scopes using CUDA launch correlation, not GPU timestamps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

# Both the CUDA and Rust FFI have compile-time assertions for this ABI size.
RECORD_BYTES = 184


def audit(path: Path, max_rows: int = 16) -> dict:
    if type(max_rows) is not int or max_rows < 1:
        raise ValueError('max_rows must be positive')
    with sqlite3.connect(path) as conn:
        scopes = conn.execute('''SELECT n.start,n.end,n.globalTid,n.endGlobalTid
            FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
            WHERE COALESCE(n.text,s.value)='w34.sampler' ''').fetchall()
        if not scopes:
            raise ValueError('No w34.sampler ranges: trace is not evidence')
        sizes = Counter()
        for start, end, tid, end_tid in scopes:
            if (start is None or end is None or tid is None or end <= start
                    or end_tid not in (None, 0, tid)):
                raise ValueError('Invalid sampler scope')
            copies = conn.execute('''SELECT m.bytes,r.returnValue
                FROM CUPTI_ACTIVITY_KIND_MEMCPY m
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=m.correlationId
                WHERE m.globalPid=? AND r.globalTid=? AND r.start>=? AND r.end<=?
                  AND m.copyKind=2''',
                (tid & ~((1 << 24) - 1), tid, start, end)).fetchall()
            # Exactly one compact output array is downloaded by this ABI. Extra,
            # duplicated or oversized transfers fail rather than being hidden by
            # a permissive largest-copy threshold.
            if len(copies) != 1:
                raise ValueError(f'Expected one D2H output array, found {len(copies)}')
            size, status = copies[0]
            if (status != 0 or type(size) is not int or size < RECORD_BYTES
                    or size > max_rows * RECORD_BYTES or size % RECORD_BYTES):
                raise ValueError(f'Invalid compact sampler output: bytes={size}, status={status}')
            sizes[size] += 1
    return {'source': str(path), 'scope': 'w34.sampler', 'passed': True,
            'scope_count': len(scopes), 'record_bytes': RECORD_BYTES,
            'max_rows': max_rows, 'dtoh_sizes': dict(sorted(sizes.items())),
            'dtoh_total_bytes': sum(size * count for size, count in sizes.items()),
            'limitations': 'Evidence covers these sampler scopes only, not other engine paths.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--max-rows', type=int, default=16)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    text = json.dumps(audit(args.trace, args.max_rows), indent=2) + '\n'
    if args.out:
        args.out.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
