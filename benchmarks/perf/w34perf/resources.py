"""Summaries of telemetry.py JSONL, never benchmark-case attribution."""

import json
import math
from pathlib import Path


_FIELDS = ('memory_used_mib', 'utilization_gpu_percent',
           'utilization_memory_percent', 'temperature_c', 'power_w')


def _number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def summarize_resources(path, *, integrate_energy=False, max_gap_s=5.0):
    """Return JSON-safe measurements; unknown measurements are None, not zero.

    Peaks and arithmetic mean power use available finite, nonnegative readings.
    Missing counts are per parsed sample, including absent known UUIDs/PIDs.
    Only explicitly recorded PIDs are reported (no process discovery or summing).
    Malformed lines break energy continuity but do not count as samples.
    Energy is partial measured coverage, not an extrapolation: adjacent rows only,
    using strictly increasing monotonic timestamps and gaps <= max_gap_s (5s by
    default, allowing the sampler's 1s wait plus bounded collection overhead).
    Duration is unknown unless all sample monotonic timestamps strictly increase.
    """
    if not _number(max_gap_s) or max_gap_s <= 0:
        raise ValueError('max_gap_s must be positive and finite')
    result = dict(sample_count=0, scope='whole sampler window, including warmup/priming; not per-case energy',
                  observed_interval=None, duration_s=None, gpus={},
                  host_peak_memory_used_kib=None, pid_peak_rss_kib={},
                  missing_counts={}, error_count=0, errors=[],
                  energy_enabled=bool(integrate_energy), energy_max_gap_s=max_gap_s)
    rows = []
    try:
        with Path(path).open(encoding='utf-8') as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('sample must be an object')
                except ValueError as exc:
                    result['errors'].append({'source': 'input', 'line': line_no, 'message': str(exc)})
                    rows.append(None)
                    continue
                rows.append(row)
    except (OSError, UnicodeError) as exc:
        result['errors'].append({'source': 'input', 'message': str(exc)})
    samples = [r for r in rows if r is not None]
    n = result['sample_count'] = len(samples)
    missing = result['missing_counts']
    times = [r.get('timestamp_monotonic_s') for r in samples]
    missing['timestamp_monotonic_s'] = sum(not _number(t) for t in times)
    if n and all(_number(t) for t in times) and all(b > a for a, b in zip(times, times[1:])):
        result['observed_interval'] = {'clock': 'monotonic', 'start_s': times[0], 'end_s': times[-1]}
        result['duration_s'] = times[-1] - times[0] if n > 1 else None
    elif n:
        result['errors'].append({'source': 'timestamps', 'message': 'missing, invalid or non-increasing monotonic timestamps'})
    if not n:
        result['errors'].append({'source': 'input', 'message': 'no samples available'})

    gpu_rows, pid_rows, host = [], [], []
    for row in rows:
        if row is None:
            gpu_rows.append({})
            continue
        errors = row.get('errors')
        if isinstance(errors, list):
            result['errors'].extend(errors)
        else:
            result['errors'].append({'source': 'schema', 'message': 'missing/invalid errors list'})
        gpus = row.get('gpus')
        gpu_rows.append({g['uuid']: g for g in gpus
                         if isinstance(g, dict) and isinstance(g.get('uuid'), str) and g['uuid']}
                        if isinstance(gpus, list) else {})
        pids = row.get('pid_rss_kib')
        pid_rows.append(pids if isinstance(pids, dict) else {})
        mem = row.get('host_meminfo_kib')
        total = mem.get('MemTotal') if isinstance(mem, dict) else None
        avail = mem.get('MemAvailable') if isinstance(mem, dict) else None
        if _number(total) and _number(avail) and total >= avail:
            host.append(total - avail)
    missing['host_memory_used'] = n - len(host)
    missing['gpus'] = sum(not g for r, g in zip(rows, gpu_rows) if r is not None)
    result['host_peak_memory_used_kib'] = max(host, default=None)
    missing['pid_rss_kib'] = {}
    for pid in sorted({p for r in pid_rows for p in r}):
        values = [r[pid] for r in pid_rows if _number(r.get(pid))]
        result['pid_peak_rss_kib'][pid] = max(values, default=None)
        missing['pid_rss_kib'][pid] = n - len(values)
    for uuid in sorted({u for g in gpu_rows for u in g}):
        summary = {'missing_counts': {}, 'energy_j': None,
                   'energy_covered_duration_s': None, 'energy_interval_count': 0}
        for field in _FIELDS:
            values = [g[uuid][field] for g in gpu_rows if _number(g.get(uuid, {}).get(field))]
            summary['missing_counts'][field] = n - len(values)
            summary['average_power_w' if field == 'power_w' else 'peak_' + field] = (
                (sum(values) / len(values) if field == 'power_w' else max(values)) if values else None)
        if integrate_energy:
            energy, covered, count = 0.0, 0.0, 0
            for i in range(1, len(rows)):
                a, b = rows[i-1], rows[i]
                if a is None or b is None:
                    continue
                t0, t1 = a.get('timestamp_monotonic_s'), b.get('timestamp_monotonic_s')
                p0 = gpu_rows[i-1].get(uuid, {}).get('power_w')
                p1 = gpu_rows[i].get(uuid, {}).get('power_w')
                if all(_number(v) for v in (t0, t1, p0, p1)) and 0 < t1 - t0 <= max_gap_s:
                    dt = t1 - t0
                    energy += (p0 / 2 + p1 / 2) * dt
                    covered += dt
                    count += 1
            summary.update(energy_j=energy if count else None,
                           energy_covered_duration_s=covered if count else None,
                           energy_interval_count=count)
        result['gpus'][uuid] = summary
    result['error_count'] = len(result['errors'])
    return result
