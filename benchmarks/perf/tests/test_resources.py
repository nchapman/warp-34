"""Exercise the public file boundary with telemetry-shaped samples."""
import json
from pathlib import Path
import tempfile
import unittest

from w34perf.resources import summarize_resources


def sample(t, power=100):
    return dict(timestamp_monotonic_s=t, timestamp_unix_s=1000+t,
                errors=[], gpus=[dict(uuid='GPU-a', power_w=power,
                memory_used_mib=200+t, utilization_gpu_percent=90,
                utilization_memory_percent=80, temperature_c=50+t)],
                host_meminfo_kib={'MemTotal': 1000, 'MemAvailable': 700-t},
                pid_rss_kib={'42': 20+t})


class ResourcesTests(unittest.TestCase):
    def summarize(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'telemetry.jsonl'
            path.write_text(''.join((r if isinstance(r, str) else json.dumps(r)) + '\n' for r in rows))
            return summarize_resources(path, **kwargs)

    def test_peaks_and_measured_energy(self):
        result = self.summarize([sample(1, 100), sample(3, 200)], integrate_energy=True)
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['duration_s'], 2)
        self.assertEqual(result['host_peak_memory_used_kib'], 303)
        self.assertEqual(result['pid_peak_rss_kib'], {'42': 23})
        gpu = result['gpus']['GPU-a']
        self.assertEqual(gpu['peak_memory_used_mib'], 203)
        self.assertEqual(gpu['average_power_w'], 150)
        self.assertEqual(gpu['energy_j'], 300)
        self.assertEqual(gpu['energy_covered_duration_s'], 2)
        self.assertIn('not per-case', result['scope'])

    def test_missing_empty_and_bad_input(self):
        with tempfile.TemporaryDirectory() as directory:
            result = summarize_resources(Path(directory) / 'missing')
        self.assertIsNone(result['duration_s'])
        self.assertGreater(result['error_count'], 0)
        for rows in ([], ['broken'], [None]):
            result = self.summarize(rows)
            self.assertEqual(result['sample_count'], 0)
            self.assertIsNone(result['host_peak_memory_used_kib'])
            self.assertGreater(result['error_count'], 0)

    def test_no_bridging_missing_power_or_malformed_lines(self):
        for middle in (sample(2, None), 'broken', dict(sample(2), gpus=None)):
            result = self.summarize([sample(1), middle, sample(3)], integrate_energy=True)
            self.assertIsNone(result['gpus']['GPU-a']['energy_j'])

    def test_invalid_timestamps_and_gaps(self):
        for t in (1, 0, float('nan'), float('inf'), 20):
            result = self.summarize([sample(1), sample(t)], integrate_energy=True)
            self.assertIsNone(result['gpus']['GPU-a']['energy_j'])
        result = self.summarize([sample(1), sample(20)], integrate_energy=True, max_gap_s=20)
        self.assertEqual(result['gpus']['GPU-a']['energy_j'], 1900)
        for gap in (0, -1, True, float('nan')):
            with self.assertRaises(ValueError):
                self.summarize([], max_gap_s=gap)

    def test_unknown_fields_errors_and_default_no_energy(self):
        row = sample(1, None)
        row.update(host_meminfo_kib={'MemTotal': 1000}, pid_rss_kib={'42': None},
                   errors=[{'source': 'gpu', 'type': 'Unavailable'}])
        result = self.summarize([row, dict(sample(2), gpus=[])])
        self.assertEqual(result['error_count'], 1)
        self.assertEqual(result['missing_counts']['host_memory_used'], 1)
        self.assertEqual(result['missing_counts']['pid_rss_kib']['42'], 1)
        self.assertEqual(result['gpus']['GPU-a']['missing_counts']['power_w'], 2)
        self.assertIsNone(result['gpus']['GPU-a']['average_power_w'])
        self.assertIsNone(result['gpus']['GPU-a']['energy_j'])


if __name__ == '__main__':
    unittest.main()
