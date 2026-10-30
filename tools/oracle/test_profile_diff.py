"""CPU-only synthetic nsys export tests. Run: python3 -m unittest discover -s tools/oracle -p test_profile_diff.py"""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from profile_diff import nvtx_candidates, parse_explicit_window, parse_nsys_sqlite


class Windows(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'trace.sqlite'
        self.meta = Path(self.tmp.name) / 'metadata.json'
        self.meta.write_text('{"rounds": 2}')
        with sqlite3.connect(self.db) as c:
            c.executescript('''
                CREATE TABLE StringIds(id INTEGER, value TEXT);
                INSERT INTO StringIds VALUES(1, 'gemm');
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(shortName, start, end, deviceId);
                INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1, 10, 20, 0), (1, 30, 40, 0), (1, 100, 120, 0);
                CREATE TABLE NVTX_EVENTS(text, textId, start, end, globalTid);
                INSERT INTO NVTX_EVENTS VALUES
                ('w34.bench.measured', NULL, 0, 50, 1),
                ('w34.decode', NULL, 10, 40, 1),
                ('w34.bench.measured', NULL, 80, 110, 1),
                ('w34.decode', NULL, 90, 105, 1),
                ('w34.decode', NULL, 91, 104, 2);
            ''')

    def test_last_nested_candidates_not_gpu_window(self):
        result = nvtx_candidates(self.db)
        self.assertEqual(result['decode_cpu_candidates'], [('w34.decode', 90, 105, 1)])
        with self.assertRaises(ValueError):
            parse_explicit_window(self.db, 90, 105, self.meta)

    def test_explicit_excludes_tail(self):
        result = parse_explicit_window(self.db, 10, 40, self.meta)
        self.assertEqual(result['rounds'], 2)
        self.assertEqual(result['categories']['marlin_gemm']['launches_per_round'], 1)
        self.assertFalse(result['gate_eligible'])

    def test_fail_closed(self):
        for start, end in [(11, 40), (10, 39), (40, 40), (50, 60), (None, 40)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                parse_explicit_window(self.db, start, end, self.meta)
        for value in [0, -1, True, 1.5, '2', None]:
            self.meta.write_text(json.dumps({'rounds': value}))
            with self.assertRaises(ValueError):
                parse_explicit_window(self.db, 10, 40, self.meta)
        with sqlite3.connect(self.db) as c:
            c.execute('DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL')
        with self.assertRaises(ValueError):
            parse_nsys_sqlite(self.db)

    def test_missing_nested_decode(self):
        with sqlite3.connect(self.db) as c:
            c.execute("DELETE FROM NVTX_EVENTS WHERE text = 'w34.decode' AND start >= 90")
        with self.assertRaises(ValueError):
            nvtx_candidates(self.db)

    def test_command(self):
        cmd = [sys.executable, str(Path(__file__).with_name('profile_diff.py')),
               '--mode', 'strict', '--w34', str(self.db), '--metadata', str(self.meta),
               '--window-start-ns', '10', '--window-end-ns', '40']
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)['measurement_mode'], 'explicit_gpu_window')
        proc = subprocess.run(cmd[:-4], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


class Correlated(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / 'trace.sqlite'
        with sqlite3.connect(self.db) as c:
            c.executescript('''
            CREATE TABLE StringIds(id, value);
            INSERT INTO StringIds VALUES(1,'gemm'),(2,'cudaGraphLaunch');
            CREATE TABLE NVTX_EVENTS(text,textId,start,end,globalTid,endGlobalTid);
            INSERT INTO NVTX_EVENTS VALUES
            ('w34.decode',NULL,1,50,16777217,NULL),
            ('w34.bench.measured',NULL,90,210,16777217,NULL),
            ('w34.decode',NULL,100,200,16777217,NULL),
            ('w34.proposer',NULL,120,160,16777218,NULL);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId,returnValue);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (130,140,16777218,7,2,0),(20,30,16777217,8,2,0),
            (130,140,33554433,7,2,0);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,deviceId,correlationId,globalPid,demangledName,graphNodeId,graphId);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (150,180,0,7,16777216,1,1,1),(220,250,1,7,16777216,1,2,1),
            (160,190,0,8,16777216,1,1,2),(170,190,0,7,33554432,1,1,3);
            ''')

    def parse(self, meta=None):
        from profile_diff import parse_nvtx_correlated
        return parse_nvtx_correlated(self.db, {'verify_rounds': 2} if meta is None else meta)

    def test_exact(self):
        p = self.parse()
        self.assertEqual(p['kernel_count'], 2)
        self.assertEqual(p['summed_gpu_kernel_ms'], 60 / 1e6)
        self.assertEqual(set(p['devices']), {'0', '1'})
        for device in p['devices'].values():
            self.assertEqual(device['kernel_count'], 1)
            self.assertEqual(device['categories']['spec_decode']['ms_per_round'], 15 / 1e6)
            self.assertEqual(device['categories']['spec_decode']['launches_per_round'], 0.5)
            self.assertEqual(device['proposer_graph_launch_count'], 1)
            self.assertEqual(device['proposer_graph_launch_threads'], [16777218])
        self.assertEqual(p['categories']['spec_decode']['launches_per_round'], 1)
        self.assertTrue(p['boundary_verified'])
        self.assertFalse(p['gate_eligible'])
        with sqlite3.connect(self.db) as c:
            c.execute("UPDATE StringIds SET value='cudaGraphLaunch_v10000' WHERE id=2")
        self.assertEqual(self.parse()['devices']['0']['proposer_graph_launch_count'], 1)

    def test_duplicate_runtime(self):
        with sqlite3.connect(self.db) as c:
            c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=7 AND globalTid=16777218')
        with self.assertRaises(ValueError): self.parse()

    def test_missing_ambiguous_markers(self):
        with sqlite3.connect(self.db) as c:
            c.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS WHERE text='w34.decode' AND start=100")
        with self.assertRaises(ValueError): self.parse()
        with sqlite3.connect(self.db) as c:
            c.execute("DELETE FROM NVTX_EVENTS WHERE text='w34.bench.measured'")
        with self.assertRaises(ValueError): self.parse()

    def test_metadata(self):
        for m in ({}, [], True, {'gen_len':128}, {'verify_rounds':0}, {'verify_rounds':True}, {'verify_rounds':2.5}):
            with self.assertRaises(ValueError): self.parse(m)

    def test_nullable_speculative_count_uses_explicit_measured_rounds(self):
        self.assertEqual(self.parse({'verify_rounds': None, 'measured_rounds': 2})['rounds'], 2)


class OracleCorrelated(Correlated):
    def setUp(self):
        super().setUp()
        with sqlite3.connect(self.db) as c:
            c.execute("UPDATE NVTX_EVENTS SET text='oracle.decode.round:oracle-measured' WHERE text='w34.decode' AND start=100")
            c.execute("UPDATE NVTX_EVENTS SET text='oracle.decode.round:oracle-warm' WHERE text='w34.decode'")

    def parse(self, meta=None):
        from profile_diff import parse_oracle_nvtx_correlated
        return parse_oracle_nvtx_correlated(self.db, {
            'verify_rounds':1, 'worker_events':[dict(kind='decode',rids=['oracle-measured'],pid=201)]
        } if meta is None else meta)

    def test_exact(self):
        p = self.parse()
        self.assertEqual(p['kernel_count'],2)
        self.assertEqual(p['rounds'],1)
        self.assertEqual(p['categories']['spec_decode']['launches_per_round'],2)
        self.assertFalse(p['gate_eligible'])
        self.assertEqual(p['summed_gpu_kernel_ms'],60/1e6)

    def test_missing_ambiguous_markers(self):
        with sqlite3.connect(self.db) as c:
            c.execute("DELETE FROM NVTX_EVENTS WHERE text='oracle.decode.round:oracle-measured'")
        with self.assertRaises(ValueError): self.parse()

    def test_nullable_speculative_count_uses_explicit_measured_rounds(self):
        with self.assertRaises(ValueError): self.parse({'measured_rounds':1})

    def test_pid_marker_collision(self):
        with sqlite3.connect(self.db) as c:
            c.execute("INSERT INTO NVTX_EVENTS VALUES('oracle.decode.round:oracle-measured',NULL,100,200,33554433,NULL)")
        with self.assertRaises(ValueError): self.parse()

    def test_count_mismatch(self):
        with self.assertRaises(ValueError):
            self.parse({'verify_rounds':2,'worker_events':[dict(kind='decode',rids=['oracle-measured'],pid=201)]})


class StrictMetadata(Windows):
    def test_nonobject(self):
        for value in ([], None, 2, 'x'):
            self.meta.write_text(json.dumps(value))
            with self.assertRaises(ValueError): parse_explicit_window(self.db,10,40,self.meta)

    def test_unnamed(self):
        with sqlite3.connect(self.db) as c:
            c.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET shortName=999 WHERE start=10')
        with self.assertRaises(ValueError): parse_explicit_window(self.db,10,40,self.meta)


if __name__ == '__main__':
    unittest.main()
