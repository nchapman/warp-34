"""Fail-closed compact-output transfer evidence using synthetic Nsight tables."""
from pathlib import Path
import sqlite3
import tempfile
import unittest

from sampler_copy_audit import audit


class SamplerCopyAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'trace.sqlite'
        with sqlite3.connect(self.path) as conn:
            conn.executescript('''
                CREATE TABLE StringIds(id,value);
                CREATE TABLE NVTX_EVENTS(text,textId,start,end,globalTid,endGlobalTid);
                INSERT INTO NVTX_EVENTS VALUES('w34.sampler',NULL,100,200,16777217,NULL);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,returnValue);
                INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(120,130,16777217,7,0);
                CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,bytes,globalPid,correlationId,copyKind);
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(210,230,2944,16777216,7,2);
                -- Another process reuses the same correlation ID.
                INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(150,160,496640,33554432,7,2);
            ''')

    def test_async_completion_and_process_isolation(self):
        result = audit(self.path)
        self.assertTrue(result['passed'])
        self.assertEqual(result['scope_count'], 1)
        self.assertEqual(result['dtoh_sizes'], {2944: 1})

    def test_no_scope_is_not_proof(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute('DELETE FROM NVTX_EVENTS')
        with self.assertRaises(ValueError):
            audit(self.path)

    def test_full_vocab_extra_and_malformed_transfers_fail(self):
        for size in (0, 185, 496640):
            with self.subTest(size=size):
                with sqlite3.connect(self.path) as conn:
                    conn.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=? WHERE globalPid=16777216', (size,))
                with self.assertRaises(ValueError):
                    audit(self.path)
        with sqlite3.connect(self.path) as conn:
            conn.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET bytes=184 WHERE globalPid=16777216')
            conn.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE globalPid=16777216')
        with self.assertRaises(ValueError):
            audit(self.path)

    def test_missing_failed_or_cross_thread_copy_is_rejected(self):
        with self.assertRaises(ValueError):
            audit(self.path, max_rows=8)
        for status, tid in ((1, 16777217), (0, 16777218)):
            with sqlite3.connect(self.path) as conn:
                conn.execute('UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=?,globalTid=?', (status, tid))
            with self.assertRaises(ValueError):
                audit(self.path)


if __name__ == '__main__':
    unittest.main()
