import unittest, io, json, zipfile
import pandas as pd
from quality_gate import assess_dataset, snapshot_zip

def frame(rows):return pd.DataFrame(rows,columns=['sample_id','label','events','apk_sha256'])

class TestQualityGate(unittest.TestCase):
    def test_clean(self):
        c,a,s=assess_dataset(frame([('a','benign','open | close','a'*64),('b','malicious','open | send','b'*64)]),require_hash=True)
        self.assertEqual(s['accepted'],2)
        self.assertEqual(s['rejected'],0)
    def test_duplicate_hash(self):
        c,a,s=assess_dataset(frame([('a','benign','open | close','a'*64),('b','benign','read | send','a'*64)]))
        self.assertEqual(s['rejected'],2)
        self.assertIn('duplicate_apk_hash',s['error_frequency'])
    def test_sequence_overlap_rejected(self):
        c,a,s=assess_dataset(frame([('a','benign','open | close','a'*64),('b','malicious','open | close','b'*64)]))
        self.assertEqual(s['accepted'],0)
        self.assertIn('conflicting_labels_for_sequence_sha256',s['error_frequency'])
    def test_bad_label(self):
        c,a,s=assess_dataset(frame([('a','unknown','open | close','a'*64)]))
        self.assertEqual(s['rejected'],1)
    def test_event_min(self):
        c,a,s=assess_dataset(frame([('a','malicious','open','a'*64)]),min_events=2)
        self.assertEqual(s['rejected'],1)
    def test_hash_required(self):
        c,a,s=assess_dataset(frame([('a','malicious','open | close','bad')]),require_hash=True)
        self.assertEqual(s['rejected'],1)
    def test_snapshot_hashes(self):
        c,a,s=assess_dataset(frame([('a','malicious','open | close','a'*64)]))
        b,m=snapshot_zip(c,a,s)
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            from hashlib import sha256
            self.assertEqual(sha256(z.read('accepted_traces.csv')).hexdigest(),m['accepted_csv_sha256'])
    def test_missing(self):
        with self.assertRaises(ValueError):assess_dataset(pd.DataFrame({'foo':[1]}))
    def test_empty(self):
        with self.assertRaises(ValueError):assess_dataset(frame([]))
    def test_duplicate_id(self):
        c,a,s=assess_dataset(frame([('a','benign','open | close','a'*64),('a','benign','read | send','b'*64)]))
        self.assertEqual(s['rejected'],2)
