import io, json, zipfile, unittest
from pathlib import Path
from experiment_registry import audit_experiment_bundle, export_audit

class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.blob=Path(__file__).with_name('example_v9_experiment_bundle_SYNTHETIC.zip').read_bytes()
    def test_demo(self):
        good,bad,summary=audit_experiment_bundle(self.blob)
        self.assertEqual((len(good),len(bad)),(2,1))
        self.assertIn('report_apk_hash_mismatch',bad.iloc[0]['reasons'])
        self.assertEqual(summary['records'],3)
    def test_export(self):
        good,bad,summary=audit_experiment_bundle(self.blob)
        with zipfile.ZipFile(io.BytesIO(export_audit(good,bad,summary))) as z:
            self.assertIn('experiment_audit.json',z.namelist())
    def test_missing_manifest(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:z.writestr('report.json','{}')
        with self.assertRaisesRegex(ValueError,'run_manifest'):audit_experiment_bundle(buf.getvalue())
    def test_reject_apk(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:z.writestr('file.apk','junk')
        with self.assertRaisesRegex(ValueError,'Only JSON'):audit_experiment_bundle(buf.getvalue())
    def test_reject_traversal(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:z.writestr('../escape.json','{}')
        with self.assertRaisesRegex(ValueError,'Unsafe'):audit_experiment_bundle(buf.getvalue())
    def test_reject_bad_timezone(self):
        import pandas as pd
        from experiment_registry import _utc
        with self.assertRaises(ValueError):_utc('2026-11-20T10:00:00')

if __name__=='__main__':unittest.main()
