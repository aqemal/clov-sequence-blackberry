import io
import json
import tempfile
import unittest
from pathlib import Path
from ingestion import apk_inventory, inventory_summary, traces_to_csv_dataframe

class IngestionTests(unittest.TestCase):
    def test_inventory_nested_and_ignores_non_apk(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'BankBot'/'variety1'
            folder.mkdir(parents=True)
            (folder/'abc.apk').write_bytes(b'NOT_AN_APK')
            (folder/'notes.txt').write_text('hello')
            df=apk_inventory(tmp,compute_sha256=True)
            self.assertEqual(len(df),1)
            self.assertEqual(df.iloc[0].family,'BankBot')
            self.assertEqual(df.iloc[0]['variety'],'variety1')
            self.assertEqual(len(df.iloc[0].sha256),64)
            self.assertEqual(inventory_summary(df)['samples'],1)
    def test_json_event_order(self):
        raw={'samples':[{'sample_id':'a','label':'malicious','family':'BankBot','events':[{'category':'api','event':'read_sms'},'outbound_transfer']}]}
        df=traces_to_csv_dataframe(io.BytesIO(json.dumps(raw).encode()),'file.json')
        self.assertEqual(df.iloc[0].events,'api:read_sms | outbound_transfer')
        self.assertEqual(df.iloc[0].malware_family,'BankBot')
    def test_csv(self):
        raw=b'sample_id,label,events\na,benign,"ui_open | file_read"\n'
        df=traces_to_csv_dataframe(io.BytesIO(raw),'file.csv')
        self.assertEqual(len(df),1)
    def test_reject_apk(self):
        with self.assertRaises(ValueError):
            traces_to_csv_dataframe(io.BytesIO(b'hello'),'file.apk')
if __name__=='__main__':
    unittest.main()
