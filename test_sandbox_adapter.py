import json
import unittest
import pandas as pd
from sandbox_adapter import adapt_report,adapt_many,training_export

class SandboxAdapterTests(unittest.TestCase):
    def test_events(self):
        result=adapt_report({'sample_id':'x','label':'benign','events':['start',{'event':'network_connect','category':'net'}]})
        self.assertEqual(result['events'],'start | net:network_connect')
    def test_processes(self):
        item={'sample_id':'y','label':'malicious','behavior':{'processes':[{'calls':[{'api':'a'},{'api':'b'}]}]}}
        self.assertEqual(adapt_report(item)['events'],'api:a | api:b')
    def test_missing_label_fails(self):
        with self.assertRaisesRegex(ValueError,'Explicit verified'):
            adapt_report({'sample_id':'x','family':'BankBot','events':['a']})
    def test_duplicate_id_fails(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            adapt_many([('a',{'sample_id':'x','label':'benign','events':['a']},{}),('b',{'sample_id':'x','label':'benign','events':['b']},{})])
    def test_manifest_not_provenance(self):
        df=pd.DataFrame([{'sample_id':'x','family':'BankBot'}])
        out=adapt_report({'sample_id':'x','label':'malicious','family':'BankBot','events':['a']},manifest=df)
        self.assertEqual(out['integrity_status'],'id_matched_only')
    def test_export(self):
        d,e=adapt_many([('a',{'sample_id':'x','label':'benign','events':['a']},{})])
        self.assertEqual(list(training_export(d).columns),['sample_id','label','events','malware_family'])
        self.assertTrue(e.empty)
if __name__=='__main__':unittest.main()
