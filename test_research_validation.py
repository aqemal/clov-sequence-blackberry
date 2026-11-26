import unittest
import pandas as pd
from research_validation import check_provenance,ordered_events,audit_and_split

class ValidationTests(unittest.TestCase):
    def test_hash_matching_is_only_declaration(self):
        h='a'*64
        m=pd.DataFrame([{'sample_id':'one','sha256':h,'family':'BankBot'}])
        r=pd.DataFrame([{'sample_id':'one','apk_sha256':h,'malware_family':'BankBot'}])
        self.assertEqual(check_provenance(m,r).iloc[0].hash_verification_status,'declared_hash_matches_manifest')
    def test_hash_family_conflict(self):
        h='a'*64
        m=pd.DataFrame([{'sha256':h,'family':'BankBot'}])
        r=pd.DataFrame([{'apk_sha256':h,'malware_family':'Triada'}])
        self.assertEqual(check_provenance(m,r).iloc[0].hash_verification_status,'family_conflict')
    def test_timestamp_sort(self):
        events=[{'api':'last','timestamp':5},{'api':'first','timestamp':2}]
        seq,stat=ordered_events({'events':events})
        self.assertEqual(seq,['first','last']);self.assertEqual(stat,'timestamp_sorted')
    def test_missing_time_keeps_order(self):
        seq,stat=ordered_events({'events':[{'api':'a'},{'api':'b'}]})
        self.assertEqual(seq,['a','b']);self.assertEqual(stat,'array_order_only')
    def test_dedup_disjoint_split(self):
        labels=['benign','malicious']*8
        d=pd.DataFrame([{'sample_id':f's{i}','label':label,'events':f'event_{i} | finish'} for i,label in enumerate(labels)])
        d=pd.concat([d,d.iloc[[0]]],ignore_index=True)
        tr,te,meta=audit_and_split(d,seed=3)
        self.assertTrue(meta['disjoint'])
        self.assertFalse(set(tr.sample_id)&set(te.sample_id))
    def test_conflicting_labels_fail(self):
        d=pd.DataFrame([{'sample_id':str(i),'label':'benign' if i%2 else 'malicious','events':'identical'} for i in range(4)])
        with self.assertRaises(ValueError):audit_and_split(d)
if __name__=='__main__':unittest.main()
