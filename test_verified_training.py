import unittest
import pandas as pd
from core import generate_demo_data
from verified_training import make_split, inspect_partitions, evaluate

class TestAuditedTraining(unittest.TestCase):
    def setUp(self):self.data=generate_demo_data(80,seed=19)
    def test_split(self):
        a,b,info=make_split(self.data,seed=42)
        self.assertTrue(info['disjoint'])
        self.assertFalse(set(a.sample_id)&set(b.sample_id))
    def test_baseline(self):
        a,b,_=make_split(self.data,seed=42)
        r,p,_=evaluate(a,b,'baseline')
        self.assertEqual(len(p),len(b))
        self.assertIn('f1',r['metrics'])
    def test_reject_duplicate(self):
        a,b,_=make_split(self.data,seed=42)
        b=b.copy();b.loc[b.index[0],'events']=a.iloc[0].events
        with self.assertRaisesRegex(ValueError,'Leakage'):inspect_partitions(a,b)
    def test_reject_one_class(self):
        a,b,_=make_split(self.data,seed=42)
        with self.assertRaises(ValueError):inspect_partitions(a,b.assign(label='malicious'))
    def test_group_disjoint(self):
        data=self.data.copy();data['malware_family']=data.label.map(lambda x:'benign' if x=='benign' else 'malware')
        with self.assertRaises(ValueError):make_split(data,group_key='malware_family',attempts=2)

if __name__=='__main__': unittest.main()
