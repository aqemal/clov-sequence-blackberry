import unittest
import numpy as np
import pandas as pd
from explainability import probability, attribution, deletion_curve, event_summary
from core import validate_data, train_baseline

class ExplainabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=[]
        for i in range(12):
            positive=i%2==1
            seq=(f'app_start | read_sms | send_data | network_{i}' if positive else f'app_start | render_ui | user_touch | benign_{i}')
            rows.append({'sample_id':f's{i}','label':'malicious' if positive else 'benign','events':seq})
        cls.train=validate_data(pd.DataFrame(rows))
        cls.fit=train_baseline(cls.train.iloc[:8],cls.train.iloc[8:])
    def test_probability(self):
        p=probability(self.fit,['app_start','read_sms','send_data'],'baseline')
        self.assertTrue(0<=p<=1)
    def test_empty(self):
        with self.assertRaises(ValueError): probability(self.fit,[],'baseline')
    def test_position_based(self):
        p, frame=attribution(self.fit,['app_start','read_sms','send_data'],'baseline')
        self.assertEqual(len(frame),3)
        self.assertTrue(np.isfinite(frame.delta_malicious_probability).all())
    def test_single_event(self):
        _,frame=attribution(self.fit,['app_start'],'baseline')
        self.assertTrue(frame.empty)
    def test_deletion_curve(self):
        events=['app_start','read_sms','send_data','network_7']
        _,frame=attribution(self.fit,events,'baseline')
        curve,cls=deletion_curve(self.fit,events,'baseline',frame.delta_malicious_probability.to_numpy())
        self.assertEqual(set(curve.strategy),{'random','top_attributed'})
        self.assertTrue(curve.remaining_events.min()>=1)
        self.assertEqual(cls in (0,1),True)
    def test_event_summary(self):
        _,f=attribution(self.fit,['app_start','read_sms','read_sms'],'baseline')
        s=event_summary(f)
        self.assertEqual(int(s.loc[s.event=='read_sms','count'].iloc[0]),2)
    def test_invalid_model(self):
        with self.assertRaises(ValueError):probability(self.fit,['abc'],'x')
    def test_attribution_cap(self):
        with self.assertRaises(ValueError):attribution(self.fit,['abc'],'baseline',max_events=1)
if __name__=='__main__':unittest.main()
