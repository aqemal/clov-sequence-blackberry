import unittest
import pandas as pd
from dashboard_v10 import frequency_ranking, impact_ranking, wordcloud_html, ordered_timeline, training_matches

class DashboardV10Tests(unittest.TestCase):
    def test_frequency_preserves_counts(self):
        self.assertEqual(frequency_ranking(['a','b','a']),[('a',2.0),('b',1.0)])
    def test_frequency_empty(self):self.assertEqual(frequency_ranking([]),[])
    def test_group_duplicate_event_impacts(self):
        f=pd.DataFrame({'event':['a','a','b'],'absolute_impact':[.2,.4,.1]})
        self.assertAlmostEqual(impact_ranking(f)[0][1],.3)
    def test_impact_empty(self):self.assertEqual(impact_ranking(pd.DataFrame()),[])
    def test_impact_required_columns(self):
        with self.assertRaises(ValueError):impact_ranking(pd.DataFrame({'event':['x']}))
    def test_html_escapes_events(self):
        s=wordcloud_html([('<script>alert(1)</script>',1)])
        self.assertNotIn('<script>',s)
        self.assertIn('&lt;script&gt;',s)
    def test_timeline_order(self):
        t=ordered_timeline(['b','a'])
        self.assertEqual(t.step.tolist(),[1,2])
        self.assertEqual(t.event.tolist(),['b','a'])
    def test_stamp_matching(self):
        f={'stamp':('dataset',42),'result':{},'model_id':'baseline'}
        self.assertTrue(training_matches(f,('dataset',42)))
        self.assertFalse(training_matches(f,('dataset',43)))
        self.assertFalse(training_matches(None,('dataset',42)))

if __name__=='__main__':unittest.main()
