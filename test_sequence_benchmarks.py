import unittest
import numpy as np
from core import generate_demo_data
from verified_training import make_split
from sequence_benchmarks import CNNClassifier, BiLSTMClassifier, compare_models
import torch

class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train,cls.test,_=make_split(generate_demo_data(n=100,seed=12),seed=5)
    def test_cnn_shapes(self):
        x=torch.tensor([[2,3,0,0],[4,5,6,0]],dtype=torch.long)
        self.assertEqual(tuple(CNNClassifier(10)(x).shape),(2,))
    def test_bilstm_shapes(self):
        x=torch.tensor([[2,3,0,0],[4,5,6,0]],dtype=torch.long)
        self.assertEqual(tuple(BiLSTMClassifier(10)(x).shape),(2,))
    def test_models_same_samples(self):
        rows,pred,audit=compare_models(self.train,self.test,epochs=1,models=['baseline','cnn','bilstm','transformer'])
        self.assertEqual(len(rows),4)
        self.assertEqual(len(pred),len(self.test))
        self.assertTrue(audit['no_exact_sequence_or_id_overlap'])
        for r in rows:self.assertTrue(0<=r['f1']<=1)
        for col in ['cnn_p_malicious','bilstm_p_malicious','transformer_p_malicious']:
            self.assertTrue(np.isfinite(pred[col]).all())
    def test_rejects_invalid_model(self):
        with self.assertRaises(ValueError):compare_models(self.train,self.test,models=['invalid'])

if __name__=='__main__':unittest.main()
