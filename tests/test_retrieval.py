import unittest
import torch
from src.retrieval import knn_prob

class RetrievalTests(unittest.TestCase):
    def test_cosine_scale_invariance_and_probability_mass(self):
        bank=torch.tensor([[1.,0.],[0.,1.],[-1.,0.]])
        labels=torch.tensor([0,1,2]);query=torch.tensor([[2.,0.]])
        p=knn_prob(query,bank,labels,k=2,num_cells=3)
        self.assertTrue(torch.allclose(p.sum(1),torch.ones(1)))
        self.assertEqual(p.argmax(1).item(),0)
        self.assertEqual(p[0,2].item(),0)
        self.assertTrue(torch.allclose(p,knn_prob(query*5,bank*3,labels,k=2,num_cells=3)))
    def test_too_few_neighbors_is_rejected(self):
        with self.assertRaises(ValueError):
            knn_prob(torch.ones(1,2),torch.ones(1,2),torch.zeros(1,dtype=torch.long),k=5)
