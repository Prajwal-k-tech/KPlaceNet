import unittest
import torch
from src.fine_retrieval import fine_predict,spherical_mean

class FineRetrievalTests(unittest.TestCase):
 def test_one_neighbor_returns_training_coordinate_not_centroid(self):
  q=torch.tensor([[1.,0.]]);bank=torch.tensor([[1.,0.],[0.,1.]])
  xy=torch.tensor([[9.1234,76.4567],[12.,77.]])
  p=fine_predict(q,bank,xy,k=1)
  self.assertTrue(torch.allclose(p['coords'],xy[:1],atol=1e-5))
 def test_consensus_does_not_average_distant_modes(self):
  q=torch.tensor([[1.,0.]]);bank=torch.tensor([[1.,.01],[1.,.02],[1.,0.]])
  xy=torch.tensor([[10.,76.],[10.01,76.],[50.,0.]])
  p=fine_predict(q,bank,xy,k=3,radius_km=25)
  self.assertEqual(p['neighbor_count'].item(),2)
  self.assertLess(abs(float(p['coords'][0,0])-10),.02)
  self.assertLess(abs(float(p['coords'][0,1])-76),.02)
 def test_mean_handles_dateline(self):
  p=spherical_mean(torch.tensor([[[0.,179.],[0.,-179.]]]),torch.tensor([[.5,.5]]))
  self.assertAlmostEqual(abs(p[0,1].item()),180,places=3)
 def test_invalid_neighborhood_rejected(self):
  with self.assertRaises(ValueError):fine_predict(torch.ones(1,2),torch.ones(1,2),torch.ones(1,2),k=2)
