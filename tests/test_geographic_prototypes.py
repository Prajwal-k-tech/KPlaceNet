import unittest
import numpy as np
import torch
from src.geographic_prototypes import build_grid,initialize_prototypes,prototype_loss
class GeographicPrototypeTests(unittest.TestCase):
 def test_bilinear_weights_reconstruct_input_coordinates(self):
  coords=np.array([[8.003,74.509],[10.2,76.4],[12.78,77.49]])
  g=build_grid(coords,5.)
  np.testing.assert_allclose(g['weights'].sum(1),1.,atol=1e-6)
  np.testing.assert_allclose((g['coords'][g['indices']]*g['weights'].numpy()[:,:,None]).sum(1),coords,atol=1e-6)
 def test_loss_has_finite_prototype_gradients(self):
  q=torch.nn.functional.normalize(torch.eye(3),dim=1);i=torch.tensor([[0,1,2,0]]*3);w=torch.tensor([[1.,0,0,0]]*3)
  p=torch.nn.Parameter(initialize_prototypes(q,i,w,3));loss=prototype_loss(q,p,i,w,torch.ones((3,3),dtype=torch.bool));loss.backward()
  self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(p.grad).all())
 def test_grid_rejects_nonpositive_spacing(self):
  with self.assertRaises(ValueError):build_grid([[9,76]],0)
 def test_reranking_can_change_reference_but_zero_weight_preserves_top1(self):
  from src.geographic_prototypes import rerank_references
  q=torch.tensor([[1.,0.]])
  bank=torch.tensor([[1.,0.],[.99,.1]])
  prior=torch.tensor([[0.,1.],[1.,0.]])
  self.assertEqual(int(rerank_references(q,bank,prior,0.,2)[0]),0)
  self.assertEqual(int(rerank_references(q,bank,prior,.1,2)[0]),1)
