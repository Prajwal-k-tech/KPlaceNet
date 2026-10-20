"""Geographic smoothing must preserve a valid target and prefer nearby cells."""
import unittest
import numpy as np
import torch
from scripts.run_l6_kerala_head import geographic_targets

class GeographicTargetTests(unittest.TestCase):
    def test_nearby_credit_and_probability_mass(self):
        coords=np.array([[10.,76.]])
        centers=np.array([[10.,76.],[10.01,76.],[12.,77.]])
        q=geographic_targets(coords,centers,np.array([0]))
        self.assertTrue(torch.all(q>=0))
        self.assertTrue(torch.allclose(q.sum(1),torch.ones(1)))
        self.assertGreater(float(q[0,0]),.9)
        self.assertGreater(float(q[0,1]),float(q[0,2]))
        hard=geographic_targets(coords,centers,np.array([0]),smoothing=0)
        self.assertTrue(torch.equal(hard,torch.tensor([[1.,0.,0.]])))

    def test_invalid_scale_is_rejected(self):
        with self.assertRaises(ValueError):
            geographic_targets(np.array([[10.,76.]]),np.array([[10.,76.]]),np.array([0]),scale_km=0)
