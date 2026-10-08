"""Dependency-free tests for spatially stratified geolocation diagnostics."""

from __future__ import annotations

import json
import math
import unittest

from src.regional_eval import spatial_distance_metrics, spatial_stratified_metrics


class SpatialStratifiedMetricsTests(unittest.TestCase):
    def test_reports_equal_area_geodesic_metrics_and_sparse_bins(self):
        result = spatial_distance_metrics(
            pred_lats=[0.0, 0.0, 10.0],
            pred_lons=[0.0, 1.0, 91.0],
            true_lats=[0.0, 0.0, 10.0],
            true_lons=[0.0, 0.0, 90.0],
            lat_bands=2,
            lon_bands=4,
            min_count=2,
        )
        equatorial = result["regions"]["lat01_lon02"]
        self.assertEqual(equatorial["n"], 2)
        self.assertAlmostEqual(equatorial["metrics"]["mean_km"], 55.597, places=2)
        self.assertAlmostEqual(equatorial["metrics"]["median_km"], 55.597, places=2)
        self.assertEqual(equatorial["metrics"]["within_200km"], 100.0)
        self.assertEqual(equatorial["metrics"]["within_25km"], 50.0)
        self.assertIsNone(result["regions"]["lat01_lon03"]["metrics"])

    def test_geodesic_strata_validates_coordinate_alignment(self):
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            spatial_distance_metrics([0.0], [], [0.0], [0.0])
        with self.assertRaisesRegex(ValueError, "latitude"):
            spatial_distance_metrics([91.0], [0.0], [0.0], [0.0])

    def test_groups_samples_and_reports_quality_and_coverage_interval(self):
        result = spatial_stratified_metrics(
            probabilities=[[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]],
            labels=[0, 1, 0],
            prediction_sets=[[True, True], [True, False], [False, False]],
            true_lats=[5.0, 7.0, -90.0],
            true_lons=[1.0, 2.0, -180.0],
            centroids=[[5.0, 1.0], [0.0, 0.0]],
            lat_bands=2,
            lon_bands=4,
            min_count=2,
        )

        northern_region = result["regions"]["lat01_lon02"]
        self.assertEqual(northern_region["n"], 2)
        metrics = northern_region["metrics"]
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["conformal_coverage"], 0.5)
        self.assertAlmostEqual(metrics["mean_prediction_set_size"], 1.5)
        self.assertEqual(metrics["within_1km"], 0.5)
        self.assertAlmostEqual(metrics["conformal_coverage_wilson_95"][0], 0.0945, places=3)
        self.assertAlmostEqual(metrics["conformal_coverage_wilson_95"][1], 0.9055, places=3)

        sparse_region = result["regions"]["lat00_lon00"]
        self.assertEqual(sparse_region, {"n": 1, "metrics": None})
        self.assertIn("do not imply conditional", result["limitations"])
        json.dumps(result, allow_nan=False)

    def test_antimeridian_maps_to_one_longitude_band(self):
        common = dict(
            probabilities=[[1.0], [1.0]],
            labels=[0, 0],
            prediction_sets=[[True], [True]],
            true_lats=[0.0, 0.0],
            centroids=[[0.0, -180.0]],
            lat_bands=2,
            lon_bands=4,
        )
        west = spatial_stratified_metrics(true_lons=[-180.0, -180.0], **common)
        east = spatial_stratified_metrics(true_lons=[180.0, 180.0], **common)
        self.assertEqual(list(west["regions"]), ["lat01_lon00"])
        self.assertEqual(list(west["regions"]), list(east["regions"]))
        west_distance = west["regions"]["lat01_lon00"]["metrics"]["mean_distance_km"]
        east_distance = east["regions"]["lat01_lon00"]["metrics"]["mean_distance_km"]
        self.assertTrue(math.isclose(west_distance, east_distance, abs_tol=1e-9))

    def test_sparse_regions_remain_visible_without_unstable_metrics(self):
        result = spatial_stratified_metrics(
            probabilities=[[1.0]],
            labels=[0],
            prediction_sets=[[True]],
            true_lats=[0.0],
            true_lons=[0.0],
            centroids=[[0.0, 0.0]],
            min_count=30,
        )
        self.assertEqual(result["regions"]["lat03_lon06"], {"n": 1, "metrics": None})

    def test_rejects_bad_shapes_coordinates_and_probabilities(self):
        valid = dict(
            probabilities=[[1.0]],
            labels=[0],
            prediction_sets=[[True]],
            true_lats=[0.0],
            true_lons=[0.0],
            centroids=[[0.0, 0.0]],
        )
        with self.assertRaises(ValueError):
            spatial_stratified_metrics(**{**valid, "true_lats": []})
        with self.assertRaises(ValueError):
            spatial_stratified_metrics(**{**valid, "true_lats": [91.0]})
        with self.assertRaises(ValueError):
            spatial_stratified_metrics(**{**valid, "probabilities": [[0.7]]})
        with self.assertRaises(ValueError):
            spatial_stratified_metrics(**{**valid, "centroids": [[0.0]]})


if __name__ == "__main__":
    unittest.main()
