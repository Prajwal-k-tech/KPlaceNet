# Calibration and abstention evaluation on OSV-5M

This is an exploratory evaluation of the existing L3 uncertainty pipeline on
the disjoint 3,000-row sample documented in the
[sample manifest](osv5m_test_confirmatory_sample_20261008.json). The model is
the seed-42, 20k-image ResNet-50 checkpoint from the
[data-scale experiment](l2_data_scale_osv_test_20261009.md). The question was
selected after inspecting an earlier test report; the rows are disjoint, but
they are from the same official OSV-5M test split and do not represent
independent geography.

## Protocol and results

With seed `20261009`, the 3,000 rows were partitioned into 500 temperature-fit
rows, 1,000 conformal-calibration rows, and 1,500 evaluation-only rows. The
split is disjoint and its indices and sample IDs are recorded in the
[machine-readable report](l3_uncertainty_osv_test_20261009.json). Temperature
scaling learned `T = 3.6901`. On the 1,500 evaluation rows, expected
calibration error fell from `0.4171` to `0.0453`; top-1 predictions and
geolocation accuracy did not improve because temperature scaling preserves
the argmax.

At `alpha = 0.10`, the empirical conformal coverage was `90.8%` on the
evaluation partition against a 90% marginal target. Mean prediction-set size
was `159.7` of 300 cells, with a fitted quantile of `0.99885`. The sets are
therefore broad. Across the 29 geographic bins with at least 20 evaluation
rows, empirical coverage ranged from `59.5%` to `100%`; this is descriptive
evidence that marginal calibration does not imply regional coverage.

The model's evaluation-partition mean geodesic error was `5,477.4 km`, with
`5.27%` of predictions within 200 km. At the 0.5 and 0.7 confidence thresholds
the abstention policy abstained on every evaluation row. At 0.3 it abstained
on `99.2%` (12 of 1,500); the retained subset's `66.7%` top-1 accuracy is too
small to support a reliable performance claim. Confidence calibration
improved, but the model remains weak and these thresholds do not form a useful
high-coverage decision policy here.

## Limits

Split-conformal marginal coverage requires calibration examples and future
examples to be exchangeable. The evaluation split is drawn from the same
spatially held-out OSV-5M test split, and these rows are spatially dependent;
the empirical result does not establish exchangeability, conditional or
regional coverage, or coverage under geographic distribution shift. The
uncertainty implementation and finite-sample order-statistic correction are
contributions in this fork; these measurements do not show improved
geolocation accuracy.

## Reproduction

```bash
python scripts/run_l3_uncertainty.py \
  --run \
  --checkpoint checkpoints/data_scale_20k/l2_imagenet_1.0_layer4/last.pt \
  --csv data/osv5m_confirmatory_test/metadata.csv \
  --output docs/results/l3_uncertainty_osv_test_20261009.json \
  --temperature-size 500 --calibration-size 1000 --alpha 0.1 \
  --threshold 0.3 0.5 0.7 --batch-size 32 --device cuda \
  --image-size 224 --num-workers 4 --seed 20261009 \
  --region-lat-bands 6 --region-lon-bands 12 --region-min-count 20
```
