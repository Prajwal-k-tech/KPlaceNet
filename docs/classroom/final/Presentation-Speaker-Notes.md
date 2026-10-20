# Presentation speaker notes

## Slide 1

Introduce visual geolocation from pixels. Distinguish our bounded regional experiment from global state of the art. Group: Prajwal Kumar K, Chandra Teja P, ShivKarthik Suresh, Hari Krishnan S. Confirm title-page roll numbers and institution separately. Image source: OSV-5M, image 319636679783973, official test archive images/test/03.zip at revision cff33609b56b54d8743b7ee7a416eb8433e9a681. License CC-BY-SA-4.0, https://huggingface.co/datasets/osv5m/osv5m .

## Slide 2

Read left to right. This slide shows the base classifier path; slide 13 adds the final KNN refinement. GPS labels supervise training but are absent from inference inputs. Evaluation converts RGB pixels, resizes the short edge to 256, center crops to 224, and normalizes with ImageNet statistics. ResNet-50 produces 2048 features. The linear head produces 300 logits. Softmax and argmax choose a cell and its saved centroid. The app does not retrieve the caption or inspect a filename for the answer. https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/demo/app.py

## Slide 3

K-means groups training GPS coordinates, not image pixels. Longitude scaling by cos(mean latitude) approximates local geometry. Each cell has a center and bounds; the boxes can overlap and assignment resolves overlap by haversine distance. The scatter plot contains all 300 actual saved centers with angular axes. It is not an administrative-boundary map. K-means builds these cells; the final optional L7 KNN extension retrieves training image features and votes over the same cells. Geometry is unchanged from L5 to L6. Source https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 4

The bounding box contains 16939 source-training images. Seed 42 holds out 1000 for validation, leaving 15939 for training. The official test has 357. All IDs are disjoint. L6 restores the same original images; it adds no new training data. This source filter is a bounding box, not an audited Kerala administrative polygon. Source https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 5

Freeze the L5 backbone and 300-cell geometry. Only the 614700-parameter linear head changes. Uniform smoothing spreads 10% mass across all classes; the geographic recipe distributes it with softmax(-haversine distance/15 km), using each training image's coordinate. The assigned class retains 90% hard-label mass. Both use AdamW lr .0001, weight decay .01, batch 256, 12 epochs, seed 42. Deterministic cached views differ from L5 stochastic training views, so do not attribute the whole gain to a single isolated ingredient. This is a boundary-awareness adaptation, not the ETH paper's prototype interpolation. Source https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 6

Both recipes start from the same original head and reset the seed. Evaluate validation within 25 km after every epoch, tie-breaking with median distance. The unchanged original head is eligible. Before looking at test performance save the winner: geographic smoothing, epoch 12. Validation improves from 64.6% to 70.9%. The predefined deployment condition is at least .5 percentage point gain with median no more than 2% worse. Selection is already frozen when the 357 official test images are evaluated. Source https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 7

On the same 357 official Kerala test photographs, L5 is within 25 km on 147 images (41.18%) and L6 on 170 (47.62%). This is an observed gain of 6.44 percentage points. Forty-one images move inside that radius while eighteen regress outside it. Median distance falls from 42.08 to 28.41 km, and mean distance from 102.85 to 85.33 km. Exact cell accuracy rises from 17.93% to 20.73%. Within 1 km remains 0.56%, only two images. The chart rounds percentages for readability; the exact counts are recorded. Selection happened on validation before this test. This is one seeded experiment with spatially dependent samples, not a significance claim. The fixed L7 hybrid later reaches 191/357 (53.50%), median 21.84km and mean 80.65km on this same reused benchmark after validation selection. Its p90 worsens to 239.82km. This follow-up is not fresh independent confirmation. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 8

The image IDs are disjoint, but geographic independence is a different question. In the recovered Kerala subset, 982 of the 1,000 random validation coordinates lie within 1 km of a training coordinate. The median nearest-training distance is about 99 meters. None of the 357 official test coordinates lies within 1 km, with minimum distance 1.005 km and median 2.520 km. Nearby street scenes can share visual structure, explaining why random validation is optimistic. This audit covers the recovered subset, not the entire source dataset. A stronger future study should define geographic and sequence groups before fitting the backbone. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/b826f63445fb537390aa5f6f0202c0f4d772c639/docs/results/l6_kerala_20261020.md

## Slide 9

Kerala L6 reports raw, uncalibrated softmax. The default global demo applies a historical temperature, but its calibration artifact has no checkpoint hash and its identity remains unverified. The UI now states this. A separate corrected 20,000-image global study uses 500 rows for temperature fitting, 1,000 other rows for conformal calibration and 1,500 other rows for evaluation. Its ECE falls from 0.4171 to 0.0453. Empirical class-set coverage is 90.8%, with a mean set containing 159.7 of 300 classes. That is not 90.8% top-location accuracy. Temperature scaling preserves the winning class, so it does not improve distance by itself. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/l3_uncertainty_osv_test_20261009.md

## Slide 10

The confirmed ETH Zurich paper is Scaling Image Geo-Localization to Continent Level, by Lindenberger and colleagues, NeurIPS 2025, with ETH Zurich, Google and Politecnico di Milano affiliations. It combines learned ground-view prototypes, aerial embeddings and geographic interpolation for retrieval. L6 adapts broad geographic boundary awareness through soft labels. It does not implement the paper's aerial encoder, S2 hierarchy, interpolation rule or multi-similarity loss, and its benchmark is not comparable. The repository's original stated inspiration is PlaNet. A future cross-view extension would need licensed aerial imagery, alignment and a separate evaluation. Sources: https://arxiv.org/abs/2510.26795 and https://scaling-geoloc.github.io/ .

## Slide 11

Click the Kottayam thumbnail to load the photograph and its matching model. Then press Locate photo. The final live GPU run predicts cell 196 at latitude 9.67606339 and longitude 76.40408516. The final hybrid combines classifier and training-only neighbor votes. Its score is uncalibrated and is not probability of being within 25 km. Do not memorize a particular score; browser decoding may change it slightly. Open in Google Maps displays that same coordinate and does not correct the prediction. Then show Southern Kerala as a failure: L6 alone predicts about 10.49834435, 76.25760402; the hybrid instead predicts 12.46153060, 75.08052619 and makes this particular error worse. All ten gallery photographs came from official test archives, but they were selected deliberately and do not estimate average accuracy. Image source: OSV-5M official images/test/03.zip, image 1083931132408866, pinned revision cff33609b56b54d8743b7ee7a416eb8433e9a681, CC-BY-SA-4.0. https://huggingface.co/datasets/osv5m/osv5m

## Slide 12

The Wikimedia collector returned 350 unique coordinate and license metadata records around seven Kerala cities. The Munnar query was temporarily unavailable. These are candidate records, not 350 reviewed training photographs. None contributed to L6 learning, selection or reported evaluation. Review actual camera coordinates, scene suitability, file licenses, visual duplicates and capture sequences before defining geographic splits. None is within 1 km of the recovered official test coordinates, but that alone does not prove label quality or independence. Mapillary also needs access and overlap checks because OSV-5M derives from that source. Remaining model limitations include its regional assumption, centroid resolution, street-view bias and raw scores. Source: https://commons.wikimedia.org/wiki/Commons:API/MediaWiki

## Slide 13

ResNet extracts a 2048-dimensional vector. Cosine similarity retrieves k=5 neighbors from the 15939 training images only. Softmax over their similarities with temperature .07 weights their cell votes. Blend 80% L6 classifier probability with 20% KNN vote and select the highest cell score. Fixed settings pass validation before test reporting: within25 improves 70.9 to74.4%, median4.78 to4.02km. On the reused357-image benchmark the hybrid reaches191/357 or53.50%, median21.84km andmean80.65km. P90 worsens230.99 to239.82km. KNN alone is diagnostic, not chosen after seeing test results. No outside-region detection is claimed. The Great Wall can still receive a regional answer. The source rectangle includes neighboring territory. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/l7_kerala_knn_20261020.md

## Slide 14

This is a separate historical global experiment, not the L6 regional model or the legacy live global checkpoint. Across three paired seeds, increasing training data from 10,000 to 20,000 images raises within-200-km accuracy from 3.756% to 5.300% and reduces mean distance from 5,977.3 to 5,521.6 km. Absolute global accuracy remains weak. Fixed cells and paired seeds make this a bounded data-scale comparison. Inverse-frequency cell sampling did not give a consistent benefit. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/l2_data_scale_osv_test_20261009.md

## Slide 15

Use scores equal to one minus the true-class probability on an independent conformal calibration subset. For desired miscoverage alpha, choose k = ceil((n + 1)(1 - alpha)) and the kth smallest score. If k exceeds n, use infinity. For four ordered scores [0.1, 0.2, 0.4, 0.8] and alpha 0.2, k is 4 and the cutoff is 0.8. With alpha 0.1, k is 5, beyond n, so the conservative cutoff is infinity. This finite-sample order statistic differs from an interpolated percentile. Coverage is marginal under exchangeability and does not guarantee performance in every town or under domain shift. The corrected workflow keeps temperature fitting, conformal fitting and evaluation rows separate. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/src/uncertainty.py