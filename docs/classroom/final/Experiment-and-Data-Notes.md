# Final L7 follow-up

## Final L7 improvement: ResNet plus training-only KNN

The live regional demo now defaults to a hybrid when its matching feature bank is installed. ResNet produces a 2,048-dimensional feature vector. We compare its direction (cosine similarity) with features from the 15,939 training photographs, take five neighbors, and softmax-weight their cell votes with temperature 0.07. The final cell scores are 80% L6 classifier softmax plus 20% neighbor votes. Argmax chooses a saved cell centroid. The KNN checkbox lets you compare with L6 alone. Neither pipeline learns from an uploaded photograph.

The fixed blend passed validation: within 25 km rose from 70.9% to 74.4%, and median error fell from 4.78 to 4.02 km. The training bank excludes all validation and official test IDs. Its checkpoint hash must match the L6 model. The blend, k and temperature were fixed before this follow-up evaluation; they were not chosen using test performance.

| Method | Within 25 km, official test | Median error | Mean error |
| --- | --- | --- | --- |
| L5 baseline | 41.18% (147/357) | 42.08 km | 102.85 km |
| L6 geographic head | 47.62% (170/357) | 28.41 km | 85.33 km |
| L7 fixed 80/20 hybrid | 53.50% (191/357) | 21.84 km | 80.65 km |

The same 357 test images were already inspected for L6, so L7 is a follow-up on a reused benchmark, not fresh independent confirmation. Hybrid p90 error worsened from 230.99 to 239.82 km even though median and mean improved. KNN alone was also recorded as a diagnostic (53.78% within 25 km, 18.90-km median); we did not replace the predeclared hybrid based on this test result. One run and spatially optimistic validation limit the conclusion.

The combined score is uncalibrated and is not probability of correctness. Retrieval does not detect outside-region inputs. A Great Wall image can still get a regional answer, and the regional data rectangle (8.0-12.8 latitude, 74.5-77.5 longitude) includes neighboring territory such as parts of Karnataka. This is a Kerala-area experiment, not a guarantee that every output lies inside Kerala's administrative boundary. Do not clamp coordinates to hide errors.

Source: docs/results/l7_kerala_knn_20261020/results.json; scripts/evaluate_l7_knn.py; src/retrieval.py.

<!-- PAGE -->

# L6 Kerala: geographic smoothing of a frozen-feature classifier

Run completed 20 October 2026. Selected by validation before test evaluation.

## Result

| Metric | L5 baseline | Selected L6 | Test samples |
|---|---:|---:|---:|
| Within 25 km | 41.18% (147/357) | 47.62% (170/357) | 357 |
| Median distance | 42.08 km | 28.41 km | 357 |
| Mean distance | 102.85 km | 85.33 km | 357 |
| Within 200 km | 80.11% | 84.31% | 357 |
| Exact cell accuracy | 17.93% | 20.73% | 357 |
| Within 1 km | 0.56% | 0.56% | 357 |

The observed within-25-km gain is 6.44 percentage points, or 23 more correct images. Forty-one images move into the radius and eighteen move out. This is one seeded experiment with spatially dependent samples, not a significance claim or a guarantee for arbitrary photos. Fine precision did not improve: only two test photos are within 1 km in either model.

## What changed

ResNet-50 feature weights, all 300 geographic cells, image size and centroid decoding remain unchanged. Cache deterministic 2,048-dimensional L5 features. Fine-tune the 614,700-parameter linear head with AdamW, learning rate 0.0001, weight decay 0.01, batch 256, for 12 epochs. Both fixed recipes start from the same L5 head and use seed 42. Recipe one uses uniform label smoothing 0.1. Recipe two mixes the original class target with a geographic neighborhood:

`q = 0.9 * one_hot(class_id) + 0.1 * softmax(-distance_to_cell_centroid_km / 15)`

The target uses each training photograph's labeled coordinate. Coordinates never enter the inference network. The geographic recipe's epoch 12 wins on validation: 70.9% within 25 km versus 64.6% for the baseline. Validation median decreases from 7.03 to 4.78 km. The selection record was saved before evaluating the official test images. No test result changed the recipe, epoch or cell geometry.

The 15-km scale was fixed before evaluation as a coarse regional-neighborhood scale. This is a smoothing temperature in geographic distance, not the probability-temperature calibration used in L3. Kerala scores remain uncalibrated.

<!-- PAGE -->
## Data and split audit

Restore the same original OSV-5M bounding-box subset: 16,939 source-training images, reconstructed as 15,939 train and 1,000 random validation, plus 357 official test images. No training image was newly added in L6. The data is CC-BY-SA-4.0 and the official archive revision remains `cff33609b56b54d8743b7ee7a416eb8433e9a681`.

All identifiers are disjoint across the three reconstructed splits. Within this recovered subset, 982 of 1,000 validation coordinates lie within 1 km of a training coordinate. The median nearest-training distance is 99 m. No official test coordinate is within 1 km; minimum distance is 1.005 km and median is 2.520 km. This directly supports the warning that random validation is optimistic. It is not a new audit of all 5.1 million source images.

## Relation to the ETH Zurich paper

Lindenberger et al., *Scaling Image Geo-Localization to Continent Level*, NeurIPS 2025, combine ground-view prototypes, aerial embeddings and geographic interpolation. Our small experiment adopts geographic boundary awareness through soft training labels. It does not implement their aerial encoder, S2 hierarchy, interpolation rule or multi-similarity objective. It is neither a reproduction nor a comparable benchmark.

Sources: https://scaling-geoloc.github.io/ and https://arxiv.org/abs/2510.26795 . The original project's stated inspiration remains PlaNet: https://arxiv.org/abs/1602.05314 .

## Hardware and reproducibility

Executed locally on an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM, PyTorch 2.7.1+cu128. A separate GPU environment and workspace-local NVIDIA 610.57.04 userspace libraries match the currently loaded kernel module. System packages and running desktop drivers were not changed. Kaggle authentication and the available 30-hour GPU quota were verified, but no Kaggle job was needed or charged for this experiment. Most work is feature extraction; cached-feature classifier optimization for both recipes took 3.22 seconds. This is not the cost of training ResNet-50 from scratch.

Run `python scripts/run_l6_kerala_head.py` after restoring data with the existing Kerala extractor and recreating the original split with `python scripts/run_l5_kerala.py --phase prep`. See the exact JSON protocol, CSV hashes, selection record, test predictions and deployment manifest in `docs/results/l6_kerala_20261020/`.

## Deployment and limits

The release pack installs `checkpoints/l6_kerala_geographic/selected.pt` and its unchanged `cells.json`. The demo prefers this published pack if installed and otherwise uses legacy L5. Preserve both for rollback. The newly collected 350 Wikimedia Commons coordinate/license records are candidates only; they were not used for learning, selection or reported evaluation. Image content, coordinate quality and visual duplicates still need review.

Regional restriction, street-view domain bias, random-validation dependence, centroid quantization, small town sample sizes and uncalibrated Kerala scores remain. The Trivandrum subgroup's median error is still about 122 km on only eleven test images. Improved aggregate error does not establish deployment reliability.


<!-- PAGE -->
# Additional Kerala data: what was actually found

## Wikimedia Commons: candidate metadata collected
The collector queried eight Kerala locations with a 10-km radius and up to 50 file records per town. Seven queries produced 350 unique candidate records. The Munnar search returned a temporary service-busy error. The manifest preserves file title, page ID, source page, photo URL, author, license and coordinate. It also records nearest distances to the restored OSV training and official-test coordinates.

The candidates are around Kasaragod, Kannur, Kozhikode, Thrissur, Kochi, Kottayam and Thiruvananthapuram. They are not yet a reviewed image dataset. No Commons photograph was used for L6. Individual file licenses and camera/subject coordinate semantics require review, as do duplicate views, unsuitable scenes and source overlap. The API responses are saved with the candidate records.

Source: https://commons.wikimedia.org/wiki/Commons:API/MediaWiki . License metadata guidance: https://commons.wikimedia.org/wiki/Help:Machine-readable_data .

## Mapillary: useful next collection source
Mapillary supplies geotagged street imagery and APIs. Kerala-specific coverage and access must be established through an authorized query. OSV-5M itself derives from Mapillary, so a newly downloaded image can overlap existing source data. Check original image IDs, sequences, capture time and visual duplicates before assigning splits. The developer documentation requires login. Source: https://www.mapillary.com/developer/api-documentation/ .

## Other screened sources
The original India Driving Dataset provides Indian road scenes around Hyderabad and Bangalore and emphasizes scene understanding. It does not provide a ready Kerala geolocation subset for this run. Source: https://idd.insaan.iiit.ac.in/ .

Mapillary Street-Level Sequences is a large place-recognition dataset, but Kerala membership must be verified from its metadata rather than assumed from global coverage. Source: https://github.com/mapillary/mapillary_sls . Reubencf/streetview-global includes image coordinates and scene annotations, but no Kerala subset was established during this review. Source: https://huggingface.co/datasets/Reubencf/streetview-global .

## Next-data acceptance checklist
Verify actual camera coordinates and image decoding; preserve author, license and source identity; remove visual and identifier duplicates; group sequences and nearby locations; exclude overlap with any final test area; measure coverage by town and scene style; freeze train/validation/test before fitting. Internet images without reliable coordinate labels are suitable for qualitative demonstrations, not supervised coordinate evaluation.
