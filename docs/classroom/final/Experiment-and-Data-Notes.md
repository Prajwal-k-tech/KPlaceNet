# Current model update: fine-coordinate retrieval
Revised 21 October 2026. Read this before older L5-L7 experiment descriptions. The current Kerala demo defaults to L10 when its assets are installed.

## What changed
The original classifier chooses one of 300 geographic cells and returns the stored cell center. L7 adds training-neighbor class votes but retains those same 300 output coordinates. L10 instead matches the uploaded photograph against 15,939 training references and returns the matched photo's GPS coordinate. Thus there are 15,939 reference positions before duplicate coordinates, rather than 300 centers. This is still reference-based localization, not unrestricted latitude-longitude regression.

A ResNet-50 encoder produces 2,048 numbers and a frozen DINOv2 ViT-S/14 encoder produces 384. Each vector is L2-normalized, scaled by the square root of 0.5 and concatenated. A cosine search chooses the most similar reference. The resulting score is an appearance similarity, not a calibrated probability. The displayed reference is from training; the selected demo query is from the official test set.

## Measured comparison
Same 357 official test images; the benchmark was reused across experiments. These are exploratory follow-ups, not independent confirmation.

| Method | Median error | Within 5 km | Within 25 km |
|---|---|---|---|
| L5 classifier | 42.08 km | historical baseline | 41.18% |
| L6 regularized head | 28.41 km | see experiment record | 47.62% |
| L7 classifier + KNN votes | 21.84 km | 8.40% | 53.50% |
| L8 ResNet local coordinate retrieval | 18.27 km | 15.13% | 59.66% |
| L10 ResNet + DINO reference retrieval | 15.92 km | 17.37% | 63.03% |

The L10 result is 225/357 within 25 km and 62/357 within 5 km. None is within 1 km: every official test photo is over 1 km from the original training reference set. On random validation, L10 is 71.1% within 5 km and median 1.07 km; that split contains many nearby locations and must not be presented as unseen-place performance.

<!-- PAGE -->
# What to explain and what remains limited

## Why two encoders?
An encoder transforms image pixels into a feature vector, or descriptor. ResNet was trained for our geographic classification task; DINOv2 provides a different, self-supervised visual representation. Their normalized cosine similarities contribute equally. DINO alone did not beat the L8 validation baseline, while the combined descriptor did. Complementary information is a plausible explanation; our small experiment does not prove the exact mechanism.

## How was the winner chosen?
L8 evaluated four fixed coordinate-retrieval recipes on validation. L9 tested DINO-only retrieval and failed its promotion gate, so it was not tested on the official test set. L10 evaluated six fixed mixtures/neighborhood choices, prioritizing validation within 5 km with a guardrail on within 25 km. The equal mixture with one neighbor won. Test results were computed after this selection. All encoders were frozen during these retrieval comparisons; this was not an end-to-end retraining run.

## Is this ETH Zurich's method?
No. ETH/Google's Scaling Image Geo-Localization to Continent Level combines learned ground prototypes with aerial embeddings and fine S2 cell codes. Its ground encoder searches that fused geographic representation. Our deployment searches ground photographs with two encoders; it has no aerial encoder, learned aerial-ground fusion or comparable training scale. ETH's reported within-200-m Europe result is not comparable to our within-25-km Kerala-area result. We have not surpassed or reproduced that system.

## Why can a foreign photo still get a Kerala-area answer?
The regional pipeline always returns its best available match. It cannot establish that an input belongs to Kerala or is represented in the bank. The collection uses a bounding rectangle that also includes parts of neighboring states. A Great Wall image may therefore get an Indian match. A large similarity does not certify location. Neither restricting the map nor adding KNN solves this open-set recognition problem.

## New data and storage
Panoramax yielded 936 candidates, 789 unique images after exact-image deduplication. These were encoded in small batches with raw downloads kept only in memory. Four review sheets and compact descriptors were retained. The first 144 images were inspected: menus, interiors and repeated frames were present. A reviewed, spatially filtered and thinned 24-reference subset did not improve validation, so it was not added to deployment. More images are not automatically better data.

## Current limitations
Sparse references, viewpoint/weather changes, similar scenery, imperfect coordinates, nearby-frame leakage and regional assumptions all matter. Meter-level localization usually needs much denser coverage, appropriate aerial imagery and/or local geometric verification. The present median is roughly 16 km, not street-address accuracy.

<!-- PAGE -->
# Dataset research and next experiments

## Sources checked
**Panoramax:** public STAC search returned936 candidates inside the regional rectangle.789 unique image bytes survived exact-image deduplication. Every retained manifest row records source URL, author, license, coordinate and hash. Full downloads were not saved. Four contact sheets cover the first144 reviewed images; these included menus, interiors and repeated road frames.118 reviewed outdoor candidates became24 after excluding within1km of existing held-out coordinates and thinning new points within100m. Adding those references left validation unchanged and was rejected.

**Global Streetscapes:** its published688-city list was inspected; it contains Indian cities but no Kerala cities. The10M headline therefore does not establish available Kerala training coverage. Its Hugging Face release is metadata; source-image acquisition is separate.

**OSV5M:** the existing regional extraction already covers the available pinned-release coordinates in this box. Re-downloading the same subset adds no new geographic information. Fresh Mapillary sequences are a stronger prospective source, but require appropriate API access and actual coverage verification. KartaView is another source to check; no new Kerala count was verified in this run.

**Aerial imagery:** Sentinel-2 is useful regional context but its finest10m bands are not equivalent to ETH's high-resolution aerial tiles. A real ETH-like implementation needs suitable licensed aerial imagery, aligned ground views and cross-view training. Merely downloading satellite pixels does not implement that method.

## Practical next protocol
First obtain new routes, not more adjacent frames from the same roads. Review coordinate quality and street-scene content; retain attribution. Split by route/date and enforce geographic exclusion before fitting. Keep a new untouched final test. Compare retrieval coverage, recall and distance errors at1/5/25/200km. Only promote a data expansion if the predefined validation objective improves. Then evaluate local feature matching/geometric verification where a relevant reference exists. OCR may help with public signs, but it adds another fallible evidence source rather than guaranteeing a location.

## Storage and hardware
For frozen encoders: download a small batch, decode in RAM, compute descriptors, persist features/metadata and release bytes. Keep shard boundaries and manifests for reproducibility. Full encoder fine-tuning needs image revisits; use a bounded disk cache or reshuffle shards across epochs, otherwise repeated downloads can dominate runtime and sequential shard training can forget earlier data. Local RTX3050 extraction was fast enough for this run; no Kaggle GPU job was needed. This does not imply Kaggle would never help a substantially larger training job.

## Primary sources
[ETH/Google project](https://scaling-geoloc.github.io/) and [released code](https://github.com/google-research/scaling-geoloc).
[DINOv2 official code](https://github.com/facebookresearch/dinov2).
[Panoramax documentation](https://docs.panoramax.fr/).
[Global Streetscapes dataset](https://huggingface.co/datasets/NUS-UAL/global-streetscapes).
[Mapillary API demo](https://mapillary.github.io/api-demo/) and [KartaView API](https://kartaview.org/doc/photos).

Experiment records: docs/results/l8_fine_retrieval_20261021, l9_dinov2_20261021, l10_precision_20261021 and fine_deployment_20261021. The compact deployment-bank verification preserves official-test threshold counts and median. Historical model-only and hybrid results remain available for comparison.

<!-- PAGE -->
# ETH-inspired prototype experiment
Experimental ground-only implementation /21 October 2026. Not active in the live app.

## What was implemented
Geographic prototype vectors summarize local appearance. A 2 km/5 km grid and bilinear interpolation connect each training location to four neighboring prototype nodes. A multi-similarity-style ground-prototype loss was tested using frozen ResNet-DINO features. We also tested using prototype context to re-rank the 50 closest training references. No aerial encoder, aerial-ground fusion, S2 cell hierarchy or end-to-end encoder training was implemented.

## Measured tradeoff
The chosen contextual re-ranker uses 2 km prototypes initialized from weighted feature averages, with no prototype optimization(epoch 0). Learned prototype versions were tested but were not the selected candidate. On validation, within 5 km rose 71.1% to 71.8% and median fell 1.072 km to 1.038 km.

| Reused official test:357 photos | Current fine mode | Experimental re-ranking |
|---|---|---|
| Within 5 km |17.37%(62)|16.53%(59)|
| Within 25 km |63.03%(225)|65.83%(235)|
| Median error |15.92 km|15.57 km|
| Mean error |61.21 km|57.70 km|

The re-ranker changed 38 references. It improved coarser localization but lost three within 5 km successes, exceeding the 0.5 percentage-point permitted loss. It was not deployed. Current fine mode and the Study Guide/Demo Playbook still describe the installed ResNet-DINO nearest-reference method.

## What the experiment establishes
Some ETH-inspired components are computationally feasible here; they are not equivalent to ETH's full aerial-ground method and do not automatically improve close-range accuracy. Full cross-view training needs suitable aligned aerial imagery, not simply a finer geographic grid or a larger GPU.

The candidate-eligibility selection code was corrected after validation inspection and before test. This is explicitly recorded as an exploratory amendment. The test benchmark has already been reused. Neither the validation gain nor the mixed test result is independent confirmation.

Implementation and evidence: [experiment record](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/eth_ground_prototypes_20261021.md). The live frontend and the two rewritten learning guides have not changed because no new method was promoted.
