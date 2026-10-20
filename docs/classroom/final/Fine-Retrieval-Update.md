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
