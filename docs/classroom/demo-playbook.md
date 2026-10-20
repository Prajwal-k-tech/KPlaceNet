# KPlaceNet: live demo playbook
Current fine-retrieval demo / 21 October 2026

## Before class: ten-minute setup
Open a terminal in the task workspace. Run `./outputs/Start-KPlaceNet.sh` and keep that terminal open. Open http://127.0.0.1:7861/ at 100% browser zoom. On this laptop the launcher selects the working isolated GPU environment, otherwise CPU. The initial prediction loads two encoders and is slower than later predictions.

If already inside the repository at `work/KPlaceNet`, run `../../outputs/Start-KPlaceNet.sh`. An ordinary separate checkout can use `python scripts/download_fine_retrieval.py` to install the verified assets, then `python -c "from demo.app import launch_demo; launch_demo(server_name='127.0.0.1',server_port=7861)"` in its installed Python environment. The source repository alone does not contain model weights or all example images.

Choose Kerala area and Fine coordinates - ResNet + DINOv2. Click the Kottayam thumbnail and then Locate photo. Verify the prediction, retrieved reference and Google Maps link before the presentation. Open the study guide and report PDFs locally for an offline backup. Internet is needed for external maps, not local inference after assets are installed.

## The central claim to memorize
We estimate an image's location from learned visual appearance. The latest method retrieves coordinates from training photos instead of restricting predictions to 300 class centers. It improves our reused benchmark, but it does not reliably recognize arbitrary locations or reject foreign images.

<!-- PAGE -->
# Five-minute route: what to click and say

## 0:00-0:40: frame the task
Show the upload and regional method controls. Say: We give the system one street photograph. It extracts visual descriptors and searches geographically labeled reference images. No movement around a panorama is used. Our main demonstration assumes a Kerala-area input.

## 0:40-1:40: run a held-out example
Click Kottayam and then Locate photo. Clicking a thumbnail loads the image and scope; it does not itself run inference. The Locate button performs the forward passes and nearest-reference search.

Point to the retrieved coordinate and the separate matched training photograph. The Kottayam smoke test retrieved training reference309630957741047 at approximately9.67611,76.42105 with cosine similarity0.812. Values can vary slightly across hardware. This is a training reference coordinate, not a verified estimate of the exact camera position.

Say: The query is from the official test archive; the reference bank contains only training IDs. These are curated examples, so a successful demonstration does not represent average accuracy. Open Google Maps to show where the returned coordinate lies. Return to the local tab promptly.

## 1:40-2:40: explain the procedure
ResNet gives2,048 features, DINOv2 gives384. Normalize each so vector length does not dominate, weight them equally and concatenate. Cosine similarity compares the query with15,939 training descriptors. The top match supplies its GPS coordinate. This is one-nearest-neighbor retrieval over neural features; the earlier KNN mode instead votes for one of300 cell classes.

## 2:40-3:30: demonstrate the change
Switch to Regional cells - ResNet + KNN and rerun the same photo. It blends80% classifier probability with20% five-neighbor class votes, then returns a class center. Switch back to Fine coordinates and rerun. Explain why moving from300 centers to reference-photo positions removes one source of quantization error, while sparse or wrong references still limit accuracy.

<!-- PAGE -->
# Finish with evidence and a failure

## 3:30-4:20: show the measured result
Say: On357 official test images, L10 achieves63.0% within25km and17.4% within5km, with15.9km median error. The previous L7 method achieved53.5% within25km and21.8km median. The test benchmark has been reused, so these are exploratory results. Random validation is easier because many photos are close to training locations.

Do not describe63% as exact-location accuracy. State the distance threshold. Do not call cosine similarity confidence or probability. Do not claim the model was trained end-to-end with DINO during this run.

## 4:20-5:00: show a known limitation
Select Southern Kerala and Locate photo. The current smoke test retrieves a northern reference near12.45793,75.09220. This is a real failure; leave it visible long enough to explain repeated visual patterns and sparse coverage. The system always picks its best reference, even if every reference is wrong.

Close with: Our next step is better distributed reference coverage, a genuinely independent route/date test and local geometric verification. ETH's system also incorporates aerial imagery and operates at much larger scale; ours is a smaller ground-image retrieval approach.

## If time is only two minutes
Run Kottayam in fine mode, explain the two descriptors and nearest reference, open Maps, state63% within25km and15.9km median, then acknowledge the failure case. Skip global mode and the comparison controls.

<!-- PAGE -->
# Recovery and likely questions

## If something fails
Missing fine option or fallback warning: install the fine assets and restart; do not claim a fallback cell prediction is L10. A missing reference photo means the bank can still return coordinates but this checkout lacks the original image file. Failed external Maps loading does not invalidate local inference; read the returned coordinate and use the local map. If a browser-to-browser image drag is blocked, save the image locally and use Upload. If prediction errors, read the terminal; do not repeatedly click Locate.

## Is it memorizing the examples?
The neural encoders transform unseen query pixels into descriptors. The bank intentionally stores training examples: this is retrieval. Official demo query IDs and training IDs are disjoint, but related routes or scenery can still make an evaluation easier. This is why spatial and sequence separation matter as well as file-ID separation.

## Can it predict more than300 places now?
Yes: fine mode returns a reference-photo coordinate from15,939 training entries. Some entries may share coordinates. It cannot freely predict any coordinate in the world, and it still needs a relevant nearby reference.

## Why not just increase to10,000 classes?
That creates more sparsely populated classes and demands more geographically balanced data. Retrieval expands output positions without retraining a huge classifier head, though retrieval storage and coverage become new limits.

## Why not average the nearest five coordinates?
Visually similar neighbors can be hundreds of kilometers apart. A blind average can land between unrelated places. We evaluated spatially local averaging, but the equal ResNet-DINO top-one method won the precision-first validation comparison.

## What did the new data contribute?
789 unique Panoramax images were encoded storage-efficiently. The reviewed24-reference expansion did not change validation enough, so it was not promoted. Do not say the final model was trained on those789 images. Keeping a negative result is better evidence than claiming an unmeasured improvement.

## What should I study tonight?
Read Fine-Retrieval-Update first, then the Study Guide's definitions, preprocessing, architecture, classification, KNN, evaluation and limitations. Explain the current pipeline aloud without notes. Rehearse the five-minute route twice, including the failure and the difference between a test query and a training reference.
