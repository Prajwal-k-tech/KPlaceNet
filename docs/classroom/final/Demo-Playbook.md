# KPlaceNet: live demonstration guide
Current application / 21 October 2026

## 1. Start and verify
Open a terminal and run these commands exactly:

```bash
cd ~/Documents/Codex/2026-10-20
cd https-github-com-asterisk-hunter-kplacenet
./outputs/Start-KPlaceNet.sh
```

Open http://127.0.0.1:7861/ . Keep the terminal open. If you are already inside `work/KPlaceNet`, use `../../outputs/Start-KPlaceNet.sh` instead. Localhost means this laptop; another device's localhost will not reach it.

The launcher chooses the working GPU environment if available, otherwise CPU. The first prediction loads model resources and is slower. After installation, local inference does not need internet; external Google Maps does.

## Rehearsal check before presenting
1. Set browser zoom to 100%. Select **Kerala area · regional model**.
2. Select **Fine coordinates · ResNet + DINOv2**.
3. Click the **Kottayam** thumbnail. Wait for its image to appear, then click **Locate photo**.
4. Check that the result says **Retrieved location**, shows coordinates and **Appearance similarity (cosine)**, and displays a separate training reference.
5. Check the **Open in Google Maps** link. It should open the same returned coordinate.
6. Rehearse the **Southern Kerala** example as a failure. Keep the Study Guide and cheat sheet locally available.

If the Fine option falls back or reports missing assets, do not present the cell result as fine retrieval. The model-artifact installer is `python scripts/download_fine_retrieval.py`, run from the repository in its prepared Python environment. It installs weights and banks, not every Python dependency or the image dataset.

## What you will demonstrate
One uploaded photograph becomes a learned visual descriptor, is matched against training references, and yields a coordinate. You will show a useful match, explain its source, compare the cell option briefly, and show one failure. The regional fine result is the focus; World is a separate optional extension.

<!-- PAGE -->
# 2. Five-minute demo: actions and explanation

## 0:00-0:35 — establish the task
**Action:** Show the upload area and geographic scope.

**Say:** KPlaceNet estimates a location from one photograph's visual content. Our main demonstration assumes a Kerala-area street image. It does not use GPS metadata from the uploaded image or search the live web for its identity.

## 0:35-1:35 — run the main example
**Action:** Choose Kerala area and Fine coordinates. Click Kottayam, wait for the photo to load, then click Locate photo.

**Say:** The gallery loads a query image; Locate runs inference. This query comes from the official test archive. The reference bank contains only training entries, so the model is matching a held-out query to a different known-location photograph.

**Point out:** The retrieved coordinate, cosine score and separate reference image. In the verified run, the reference is ID 309630957741047, near 9.67611, 76.42105, with similarity about 0.812. Treat this as a rehearsal expectation, not a required score on every device or image decoding path.

**Say:** The coordinate belongs to the retrieved training photograph. The similarity is an appearance score, not an 81.2% probability that the location is correct.

**Action:** Open Google Maps and return to the app.

**Say:** Maps shows the returned coordinate; it does not correct or independently validate it.

## 1:35-2:35 — explain the method
**Say:** ResNet produces 2,048 features, and DINOv2 produces 384. A feature vector is a learned numerical description of an image. We normalize each vector to unit length, combine them with equal similarity weight, and compare the result with 15,939 stored training descriptors. The most similar reference supplies its GPS coordinate.

**Define if asked:** An encoder makes a descriptor from pixels. Cosine similarity compares vector direction. Nearest-neighbor retrieval chooses the most similar stored example. No training or weight update occurs when Locate is clicked.

**Clarify:** Fine retrieval uses one best reference; it does not average five coordinates or adjust the fine coordinate with the classifier's predicted cell.

<!-- PAGE -->
# 3. Finish the five-minute demonstration

## 2:35-3:20 — compare the cell method
**Action:** With the same Kottayam image, select **Regional cells · ResNet + KNN**, then Locate photo. Return to Fine coordinates and rerun before continuing.

**Say:** This option uses ResNet features to find five training neighbors and blend their geographic class votes with the classifier: 80% classifier, 20% neighbors. It still selects one of 300 cells and returns its center. Fine mode uses ResNet plus DINOv2 and returns a reference photograph's coordinate instead.

**Define:** A class is a category; a geographic cell is a location category; a centroid is its stored representative center. KNN means K-nearest neighbors. K-means, which creates regional classes from GPS coordinates, is a different algorithm.

## 3:20-4:00 — state the measured result
**Say:** On 357 official test photographs, the current fine method puts 225, or 63.03%, within 25 km and 62, or 17.37%, within 5 km. The median error is 15.92 km. Median means the middle of the ordered errors. This is not exact-street accuracy, and the test benchmark has been reused during the project.

**If asked about stronger validation numbers:** Many random validation photos are close to training locations. Their 1.07-km median is an easier setting and is not the official test result. Do not substitute it in the presentation.

## 4:00-5:00 — demonstrate a real limitation
**Action:** Select Southern Kerala, wait for the image, then Locate photo in Fine mode.

**Say:** This example retrieves a northern reference near 12.45793, 75.09220 in our checked run. The result is wrong. Different roads can look similar, and a sparse bank may not contain a suitable nearby reference. The system always chooses its best available regional match; it has no validated outside-region rejection.

**Conclude:** We removed the fixed 300-center restriction for fine mode, but reference coverage and visual ambiguity still limit precision. The reference display lets us inspect the evidence rather than treating a coordinate as automatically correct.

## If only two minutes are available
Run Kottayam in Fine mode. Explain two encoders, normalized features, nearest reference and returned GPS. Open Maps. State the 25-km result and median with their limits. Show or mention the Southern Kerala failure. Skip comparison controls and World mode.

<!-- PAGE -->
# 4. Optional World-mode demonstration
Use this only after the regional core, or when asked about the other mode.

## What to do
Choose the Global examples tab, select a photo, wait for it to load, and confirm that geographic scope is World. Click Locate photo. Expand Prediction details. Switching modes changes resources, not merely the map zoom.

## What to explain
World uses a separate ResNet classifier over 300 global geographic cells. It does not use the Kerala DINO reference bank. The output is a cell center, not the coordinate of a retrieved global photograph. No regional fine-accuracy figure applies to this mode.

**Softmax** converts class scores into values summing to one. **Temperature scaling** divides class scores by a positive fitted number before softmax, changing their concentration without changing the winning class. The installed value is about 3.9387.

**ACCEPT** means the maximum scaled score is at least 0.5. **ABSTAIN** means it is below 0.5. The app can still show a coordinate when abstaining: that is its highest-ranked candidate, not an accepted location. Neither decision proves correctness or reliably detects foreign input.

A **conformal prediction set** contains candidate class labels under a cutoff derived from calibration data. Set size counts labels, not kilometers. A large set can cover the true class while being geographically imprecise. Target set coverage is not top-one location accuracy.

## The installed calibration limitation
**Calibration** concerns whether model scores agree with observed correctness on suitable data. The current World calibration file lacks a checkpoint fingerprint, so the app marks its identity match as unverified. Say this plainly. Do not claim a proven live 90% coverage guarantee, calibrated correctness probability, or distance improvement from temperature scaling.

## If asked which mode is better
They solve different geographic settings with different methods. Regional fine retrieval is the project's main demonstration and has the stated regional evaluation. World provides broad but coarse classification and uncertainty controls. A wider map is not evidence of better localization.

**Return to:** Kerala area and Fine coordinates when the optional explanation is finished.

<!-- PAGE -->
# 5. Questions you should answer directly

## Are we using KNN?
Yes, in two different ways. Fine mode uses one-nearest-neighbor search over combined ResNet-DINO descriptors and returns a reference coordinate. The regional cell-plus-KNN option uses five ResNet neighbors to vote for classes, then returns a class center.

## Were the examples used in training?
The gallery query IDs come from official test archives. The search bank contains training IDs only. The separately displayed match is supposed to be a training photo. Disjoint IDs do not rule out nearby roads or similar capture sequences, so spatial independence still matters.

## Does the place caption give the answer away?
No. Selecting a thumbnail loads its image and matching scope. The image encoder receives processed pixels, not the displayed place caption or file ID. The label is for the audience.

## Can it predict more than 300 locations?
Fine mode can select coordinates from 15,939 training entries, some with repeated coordinates. The class methods still have 300 center outputs. Fine mode is reference lookup, not unrestricted coordinate regression.

## Why not average several nearby-looking references?
They may look similar while being geographically far apart. Averaging their GPS could produce an unrelated point. The deployed precision-selected method uses one best combined-descriptor match, not a coordinate average.

## Is it overfitting?
Overfitting means exploiting development examples in a way that fails on new data. One successful curated photo cannot rule it out, and one failure cannot prove it. The concrete concerns are nearby validation images, reused test evaluation and limited coverage. The bank intentionally stores training references because that is how retrieval works.

## Does ETH use our method?
No. ETH/Google combines learned ground prototypes with aerial embeddings and fine geographic cells. We combine two ground-image encoders and retrieve actual reference photographs. We have not implemented their aerial fusion or reproduced their result.

## Did the new dataset improve deployment?
Not yet. We encoded 789 unique Panoramax candidates, reviewed a subset and tested a filtered 24-reference expansion. Validation did not improve, so it was not added to the deployed bank. Do not say the final encoder was trained on those images.

<!-- PAGE -->
# 6. Recovery and final rehearsal

| Symptom | What to do and explain |
|---|---|
| Site will not load | Run the launcher from page 1 and keep its terminal open. If another app instance is already running on 7861, open it rather than launching duplicates. |
| First prediction is slow | Wait for model loading. Do not repeatedly click Locate. Later queries reuse cached models. |
| Browser image drag fails | Save the actual image to the laptop and use Upload. Some source websites provide page links or block image fetching. |
| Fine assets missing | Install with scripts/download_fine_retrieval.py in the prepared environment and restart. Do not label a fallback cell answer as fine retrieval. |
| Coordinate appears but reference photo is absent | The compact bank can predict without the original image files; this checkout needs the corresponding training image to display it. |
| Maps link fails | Internet may be unavailable. Read the returned coordinate and show the local map; local inference can still work. |
| Prediction raises an error | Read the terminal. Preserve the working environment and artifact paths; do not start retraining during the demo. |
| Model is wrong | State the error, show the reference, and explain coverage or ambiguity as possible causes without inventing a tested explanation. |

## Useful output distinctions
A cosine score is not probability. A cell box is not a confidence interval. A map marker is an estimate, not ground truth. A conformal set counts class labels, not a GPS radius. A selected test example is a demonstration, not an accuracy estimate. The 300 background map dots are cell centers, not the entire fine reference bank.

## Rehearse this sequence twice
Start the server. Run Kottayam in Fine mode. Explain where its coordinate comes from. Show the reference and Maps. Compare the class-plus-KNN option. Return to Fine mode. State the regional test metric with its distance threshold. Run Southern Kerala and explain the failure. Answer why World is a separate classifier and why its loaded calibration is unverified.

You are prepared when you can perform this sequence without reading a script and can explain encoder, descriptor, normalization, cosine similarity, reference bank, class, centroid, calibration and abstention in your own words. The Study Guide defines each and explains the full implementation.
