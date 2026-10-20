# KPlaceNet: Kerala demo guide
Kerala-area demonstration / 21 October 2026

## 1. Start the application
Open a terminal and run:

```bash
cd ~/Documents/Codex/2026-10-20
cd https-github-com-asterisk-hunter-kplacenet
./outputs/Start-KPlaceNet.sh
```

Open http://127.0.0.1:7861/ and leave the terminal open. From inside the repository at `work/KPlaceNet`, use `../../outputs/Start-KPlaceNet.sh` instead. Localhost is this laptop; 7861 is the app's port, a numbered network endpoint.

The launcher chooses the working GPU environment if available and otherwise uses CPU. The first query loads model resources, so it is slower. Later predictions reuse loaded resources. Once installed, model inference runs locally; external Google Maps requires internet.

## Keep these three choices straight
Leave geographic scope on **Kerala area · regional model**. The current app has a **Regional prediction method** selector rather than a separate KNN checkbox.

| Choice | What to explain |
|---|---|
| Fine coordinates · ResNet + DINOv2 | Two visual encoders find one training reference; return its GPS |
| Regional cells · ResNet + KNN | Classifier scores plus five-neighbor class votes; return the winning cell's center |
| Regional cells · classifier only | Classifier scores alone; return the winning cell's center |

When demonstrating the KNN toggle, compare the last two options on the same image. Fine retrieval is a separate method; it also uses nearest-neighbor search, but not the five-neighbor class blend.

## Before class
Set browser zoom to 100%. Run Kottayam in Fine mode once and check that its image, coordinate, cosine score and separate reference photo appear. Test the Maps link. Rehearse classifier-only versus class-plus-KNN, then the Southern Kerala failure. Keep the Study Guide locally available.

A gallery click loads a photograph. Wait for it to appear, then click **Locate photo**. Changing a method clears the old output; click Locate again to run the new choice.

<!-- PAGE -->
# 2. Demonstrate the main Kerala prediction

## 0:00-0:35 — explain the task
**Action:** Show the upload area, keeping Kerala area selected.

**Say:** We estimate where a photograph was taken from its visible appearance. This demonstration assumes a Kerala-area street image. The model receives pixels, not a GPS answer from the uploaded image's metadata, its file name or the gallery caption.

**Define if asked:** Visual geolocation is estimating a place from an image. Latitude and longitude are numbers describing a position on Earth. Our returned coordinate is an estimate, not ground truth.

## 0:35-1:35 — run the example
**Action:** Select Fine coordinates, click Kottayam, wait for the photograph, then Locate photo.

**Say:** ResNet produces 2,048 features and DINOv2 produces 384. A feature vector, or descriptor, is a learned numerical description of appearance. We normalize both to unit length, combine them with equal similarity weight and search 15,939 training references. The highest-similarity reference supplies its stored coordinate.

**Point out:** The query image, result marker, coordinate, **Appearance similarity (cosine)** and separate reference image. In the checked run, the reference ID is 309630957741047, near 9.67611, 76.42105, with cosine similarity about 0.812. Treat these as rehearsal expectations rather than a required score on every hardware or decoding path.

**Say:** This query comes from the official test archive. The bank contains training images only, so the displayed match is a different reference photo. Similarity measures visual resemblance; 0.812 does not mean an 81.2% probability of correct location.

## 1:35-2:00 — show the location
**Action:** Open in Google Maps, show the coordinate's surroundings, then return.

**Say:** Maps displays the coordinate our model already returned. It does not identify the real location independently or improve the prediction. This output is the matched reference's GPS, not a newly generated coordinate.

**If asked about the small map dots:** They are the 300 class centers used by the cell methods, not the full fine-retrieval bank. Fine mode can return coordinates beyond those dots.

<!-- PAGE -->
# 3. Demonstrate what the KNN toggle changes

## 2:00-2:40 — classifier only
**Action:** Keep the same Kottayam image. Select **Regional cells · classifier only**, then Locate photo.

**Say:** ResNet describes the image. A classifier head uses those features to score 300 geographic cells. A cell is a location category; its centroid is a saved center. The highest-scoring cell supplies the coordinate. This option does not search for a particular training photograph to choose its output.

**Define:** A classifier chooses a category. Softmax converts raw scores into values summing to one, but those values are not automatically reliable correctness probabilities.

## 2:40-3:30 — enable regional KNN
**Action:** Select **Regional cells · ResNet + KNN**, then Locate photo.

**Say:** KNN means K-nearest neighbors. Here k is five. We compare ResNet visual features with the training bank, find five most-similar photos and add their similarity-based weights by cell label. We blend 80% classifier scores with 20% neighbor class votes, then return the winning cell's center.

**Clarify:** This blends class scores, not GPS coordinates. It does not move the predicted point within a cell. It may choose another cell, keep the same cell or make a wrong result worse. No training or parameter update happens when the option changes.

## A short numerical explanation if asked
Suppose the classifier gives A = 0.48 and B = 0.45. The neighbors vote A = 0.20 and B = 0.70. The blended score for A is 0.8 x 0.48 + 0.2 x 0.20 = 0.424. For B it is 0.8 x 0.45 + 0.2 x 0.70 = 0.500. B now wins, so its center is returned. Other classes account for the remaining score mass.

**If the marker does not change:** Explain that the classifier and neighbors may agree, or that a 20% vote contribution is too small to overturn the classifier's lead. An unchanged winning cell is a valid outcome.

**Distinguish K-means:** K-means groups training GPS positions to create cells. KNN searches image descriptors during prediction. They are separate algorithms despite their similar names.

**Action:** Return to Fine coordinates and rerun before the final result and failure explanation.

<!-- PAGE -->
# 4. State the result and show the limitation

## 3:30-4:05 — measured performance
**Say:** The current fine method places 225 of 357 official test photos, or 63.03%, within 25 km. It places 62, or 17.37%, within 5 km. Median error is 15.92 km: the middle of the ordered errors. This is not exact-street accuracy, and these figures apply to fine retrieval, not the cell-plus-KNN option.

**If asked how the alternatives compare:** On the same official subset, classifier-plus-KNN achieves 53.50% within 25 km with a 21.84-km median; classifier only achieves 47.62% with a 28.41-km median. These measured aggregate differences do not mean KNN helps every image. The benchmark has been reused during the project, so it is not fresh independent confirmation.

**If asked about stronger validation results:** Many random validation places are near training references. Their results describe an easier setting and must not replace the official test figures.

## 4:05-5:00 — the known failure
**Action:** In Fine mode, select Southern Kerala, wait for the photo, then Locate photo.

**Say:** Our checked run retrieves a northern reference near 12.45793, 75.09220. That is wrong for this example. Similar-looking roads can confuse the descriptors, and the bank may lack a useful nearby reference. The system always returns its best available match; it cannot verify that the image belongs to Kerala.

**Clarify:** The source data were filtered using a rectangle that includes neighboring territory. A point appearing on our regional map does not prove it lies inside Kerala or is correct. A foreign landmark may still receive a regional answer.

**Finish:** The project demonstrates learned visual features, retrieval and a classifier-plus-neighbor combination. Fine mode removes the 300-center output restriction, but its output still depends on reference coverage and appearance matching.

## If only two minutes are available
Run Kottayam in Fine mode. Explain the two descriptors, nearest reference, returned GPS and Maps link. State the 25-km threshold and median. Explain the KNN option in one sentence: five training-neighbor class votes contribute 20% to classifier scores and still return a cell center. Mention the known failure.

<!-- PAGE -->
# 5. Questions, recovery and rehearsal

## Questions to answer directly
**Are the example queries training images?** The gallery queries come from official test archives; the bank contains training IDs only. Nearby scenes can still overlap across splits, so different IDs do not prove geographic independence.

**Is this overfitting?** Overfitting means doing well on development examples without generalizing to new ones. A curated success cannot rule it out and a single failure cannot prove it. Nearby validation locations, reused test evaluation and sparse coverage are concrete limitations.

**Can it predict more than 300 places?** Fine mode selects coordinates from 15,939 reference entries, some with repeated GPS. The cell options have 300 center outputs. Neither is unrestricted coordinate regression, which would directly estimate new numerical coordinates.

**Does KNN correct GPS?** It can change the winning class using similar training examples; it does not correct the true GPS, average reference coordinates or guarantee a closer answer.

**What does the score mean?** Fine mode shows cosine similarity. Cell methods show uncalibrated class scores. Neither tells you the probability that the location is within a specified radius.

## Recovery
| Problem | Response |
|---|---|
| Site unavailable | Run the launcher from page 1 and keep its terminal open. If the app is already running on 7861, open it instead of starting duplicates. |
| First query slow | Wait for resources to load; avoid repeated clicks. |
| Browser-image drag fails | Save the image locally and use Upload. Some sources send a page link or block access. |
| Fine assets missing | Run scripts/download_fine_retrieval.py in the prepared repository environment, then restart. Do not call a fallback cell answer fine retrieval. |
| Reference image absent | The bank can return coordinates without the original image files, but displaying the match requires that training photo locally. |
| Maps unavailable | Show the local map and coordinate; external Maps needs internet. |
| Prediction error | Read the terminal. Do not retrain or change the environment during the demo. |

## Rehearse twice
Start the app. Run Kottayam in Fine mode. Show reference GPS, similarity and Maps. Compare classifier-only with class-plus-KNN on the same photo. Explain the 80/20 blend. Return to Fine mode, state the measured result and run the known failure. Be able to define descriptor, cosine similarity, reference bank, classifier, cell, centroid and KNN in your own words.
