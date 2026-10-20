# KPlaceNet demo playbook
## A Kerala-first presentation you can rehearse
Use this during practice and keep it beside the laptop during the live demonstration. The Study Guide explains the concepts; this playbook tells you what to do, what the audience should see and how to explain it.

## The aim of the demonstration
Show the complete path from photograph to regional estimate, explain the ResNet + KNN combination, distinguish a model score from benchmark accuracy, and demonstrate one real limitation. A successful presentation includes an honest failure; it is not a sequence of selected correct markers.

**Main route:** Kerala-area mode, Kottayam example, prediction and Google Maps, KNN comparison, southern-edge failure, measured results. World mode is optional after the main route.

## Start the site without assistance
Open a terminal and run these two commands. The first enters the task workspace; the second starts the prepared launcher.

```bash
cd /home/prajwal-k/Documents/Codex/2026-10-20/https-github-com-asterisk-hunter-kplacenet
./outputs/Start-KPlaceNet.sh
```

Open **http://127.0.0.1:7861/**. Keep the terminal open for the whole demonstration. Press Ctrl+C in that terminal when you want to stop your server. If the site is already working, use the running instance rather than starting another on the same port.

The launcher selects the prepared GPU environment if CUDA works, otherwise CPU. The first prediction loads resources; later predictions reuse them. Do not change drivers or install packages immediately before presenting.

## The exact final setup
Select **Kerala area - regional model** and keep **Refine with training-photo neighbors (KNN)** checked. The interface uses a centered dot instead of the hyphen in the scope label; the meaning is the same.

The final regional system is L6 ResNet plus the optional L7 bank. L6 alone is the comparison when you uncheck KNN. The bank contains training photographs only. Neither setting can verify that an arbitrary upload comes from Kerala.

<!-- PAGE -->
# 1. Prepare before the audience arrives
## Ten-minute preflight
| Do this | Confirm this | If it fails |
| --- | --- | --- |
| Open the local site | Controls and map appear; no error page | Use the launcher and read its terminal output |
| Choose Kottayam, wait for the preview, then Locate photo | Regional cell 196, approximately 9.6761 N, 76.4041 E | Check that regional scope and installed model match |
| Inspect the score label with KNN checked | Combined retrieval score | Raw model score means classifier-only; check bank warning |
| Open the Maps link | A new tab opens at the predicted coordinate | Use the in-app coordinate; Maps requires internet |
| Rehearse the southern-edge example | You can explain the failure without guessing a cause | Review the example notes in Section 4 |

Warm the same examples you will actually show. Then return to Kottayam with KNN checked. Use a readable browser zoom and keep the page at the top before starting. Close unrelated notifications and tabs that could distract from the explanation.

## Keep these files available
Open the final presentation and its PDF backup. Keep Demo-Screenshot.png as a recorded-result fallback, the one-page cheat sheet for numbers, and this playbook for recovery steps. The study guide is for preparation rather than reading aloud on stage.

## Verify assets if needed
From the task workspace, this checks the model files for both modes; it is not a substitute for a real prediction:

```bash
cd work/KPlaceNet
.venv/bin/python scripts/check_demo_ready.py --mode all
```

The regional files are checkpoints/l6_kerala_geographic/selected.pt, its matching cells.json, and checkpoints/l7_kerala_hybrid/train_bank.pt. The bank is required for the hybrid. The training image collection itself is not needed for an already-installed feature bank, but the gallery's example image files must be present.

On another properly set-up checkout, install the final regional artifacts with `python scripts/download_kerala_l7.py`. It downloads the matching L6 model and L7 bank with file checks. Code, weights, bank and demo images are separate resources; copying only the Git repository does not supply all of them.

<!-- PAGE -->
# 2. Opening and first prediction
## 0:00-0:30 - State the task and the assumption
**Action:** show the regional controls before uploading anything. Point to the scope selection and KNN checkbox.

**Say:** 'KPlaceNet estimates a photograph's geographic region from its pixels. Our main demonstration is the Kerala-area specialist. It uses a ResNet image representation and an optional nearest-neighbor refinement. The output is the center of a predicted geographic cell, not an exact camera address.'

**Add the assumption:** 'Regional mode assumes an image from its training area. It does not first prove that the image was taken in Kerala, and the dataset rectangle includes nearby territory.'

This opening prevents the audience from confusing a regional classifier with a worldwide image-search engine or a GPS reader.

## 0:30-1:10 - Load the known example
**Action:** choose the Kerala examples tab, then click **Kottayam**. Wait until the photograph appears. Confirm regional scope and the checked KNN switch. Press **Locate photo** once.

**Expected:** the highlighted result is cell 196, approximately latitude 9.6761 and longitude 76.4041. With the final lossless gallery path and matching artifacts, the recorded hybrid score is about 59.3%; treat that as an observed score, not a value to promise for differently encoded input.

**Say:** 'The example is a selected official-test photograph. Clicking it only loads the image. Locate photo runs the prediction; it does not learn from the image or use the caption as the answer.'

## 1:10-1:45 - Explain the pipeline using the screen
**Action:** point from the photograph to the marker, then to the score.

**Say:** 'We resize and center crop the RGB image to 224 by 224 pixels and normalize it. ResNet extracts 2,048 visual features. A learned head scores 300 geographic classes. With KNN enabled, five similar training-image features contribute a smaller class vote. The winning class supplies the displayed coordinate.'

Do not claim to know exactly which tree, sign or building caused this prediction; the current demo does not establish such an interpretation. Explain the computation you implemented rather than inventing a visual explanation.

<!-- PAGE -->
# 3. Maps, the score and the KNN switch
## 1:45-2:10 - Give the coordinate geographic context
**Action:** select **Open in Google Maps**. Point to the location, then return to the app. If internet is slow, continue using the displayed coordinate instead of waiting.

**Say:** 'The link sends this same predicted coordinate to Google Maps. Maps gives us geographic context; it does not verify the result or refine the model.'

For this selected photograph, the dataset reference is approximately 9.6977 N, 76.4195 E. Comparing that labeled reference with the predicted centroid gives about **2.93 km great-circle error**. This error was computed from the saved official-test metadata; the website cannot calculate error for an arbitrary upload without a reference label.

## 2:10-2:55 - Explain and compare KNN
**Action:** uncheck the KNN refinement, then select Locate photo again. Inspect the result. Recheck KNN and select Locate photo again.

**Expected:** this Kottayam photograph remains in cell 196 under both settings. The score changes. With KNN off the recorded classifier score is about 61.8%; with it on, the combined score is about 59.3%. A changed score need not change the winning class.

**Say:** 'Unchecked is the L6 classifier alone. Checked combines 80% classifier scores with 20% votes from the five cosine-nearest training descriptors. Both branches use the same ResNet features. KNN searches image appearance, not the unknown GPS of this upload.'

**Then clarify:** 'This example stays in the same cell. On other images the vote can change the winning class. Whether that helps has to be measured across the test set; a switch that moves the marker is not automatically an improvement.'

## Explain the score without mixing quantities
The website's **combined retrieval score** is relative support for the winning class. The **53.5% within-25-km benchmark** is the fraction of 357 labeled test images with error at most 25 km. One is an output for an image; the other is a measured frequency over a dataset. Neither is a promise for the next upload.

If someone asks why the hybrid score is lower on this successful example, explain that the weighted neighbor vote distributes support differently. A lower uncalibrated maximum is not itself evidence of lower location accuracy. Here both settings return the same coordinate.

<!-- PAGE -->
# 4. Failure, measured improvement and close
## 2:55-3:40 - Show the southern-edge failure
**Action:** with KNN checked, select the final regional gallery example, captioned **Southern Kerala**, wait for loading and press Locate photo.

**Expected:** the final hybrid predicts cell 190 around 12.4615 N, 75.0805 E. Its labeled dataset reference is around 8.1469 N, 77.4784 E, so the error is approximately **546.76 km**. The gallery caption is a descriptive example label, not an audited administrative-state label; this record comes from the regional bounding-box dataset.

**Say:** 'This is a failure. Similar-looking training appearances can pull the result far north. The model still has to choose among its available regional classes. A plausible marker or a high-looking score does not prove the location is correct.'

If time permits, uncheck KNN and predict again: L6 alone returns about 10.4983 N, 76.2576 E, with approximately 293.78 km error. Both fail, and retrieval makes this particular case worse. Recheck KNN before returning to the main setup.

## 3:40-4:30 - Connect examples to the benchmark
**Action:** show the results slide or point to this table. Do not estimate accuracy from how many curated gallery examples look good.

| Method | Within 25 km on 357 test images | Median error |
| --- | --- | --- |
| L5 regional baseline | 41.18%: 147 images | 42.08 km |
| L6 geographic smoothing | 47.62%: 170 images | 28.41 km |
| L7 ResNet + KNN | 53.50%: 191 images | 21.84 km |

**Say:** 'L6 updates only the classifier head using geographic soft labels. The final fixed KNN blend improves the median and within-25-km result further. It is not better on every image: its upper-tail p90 error is worse than L6. L7 also reuses the already-inspected test benchmark, so we present it as a follow-up, not fresh independent confirmation.'

## 4:30-5:00 - Close with a clear contribution
**Say:** 'The project demonstrates a complete image-to-region pipeline, an evaluated training improvement and a training-only retrieval refinement. My documented work also strengthens evaluation partitions and provenance. The remaining limits are regional assumptions, coarse cell centers, spatially optimistic validation and uncalibrated scores.'

Return to the Kottayam example with KNN checked if you want a neutral screen for questions. World mode is an optional extension, not required for this five-minute route.

<!-- PAGE -->
# 5. Controls and optional extensions
## What every action does
| Control | Actual behavior |
| --- | --- |
| Geographic scope | Chooses regional or global weights and geographic classes; clears the previous result |
| Gallery tab | Shows that group of examples; changing only the tab is not inference |
| Gallery photograph | Loads its pixels and associated scope; does not run prediction yet |
| Upload / local file drop | Supplies a photograph; does not train the model |
| KNN checkbox | Enables regional refinement when the matching bank is installed; clears the old result |
| Locate photo | Runs inference for the loaded image and current configuration |
| Map pan / zoom | Changes the view, not the predicted coordinate |
| Prediction details | Shows the pipeline, score details and stored geographic bounds |
| Google Maps link | Opens the same predicted centroid on an external map |

## If asked to upload an unfamiliar photograph
Prefer a saved local JPEG or PNG of a street scene whose true location is known independently. Load it, confirm scope and predict. An arbitrary scenic, aerial or indoor photograph can be outside the training appearance domain even if it was taken near Kerala.

Explain what the model produces without announcing an exact error unless you have the labeled reference. Do not keep changing photos until a favorable result appears and present that sequence as representative evaluation.

Browser-image dragging can send a page URL rather than an image file, and some sites block cross-origin fetching. If it fails, save the actual image locally and upload the file. This is an input-transfer problem; it is not evidence about the model's geographic accuracy.

## Optional World-mode demonstration
Choose Global examples, select United Kingdom, wait for the loaded scope, then Locate photo. The recorded result is around 52.6339 N, 1.5874 W and is marked abstained. The global model displays a historical temperature-scaled score with unverified calibration-artifact identity. The KNN switch does not refine World mode.

Say: 'This is a separate global model and class map. Abstained means it did not meet the acceptance threshold, not that the software crashed or that the candidate must be wrong.' Do not apply the separate corrected global study's 90.8% set coverage to this live model or the regional hybrid.

<!-- PAGE -->
# 6. Answers for interruptions and viva
## 'Are these examples from training?'
The gallery images come from official test archives. The regional KNN bank contains only 15,939 training descriptors. The examples are curated, however, so their success rate is not the benchmark. A thumbnail click loads pixels; it is not a caption-to-answer lookup.

## 'What is KNN doing that ResNet is not?'
ResNet extracts the visual descriptor. The classifier maps it to learned class scores. KNN compares that descriptor with stored training descriptors and votes over their labels. The final blend is 80% classifier and 20% neighbor votes, not an average of two map coordinates.

## 'Why does the Great Wall get Mysuru or another regional place?'
Regional mode assumes an input from its dataset area. Neither the classifier nor its neighbor bank has an established outside-region rejection rule. The source rectangle also includes nearby states. A forced winner among regional choices is not a verified match to China or proof of Kerala membership.

## 'Is the displayed percentage the chance it is correct?'
No. Regional classifier and hybrid scores are uncalibrated relative class scores. The 53.5% figure is measured within-25-km frequency on the labeled benchmark, which is a different quantity.

## 'Is overfitting solved?'
No. The method improves measured aggregate results, but the validation set has many nearby training views. Test performance is lower, tail failures remain and L7 reuses an inspected benchmark. Geographic/sequence holdouts and fresh data are needed for stronger evidence.

## 'Did you reproduce the ETH paper?'
No. The paper is related work involving ground prototypes, aerial representations and geographic interpolation. This project uses geographic label smoothing and training-image KNN, not its complete aerial/prototype system. PlaNet is the original stated classification inspiration.

## 'Did the new 350 images train this improvement?'
No. Those are Commons candidate metadata records needing review. L6 and L7 use the restored original regional training subset. Do not call the records a finished training dataset.

## 'Why show a failure if the project improved?'
Average improvement and individual reliability are different. The failure demonstrates that limitation and prevents selected examples from being mistaken for representative accuracy.

## If you do not know an answer
State what is implemented and measured, then identify what would need checking. Do not invent an exact visual cause, a calibrated guarantee, a training run or a source-paper result. Use the Study Guide and saved experiment records for follow-up.

<!-- PAGE -->
# 7. Recovery and final rehearsal
## Recover without derailing the talk
| Problem | Immediate action |
| --- | --- |
| Localhost does not load | Check the launcher terminal and printed address; run the two commands on the opening page if needed |
| Address already in use | Check whether the site already works; stop only your own previous server with Ctrl+C before restarting |
| First prediction takes longer | Allow resource loading to finish; warm the demo before the talk |
| Preview changes but old result disappears | Expected reset behavior; wait for loading, then press Locate photo |
| KNN result says classifier-only | Read the missing-bank warning; install the matching bank before class or clearly demonstrate L6 alone |
| Prediction raises a model error | Read the terminal; check checkpoint/cells/bank compatibility; do not replace weights with a random model |
| Maps or map geography will not load | Use the displayed coordinate and saved screenshot; external assets may need internet or cache |
| Browser-image drop fails | Save a real image file and use Upload |

If the live system is unavailable, open Demo-Screenshot.png and the presentation PDF. Explicitly call the screenshot a recorded run. It is a backup for explanation, not a live prediction. Avoid an untested driver change or dependency upgrade during the presentation.

## Rehearse once slowly, then once to time
First pass: explain why each step exists, including the meaning of the score and cell center. Second pass: perform the five-minute sequence without reading complete paragraphs. Keep these prompts visible: **task - image - pipeline - Maps - KNN - failure - evidence - limits**.

A good rehearsal includes an unsuccessful upload recovery and a question about the Great Wall. Confirm you can distinguish K-means from KNN, explain the 80/20 blend and give the correct L5/L6/L7 result figures.

## Evidence for the two scripted examples
The dataset references come from data/kerala_osv5m/kerala_test.csv. Kottayam image ID: 1083931132408866. Southern-edge image ID: 905069160348732. Final predictions were checked with the installed L6 checkpoint and matching training bank. Errors above use src/eval.py's haversine function on those coordinates; they are per-example calculations, not benchmark averages.

For full results use docs/results/l6_kerala_20261020/ and docs/results/l7_kerala_knn_20261020/. For concepts and definitions use the companion Study Guide. Before submitting the separate report, fill its institution, department, instructor and roll-number fields and confirm group attribution.
