# KPlaceNet Live Demo Playbook

Prajwal Kumar K

CSE 311 Artificial Intelligence

## Your job during the demonstration

Lead the audience through one image, one model choice, one prediction, and one explanation. You should be able to trace every screen element back to the computation that produced it. The frontend is a Gradio application with Plotly maps; the geographic prediction comes from the PyTorch ResNet model behind it.

Use this opening: We estimate where a street photograph was taken using its pixels. We predict a geographic class and display its saved centroid. The two modes have different geographic scopes, and their confidence outputs have different calibration status.

## Recommended order

1. Start with Kerala regional mode and a previously verified held out example. Explain the restricted task before clicking anything.
2. Select Locate photo and describe the marker, class bounds, coordinate readout, and raw score.
3. Open Research details. Explain why raw softmax is not a measured probability of being within 25 km.
4. Show one verified failure case. State what remains uncertain and why this is expected for ambiguous street images.
5. Switch to World global mode, select a matching global example, and explain calibrated confidence, the prediction set, and abstention.
6. If time and the required dataset are available, use the separate GeoGuessr game as an optional audience interaction.

A good first example is chosen because you have verified its behavior on the exact model installed for class. The supplied example captions are curated claims from a previous setup, not a fresh guarantee that the same image works on every checkpoint.

## Current readiness on this laptop

The frontend has been launched and inspected locally. Scope switching, the research accordion, UI construction, and software tests have been checked. No real image prediction has been verified here because the trained checkpoints, matching regional cell file, and example images are absent. The repository ignores checkpoints and downloaded data intentionally. A visible interface does not mean its trained model is available.

The integrated code is in Prajwal-k-tech's fork and existing PR 1. Asterisk-Hunter main needs a maintainer merge because Prajwal-k-tech lacks direct push permission. No force push is necessary for this integration.

<!-- PAGE -->
# Know every frontend control

## Geographic scope

Kerala regional model loads the L5 regional checkpoint and regional cell map. World global model loads the configured global checkpoint and any matching calibration artifact. Changing scope changes the model and the meaning of all 300 class IDs. It is not simply a zoom operation. The same number of outputs does not mean the same classes.

The page starts in Kerala mode. Changing the mode clears the displayed prediction and resets the map so that an older result is not mistaken for a prediction from the newly selected model.

## Street scene and Locate photo

Upload or paste an image. Gradio provides it to Python as a PIL image. Merely choosing an image resets the old result; Locate photo triggers inference. An empty input produces a prompt to choose a photo. A model loading failure is an operational error, not an abstention decision.

## Map and prediction summary

The small region dots represent saved class centroids. The highlighted marker is the centroid of the winning class. The selected cell bounds show a stored geographic extent. They are not a 90 percent confidence interval. Hover and zoom help inspect geometry; changing the map view does not run the neural network again or alter the predicted coordinate.

Latitude and longitude are the decoded class centroid, not a separate coordinate regression output. A marker with four decimal places should not be described as precise to that many places. Display formatting does not improve geographic accuracy.

## Benchmark strip and demonstration examples

The Kerala benchmark strip shows aggregate historical results on 357 official test photos, even if global mode is selected. Read the Kerala label aloud; it is not the accuracy of the current upload or a global benchmark. The reported figures are approximately 42 km median error and 41 percent within 25 km.

Examples appear only if their image files exist in the expected folders. Empty galleries currently reflect missing local image files. Selecting an available example sets its image, matching mode, and caption; press Locate photo to compute its output. Curated successes and failures illustrate behavior, not average accuracy.

## Research details

Expand this panel to inspect cell ID, class bounds, raw and scaled scores, temperature, conformal cutoff, set size, and threshold decision. Kerala displays no fitted temperature, no conformal set, and no accept or abstain threshold. Global displays calibrated fields only when the matching artifact is available; an absent artifact is explicitly a raw fallback.

<!-- PAGE -->
# Explain what happens after the click

## The inference path

Locate photo calls run_prediction in demo/app.py. That passes the image and chosen mode to predict_one. Kerala routes to its regional function; global uses the global loader. A resource cache keeps each loaded model available for later predictions. The first successful inference can take longer because it loads weights; later calls reuse them.

Both paths convert the image to RGB, apply deterministic evaluation transforms, and create a batch with one image. Resize to 256 pixels and center crop to 224 are the defaults. Tensor conversion and ImageNet normalization align the input with the pretrained encoder's expected scale. The final input has shape 1 x 3 x 224 x 224.

The ResNet 50 backbone produces 2,048 features after global average pooling. The geographic linear head maps these to 300 logits. Softmax yields a distribution over the model's cells. The largest probability identifies the predicted cell, and that cell's saved centroid supplies latitude and longitude.

Global mode may divide logits by a positive fitted temperature before softmax. It then forms a candidate class set and compares maximum probability with the configured 0.5 abstention threshold. The top class and centroid remain unchanged by positive temperature scaling. Kerala uses raw softmax and no fitted uncertainty policy.

Finally Python builds a Plotly map, a summary card, and research text. Gradio updates those outputs in the browser. PyTorch performs inference server side; the browser does not train a CNN or obtain location from the upload's GPS metadata.

## Why there is no distance error for an arbitrary upload

Distance error requires a known reference location. The model has a prediction, but a normal uploaded photo does not automatically provide a trusted ground truth. On labeled test data, haversine distance compares the predicted centroid and known coordinate. For a live upload, describe the estimate and uncertainty without inventing a measured error.

## Why saved cells and weights must travel together

Head row j was trained to mean cell j in one saved map. A different 300 cell map changes the meaning of that same output. Strict weight loading checks architecture compatibility; checkpoint metadata and centroid checks also protect geographic meaning. Global and regional heads cannot be interchanged just because both have 300 rows.

Code route to memorize: build_ui -> run_prediction -> predict_one -> resource loader -> get_transforms -> GeoClassifier -> softmax -> cell centroid -> render_map -> result formatting.

<!-- PAGE -->
# Explain why the design choices make sense

## Why a CNN and transfer learning

Street scenes contain spatial visual patterns, so convolutional features provide a natural image representation. ResNet uses shortcut connections to make deep feature learning easier. ImageNet initialization supplies reusable visual features and reduces the amount of geographic training required. The original classifier is replaced by a geographic head; this is adaptation rather than training every visual filter from scratch.

Why ResNet 50 rather than another backbone: it is the implemented and measured choice within the laptop compute budget. Do not claim that every larger model ran out of memory or every smaller model failed unless you can show a corresponding experiment.

## Why cells rather than direct latitude longitude regression

Classification gives a distribution over several possible places and a straightforward supervised objective. Direct coordinate regression must handle spherical geometry, longitude wraparound, and multiple plausible locations. It remains possible, but it is not the model implemented here. Cells impose quantization error: the predicted centroid can be far from the true point even if the cell is correct.

## Why 300 cells and a Kerala specialist

At 10,000 global training images, 300 classes average only about 33 examples each. More classes give finer resolution but reduce evidence per class. Regional specialization makes each of 300 cells cover a much smaller part of the map. It also gives the model a regional prior by construction. Kerala's results therefore describe a different, restricted task.

The Kerala model warm starts the global backbone, replaces the global head, and fine tunes layer4 plus the regional head. Global features can transfer; the global class meanings cannot. The regional cells are built from regional training coordinates only. No regional calibration has been fitted in the recorded setup.

## Why fine tune only layer4 and the head

This controls the number of trainable parameters and preserves much of the pretrained representation. Early filters are reused while the final block adapts to the task. In the current training loop, frozen parameters do not necessarily freeze batch normalization running statistics because model.train is called. Mention this distinction if asked about exact implementation behavior.

## Why explicit failure and provenance checks

A frontend can display a convincing map even with random weights, a mismatched class map, or calibration from another model. The integrated pipeline rejects incompatible artifacts, records sample identities and hashes, and clearly labels raw confidence. These engineering checks protect the meaning of the demonstration; they do not themselves increase geolocation accuracy.

<!-- PAGE -->
# Explain confidence without overclaiming

## Raw score, calibrated confidence, and distance accuracy

A raw score is the largest softmax value. It reflects the model's relative preference among its classes. It is not automatically the chance of being within 25 km. Temperature scaling fits one positive scalar on separate labeled examples to improve agreement between confidence and observed class correctness. It cannot move the winning centroid because class ordering stays the same.

Expected calibration error groups predictions by confidence and compares their confidence with observed class accuracy. The revised 20k model report records ECE decreasing from 0.4171 to 0.0453 on 1,500 evaluation rows. Those figures apply to that exact model and split; the default historical L2 demo checkpoint is a different artifact unless explicitly configured otherwise.

## Candidate sets and the finite sample correction

For each conformal calibration image, score one minus its assigned true class probability. With n examples and target miscoverage alpha, select the kth smallest score where k = ceil((n + 1)(1 - alpha)). Include candidate classes whose score is at most that cutoff. If k exceeds n, the conservative cutoff is infinity.

The three partitions have distinct jobs: 500 rows fit temperature, 1,000 different rows fit the conformal cutoff, and 1,500 different rows evaluate the finished procedure. Exact identities are recorded. The standard guarantee is marginal and assumes exchangeability; it does not guarantee coverage in each town or under geographic shift.

The revised report gets 90.8 percent empirical class set coverage, but the mean set includes 159.7 of 300 cells. This is broad uncertainty. Do not say that the system is 90.8 percent accurate at locating photographs or within a distance radius. The set can contain many geographically dispersed classes.

## Abstention

Global mode labels a prediction accepted when its maximum probability reaches the configured threshold. Abstained means the policy declines to trust the estimate, even though a candidate marker can still be displayed for inspection. It does not mean the app crashed or proved the estimate wrong. Accepted also does not prove correctness.

At threshold 0.5 the revised evaluation abstained on every row. This is a legitimate result and illustrates why thresholds need validation at the desired operating point. Do not lower the threshold during class and claim improved model accuracy. Kerala has no equivalent fitted decision policy in this setup.

<!-- PAGE -->
# A five minute live script

## First minute

Open the page full screen and choose Kerala regional model. Say: This mode assumes the photo comes from the Kerala domain. It uses regional classes and a specialized checkpoint. The global mode has different cells and a different model, not simply a different map zoom.

Select a verified held out example. Say: This is a curated demonstration example, not evidence that every upload behaves the same way. We also have an explicit failure example.

## Second minute

Press Locate photo once. While it runs: The image is converted to RGB, resized and cropped to 224 pixels, normalized, and passed to ResNet 50. Its 2,048 features become 300 class logits. We select a class and display its saved centroid.

Point at the marker, cell dots, and bounds. Say: The marker is a representative class center. Displayed decimal places do not mean GPS precision. These bounds describe the class geometry, not a calibrated confidence interval.

## Third minute

Open Research details. Say: Kerala reports raw softmax, without a fitted temperature or conformal policy. This score is not a measured probability of being within 25 km. The benchmark strip is a separate aggregate test result: about 42 km median error and 41 percent within 25 km across 357 official test photos.

Show a failure example. If the output is far away, explain that similar looking street scenes and limited regional coverage can lead to errors. Do not claim a specific causal visual cue was used unless a separate interpretability analysis supports it.

## Fourth minute

Switch to global mode and choose its verified example. Say: This mode predicts among worldwide cells. With the matching calibration artifact it also shows temperature scaled confidence, a candidate class set, and a threshold based decision. Calibration adjusts how confident the model is; it does not improve the predicted coordinate by itself.

If it abstains, use that outcome: The system still computes a candidate region, but its decision policy declines to trust it. This is why we report uncertainty separately from localization accuracy.

## Fifth minute

Explain your contribution: I corrected the finite sample conformal cutoff, separated the fitting and evaluation subsets, added checkpoint and provenance checks, and implemented controlled follow up tools. The original model pipeline and its historical results are attributed separately.

Close with the bounded findings: more data helped across three paired seeds; balanced sampling did not consistently help; global localization remains weak; regional specialization is more useful in its restricted domain. Take questions with the code route and metric distinctions ready.

<!-- PAGE -->
# Get the exact demo pack and rehearse

## Files needed from the training machine

For Kerala, copy checkpoints/l5_kerala_kmeans300/last.pt and checkpoints/l5_kerala_kmeans300/cells.json together. Also copy a few agreed demonstration images from data/kerala_osv5m/test/images while retaining their filenames. The UI's configured examples include 319636679783973.jpg, 1083931132408866.jpg, 313393843511505.jpg, 288990592896618.jpg, and failure example 905069160348732.jpg. Verify each after transfer.

For the default global demo, copy checkpoints/l2_imagenet_1.0_layer4/last.pt, its matching checkpoints/l3_uncertainty_results.json, and the matching full cell file used as checkpoints/l2_cells_full.json. Copy selected images into data/osv5m_test/images. Example filenames are listed in demo/app.py. The calibration and full bounds files are optional for raw inference, but their absence changes what can be demonstrated.

If using the later 20k checkpoint instead, use its own calibration and cell map. The committed 9 October uncertainty JSON is associated with checkpoint SHA 256 588b572a3c524a244b20a47d2a5ff2dc2e206e54a87f9144cf49ad19910d0e85. A different checkpoint must not borrow that calibration. The KPLACENET_CHECKPOINT, KPLACENET_CALIBRATION, and KPLACENET_CELLS environment variables select the global artifacts. Restart the server after changing them because resources are cached.

The whole training dataset is unnecessary for the upload demo. The separate GeoGuessr game additionally needs its metadata CSV and referenced test images. Ask for a small self consistent game subset if you intend to use it; arbitrary images without matching metadata cannot be scored honestly.

## Start and verify

From the repository root, use the project's Python environment. Run python scripts/check_demo_ready.py --mode all for file presence and available calibration checks. A success here is preliminary; load each model and run actual images afterward. Then start python demo/app.py and open the printed localhost URL. The classroom theme is supported on Gradio 5 and 6.

Rehearse with one verified success and one failure in each available mode. Record the selected checkpoint, raw or calibrated status, and observed output for those exact examples. Warm both models once before class. Check map loading with the classroom network; Plotly geographic assets may need network or a previously cached copy. Keep the existing slide PDF and saved genuine predictions as a backup, clearly identified as recorded outputs.

If samples do not appear, check the image folder and filenames. If loading fails, check the checkpoint and cell map before changing code. A calibration hash mismatch requires the matching model artifact. Slow first inference may be model loading or CPU execution. Do not replace missing trained weights with random weights for a presentation.

## Questions to answer before class

What changes when scope changes? Why is the output a centroid? Why is the Kerala score raw? What does abstain mean? Why does 90 percent coverage not mean 90 percent localization accuracy? Where is ground truth used? Which components did you implement? Why did the official test perform worse than random validation? What is the difference between file readiness, software tests, and real model verification?
