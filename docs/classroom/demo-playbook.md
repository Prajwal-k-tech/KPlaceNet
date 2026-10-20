# KPlaceNet: final live-demo playbook
20 October 2026. Final Kerala model: L6 geographic smoothing.

## Start on this laptop
Open a terminal in the task workspace and run:

./outputs/Start-KPlaceNet.sh

Open http://127.0.0.1:7861/ and refresh with Ctrl+Shift+R after server updates. Keep the terminal open. The launcher prefers the separate GPU environment when its CUDA check succeeds and otherwise uses the established CPU environment. It applies workspace-local matching NVIDIA libraries only while the loaded kernel is 610.57.04; it does not replace system drivers.

If a server is already running, use it instead of starting a second process on the same port. Stop your server with Ctrl+C before restarting. The first prediction loads weights and later predictions reuse them. Run one photo in each mode before class to warm the caches.

## Confirm the installed model
From work/KPlaceNet, run:

.venv/bin/python scripts/check_demo_ready.py --mode all

The Kerala checkpoint should be checkpoints/l6_kerala_geographic/selected.pt, with its matching cells.json. The app prefers this published pack if installed, otherwise it falls back to L5. File presence is preliminary: actually run a photograph after the check.

To install L6 on another checkout, use python scripts/download_kerala_l6.py. It verifies SHA-256. The standard python scripts/download_weights.py supplies the historical global/L5 release assets. Install Python dependencies and download assets before class. Git does not contain trained weights or the image dataset.

## Screen setup
Use the browser full screen at a readable zoom. The desktop layout puts photo controls, map and prediction in three columns. At narrower sizes it wraps vertically. Choose Kerala mode before your first example. Keep the final presentation PDF and genuine prediction screenshot open as an offline backup.

<!-- PAGE -->
# A five-minute demonstration

## First minute: explain the task
Say: This model estimates a region from the pixels in a street photograph. Kerala mode assumes a Kerala-domain input; it uses different weights and cells from global mode. The output is a representative cell center, not an exact camera position.

Open the Kerala examples tab and click Kottayam. A thumbnail click loads the image and scope. It does not predict yet and it does not learn from the example. All ten examples came from official test archives, but they are selected demonstrations rather than a random accuracy sample.

## Second minute: run inference
Click Locate photo once. Say: We convert to RGB, resize and center crop to 224, normalize, extract 2,048 ResNet features and choose one of 300 classes. We then look up that class's centroid.

Point at the highlighted marker, coordinates and raw score. For the Kottayam example the verified final run predicts cell 196 at approximately 9.6761 N, 76.4041 E. A regional result can be useful without being an exact address. Avoid quoting its error without the stored ground-truth record in front of you.

Click Open in Google Maps. It opens a new tab at the same predicted coordinate. Maps adds geographic context; it does not verify or refine the neural estimate. Return to the app.

## Third minute: show the limitation
Select Southern Kerala and click Locate photo. The final L6 run predicts around 10.4983 N, 76.2576 E, still substantially north of the source scene. Explain that plausible-looking visual patterns can lead to a wrong region and the model has no Kerala calibration or abstention policy. Do not describe the raw score as the chance of being within 25 km.

## Fourth minute: explain the improvement
Say: We froze the backbone and cells and trained the classifier using a target that gives some credit to nearby geographic cells. We selected the recipe and epoch on validation, then evaluated 357 official test images. Within 25 km improves from 41.2% to 47.6%, with median error falling from 42.1 to 28.4 km. Some individual images still regress and fine precision remains weak.

## Fifth minute: global mode and your contribution
Select Global examples, then United Kingdom, then Locate photo. The verified run predicts around 52.6339 N, 1.5874 W and abstains. Its score uses historical temperature scaling with artifact identity unverified. The newer corrected 20k calibration study is a separate experiment.

Close by identifying your own evaluation, reproducibility, integration and L6 contributions while crediting the existing project pipeline. Invite methodology questions with the code files ready.

<!-- PAGE -->
# Every interaction and fallback

## Uploads
Saved JPEG/PNG files can be dropped into Street scene or chosen using Upload. The browser-link drop helper also accepts image URLs or dragged image HTML when the source permits fetching. Some search engines provide page links or block cross-origin image requests. In that case save the actual image locally and upload it. Clipboard access depends on browser permissions.

A new image or changed scope clears the old prediction. A gallery selection chooses the associated scope. The two example tabs expose photographs; switching a tab alone is not an inference run. The result appears only after Locate photo.

## Map and details
The small dots are saved cell centers. The large marker is the selected centroid. The outline is stored cell extent, not a confidence interval. Pan and zoom change the view only. Prediction details contains technical confidence and class-bound information. The benchmark banner and course labels have been removed from the website.

## Failure recovery
If the site does not open, check that the server terminal is running and use its printed address. If the port is occupied, use the existing process or stop that server before restarting. If inference fails, check the checkpoint and matching cell file rather than substituting random weights.

If CUDA is unavailable, the launcher falls back to CPU. The original desktop NVIDIA library mismatch does not require changing system drivers for this demo. The GPU environment is separate from the working CPU environment.

Google Maps and external image fetching require internet. Plotly's geographic assets can also require network or a warmed browser cache. If the classroom network fails, show the saved real prediction screenshot and the final slides PDF, identifying them as recorded outputs. Do not imply an offline screenshot is a new inference result.

## What to carry
Keep the final presentation PPTX and PDF, this playbook, the study guide, report, cheat sheet and original result JSONs. Keep model weights and matching cells on the demo laptop. The full restored training dataset is not needed for inference. The separate GeoGuessr game still needs global metadata and is not part of the rehearsed final demo.

## Code route
For image preprocessing open src/dataset.py. For the network open src/model.py. For prediction and Maps links open demo/app.py. For geographic smoothing open scripts/run_l6_kerala_head.py. For measured evidence open docs/results/l6_kerala_20261020/results.json and selection_before_test.json.
