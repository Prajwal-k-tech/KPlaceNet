# KPlaceNet: presentation study guide

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

Updated 20 October 2026. Use this guide for the final L6/L7 demo and tomorrow's viva.

## Your opening explanation
KPlaceNet estimates a photograph's geographic region from its pixels. It converts geography into 300 classes, uses a pretrained ResNet-50 to predict a class, and displays that class's saved centroid. The Kerala specialist assumes a Kerala-domain input. The global mode has different weights and different geographic cells.

My follow-up work strengthens evaluation integrity and reproducibility, integrates the demonstration, and tests a geographic smoothing improvement. The final Kerala run increases the fraction of official test photos within 25 km from 41.2% to 47.6%, and reduces median error from 42.1 to 28.4 km. These are measured results on 357 images, not a guarantee for a new upload.

## The facts to learn first
- Input: RGB image pixels. GPS coordinates provide training labels and evaluation truth, not inference features.
- Model: ResNet-50, 2,048 features, linear layer with 300 outputs, softmax, highest-scoring cell, saved centroid.
- Kerala cells: K-means over training coordinates. K-means is not KNN.
- Kerala data: 15,939 training images, 1,000 random validation images, 357 official test images.
- L5 fine-tunes layer4 and the classifier. L6 freezes that backbone and adjusts only the classifier using geographic soft targets.
- L6 official test: 47.6% within 25 km, median 28.4 km, mean 85.3 km, exact cell accuracy 20.7%.
- Kerala confidence is raw and uncalibrated. The global demo uses historical temperature scaling with unverified calibration identity.
- All ten gallery examples came from official test archives. They are selected examples, not a representative benchmark.

## What you should be able to do
Explain the procedure without slides, predict what each frontend action does, explain one success and one failure, distinguish every metric, and identify exactly which improvements are yours. Read the Demo Playbook after this guide and rehearse with the actual running app.

<!-- PAGE -->
# 1. Data and geographic classes

## What the dataset contains
OSV-5M supplies street-level photographs and labeled coordinates. The project uses bounded subsets, not all 5.1 million source images. The regional extraction uses latitude 8.0 to 12.8 and longitude 74.5 to 77.5. That is a bounding-box slice associated with Kerala, not a carefully audited state polygon. It can include nearby border areas.

The 16,939 source-training rows are split with RandomState seed 42: 1,000 validation rows and 15,939 training rows. The 357 official test rows retain their official split. The current recovered splits have disjoint image IDs. The source imagery is CC-BY-SA-4.0 and requires attribution.

## Why we create cells
A classifier needs discrete target labels. We turn geographic coordinates into 300 classes. Each class stores a cell ID, centroid latitude/longitude and geographic bounds. A training photograph's GPS coordinate determines its target class. The image network then learns to associate visual features with these labels.

K-means groups training coordinates around centers. The implementation scales longitude by cos(mean training latitude) before clustering to approximate local geographic geometry. It does not cluster image pixels. It does not use test coordinates to construct the cells. Saved class geometry stays fixed throughout the L5-to-L6 comparison.

When assigning labels, the implementation first checks cell bounds, resolves overlaps using nearest-centroid haversine distance, and falls back to the nearest centroid if no bounds contain the coordinate. That matters because K-means cluster boxes can overlap.

## K-means, KNN and neural classification
K-means constructs geographic groups during preparation. KNN searches a stored bank of similar image descriptors. L6 alone uses the neural classifier. The final optional L7 extension uses five cosine-nearest training-only ResNet descriptors and blends their votes with L6, as described at the start of this guide.

## Why centroids limit precision
The network chooses a class, not a unique house or camera position. Every image predicted into the same class receives the same centroid. Even a correct class can produce nonzero distance error. More displayed decimal places do not make the model GPS-accurate. More cells can improve resolution but leave fewer examples per class and can make learning harder.

<!-- PAGE -->
# 2. Preprocessing and the complete procedure

## Training preprocessing in L5
1. Read the image path and its labeled latitude/longitude.
2. Decode the photograph and convert it to RGB.
3. Apply RandomResizedCrop to 224 by 224 pixels with crop scale 0.8 to 1.0.
4. Randomly flip horizontally with probability 0.5.
5. Convert pixels to a tensor and normalize channels with ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].

Cropping and flipping expose varied appearances and reduce reliance on one exact view. Normalization matches the expected pretrained backbone input. Augmentation is a training tool, not proof that the model handles every viewpoint or phone photograph.

## Inference preprocessing
Convert to RGB, resize the short edge to 256, center crop to 224 by 224, convert to a tensor and apply the same normalization. A batch has shape B x 3 x 224 x 224. Evaluation transforms are deterministic so the model comparison uses the same view of every image.

## What happens after Locate photo
1. Select the correct cached model and geographic cell map for the chosen scope.
2. Preprocess the uploaded PIL image. The network does not read its filename, gallery caption or GPS tag as location evidence.
3. Run the model in evaluation mode under torch.no_grad(). No weights update.
4. Produce 300 logits. Softmax converts them into a class-score distribution.
5. Select the highest-scoring class and look up its saved centroid.
6. Draw the centroid and class bounds. Show coordinates, score and the Google Maps link.

The gallery click only loads the photograph and selects its scope. Locate photo runs the model. Map zooming and opening Google Maps do not run another neural prediction or make the predicted location more accurate.

## What L6 changes in preprocessing
L6 caches deterministic evaluation-transform features for the existing training, validation and test images. It does not rerun L5's stochastic image augmentation during classifier training. The head-only update therefore changes both the target regularization and the views supplied to the head. Do not attribute the entire gain to one isolated ingredient.

<!-- PAGE -->
# 3. ResNet-50 and transfer learning

## What ResNet contributes
A convolutional backbone converts pixels into a representation of visual patterns. Earlier layers can detect simple structures; deeper representations combine information useful for classification. Residual connections add a block's input to its learned transformation, helping optimization through a deep network. Avoid claiming that a specific neuron detects coconut trees or Malayalam signs unless a separate interpretability analysis demonstrates it.

ResNet-50 ends with global average pooling. The resulting 2,048-dimensional feature vector enters a linear classification head:

z = W f(x) + b

Here f(x) has 2,048 features, W has 300 rows and b has 300 values. The head has 300 x 2,048 + 300 = 614,700 parameters. Its outputs are logits, which can be positive or negative and need not sum to one.

p_i = exp(z_i) / sum_j exp(z_j)

The winning cell is argmax(p). Its centroid becomes the displayed coordinate. The network does not separately regress latitude and longitude.

## Why pretrained weights
Transfer learning starts from a representation trained on many images rather than learning all visual structure from scratch. KPlaceNet uses ImageNet initialization in the global experiments. Kerala L5 warm-starts from the selected global L2 backbone, replaces the regional classifier and fine-tunes layer4 plus the head.

L5 ran ten epochs with batch size 16 and learning rate 0.0003 on the original experiment machine. That historical machine was an RTX 4050. The new L6 experiment ran on this laptop's RTX 3050 6 GB. Keep these hardware histories distinct.

## Evaluation mode and no_grad
model.eval() changes the behavior of layers such as batch normalization and dropout. torch.no_grad() disables gradient tracking. Neither command performs training. An epoch means one traversal of the training set, while a batch is one smaller group processed together. The optimizer updates parameters after training batches.

## Why this architecture
ResNet-50 is the implemented, measured backbone within the project's compute budget. A larger foundation model might help, but the project has not established that every larger model fails or that ResNet-50 is optimal. Explain the design choice as a bounded engineering decision.

<!-- PAGE -->
# 4. Training, loss and the final improvement

## L5 supervised learning
A hard class target identifies one correct geographic class. Cross entropy penalizes low probability on that class: loss = -log(p_true). If the true class receives probability 0.6, its loss is about 0.511. AdamW updates trainable weights and applies weight decay. Learning rate controls update size. Training loss falling does not establish generalization.

L5's recorded training cell accuracy rises to 97.6%. Validation cell accuracy is about 43% and official-test cell accuracy about 17.9%. This is a large generalization gap. It is consistent with overfitting and also reflects spatial distribution differences between splits.

## L6: geographic smoothing
Freeze the already-trained L5 backbone and its 300 cells. Cache the 2,048 image features. Compare two fixed classifier-training recipes, initialized from the same original head: uniform label smoothing, and geographic label smoothing. Each uses AdamW, learning rate 0.0001, weight decay 0.01, batch 256 and 12 epochs, with seed 42.

For geographic smoothing, use a training image's coordinate to compute its distance d_i to each saved cell centroid. Define:

q_i = 0.9 * one_hot_i + 0.1 * softmax(-d_i / 15 km)

This gives most target mass to the assigned class and some credit to geographically nearby alternatives. Ordinary label smoothing spreads that extra mass uniformly across all classes. The 15-km scale is fixed before evaluation. It is unrelated to the temperature fitted for probability calibration.

The soft-target loss is -sum_i q_i log(p_i). It changes the learning objective. It does not inject GPS into inference. At inference the same image-only ResNet classifier still predicts a cell.

## Selection and timing
Select the candidate with the best validation within-25-km rate, breaking ties with median error. Keep the original head as an eligible candidate. Save selection before test evaluation. Geographic smoothing at epoch 12 wins: validation within 25 km is 70.9%, versus 64.6% for L5.

Only the head receives gradients in L6. The two recipes' cached-feature optimization took 3.22 seconds on the GPU. Feature extraction took substantially longer. This is not training an entire ResNet from scratch in three seconds. The feature cache and unchanged cells make the small experiment efficient.

<!-- PAGE -->
# 5. Results and evaluation

## Which numbers belong to which experiment
| Metric | L5 baseline | L6 classifier | Evaluation |
|---|---:|---:|---|
| Within 25 km | 41.18% | 47.62% | 357 official Kerala test images |
| Correct within 25 km | 147 | 170 | Same 357 images |
| Median error | 42.08 km | 28.41 km | Same images |
| Mean error | 102.85 km | 85.33 km | Same images |
| Within 200 km | 80.11% | 84.31% | Same images |
| Exact cell accuracy | 17.93% | 20.73% | Same 300 cells |
| Within 1 km | 0.56% | 0.56% | No improvement |

The within-25-km gain is 6.44 percentage points. Forty-one test images become correct under that radius and eighteen regress. Do not claim every image improved. One seeded run and spatially related samples do not establish statistical significance or universal reliability.

## How distance is measured
Compare the predicted centroid with the labeled coordinate using haversine great-circle distance, in kilometers. Latitude and longitude are angular coordinates on Earth, so a flat pixel-style distance is inappropriate. Longitude wraps around and physical distance per degree changes with latitude.

Within-25-km accuracy counts predictions whose distance is at most 25 km and divides by the number of test images. Median is the middle distance and describes a typical image. Mean averages all distances and is sensitive to large failures. The p90 distance marks the point below which 90% of errors fall; it is not a probability-confidence guarantee.

Exact cell accuracy asks whether the model selected the assigned class. Distance accuracy asks whether the centroid is near enough. A neighboring class can be wrong as a label but still within 25 km. Conversely, a wide correct class can have a distant centroid.

## Results you must not mix
The three-seed global 10k-to-20k study reports within-200-km accuracy 3.756% to 5.300%, and mean error 5,977.3 to 5,521.6 km. Its scope, training data and checkpoints differ from regional L6. The newer 20k global uncertainty evaluation is also different from the historical global checkpoint loaded by default in the app. Never present one experiment's numbers as another model's live accuracy.

<!-- PAGE -->
# 6. Why validation is optimistic

## The measured spatial audit
The restored training, validation and official-test CSVs have no overlapping image IDs. However, 982 of the 1,000 random validation photos are within 1 km of a training photo. Median nearest-training distance is about 99 meters. Nearby street scenes can share vegetation, roads and buildings even when their file IDs differ.

For the 357 official test photos, no coordinate in the recovered subset is within 1 km of a training coordinate. The nearest is 1.005 km and the median nearest-training distance is 2.520 km. These measurements audit our recovered subset, not every image in the full source dataset.

This explains why validation can be substantially easier. L6 validation median is 4.78 km, while official-test median is 28.41 km. Report the official-test number for generalization. A clean identifier split is necessary but does not eliminate spatial dependence.

## How the comparison remains useful
Both models use the same images, backbone, class map and evaluation coordinates. The validation split chooses the new head before its official test score is computed. The observed test gain is therefore a controlled follow-up under this protocol, with the existing validation limitation stated explicitly.

A stronger future study would create validation groups by geographic area and capture sequence, enforce distance buffers, use multiple seeds and preserve a final holdout. Retraining a backbone after selecting geographic holdouts would be necessary if it had already seen those images. Simply moving previously trained-on photos into a new folder does not make them unseen.

## Examples and overfitting
All ten demonstration examples were recovered from official test archives. Their filenames and byte sizes match saved provenance. Clicking a thumbnail does not retrieve a memorized answer from the caption; Locate photo runs inference.

They were deliberately chosen, however, so they cannot estimate average accuracy. Showing only selected successes would hide failure modes. Use one successful regional example, one failure, and an unfamiliar image if time permits. For the unfamiliar image, do not announce an error in kilometers without independently known ground truth.

<!-- PAGE -->
# 7. Confidence, calibration and conformal prediction

## Three different outputs
A softmax score measures relative class preference inside the model. It is not automatically the probability of being correct, within 25 km, or anywhere inside the displayed rectangle. Kerala L6 uses raw softmax without a fitted calibration or abstention threshold.

Temperature scaling divides global logits by a fitted positive T before softmax. A larger T usually softens the distribution. It preserves class ordering, so it changes confidence rather than the winning class or centroid. It cannot by itself improve localization distance.

The live global calibration is historical and contains no checkpoint identity hash. The UI therefore says temperature-scaled score with identity unverified. The later, corrected 20k study has separately recorded model provenance; do not transfer its stronger calibration evidence to the legacy demo.

## Conformal sets
On separate calibration images, score each true class as s = 1 - p_true. For n calibration examples and desired miscoverage alpha, choose rank k = ceil((n + 1) * (1 - alpha)). The cutoff is the kth smallest score. If k exceeds n, use infinity. Include classes with 1 - p_class at most that cutoff.

Example: four scores [0.1, 0.2, 0.4, 0.8] with alpha = 0.2 give k = 4 and cutoff 0.8. With alpha = 0.1, k = 5, beyond n, so the conservative cutoff is infinity. A naive interpolated percentile can give a different finite-sample cutoff.

## Your evaluation-integrity contribution
The corrected global workflow separates 500 rows for temperature fitting, 1,000 different rows for conformal calibration and 1,500 different rows for final evaluation. It records sample identities and model hashes. On that specific 20k-model study, ECE falls from 0.4171 to 0.0453 and empirical coverage is 90.8%, with a mean set of 159.7 of 300 classes.

Coverage means the set contains the true class, not that the top location is correct. Large sets can achieve coverage while being unhelpful geographically. The usual guarantee is marginal under exchangeability, not a guarantee for every town, image source or distribution shift. Abstention is a policy threshold, not a crash and not a proof that a prediction is wrong.

<!-- PAGE -->
# 8. The ETH paper and more Kerala data

## What the ETH paper contributes
Lindenberger et al., Scaling Image Geo-Localization to Continent Level (NeurIPS 2025), combine learned ground-view prototypes with aerial embeddings. They interpolate geographic representations and retrieve over cell codes to address precision and coverage. Their system uses different data, encoders, losses and geographic resolution from KPlaceNet. Its reported accuracy cannot be borrowed for our project.

Our L6 experiment takes the broad boundary-awareness idea and implements geographic soft training labels. It does not implement their aerial encoder, S2-cell hierarchy, prototype interpolation or multi-similarity loss. The repository's original stated inspiration remains PlaNet. Treat the ETH work as relevant later literature and an improvement direction, rather than rewriting the project's origin story.

Source: https://scaling-geoloc.github.io/ and https://arxiv.org/abs/2510.26795 .

## Additional-data research
Wikimedia Commons yielded 350 candidate records around seven Kerala cities, with geographic coordinates, source pages, author fields and recorded free licenses. The Munnar query was unavailable during collection. These records are candidates, not 350 reviewed training photographs. No Commons image contributed to L6 training, validation or official-test results.

The candidate manifest includes nearest-distance checks against recovered OSV training and test coordinates. None of these candidates is within 1 km of the recovered official test coordinates, but coordinate separation alone does not rule out duplicate imagery, incorrect metadata or unsuitable views. Review the actual photos before any use.

Mapillary can supply street-level imagery and geographic metadata, but collection requires API access and attention to source overlap with OSV-5M. The India Driving Dataset's original scenes are around Hyderabad and Bangalore and focus on road-scene understanding; it is not a ready Kerala-coordinate training set. Global Mapillary collections require an actual coordinate filter before claiming Kerala coverage.

## A credible next experiment
Validate new photo licenses and camera coordinates, remove visual duplicates, group by location and sequence, freeze a geographically separate validation/test protocol, and only then train. A cross-view extension additionally needs appropriately licensed aerial imagery aligned with ground photos. New data can help only if the labels, coverage and evaluation are trustworthy.

<!-- PAGE -->
# 9. Your code walkthrough and contribution

## Trace one prediction
Open src/dataset.py for get_transforms and RGB decoding. Open src/model.py for GeoClassifier, the 2,048 features and the linear head. Open src/cells.py for cell construction and coordinate-to-class assignment. Open demo/app.py for predict_one, _predict_kerala and _result_html. This is enough to explain one frontend prediction end to end.

For L6, open scripts/run_l6_kerala_head.py and geographic_targets. Show that it computes target distances only from training coordinates, freezes features, trains both fixed recipes and saves selection_before_test.json before evaluating the official test. Show docs/results/l6_kerala_20261020/results.json for measured results and test_predictions.csv for per-image evidence.

For uncertainty, open src/uncertainty.py at fit_conformal_quantile. Explain ceil, kthvalue and the rank-beyond-n case. Open the corrected global study's recorded partition and provenance files. Strict state loading and checkpoint class-map checks prevent silently pairing incompatible weights and classes.

## A defensible contribution statement
The existing project supplied the global and Kerala modeling pipeline and historical L5 weights. My contributions include the conformal finite-sample correction, separation of calibration/evaluation roles, provenance and checkpoint checks, controlled follow-up experiments, integration of the current demo, and the L6 geographic smoothing experiment and spatial split audit. Attribute prior work separately rather than claiming the full project or published papers as your own.

## Repository status
Changes are pushed to Prajwal-k-tech's fork and the existing PR into Asterisk-Hunter/KPlaceNet. The original main still needs maintainer approval and merging. The L6 weights are distributed separately in a release because large trained checkpoints are not stored in Git. A code merge alone does not install the weights.

## Questions to answer with the code
Where are labels created? Which weights change in L5 and L6? What is the model input? What happens when an image is selected? Why do two scopes both have 300 outputs? Where do coordinates come from? Does Google Maps improve the estimate? What does an abstention mean? Which result belongs to the exact installed checkpoint?

<!-- PAGE -->
# 10. Viva practice and rehearsal

## Short answers
Do we use KNN? Yes, the final optional L7 refinement searches five training-only ResNet feature neighbors and blends votes 80/20 with L6. K-means separately constructs the geographic classes.

Why Kerala? Restricting the region lets training focus on finer local distinctions. The user supplies that regional assumption. The app does not prove a photograph comes from Kerala.

Can any photo work? The model always computes scores for valid images, but street-level training limits generalization to aerial views, indoor scenes and other photo styles.

Is 47.6% the score shown for an upload? No. It is the proportion of 357 labeled test images within 25 km. The live raw score is a different quantity.

Did you add new training data today? No. We restored the existing Kerala slice and improved the head. We separately collected 350 additional-data candidate metadata records for future review.

Did the ETH method run here? No. L6 is a small geographic-label adaptation. The aerial and prototype-retrieval system remains future work.

Is overfitting fixed? No. Official-test performance improves, but a substantial gap and domain limitations remain. Within-1-km accuracy is still only 0.56%.

Why can the failure photo look like another region? Similar appearances and incomplete learned distinctions can confuse the classifier. Without interpretability analysis, do not claim the exact visual cause of a particular failure.

## A two-hour study plan
Spend 30 minutes explaining data, preprocessing and the model using pages 1 to 4. Spend 25 minutes on L6/L7, metrics and the spatial audit. Spend 25 minutes on uncertainty, contribution and the ETH comparison. Spend 25 minutes rehearsing the live demo twice, including the failure and the Maps link. Use the final 15 minutes for the cheat sheet and questions without notes.

## Before you sleep
Be able to give the opening in 45 seconds, trace the pipeline in two minutes, explain the improvement in one minute and recover from a failed upload. Confirm the report's institution, department, instructor and roll numbers before submission on 22 October 2026. Keep the final slides PDF and genuine prediction screenshots available offline.

Primary evidence: project code and docs/results/l6_kerala_20261020/; OSV-5M dataset https://huggingface.co/datasets/osv5m/osv5m ; PlaNet https://arxiv.org/abs/1602.05314 .

<!-- PAGE -->

# Explain the final KNN refinement

## Why this particular KNN blend?
The neighbors act as a local memory of training appearances. The classifier compresses those appearances into learned weights; retrieval can recover useful local evidence that the head spreads across several cells. We blend cell probabilities rather than averaging coordinates, which could place a result between unrelated places. Five neighbors, a similarity temperature of 0.07 and a 20% vote were fixed as a small conservative experiment, not established as optimal settings. More weight could amplify bad visual matches. The Southern Kerala failure demonstrates that risk.

KNN here searches image features, not GPS coordinates. Training GPS supplies each neighbor's class label; an uploaded photograph supplies no GPS. The bank uses roughly 125 MiB for features and needs extra similarity computation. It does not train the ResNet again. KNN memory is normal for that method; leakage would occur if validation/test photographs were included in its bank, which this run explicitly forbids. Spatially adjacent training scenes can still make random validation optimistic even with disjoint IDs.


## A twenty-second answer
ResNet extracts a visual descriptor. We find five training photographs with similar descriptors using cosine similarity. Their location-class votes contribute 20% and the learned classifier contributes 80%. We predict the class with the highest combined score and display its stored centroid. Test photographs never enter the neighbor bank.

## If asked why it can still fail
A nearest neighbor always exists, even when no photograph in the bank is a good match. Similar greenery, masonry or road texture does not establish a country. The regional model has no learned foreign-image rejection class. We must evaluate an outside-region detector on independent examples before claiming one. The Great Wall screenshot demonstrates this limitation.

## If asked whether this proves generalization
No. The hybrid improved a reused official benchmark after validation selection. Its high-error tail worsened, the random validation split has nearby street scenes, and only one fixed retrieval recipe was tested. A new geographically separated dataset and repeated runs are the next evidence needed. Do not describe test-set reuse as an untouched final confirmation.
