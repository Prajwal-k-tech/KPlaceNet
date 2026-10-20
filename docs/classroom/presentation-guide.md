# KPlaceNet Presentation and Viva Guide

Focus: explain what happens, how it happens, and why each choice was made. This guide follows the teacher's emphasis in the supplied screenshot.

## Your opening explanation

Our project estimates where a photograph was taken using its pixels. We turn geography into 300 classes, use a pretrained ResNet 50 to predict a class, and decode that class to a saved geographic centroid. We evaluate distance error separately from confidence. My contributions focus on conformal correctness, evaluation integrity, reproducibility, and controlled follow up experiments.

## The procedure to explain without looking at notes

1. Obtain bounded OSV 5M train and official test subsets. Keep their source identities separate.
2. Read image_path, lat, and lon. Coordinates provide supervision; they are not image input features.
3. Build geographic cells using training coordinates. Save the cells and reuse them across compared models.
4. Assign each training coordinate a cell ID. Load the image as RGB and apply training transforms.
5. Feed B x 3 x 224 x 224 tensors through ResNet 50, producing B x 2048 features.
6. Use a linear head to produce B x 300 logits. Cross entropy compares logits with cell labels.
7. Optimize head parameters, or layer4 plus head, with AdamW. AMP reduces GPU memory and compute costs.
8. Save weights, cell definitions, class count, arguments, and provenance together.
9. At evaluation, use deterministic transforms, strict weight loading, and only the checkpoint's saved class map.
10. Predict the top class, look up its centroid, and calculate haversine error against the known label.
11. Fit temperature on 500 held out rows, the conformal cutoff on a separate 1,000, and evaluate on the remaining 1,500.
12. Report error thresholds, calibration, coverage, set size, retained count, and geographic slices.

## One concrete image example

Suppose an image's true coordinate is near Kochi. Its coordinate assigns it a regional target cell during training. The photo becomes a normalized 224 x 224 RGB tensor. The network emits 300 logits; the highest logit identifies a saved cell, whose centroid is returned as the location. If that centroid is 18 km from the label, this example counts as within 25 km and 200 km, but not within 1 km. A high confidence score does not itself tell us the distance. Kerala confidence is currently uncalibrated.

<!-- PAGE -->
# The neural network and training

## Why classification instead of raw coordinate regression

Cells let the model express multiple competing geographic hypotheses and use standard cross entropy. A raw latitude longitude regression can average distinct plausible locations and must handle spherical geometry and longitude wraparound. Classification has its own limitation: a centroid is a coarse representative and class boundaries can separate nearby places. Do not say that regression is impossible.

## What ResNet contributes

Convolutions learn visual patterns, residual blocks learn transformations added to shortcut features, and global average pooling produces a 2,048 dimensional representation. ImageNet pretraining supplies a useful starting point. The original 1,000 class ImageNet head is removed and replaced with a geographic head.

The 300 class head has 2048 x 300 + 300 = 614,700 parameters. Frozen means backbone parameters have requires_grad false. Layer4 fine tuning updates the last residual block plus the head. Because the current loop calls model.train(), batch normalization running statistics may still update in earlier blocks. Be precise about frozen parameters versus all model state.

## Explain the loss and optimizer

For an image with true class y, cross entropy penalizes low assigned probability for y. In PyTorch the loss receives raw logits and integer targets; it handles log softmax internally. Backpropagation calculates gradients, and AdamW updates the selected trainable parameters with decoupled weight decay. An epoch is a pass through the configured sampling budget; a batch is the group of examples used for one update.

Training transforms: RandomResizedCrop 224 with scale 0.8 to 1.0, horizontal flip probability 0.5, tensor conversion, ImageNet normalization. Evaluation transforms: resize 256, center crop 224, tensor conversion, identical normalization. Explain why evaluation is deterministic.

## Fair comparisons and their limits

The 10k versus 20k study uses seeds 42, 43, 44, one fixed 300 cell map, identical test rows, layer4 fine tuning, batch 16, ten epochs, and final epoch checkpoints. Within each seed the 10k rows are nested in the 20k pool. The test sample does not choose the checkpoint.

Ten epochs on 20k means roughly twice as many optimizer updates as ten epochs on 10k. Say that the experiment tests a larger dataset under a shared epoch recipe. It does not isolate data diversity at fixed compute. Three seeds support a consistent observed direction, not a broad significance claim.

<!-- PAGE -->
# Your contribution in detail

## The conformal bug and correction

For each conformal calibration row, calculate s = 1 - probability assigned to its true cell. Sort the n scores. For desired miscoverage alpha, choose rank k = ceil((n + 1)(1 - alpha)). Use the kth score, not a linearly interpolated percentile. If k exceeds n, use infinity and include every class.

Worked example: scores 0.1, 0.4, 0.6, 0.9 with n = 4 and alpha = 0.4 give k = ceil(5 x 0.6) = 3. The cutoff is 0.6. Candidate classes with probability at least 0.4 enter the set. In the small sample edge case n = 9, alpha = 0.01, rank is 10; clamping to score 9 is incorrect.

## Why three partitions

Temperature fitting learns T. Once T is fixed, the conformal subset measures scores and fits q. The evaluation subset measures the completed procedure. Sharing the temperature fitting labels with conformal fitting breaks the clean split conformal setup because the scoring transformation was fitted on those same examples. Disjoint seeded membership is recorded explicitly.

## What temperature changes

For positive T, softmax(logits / T) changes probability sharpness and preserves class ordering. The top cell, centroid, and distance error stay unchanged. ECE improved from 0.4171 to 0.0453 on 1,500 rows, but that is confidence calibration rather than an accuracy gain.

## What the guarantee actually says

Under exchangeability and a fixed scoring function, split conformal targets marginal coverage over future examples. It does not guarantee 90% coverage in every country or for every photograph. Spatial dependence and geographic shift matter. The measured 90.8% overall coverage coexists with regional values between 59.5% and 100% in sufficiently populated bins.

## What provenance means

Record sample identities, split indices, seeds, metadata and checkpoint SHA 256 hashes, code revision, arguments, and library versions. A hash identifies file contents; it is not proof that data labels are correct. Checkpoint validation rejects absent weights, wrong head dimensions, invalid cell IDs, and known calibration model mismatches. These checks prevent meaningless evaluation from silently succeeding.

Your defensible attribution: I implemented the conformal rank correction, separate fitting partitions, provenance and regional diagnostics, checkpoint and evaluation checks, optional cell balancing, controlled follow up experiment tooling, tests, and documentation. The original ResNet pipeline and historical upstream metrics belong to the original work.

<!-- PAGE -->
# Results to remember and questions to expect

| Experiment | Number to remember | Meaning |
| Paired 10k to 20k | 3.756% to 5.300% within 200 km | More data helped under this recipe |
| Paired mean error | 5,977.3 to 5,521.6 km | Error still large globally |
| Sampling comparison | 5,849.9 versus 5,859.0 km | No consistent balancing benefit |
| Revised calibration | ECE 0.4171 to 0.0453 | Confidence improved |
| Revised conformal | 90.8%, mean 159.7 of 300 cells | Coverage is broad, not precise |
| Kerala official test | 41.2% within 25 km, median 42.1 km | Separate restricted geography task |

## Is the model 90 percent accurate

No. The 90.8% figure is prediction set coverage on a specific 1,500 row partition. Many candidate cells are included. The same model gets 5.27% within 200 km on that partition. Coverage, cell accuracy, and geographic accuracy measure different things.

## Why use haversine

Latitude longitude lie on a sphere, not a flat Cartesian plane. Haversine measures great circle distance. One degree of longitude at the equator is about 111 km, while the physical distance changes with latitude. Longitude also wraps at the antimeridian.

## Why is validation so much better in Kerala

The random holdout can share nearby street environments with training. The official test uses the dataset's held out protocol and is the number to emphasize for generalization. Validation median 7 km versus test median 42.1 km reveals an important gap. Kerala also constrains every prediction to one region, so compare against a measured region restricted baseline before attributing all improvement to visual learning.

## Why did cell balancing fail

It did not consistently improve the measured setup. Inverse frequency weights equalize expected class sampling, but can repeat sparse class images and do not create new geographic evidence. The three seed results have mixed directions. Do not invent a single proven causal explanation.

## Why 300 cells and why not a larger model

With 10k images, 300 cells give only about 33 examples per class on average. A larger output space gives finer resolution but less supervision per class. The small data and laptop budget motivated transfer learning and partial fine tuning. A larger head alone does not solve missing geographic evidence.

<!-- PAGE -->
# Rehearsal and code walkthrough

## Ten minute presentation sequence

Minute 0 to 1: state the problem and show a street photo. Explain why visual clues are ambiguous.

Minute 1 to 3: draw image to preprocessing to ResNet to features to logits to cell centroid. Explain that training coordinates provide labels and are absent from inference inputs.

Minute 3 to 5: explain the controlled experiments, fixed cells, paired seeds, nested samples, and final epoch evaluation. Show the data scale improvement and sampling negative result.

Minute 5 to 7: explain temperature scaling and your conformal correction using the four score example. Draw the 500 / 1,000 / 1,500 split. Distinguish coverage and set size.

Minute 7 to 9: show a demo if the exact weights and cells are available. Clearly identify global or Kerala mode. Show one success and one failure rather than implying every photo works.

Minute 9 to 10: conclude with limitations and your exact contribution. Keep three facts ready: low global accuracy, broad prediction sets, and optimistic random regional validation.

## Files to open in order

src/dataset.py: get_transforms and GeoDataset. Explain CSV validation, RGB decoding, and train versus evaluation transforms.

src/cells.py: build_cells and assign_cells. Explain training only class geometry, centroid metadata, box ambiguity, and haversine fallback.

src/model.py: GeoClassifier and unfreeze_backbone. Show 2,048 features, a 300 class head, and layer4 at sequential index 7.

src/train.py: model construction, optimizer, training loop, and checkpoint dictionary. Explain loss, gradient updates, AMP, and saved cells.

src/eval_contract.py and src/eval.py: show why missing or incompatible artifacts cause a clear failure.

src/uncertainty.py: fit_conformal_quantile. Point directly to ceil, kthvalue, and the rank greater than n case.

scripts/run_l3_uncertainty.py and src/reproducibility.py: show the three partitions, sample manifests, and model hash.

docs/results/l2_data_scale_osv_test_20261009.md and docs/results/l3_uncertainty_osv_test_20261009.md: trace a slide number to a dated report and associated JSON.

## Practical preparation before class

Run unit tests and plan only CLIs first. The real demo additionally needs local model checkpoints, matching cells, and any calibration artifacts. These are not in Git. Existing historical reports are evidence of recorded runs; passing synthetic tests does not reproduce them. Keep the existing presentation PDF as an offline backup. Add Kerala results as a separately labeled optional section if using that demo.

Complete the report title page with confirmed roll numbers, institution, department, and instructor. The report includes all thirteen requested sections and cites the original papers and project records.

## Final self check

Explain the complete procedure in two minutes without slides. Calculate the conformal rank by hand. Distinguish training accuracy, within 25 km accuracy, ECE, coverage, set size, and abstention. Explain one failure case and identify precisely which work is yours.
