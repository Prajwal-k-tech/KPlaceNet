# KPlaceNet

## Image Based Geolocation with Transfer Learning and Calibrated Uncertainty

CSE 311 ARTIFICIAL INTELLIGENCE

Project Report

Submission deadline 22 October 2026

Group members

Prajwal Kumar K

Chandra Teja P

ShivKarthik Suresh

Hari Krishnan S

Institution ______________________________

Department ______________________________

Instructor ______________________________

Roll numbers ____________________________

Project repository https://github.com/Asterisk-Hunter/KPlaceNet

Contributor fork https://github.com/Prajwal-k-tech/KPlaceNet

<!-- PAGE -->
# 1 Abstract

KPlaceNet investigates whether a single RGB photograph can be assigned a useful geographic location using visual information alone. The system treats geolocation as classification: training coordinates define 300 geographic cells, a pretrained ResNet 50 predicts a cell, and its saved centroid becomes the latitude and longitude estimate. The implementation includes image preprocessing, transfer learning, several cell construction methods, geodesic evaluation, confidence calibration, conformal prediction sets, and an interactive demonstration.

The project contains several distinct experiments. Historical global experiments compare pretrained initialization, training fractions, and cell construction. A later controlled study compares 10,000 and 20,000 training images using the same 300 cell definitions and three paired seeds. Reported accuracy within 200 km rises from 3.756% to 5.300%, while mean geodesic error falls from 5,977.3 to 5,521.6 km. A separate comparison finds no consistent benefit from inverse frequency cell sampling. These results support a bounded data scale observation, rather than a claim of accurate worldwide localization. [R1, R2, R3]

For the seed 42 model trained on 20,000 images, three disjoint subsets are used for temperature fitting, conformal calibration, and final evaluation. Expected calibration error decreases from 0.4171 to 0.0453. Empirical conformal coverage is 90.8%, but the average set contains 159.7 of 300 classes. Better confidence estimates therefore coexist with weak geographic precision. The correction to the finite sample conformal rank and recorded sample provenance make this evaluation auditable. [R4]

A separate Kerala specialist uses regional training images and regional cells. Its reported official test accuracy within 25 km is 41.2%, with median error 42.1 km. These figures describe a restricted task and cannot be compared directly with global results. The overall contribution is a working visual geolocation pipeline, controlled experimental comparisons, and a clear account of what confidence and evaluation results do and do not establish. [R5]

Keywords: visual geolocation, ResNet 50, transfer learning, geographic cells, temperature scaling, conformal prediction.

<!-- PAGE -->
# 2 Introduction

Photographs contain geographic clues in vegetation, road layout, terrain, building appearance, and scene structure. A model can learn associations between these visual patterns and geographic regions. The task remains ambiguous: similar roads or forests occur in distant countries, and many images contain few distinctive cues. KPlaceNet studies this ambiguity through both location error and uncertainty estimates.

The project follows the broad classification idea of PlaNet, which subdivides geography into classes and predicts a distribution over them. KPlaceNet is an educational prototype with a different backbone, cell implementation, and much smaller training budget. It is not a reproduction of PlaNet or its published benchmark results. [1]

OSV 5M supplies street level imagery with geographic labels and an official spatially separated test split. This is more demanding than a random image split because nearby views of the same street can otherwise appear on both sides. The dataset authors describe a 1 km train test spatial separation. Our subsets preserve the official split labels; this report does not claim a new audit of every source coordinate. [2]

The central experimental distinction is between learning a location and estimating whether to trust a prediction. More data or a better representation may improve localization. Temperature scaling can improve probability calibration without changing the predicted class. A large prediction set can cover the truth often while still being geographically uninformative. The project measures these outcomes separately.

# 3 Problem Statement

Given a photograph x, estimate the latitude and longitude at which it was captured, without using GPS or EXIF as input features. During supervised training, geographic metadata supplies the target label. At inference, the model receives only image pixels and the geographic definitions saved with its checkpoint.

Let K be the number of training derived cells. The classifier produces K logits. The winning class is mapped to its saved centroid, and great circle distance to the ground truth measures localization error. The uncertainty task additionally reports calibrated confidence, a candidate class set, and a threshold based abstention decision.

The practical constraints are limited labeled subsets, a single laptop GPU, class imbalance, coarse geographic cells, and spatial dependence. Success means implementing the complete pipeline, testing bounded hypotheses fairly, and identifying limitations. It does not require claiming street level accuracy worldwide.

<!-- PAGE -->
# 4 Objectives

1. Build a complete image to cell to coordinate pipeline using a pretrained ResNet 50 and saved geographic metadata.
2. Compare cell construction, transfer initialization, data scale, and sampling under explicit experimental controls.
3. Evaluate mean and median geodesic error and accuracy within 1, 25, and 200 km.
4. Separate temperature fitting, conformal calibration, and evaluation, and measure both prediction set coverage and size.
5. Record sample IDs, hashes, seeds, and checkpoint compatibility so that reported comparisons can be checked.
6. Demonstrate global and Kerala modes and explain their different geographic scope and calibration status.

# 5 Dataset Description

OSV 5M is the dataset family used for the reported image experiments. The project does not train on the complete collection. Image files, downloaded CSVs, and large checkpoints are intentionally excluded from Git. Committed result documents and manifests record the historical measurements. [2, R1-R5]

| Study | Training sample | Evaluation sample | Important control |
| Historical global L1 to L4 | 10,000 images | 3,000 official test rows | Different from later fork samples |
| Sampling comparison | 10,000 from shards 00 and 01 | 3,000 official test rows | Three paired seeds |
| Data scale comparison | 10,000 nested in 20,000 | A second 3,000 row test sample | Four train shards and fixed cells |
| Revised uncertainty | Seed 42 model trained on 20,000 | 500 fit plus 1,000 calibrate plus 1,500 evaluate | Three disjoint partitions |
| Kerala specialization | 15,939 train and 1,000 validation | 357 official test rows | Separate regional cell map |

The data scale training pool draws 5,000 rows from each of shards 00 to 03. Its test sample spans 169 countries and has no recorded ID overlap with either training or the earlier sampling test sample. ID separation does not imply independent geography. The follow up was chosen after inspecting earlier results and is explicitly exploratory. [R3]

The training CSV requires image_path, lat, and lon. Source IDs support overlap checks where available. Paths are resolved relative to the CSV, coordinates are range checked, and images are loaded as RGB. Metadata labels teach the model but are not concatenated to image features. The Kerala slice is a restricted geographic subset with its own training and test counts. [R5]

<!-- PAGE -->
# 6 Methodology
## 6 1 Overall procedure

The procedure has two inputs during training: images and their known coordinates. Coordinates build the class definitions and assign each image its target class. Images pass through the neural network. The cross entropy loss compares predicted logits with target cell IDs, and the optimizer updates the selected trainable parameters.

Training coordinates -> cell construction -> saved cell definitions

RGB image -> preprocessing -> ResNet 50 -> 2,048 features -> linear head -> 300 logits

Logits -> softmax -> highest probability class -> saved centroid -> predicted coordinates

Held out logits -> temperature fitting -> conformal calibration -> evaluation and abstention

The class map is part of the model contract. Class 17 is meaningful only with the cell definition used while training that classifier. Rebuilding cells from test labels would alter the meaning of outputs and leak evaluation information. The evaluator therefore requires saved model weights, the declared class count, compatible head dimensions, and valid contiguous cell IDs.

## 6 2 Geographic discretization

Three cell builders are compared. The baseline recursively subdivides a dense geographic box, retaining nonempty children and prioritizing cells with many training points. This allocates more classes where data is dense, but produces axis aligned boundaries. Its effective cell count is recorded rather than assumed.

MiniBatchKMeans groups coordinates into approximately compact clusters. The code scales longitude by the cosine of mean training latitude before clustering. This is a local distance approximation; it is not an exact global equal area projection, does not fully handle longitude wraparound, and does not make every cluster an equal area region. This qualification matters particularly for worldwide data.

DBSCAN starts from density clustering, searches its neighborhood radius, reassigns noise, and merges or splits clusters toward the requested count. The result is an adapted fixed class representation rather than unmodified DBSCAN output. Each cell stores a centroid and bounds. At assignment, containing boxes are considered first; among competing boxes the closest centroid by haversine distance is selected. If none contains the point, the nearest centroid overall is used.

Cell count K = 300 balances a manageable classifier against sparse supervision. At 10,000 images this is only about 33 images per class on average. Coarse classes also introduce quantization error: even a correct cell prediction may have a centroid far from the true point. [R1, R6]

<!-- PAGE -->
# 6 Methodology continued
## 6 3 Learning and experimental controls

The backbone is a pretrained ResNet 50 with its original classifier removed. Global average pooling produces 2,048 features. A new linear layer outputs K logits; for K = 300 it has 2048 x 300 + 300 = 614,700 parameters. Pretraining supplies general visual features, while the new head learns geographic distinctions. [3]

The frozen regime updates the classifier head parameters. The layer4 regime also updates the final residual block. Earlier convolutional parameters remain frozen, reducing the optimization budget. In the current training implementation model.train() can still update batch normalization running statistics; frozen parameters do not imply that every backbone state value remains unchanged. This is a reproducibility detail when explaining the exact regime.

For the controlled data scale study, ImageNet initialization, layer4 fine tuning, 300 quadtree cells, AdamW, learning rate 0.001, weight decay 0.0001, batch size 16, image size 224, AMP, and ten epochs are shared. Seeds 42, 43, and 44 define paired runs. Within each seed the 10k subset is contained in the 20k training pool. Cells are built once from the full training pool, so changing data size does not change the label space. The final epoch is evaluated; test scores do not select a checkpoint. [R3]

Equal epochs are not equal compute: 20k rows produce about twice as many minibatch updates as 10k at the same batch size. Consequently the study measures the larger dataset under a fixed epoch recipe, not the isolated effect of more distinct images at an identical optimization step budget. Three seeds describe repeatability in this bounded setup and do not establish statistical significance across all geography.

The separate sampling experiment holds training rows fixed. Uniform sampling shuffles without replacement. Cell balanced sampling weights each row inversely to its cell frequency and samples with replacement while preserving epoch length. This makes each class approximately equally likely in expectation but can repeatedly present sparse class images. The experiment tests the effect instead of assuming balancing improves generalization. [R2]

Historical runs used an RTX 4050; the fork experiment report records an RTX 3050 laptop GPU with 6 GB memory. These are different recorded experiment environments, not a claim that all work used the same device. New integration checks are CPU checks and do not rerun the full image experiments.

<!-- PAGE -->
# 6 Methodology continued
## 6 4 Calibration and prediction sets

Softmax converts logits z into probabilities. Temperature scaling instead uses softmax(z / T), where T is positive and is fitted by minimizing cross entropy on a held out temperature subset. Increasing T generally softens the distribution. Because division by a positive scalar preserves ordering, temperature scaling preserves the top class and its decoded centroid. It can improve confidence estimates without moving a single predicted location. [4]

The revised evaluation partitions 3,000 rows into 500 temperature fitting rows, 1,000 conformal calibration rows, and 1,500 evaluation only rows using a saved seed. The split records row indices and sample identities. These partitions have different responsibilities: fitting T is a learned transformation; conformal calibration fits the cutoff after that transformation is fixed; evaluation measures the resulting policy on unused rows. [R4]

For calibration example i with true class y, the nonconformity score is s_i = 1 - p(y | x_i). Low true class probability means high nonconformity. With n calibration examples and target miscoverage alpha, the corrected finite sample rank is k = ceil((n + 1)(1 - alpha)). The cutoff q is the kth smallest score. If k exceeds n, q is positive infinity, yielding the conservative full set. The candidate set contains classes j satisfying 1 - p(j | x) <= q. [5]

The original code used an interpolated quantile and clamped the requested rank to the sample range. Prajwal's contribution uses the discrete order statistic and explicitly handles the out of range rank. With n = 9 and alpha = 0.01, k = 10; using the observed maximum as a shortcut is not the required correction. Unit tests and a synthetic score diagnostic cover this case.

The usual guarantee is marginal and relies on exchangeability of calibration and future examples with a fixed scoring function. Random row splitting does not establish geographic exchangeability. The report therefore gives empirical coverage and regional diagnostics without promising conditional coverage. Prediction set size is reported alongside coverage because a set containing most classes is often unhelpful. [5, R4]

Abstention accepts only predictions whose maximum probability exceeds a chosen threshold. It trades retained sample count against reliability. When no rows are retained, conditional error is undefined and represented as missing, rather than assigned zero or NaN.

<!-- PAGE -->
# 7 Data Preprocessing

The loader first reads the CSV and checks required columns and coordinate ranges: latitude lies within -90 to 90 degrees and longitude within -180 to 180 degrees. Each path is resolved using pathlib relative to the metadata directory when necessary. The image is decoded and converted to three channel RGB so that the encoder always receives the expected channel shape.

Training augmentation uses RandomResizedCrop at 224 pixels with scale 0.8 to 1.0, followed by a horizontal flip with probability 0.5. The tensor is normalized using ImageNet channel means 0.485, 0.456, 0.406 and standard deviations 0.229, 0.224, 0.225. These values align inputs with the pretrained encoder's expected distribution. Augmentation provides varied views of training images but does not create new independent locations.

Evaluation uses deterministic resizing to 256 pixels for the default 224 configuration, a center crop to 224 x 224, tensor conversion, and the same normalization. Random training augmentation is not applied while calculating test metrics. After batching, an image tensor has shape B x 3 x 224 x 224, the backbone feature tensor becomes B x 2048, and the head output is B x K.

Coordinates are assigned cell labels using the saved cell map. In fixed cell comparisons this map is reused across all conditions. For the Kerala specialist, cells are rebuilt from its regional training coordinates only; global and regional label maps cannot be exchanged. The regional model warm starts transferable backbone weights but uses a head appropriate to its regional classes. [R5]

The experiment tools additionally validate unique held out IDs, reject overlap with recorded training identities, verify training metadata and cell hashes, and retain source subset indices. Source pool provenance is checked rather than trusting a filename such as metadata_20k.csv. Absolute host paths are omitted from public comparison summaries where possible.

There is a difference between implemented validation and a fresh data audit. The historical reports record successful image decoding and overlap checks for their sampled runs. The current integration does not include their images and checkpoints and therefore does not independently repeat decoding, training, or real image evaluation. The recorded metrics remain attributed to the dated reports.

<!-- PAGE -->
# 8 Exploratory Data Analysis

EDA focuses on geographic coverage, the amount of supervision per class, imbalance, and differences between validation and official test results. These properties determine whether a visually strong network has enough geographic evidence to learn useful distinctions. The following values are summaries of the committed experiment records, rather than newly generated observations from image files.

| Cell builder | Mean rows per cell | Minimum | Maximum | Gini |
| Quadtree | 33.3 | 1 | 99 | 0.370 |
| K means | 33.3 | 0 | 113 | 0.324 |
| DBSCAN adapted | 33.3 | 3 | 1,864 | 0.559 |

Source: historical L4 cell analysis. The zero minimum in the recorded k means assignment distribution should not be replaced with a claim that every assigned class is nonempty. Construction membership and the later bounding box based assignment rule can differ. The global mean is small even before considering imbalance. [R6]

K means gives the lowest recorded Gini, while DBSCAN produces a highly skewed distribution. A dense cluster may dominate training while many other cells have little support. However, class balance is not geographic fairness. A count based dense versus sparse proxy in L4 is not a verified urban versus rural label; the report should describe it as training density only.

Geographic breadth also differs by experiment. The historical balanced 10k training subset covers 161 countries and its test subset 168. The later sampling subset covers 166 training countries; the separate data scale test sample covers 169 countries. These counts describe their own subsets and cannot be merged into a single coverage claim. [R1-R3]

The uncertainty analysis reports geographic slices using latitude bands uniform in sin(latitude) and uniform longitude bands. Unlike the approximate coordinate scaling used for k means, this grid is designed to give equal spherical area cells. Twenty nine regions with at least 20 evaluation rows have reported conformal coverage ranging from 59.5% to 100%. This variation motivates regional inspection even when overall coverage is close to target. Sparse regions remain visible without unstable rates. [R4]

The Kerala results show another distribution effect: random validation median error is 7.0 km, while official test median error is 42.1 km. Nearby street views can inflate random split results. This is evidence to prefer the official held out result when presenting generalization, without claiming that the gap alone proves a single causal mechanism. [R5]

<!-- PAGE -->
# 9 Models Used

## 9 1 ResNet 50 geographic classifier

Residual connections allow deep convolutional blocks to learn changes to an identity mapping. The pretrained network supplies visual features; the geographic head maps those features to class logits. The model is a supervised CNN classifier. The cell builders use coordinate clustering, but they do not replace the image model or independently predict a location from pixels. [3]

| Configuration | Role | Trainable part |
| ImageNet initialization and frozen regime | Historical L1 baseline | Linear geographic head parameters |
| ImageNet initialization and layer4 regime | Selected global regime | Final residual block plus head |
| Places365 initialization | Scene transfer comparison | Head or final block plus head |
| Kerala warm start | Regional specialist | Final block plus regional head |

The historical L2 matrix tests 1%, 10%, and 100% of its 10k sample under two initializations and two training regimes, giving twelve combinations. ImageNet plus full data plus layer4 has the strongest within 200 km score in that matrix, 3.33%. Places365 with full data plus layer4 has lower mean error, 5,800.3 versus 6,029.7 km, but lower within 200 km accuracy, 2.87%. A winner must therefore be identified by the selected metric, rather than described as best on every criterion. [R1]

The original hypothesis that a tenth of the data could match the full baseline is not established by that matrix. Places365 also does not consistently improve transfer. These negative results are part of the project, not reasons to discard completed measurements.

## 9 2 Regional specialization

Kerala mode warm starts the selected global backbone and trains a regional head with 300 k means cells. It uses ten epochs, batch size 16, learning rate 0.0003, image size 224, and layer4 plus head adaptation. It has no fitted L3 calibration artifact; its demo score is raw softmax confidence. The restricted region itself gives a strong geographic prior, so improvement over a global model should be supplemented by a region restricted baseline. [R5]

## 9 3 Uncertainty components

Temperature scaling is a post hoc probability transformation, conformal calibration supplies a class set cutoff, and abstention is a decision policy. None is an additional geographic backbone. Their evaluation requires the exact checkpoint they were fitted against, which is why calibration artifacts record and verify model hashes.

<!-- PAGE -->
# 10 Results and Evaluation
## 10 1 Metrics and controlled global results

For each image, the predicted class centroid is compared with its true coordinate using haversine great circle distance. In radians, a = sin squared(delta latitude / 2) + cos(latitude 1) cos(latitude 2) sin squared(delta longitude / 2), and distance = 2 R asin(sqrt(a)), with Earth radius approximately 6,371 km. Numerical clipping protects the inverse trigonometric calculation from floating point roundoff.

Accuracy within radius r is 100 times the fraction of distances at most r. Mean error is sensitive to severe mistakes; median describes a typical ranked observation; threshold accuracy captures a specific geographic tolerance. Cell top one accuracy is a separate statistic and does not imply kilometer precision. A correct cell can still have a distant centroid.

| Metric | 10k images | 20k images | Paired change |
| Within 200 km | 3.756 +/- 0.158% | 5.300 +/- 0.240% | +1.544 percentage points |
| Mean distance | 5,977.3 +/- 77.0 km | 5,521.6 +/- 129.7 km | -455.7 km |
| Median distance | 4,658.5 +/- 292.6 km | 3,460.2 +/- 165.1 km | -1,198.3 km |
| Within 25 km | 0.044 +/- 0.019% | 0.067 +/- 0.058% | +0.022 percentage points |

Values are mean and sample standard deviation across seeds 42, 43, and 44 on the same 3,000 row evaluation sample. All three paired seeds improve mean error, median error, and within 200 km accuracy. The majority training cell baseline has 9,713.3 km mean error and 0.067% within 200 km on these same rows. These observations support the larger sample under this recipe, but absolute localization remains weak. [R3]

In the separate sampling comparison, uniform sampling yields mean error 5,849.9 km and balanced sampling 5,859.0 km. Both average 3.38% within 200 km. The paired mean distance difference is +9.1 km, with sample standard deviation 222.7 km and inconsistent directions across seeds. There is no consistent demonstrated gain from balancing here. These are different models and test rows from the data scale study. [R2]

Historical L4 within 200 km accuracy is 3.33% for quadtree, 4.07% for k means, and 3.47% for adapted DBSCAN. Mean distances remain around 6,000 km. These modest differences do not identify cell geometry as the only bottleneck. [R6]

<!-- PAGE -->
# 10 Results and Evaluation continued
## 10 2 Uncertainty and Kerala results

| Revised uncertainty metric | Reported value |
| Evaluation count | 1,500 images |
| Temperature | 3.6901 |
| Raw ECE | 0.4171 |
| Temperature scaled ECE | 0.0453 |
| Empirical conformal coverage | 90.8% |
| Mean set size | 159.7 of 300 classes |
| Mean geodesic error | 5,477.4 km |
| Within 200 km | 5.27% |

These results use only the seed 42 model trained on 20k images and the three way split. ECE bins predictions by confidence and compares average confidence with observed class accuracy, weighted by bin occupancy. A smaller ECE means better agreement for that evaluation and binning, not better location predictions. Calibration leaves the top class unchanged. [4, R4]

Coverage is the fraction of evaluation examples whose assigned true class belongs to the prediction set. It is not the fraction predicted within a kilometer radius. The mean set size is more than half of all classes, exposing unresolved uncertainty. Marginal coverage near 90% also coexists with much lower coverage in some geographic slices.

At confidence threshold 0.5 or 0.7 the model abstains on all 1,500 rows. At 0.3 it abstains on 99.2%, retaining only 12 rows. The reported retained top one accuracy is 66.7%, but twelve observations are too few to support a reliable operating policy. A threshold curve and a separately chosen operating point would be more useful. [R4]

| Kerala split | Global before median | Regional after median | After within 25 km | After within 200 km |
| Random validation 1,000 | 6,253.1 km | 7.0 km | 64.6% | 92.9% |
| Official test 357 | 6,330.0 km | 42.1 km | 41.2% | 80.1% |

The regional official test mean error is 102.8 km, with 0.6% within 1 km. This is useful coarse regional localization rather than reliable pinpoint GPS. The random holdout is optimistic relative to the official test. Town level sample counts can be small, so their rates should not be overinterpreted. No L3 calibration is claimed for Kerala. [R5]

All values above are the dated repository's recorded experiment results. The integration validates software compatibility and report consistency; it does not newly rerun these image experiments without their data and model artifacts.

<!-- PAGE -->
# 11 Discussion
## Comparison with existing methods

PlaNet establishes the large scale classification approach to photo geolocation, with multiscale geographic classes and a large image training corpus. KPlaceNet uses the same high level idea but a ResNet 50 encoder, approximately 300 classes, laptop scale subsets, and different cell construction. Its reported numbers are not a head to head reproduction on PlaNet's data and benchmark. Comparing published percentages directly would mix experimental protocols. [1]

The OSV 5M work supplies an open street view benchmark and a spatially separated evaluation framework. KPlaceNet adopts subsets and preserves those official split identities. It does not reproduce the full dataset baseline or demonstrate superiority over it. Street level training also limits extrapolation to indoor photographs, portraits, artistic images, and arbitrary phone photographs. [2]

Temperature scaling follows an established calibration method. The project contribution is not inventing temperature scaling; it is fitting and auditing it in this pipeline. Likewise, the conformal contribution applies the correct finite sample order statistic and independent fitting partitions, with sample provenance and regional diagnostics. An established method can still require substantial implementation care. [4, 5]

The most useful direct comparisons are internal controls. The 10k and 20k models share cells and test rows within paired seeds. The sampler study shares training examples and varies the sampling distribution. The historical cell experiment holds the backbone recipe approximately fixed while changing geographic discretization. Each comparison answers a bounded question and should retain its own sample identity.

Several limitations explain why cautious interpretation is necessary. Geographic classes are coarse and uneven; the training subset covers only a few archives; all reported global accuracies remain low; and three seeds offer limited evidence about variability. Equal epochs give the larger sample more optimization updates. Follow up questions were selected after inspecting earlier results. ID disjointness helps prevent row reuse but is not evidence of independent geographic populations.

The Kerala model changes both training geography and task scope. Its large improvement over the global model partly reflects restricting all predictions to Kerala. A proper regional evaluation should compare against a training only regional prior or a random regional centroid baseline measured on the same rows. The existing report's rough random distance estimate is context rather than a measured controlled baseline. [R5]

Privacy and deployment also constrain usefulness. Visual localization can expose sensitive places and confidently wrong outputs can mislead users. The demo should communicate coarse estimates and uncertainty without implying that a cell bounding rectangle is a geographic confidence interval.

<!-- PAGE -->
# 12 Conclusion

KPlaceNet implements the complete chain from image preprocessing to geographic classification, coordinate decoding, and geodesic evaluation. Transfer learning makes the experiment feasible on laptop hardware, while saved class definitions keep model outputs geographically meaningful. The project goes beyond producing a map marker by testing data scale, sampling, cell geometry, and uncertainty under explicit controls.

The controlled data scale study records consistent improvements across three paired seeds when training increases from 10,000 to 20,000 images under the same epoch recipe. The separate balanced sampling study finds no consistent benefit. The historical cell study shows modest differences among three geographic representations. Together these results motivate broader training coverage and more careful evaluation rather than attributing all error to one component.

Calibration improves confidence agreement but does not improve argmax location accuracy. Empirical conformal coverage of 90.8% is accompanied by sets averaging 159.7 of 300 classes, and geographic slices vary substantially. The corrected rank, three way split, checkpoint verification, and provenance manifests strengthen the scientific procedure while preserving the distinction between mathematical assumptions and measured behavior.

The Kerala specialist produces stronger coarse regional results, with 41.2% within 25 km on 357 official test examples. It should be presented as a separate restricted geography model, with raw uncalibrated confidence and a substantial random validation versus official test gap. These limitations are essential to an accurate account of the outcome.

Future work should use broader training archives, independent spatial validation for model and threshold choices, measured region restricted baselines, and repeated evaluation beyond three seeds. Step matched comparisons would help separate more data from more optimization updates. Larger or hierarchical cell representations may reduce quantization error if supervision and compute permit. Regional uncertainty assessment and fresh held out geography are needed before making deployment claims.

## Contribution attribution

The original repository provides the ResNet pipeline, dataset acquisition tools, historical L1 to L4 studies, and later demo and Kerala work. Prajwal Kumar K's fork adds conformal correctness, disjoint fitting partitions, evaluation integrity, provenance, regional diagnostics, controlled sampler and data scale follow ups, and focused tests. The group should attribute these components explicitly and avoid claiming that historical upstream results were generated by the fork changes. The members listed on the title page are taken from the existing group presentation; individual assignments beyond the documented contribution remain for the group to confirm.

<!-- PAGE -->
# 13 References

[1] Weyand T, Kostrikov I, and Philbin J. PlaNet Photo Geolocation with Convolutional Neural Networks. ECCV 2016. https://arxiv.org/abs/1602.05314

[2] Astruc G et al. OpenStreetView 5M The Many Roads to Global Visual Geolocation. CVPR 2024. https://arxiv.org/abs/2404.18873 and https://osv5m.github.io/

[3] He K, Zhang X, Ren S, and Sun J. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385

[4] Guo C, Pleiss G, Sun Y, and Weinberger K Q. On Calibration of Modern Neural Networks. ICML 2017. https://proceedings.mlr.press/v70/guo17a.html

[5] Angelopoulos A N and Bates S. A Gentle Introduction to Conformal Prediction and Distribution Free Uncertainty Quantification. Foundations and Trends in Machine Learning 2023. https://arxiv.org/abs/2107.07511

Project implementation and experiment records accessed 20 October 2026

[R1] KPlaceNet historical baseline and transfer experiments. docs/results/l1_baseline.md and docs/results/l2_data_efficiency.md. Upstream https://github.com/Asterisk-Hunter/KPlaceNet at revision bab61f9.

[R2] Prajwal Kumar K. Cell balanced sampling on a fixed OSV 5M subset. 8 October 2026. docs/results/l2_sampling_osv_test_20261008.md and associated JSON report.

[R3] Prajwal Kumar K. Training set size comparison on OSV 5M. 9 October 2026. docs/results/l2_data_scale_osv_test_20261009.md, per condition JSON reports, comparison JSON, training pool manifest, and held out sample manifest.

[R4] Prajwal Kumar K. Calibration and abstention evaluation on OSV 5M. 9 October 2026. docs/results/l3_uncertainty_osv_test_20261009.md and JSON report.

[R5] KPlaceNet Kerala Pinpoint Mode. docs/results/l5_kerala_pinpoint.md. Upstream revision bab61f9.

[R6] KPlaceNet Adaptive Cell Construction. docs/results/l4_adaptive_cells.md. Upstream revision bab61f9.

R2 to R4 source revision c02d320 on branch prajwal/eval-integrity at https://github.com/Prajwal-k-tech/KPlaceNet. Historical and follow up results use distinct evaluation subsets. Source code examined includes src/dataset.py, src/model.py, src/train.py, src/cells.py, src/eval.py, src/uncertainty.py, and experiment runners. Measurements are cited to the recorded reports rather than represented as newly reproduced runs.
