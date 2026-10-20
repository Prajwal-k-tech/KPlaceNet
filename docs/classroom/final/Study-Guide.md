# KPlaceNet: understand the current system
A learning guide / 21 October 2026

## 1. The problem we solve
**Visual geolocation** means estimating where a photograph was taken by examining its visible content. KPlaceNet takes one image and returns a location estimate. Roads, buildings, vegetation, terrain and other visual patterns can help distinguish places, but similar-looking places can cause errors.

The current app has two separate **modes**, meaning two different prediction workflows. They use separate model resources; changing mode changes the method and geographic coverage, not just the map zoom.

| Mode | What happens | What the coordinate means |
|---|---|---|
| Kerala area: Fine coordinates | ResNet and DINOv2 describe the image; search regional training references | GPS coordinate of the best-matching reference photograph |
| Kerala area: Regional cells | ResNet predicts a geographic class, with optional KNN class votes | Saved center of the selected class |
| World | A separate ResNet predicts a global geographic class; score-scaling and uncertainty rules are applied | Saved center of the selected global class |

**GPS coordinates** are a latitude-longitude pair. Latitude measures north/south position; longitude measures east/west position. A displayed coordinate such as 9.6761, 76.4211 identifies a point on Earth. Four decimal places describe how the number is displayed, not how accurate the model is.

## What our system does not do
It does not use the uploaded image's GPS metadata to solve the task, search the live web for its identity, move around a panorama or perform live navigation. Google Maps is an output link: it displays the coordinate our model already produced. It does not improve that prediction.

Regional mode assumes a Kerala-area scene. Its data were collected using a rectangle, not a verified Kerala state boundary. That rectangle includes nearby territory in neighboring states. The app cannot establish that an arbitrary input belongs to Kerala.

**Checkpoint:** Explain the three possible meanings of a returned coordinate before studying how any network works.

<!-- PAGE -->
# 2. Data, labels and honest separation

## What one data record contains
A **dataset** is a collection of examples. A KPlaceNet record associates an image file with its known latitude and longitude. An image ID identifies the record; it is not a visual feature. A **label** is the answer supplied during supervised learning: here it can be a geographic class derived from GPS.

The main regional imagery comes from **OSV-5M**, a geotagged street-image dataset. **Geotagged** means an image has associated geographic coordinates. The regional selection uses approximately 8.0-12.8 degrees latitude and 74.5-77.5 degrees longitude. This is a collection filter, not a guarantee that every photograph is inside Kerala or that its coordinate is perfect.

| Regional split | Number | Purpose |
|---|---|---|
| Training | 15,939 | Learn model parameters and populate retrieval references |
| Validation | 1,000 | Choose settings without fitting on the official test |
| Official test | 357 | Measure distance error after choosing the method |

A **split** is one of these distinct subsets. No validation or test image IDs are in the deployed retrieval bank. However, different IDs can still show almost the same place: a nearby road frame can be easier to match than an unfamiliar location.

## Three different kinds of leakage
**Data leakage** means information that should be held out influences learning or evaluation. Exact-image leakage is reusing the same photo. Spatial leakage is letting very nearby places appear across splits. Selection leakage is repeatedly using test results to decide what to deploy.

In our audit, 982 of 1,000 random validation positions are within 1 km of a training position. None of the 357 official test positions is within 1 km of the original training references. Validation therefore measures an easier setting with many nearby places. Its stronger result must not be presented as the official test result.

The official test has been inspected across multiple project experiments. The current result is a useful measured follow-up, but it is not a newly collected independent confirmation. An independent route/date test would provide stronger evidence.

**Checkpoint:** Why can two disjoint image-ID lists still produce an optimistic evaluation? Because disjoint files do not guarantee different places or capture sequences.

<!-- PAGE -->
# 3. Turning a photograph into model input

## Pixels, channels and tensors
A **pixel** is one small color sample in an image. **RGB** represents color using red, green and blue channels. The app converts the uploaded photograph to RGB, including images originally stored in other color formats.

A **tensor** is a multidimensional array of numbers. A single model input has shape 1 x 3 x 224 x 224: one image, three color channels, and 224 rows and columns. **Shape** describes the sizes of an array's dimensions.

## The inference preprocessing steps
**Preprocessing** means applying consistent transformations before a model reads an image.

1. Resize the short edge to 256 pixels while preserving the aspect ratio, so the image is not stretched.
2. Take a central 224 x 224 crop. A crop selects a portion of the image; this can remove useful signs near the edges.
3. Convert pixel values to floating-point numbers, initially in the range 0 to 1.
4. Normalize each color channel using ImageNet mean and standard deviation values.

**ImageNet** is an image-recognition dataset used in pretraining. The stored channel means are [0.485, 0.456, 0.406], and standard deviations are [0.229, 0.224, 0.225]. For a red-channel value of 0.5, normalization gives (0.5 - 0.485) / 0.229, approximately 0.066. A **standard deviation** describes variation around a mean. This transformation puts input values on the scale expected by the encoders.

Both regional encoders receive this same processed image tensor. The bank was built using matching evaluation preprocessing. If query images and reference images were processed differently, their descriptors could become less comparable.

## Training images can be transformed differently
**Data augmentation** makes altered training views, such as a random crop or horizontal flip, to reduce reliance on one exact appearance. The project's general training transform supports these changes. Inference and cached-feature extraction use the deterministic resize-and-center-crop pipeline instead. **Deterministic** means the same transformation is applied consistently, rather than drawing a fresh random view each time.

**Checkpoint:** Image normalization changes pixel scales. Descriptor normalization, introduced later, changes feature-vector lengths. They are different operations.

<!-- PAGE -->
# 4. ResNet: learning useful visual features

## Neural networks and weights
A **neural network** is a sequence of numerical operations with adjustable parameters called **weights**. Training adjusts those weights so useful patterns become easier to recognize. Inference applies the saved weights to an input without changing them.

A **convolutional neural network**, or CNN, learns small filters that are applied across image regions. A **filter** combines nearby pixel or feature values to detect a pattern. Early layers can respond to edges and textures; deeper layers combine patterns into more complex representations. This is learned computation, not a manually coded rule saying that coconut trees imply Kerala.

## Why ResNet-50?
**ResNet** means residual network. A residual connection adds a block's input to a learned transformation: output = input + F(input), when dimensions match. This allows a block to learn a change to its input and helps optimization in a deep network. The 50 identifies the architecture's conventional depth; it does not describe the number of places predicted.

The project uses a ResNet-50 visual backbone initialized from pretrained image-recognition weights and adapted for geolocation. A **backbone** is the main feature extractor. **Pretraining** means learning useful visual representations before our task. **Transfer learning** reuses that knowledge. **Fine-tuning** adjusts some pretrained weights for a new task; the project supports adapting the final ResNet block while earlier layers remain fixed.

The original object-classification layer is removed. After the convolutional blocks, **global average pooling** averages each feature map across its spatial positions. A feature map is a grid of responses produced by learned filters. Pooling produces a 2,048-number vector for each image.

## Descriptor versus classifier head
A **descriptor**, also called an embedding or feature vector, is this compact numerical representation of the image. Its 2,048 values are learned features, not 2,048 named landmarks or geographic coordinates.

A **classifier head** is a separate linear layer that maps the descriptor to geographic class scores. Fine retrieval uses the ResNet descriptor itself. The regional cell options use the head as well. The app's shared regional code may compute head scores during a fine request, but those scores do not choose the fine coordinate.

**Checkpoint:** Why is ResNet useful in retrieval even when its classifier head does not select the answer? Its trained backbone supplies the representation used to compare photographs.

<!-- PAGE -->
# 5. DINOv2: a second visual representation

## What DINOv2 contributes
DINOv2 is a pretrained visual encoder developed through **self-supervised learning**. Self-supervision builds a learning signal from the images themselves rather than requiring a human geographic label for every training example. The current KPlaceNet uses the official pretrained DINOv2 encoder as a feature extractor; it does not train DINOv2 from scratch or fine-tune its weights on Kerala images.

The installed variant is **ViT-S/14**. ViT means **Vision Transformer**, S denotes the small variant, and 14 is the patch width in pixels for this model. At our 224 x 224 input size, dividing the image into 14 x 14 patches creates a 16 x 16 arrangement of 256 patches.

A **patch** is a small image region. A **token** is a numerical representation of a patch or special model element. A transformer processes relationships among tokens. **Attention** lets a token combine information from other tokens with learned weights; it is not a human explanation or proof that the model noticed the correct landmark.

The model produces a 384-number image descriptor. The project uses the complete image descriptor rather than matching individual patch coordinates or reconstructing a 3D scene.

## Why use two encoders?
ResNet and DINOv2 encode appearance differently. Combining them lets both representations contribute to reference matching. This can help when one encoder's strongest similarity is misleading. It can also fail if both encoders find the same wrong-looking place.

In deployment both encoders are **frozen**, meaning their weights remain fixed. Offline reference extraction saves their descriptors. For an uploaded query, the app computes a new descriptor with each encoder. There is no parameter update when you click Locate photo.

**Ensemble** means combining multiple model outputs or representations. Our fine method is a descriptor ensemble: it combines normalized features before searching. It is not a vote between two independently predicted latitude-longitude pairs.

**Checkpoint:** DINOv2 does not supply a second GPS prediction. It supplies another visual descriptor. The bank supplies the returned GPS coordinate.

Source for DINOv2's model family and pretrained encoders: [official DINOv2 repository](https://github.com/facebookresearch/dinov2).

<!-- PAGE -->
# 6. Combining features: a worked explanation

## Why normalize a descriptor?
A vector's **L2 norm** is its length: the square root of the sum of its squared values. L2 normalization divides every value by this length, producing a unit-length vector. For example, [3, 4] has length 5, so its normalized version is [0.6, 0.8].

This keeps vector magnitude from dominating a comparison. It also prevents the 2,048-dimensional ResNet descriptor from automatically outweighing the 384-dimensional DINO descriptor merely because it has more components.

Let r be the normalized ResNet descriptor and d be the normalized DINO descriptor. **Concatenation** means placing one vector after the other to create a longer vector. Our combined descriptor is:

`z = [sqrt(0.5) * r, sqrt(0.5) * d]`

Its dimension is 2,048 + 384 = 2,432. Each part has half of the squared length, so the combined vector has unit length before storage rounding. The app normalizes again when searching.

## What cosine similarity measures
**Cosine similarity** measures how closely two vectors point in the same direction. It is their dot product divided by their lengths. A **dot product** multiplies corresponding components and adds the products. For unit vectors, the dot product is already cosine similarity.

Because of the weighting above, the combined similarity is equally influenced by the ResNet and DINO similarities:

`combined similarity = 0.5 * ResNet cosine + 0.5 * DINO cosine`

Illustrative example, not actual project measurements:

| Reference | ResNet similarity | DINO similarity | Combined |
|---|---|---|---|
| A | 0.90 | 0.50 | 0.70 |
| B | 0.75 | 0.80 | 0.775 |

ResNet alone prefers A. The combined descriptor prefers B. This demonstrates how the second encoder can change the match without any geographic coordinate averaging.

A cosine score such as 0.812 is a similarity value. It does not mean an 81.2% chance that the location is correct, nor does it specify a distance error. The project's fine pipeline has no calibrated conversion from similarity to probability.

**Checkpoint:** Explain why the square-root weights result in equal similarity contributions: multiplying query and reference weights gives sqrt(0.5) x sqrt(0.5) = 0.5.

<!-- PAGE -->
# 7. Fine-coordinate retrieval, end to end

## Query and reference bank
A **query** is the image whose location we want to estimate. A **reference** is a training image with a known coordinate. A **reference bank** stores each reference's descriptor, image ID and coordinate. The deployed fine bank contains 15,939 original training entries. Some may have the same coordinates; this is not a claim of 15,939 distinct places.

The query and references pass through matching encoders and preprocessing. Their descriptors can therefore be compared in the same **feature space**, meaning a shared numerical representation where similarity is computed.

For every reference j, the app calculates a cosine score s(j). It selects the largest score using **argmax**, which means the index of the largest value. The selected index retrieves the saved latitude and longitude. This is **top-one nearest-neighbor retrieval**: one best reference supplies the answer.

## What is actually predicted?
The neural networks predict descriptors; the search selects a reference; the lookup returns that reference's GPS. The result is not restricted to 300 class centers. It is also not **coordinate regression**, which would directly produce numerical latitude and longitude through a learned output function.

Fine mode does not average the top five reference coordinates, use classifier probabilities to adjust the chosen fine coordinate or run geometric verification. **Geometric verification** would test whether detailed image correspondences support a physically consistent scene match. That is a possible extension, not an installed feature.

## Why retrieval still has a precision limit
If the bank has no relevant nearby photograph, even the best match can be distant. A top-one output cannot be closer to the true location than the closest bank coordinate, and appearance matching can select something much farther away. Increasing coverage with accurate, varied references is therefore as important as improving the encoder.

This is not a direct pixel comparison: the query can differ in lighting or viewpoint and still have a similar learned descriptor. Conversely, two different places can share a similar descriptor and produce an incorrect match.

**Checkpoint:** The returned coordinate belongs to the reference photo. Without the query's true coordinate, we cannot know the query-reference distance error just by reading the output.

<!-- PAGE -->
# 8. The two regional cell methods

## Cells, classes and centroids
A **geographic cell** is a group or region used as a classification label. A **class** is one possible output category. The regional classifier has 300 classes. Each has a stored **centroid**, or representative center coordinate, and latitude-longitude bounds.

The regional cells are built from training GPS positions with **K-means**, a clustering algorithm that repeatedly assigns positions to nearby cluster centers and updates the centers. A cluster is a group of related examples. Here clustering groups coordinates, not image pixels. These cells are not Kerala districts, uniform-size grid squares or a verified administrative partition.

A class's bounding rectangle describes its spatial extent. It is not a confidence interval and does not guarantee the query lies inside it. Overlapping bounds and uneven sampling further limit a simple geometric interpretation.

## Classifier only
A linear head maps the 2,048 ResNet features to 300 **logits**, which are unrestricted class scores. **Softmax** converts logits into nonnegative values that sum to one: p(j) = exp(logit(j)) / sum exp(logit). The largest score selects the class; its center becomes the output.

A score summing to one with the others is not automatically a calibrated real-world probability. Kerala class scores are uncalibrated.

## Classifier plus KNN
**K-nearest neighbors**, or KNN, finds k references most similar to the query. This regional option uses ResNet descriptors only, with k = 5. Similarities are converted into neighbor weights using softmax(similarity / 0.07). The number 0.07 controls how strongly the most similar neighbors dominate. Their weights are added to their geographic class labels.

The app combines 80% classifier scores with 20% neighbor class votes, selects the winning class and returns its centroid. Thus this option still has only 300 center outputs. Fine mode instead uses two encoders, k = 1 and a reference coordinate.

**Do not confuse K-means and KNN:** K-means builds geographic classes offline. KNN searches visual training descriptors at inference. The 300 classes and the five neighbors count different things.

**Checkpoint:** State which regional options return a center and which returns a reference GPS coordinate.

<!-- PAGE -->
# 9. How the learned model is trained

## Supervised learning and loss
**Supervised learning** uses examples with known answers. For geographic classification, a training photo's GPS position determines its target cell label. The network receives image pixels; GPS is used to define the answer during training, not to give away a query's location at inference.

A **forward pass** computes the network's scores. A **loss** measures disagreement with the target. **Cross-entropy** penalizes placing little probability on the correct class. With one hard target y, the loss is -log p(y). Assigning p(y) = 0.8 gives a lower loss than p(y) = 0.1.

**Backpropagation** computes how each trainable weight affects the loss. A **gradient** is that rate of change. An **optimizer** uses gradients to update weights. The **learning rate** controls the update step size. A **batch** is a group processed together; an **epoch** is one pass through the chosen training examples.

## The currently installed regional head
The deployed ResNet descriptor comes from the geolocation-adapted backbone. Its installed classification head was fitted while keeping that backbone fixed. **Freezing** weights means not optimizing them. This lets the project cache the 2,048 features and train the head cheaply.

Its **geographic label smoothing** uses a soft target: 90% assigned class plus 10% distributed according to distance from the training GPS to cell centers. Nearby centers get more of that remaining mass, using a 15-km distance scale. A **soft target** distributes credit across classes rather than assigning all credit to one. This can reduce an overly sharp penalty at arbitrary geographic boundaries.

**AdamW** is an optimizer that adapts updates using running gradient statistics and applies weight decay separately. The head-fitting configuration is AdamW, learning rate 0.0001, weight decay 0.01, batch size 256 and 12 epochs per recipe. **Weight decay** discourages unnecessarily large weights; it is a form of regularization, meaning a constraint intended to improve generalization. These settings describe head fitting, not a fresh end-to-end training of both fine-retrieval encoders.

The fine coordinate is selected from frozen descriptors, so changing the head alone does not directly change the deployed fine search.

**Checkpoint:** Training changes parameters. Clicking Locate only runs inference and lookup; it does not teach the model a new place.

<!-- PAGE -->
# 10. Building and selecting the retrieval system

## Offline work versus a live prediction
**Offline** here means preparation performed before serving a user query, not necessarily without internet. Build the bank by applying the frozen ResNet and DINO encoders to each training reference once. Save the combined descriptors, coordinates and IDs. A **cache** is saved work reused later instead of recomputing it.

At inference, only the query needs fresh encoder passes. Comparing one 2,432-component descriptor against 15,939 cached references is a matrix multiplication. A matrix is a table of numbers; multiplying the query by the reference matrix calculates all similarity scores efficiently.

The bank stores descriptors in **FP16**, a 16-bit floating-point format that saves space. At search time the app converts them to FP32, a higher-precision 32-bit format, and normalizes them. Storing fewer bits can introduce small rounding differences, so the compressed deployment bank was checked separately. Its official-test threshold counts and median match the reported current result.

## What was selected, and how?
A **hyperparameter** is a setting chosen rather than learned by gradient updates: for example, encoder weight or neighbor count. The current equal encoder weighting and k = 1 were selected from six fixed mixture/neighborhood options on validation.

The main selection objective was the percentage within 5 km, with a median-error tie-break and a guardrail against excessive worsening within 25 km. A **guardrail** is a minimum acceptable condition on another metric. The winning method was fixed before its test evaluation. The reused-test caveat still applies because the overall project has examined that benchmark repeatedly.

## Why not store only training images and search them later?
Neural feature extraction is the expensive part. Recomputing every reference descriptor per query would waste time. The bank makes inference practical. Keeping query and reference encoder versions consistent is essential: a descriptor bank from a different encoder may no longer be comparable.

A **checkpoint** is a saved model state. A **SHA-256 fingerprint** is a file-content digest used to check identity. Installers verify asset fingerprints, and runtime checks tie the fine bank to the matching ResNet and DINO weights. This prevents accidental resource mismatch; it does not prove predictive accuracy.

**Checkpoint:** What needs rebuilding after an encoder's weights or preprocessing changes? The reference descriptors, and the validation/evaluation needed to justify the new system.

<!-- PAGE -->
# 11. World mode is a separate classifier

## The global prediction path
World mode loads a different ResNet checkpoint and its global cell definitions. It processes the image, calculates 300 class logits, scales their scores, selects a class and returns that class's centroid. It does not use the regional DINO encoder or the regional nearest-photo bank.

A **global model** means its geographic class map spans the world; it does not mean every place has useful training coverage or that any input can be localized precisely. Each cell can cover a large area. World mode is a coarse classifier, not a world-scale version of our fine regional retrieval.

| Property | Regional fine mode | World mode |
|---|---|---|
| Visual representation | ResNet plus DINOv2 | Separate ResNet classifier |
| Output selection | Most similar training reference | Highest-scoring global cell |
| Coordinate source | Reference GPS | Cell center |
| Displayed score | Cosine similarity | Softmax / temperature-scaled class score |
| Decision rule | Always returns best available match | ACCEPT or ABSTAIN at score threshold 0.5 |
| Uncertainty set | Not fitted | Conformal set machinery if a quantile is loaded |

## Temperature scaling
**Calibration** asks whether scores reflect observed correctness over comparable examples. A model can rank classes well but be overconfident. **Temperature scaling** divides all logits by one positive number T before softmax. T greater than one usually makes the score distribution less concentrated.

The installed World artifact supplies T approximately 3.9387. Dividing all logits by the same positive number preserves their ranking. It therefore does not correct the selected location or improve top-one distance by itself. It changes scores used by uncertainty and threshold rules.

The installed artifact has no checkpoint fingerprint. The app correctly labels its match to the current checkpoint as unverified. A temperature value being loaded is not proof that it is a valid calibration for this exact deployed model. Do not attach a measured calibration or coverage guarantee to this live World instance without verifying that identity and the evaluation setup.

**Checkpoint:** Temperature scaling can change confidence and ACCEPT/ABSTAIN behavior while leaving the predicted cell unchanged.

<!-- PAGE -->
# 12. Understanding World uncertainty outputs

## ACCEPT and ABSTAIN
The app compares the largest temperature-scaled class score with 0.5. **ACCEPT** means the score meets that threshold. **ABSTAIN** means it does not. These are workflow decisions, not externally verified facts. A wrong prediction can be accepted; a correct prediction can be rejected.

The app still displays a best cell coordinate when it abstains. That coordinate means “highest-ranked candidate,” not “the system approved this location.” A score threshold is not a reliable foreign-image detector, and the regional mode has no equivalent ACCEPT/ABSTAIN threshold.

## Conformal prediction
A **prediction set** contains several candidate class labels rather than just one. **Conformal prediction** uses labeled calibration examples to choose a rule for forming such sets. For each calibration example, our score is 1 minus the model's probability for the true class. This is a **nonconformity score**: a larger value means the true label was less compatible with the model output.

For n calibration examples and desired error fraction alpha, choose the kth smallest score, where k = ceil((n + 1) * (1 - alpha)). **Ceil** rounds upward. The chosen value q is a **quantile threshold**, meaning a cutoff derived from an ordered sample. If k exceeds n, the conservative cutoff is positive infinity. At inference, include classes for which 1 - p(class) is at most q.

Example: q = 0.8 includes every class with p at least 0.2. A larger q usually includes more classes. These sets contain class labels, not a continuous GPS error circle.

**Coverage** is the proportion of examples whose true class is inside the set. Under appropriate **exchangeability**, meaning calibration and future examples behave like interchangeable draws, split conformal can provide marginal coverage. **Marginal** means over the population, not for every individual image or town. Distribution shift can break that interpretation.

The live artifact's checkpoint identity is unverified. Its loaded q is approximately 0.99891. Treat the resulting set as an artifact-driven output, not a proven current coverage guarantee. “90% target set coverage” would never mean 90% exact-coordinate accuracy.

## ECE
**Expected calibration error**, or ECE, groups predictions by confidence and compares average score with observed class accuracy in each group. Lower is better for that specific evaluation. It measures score reliability, not geographic distance, and is not currently a verified live World result.

**Checkpoint:** A set containing many classes can have high coverage while being too broad to be useful for precise localization.

<!-- PAGE -->
# 13. Reading the actual regional result

## Distance error and thresholds
**Ground truth** is the recorded correct coordinate used for evaluation. Distance error compares the predicted coordinate with that recorded position. The project uses **haversine distance**, a great-circle distance calculation on a spherical Earth approximation. A great-circle path is the shortest surface path on a sphere; simply subtracting latitude and longitude degrees would not give kilometers.

For each labeled query, calculate one error in kilometers. Then summarize the distribution. A **metric** is a numerical performance measure.

| Current fine mode: 357 official test photos | Result |
|---|---|
| Within 1 km | 0 of 357 |
| Within 5 km | 62 of 357: 17.37% |
| Within 25 km | 225 of 357: 63.03% |
| Within 200 km | 90.76% |
| Median error | 15.92 km |
| Mean error | 61.21 km |
| 90th-percentile error | 197.75 km |

**Median** is the middle of the ordered errors; roughly half fall on each side. **Mean** is the arithmetic average and is pulled upward by large failures. The **90th percentile** is an error cutoff at roughly the 90% position of the ordered results; it is not the accuracy of one uploaded photograph.

“63.03% within 25 km” means 225 queries had an error at most 25 km. It does not mean 63.03% were on the exact street. Reporting an accuracy without its distance threshold is misleading for this task.

## Why zero within 1 km?
Every official test position is more than 1 km from the original reference bank, and fine mode returns a reference coordinate. That creates a geometric ceiling: no possible top-one match from this bank can meet the 1-km threshold on this subset. The reference coverage must change, or the output method must estimate beyond reference coordinates, to overcome that ceiling.

Random validation gives 71.1% within 5 km and 1.07-km median, but contains many near-training positions. Those numbers are not interchangeable with the official test. All reported values here concern regional fine mode, not World mode.

**Checkpoint:** Why can the median be 15.92 km while the mean is 61.21 km? A smaller number of very distant errors can strongly increase the mean.

<!-- PAGE -->
# 14. Failure, generalization and overfitting

## Four different reasons for a wrong location
**Coverage failure:** no geographically useful reference exists. **Appearance ambiguity:** the bank has relevant places, but another location looks more similar. **Domain shift:** the query differs from training imagery, such as a drone view, indoor photo, heavy fog or a famous foreign landmark. **Label error:** the recorded reference or test GPS is inaccurate.

A **domain** is a data setting, such as street-level camera imagery in a region. **Out-of-distribution**, or OOD, describes inputs outside the patterns the model was developed for. The regional system has no validated OOD rejection method. Giving it the Great Wall of China can still return an Indian reference because the algorithm always selects its highest available similarity.

## What overfitting means
**Overfitting** occurs when learning or method selection exploits details of development data that do not generalize to new data. **Generalization** means performing well on genuinely new cases. A curated success cannot prove generalization, and a single foreign-image failure cannot by itself establish training overfitting.

The retrieval bank intentionally stores training references. Storing them is how retrieval works, not automatically evidence of improper memorization. The important questions are whether held-out queries were excluded, whether scenes or sequences overlap across splits, and how performance changes on new routes and conditions.

Current warnings are specific: random validation has many nearby references; official test is spatially harder; and the benchmark was reused. These limit the strength of the performance claim even though the bank has no held-out IDs.

## Explainability versus verification
Showing the retrieved photograph provides **inspectability**: a person can compare the visual match and see where the coordinate came from. It is not a complete explanation of all neural features and is not geometric proof of place identity. Attention is not a causal explanation either.

A local map makes the result visible; it cannot make it correct. The Google Maps link can help inspect roads and surroundings but does not retrospectively certify the model output. A displayed similarity should therefore always be described as similarity.

**Checkpoint:** Diagnose a northern match for a southern road photo as an observed retrieval error. Do not invent a reason such as “the model read this sign” without testing that claim.

<!-- PAGE -->
# 15. Reading and operating the frontend

## What each control does
**Frontend** means the user-facing interface. **Backend** means the Python logic handling model loading and prediction. **Gradio** connects the interface components to Python callbacks; a callback is a function triggered by a user action. **Plotly** draws the interactive geographic figure.

Geographic scope chooses regional or World resources. The regional method control chooses fine retrieval, classifier-plus-KNN or classifier only. Upload accepts an image file; clipboard input can paste an available image. A browser-image drag may supply a URL rather than a file. If that source blocks access, save the image and upload the saved file.

Clicking a gallery thumbnail loads an example and its scope. It does not itself run the encoders. **Locate photo** runs inference. Changing the image, mode or method clears the previous result so an old coordinate is not presented as the new query's answer.

## Understanding the displayed outputs
In fine mode, the marker is the matched training reference coordinate. The separate retrieved-reference image, when locally available, is not the uploaded query. Its ID identifies the bank entry. In cell modes, the marker is a class centroid and the cell bounds may be drawn. Those bounds are geographic extents, not uncertainty intervals.

The background map dots represent the 300 regional class centers, including in fine mode. They do not show all 15,939 retrieval references or every place the fine method can return. The current fine coordinate does not need to coincide with a background class-center dot.

**Prediction details** contains method-specific information. In World mode, distinguish the best coordinate from the ACCEPT/ABSTAIN decision, the scaled score from the raw score, and set size from geographic precision. Fine mode has neither a conformal set nor an accept threshold.

## What the examples demonstrate
The gallery query photographs came from official test archives and were deliberately selected. **Curated** means chosen for demonstration, not sampled to estimate average accuracy. Loading a named thumbnail does not feed its place caption into the model. The callback loads the image; encoders receive processed pixels.

A known failure is included. In our current smoke check, the Southern Kerala example retrieves a northern reference near 12.45793, 75.09220. This should be explained as a limitation, not hidden or relabeled a successful regional prediction.

**Checkpoint:** Fine mode can show 300 map dots while searching 15,939 references because the dots and retrieval bank are different data structures.

<!-- PAGE -->
# 16. Running and understanding the implementation

## Start this laptop's app
From the task workspace, run:

```bash
cd ~/Documents/Codex/2026-10-20
cd https-github-com-asterisk-hunter-kplacenet
./outputs/Start-KPlaceNet.sh
```

Open http://127.0.0.1:7861/ and keep the terminal open. **Localhost**, including 127.0.0.1, refers to this machine; 7861 is the service's **port**, a numbered network endpoint. A different device's localhost is a different machine.

From inside the repository at `work/KPlaceNet`, the same launcher is `../../outputs/Start-KPlaceNet.sh`. The launcher uses the installed GPU environment when it works, otherwise the CPU environment. A **GPU** accelerates parallel numerical work. **CUDA** is NVIDIA's GPU-computing platform. Inference can run on CPU more slowly.

A **virtual environment** keeps a project's Python packages separate. The app uses PyTorch for tensor operations and neural networks, Torchvision for ResNet and image transforms, Pillow for image decoding, Gradio for the UI and Plotly for maps. The first prediction is slower because resources are loaded **lazily**, when first needed, then cached in memory. Models do not need to reload for every query.

## Files to recognize
| File or location | Responsibility |
|---|---|
| demo/app.py | UI, mode dispatch, model loading, maps and output labels |
| src/dataset.py | Image loading and transforms |
| src/model.py | ResNet backbone and class head |
| src/fine_inference.py | DINO loading, descriptor fusion and fine search |
| src/retrieval.py | Five-neighbor regional class voting |
| src/cells.py | Cell construction, assignment and metadata |
| src/uncertainty.py | Temperature, ECE, conformal and threshold functions |
| src/demo_config.py | Artifact selection and identity checks |

Model weights and banks are **artifacts**, meaning saved resources required at runtime. They are downloaded from releases rather than stored as large Git files. In a separately prepared Python environment, `python scripts/download_fine_retrieval.py` installs the verified fine assets and matching regional model artifacts. It does not install Python packages or download the entire image dataset. Reference thumbnails require their corresponding image files.

**Checkpoint:** Cloning source code alone is not enough to reproduce the installed demo: packages, weights, cell metadata and matching banks are also needed.

<!-- PAGE -->
# 17. Additional data and research boundaries

## What the current bank actually uses
The deployed bank uses the 15,939 OSV-5M training entries. New Panoramax photos are not in that bank. The data pipeline found 936 regional candidates and encoded 789 unique image byte sequences after exact deduplication. **Deduplication** removes repeated content; exact-byte deduplication does not detect all visually near-identical frames.

Images were downloaded in small batches, decoded in memory and converted to descriptors. The pipeline retained metadata, descriptors and limited review sheets rather than a full new image archive. **Metadata** means information about an image, such as author, source URL, license and coordinates.

Review found menus, indoor images and repeated road views. After outdoor review and spatial filtering, a 24-reference candidate expansion left validation unchanged and was not promoted. Acquiring data and using it to improve a deployed model are separate achievements.

For frozen encoders, saving descriptors avoids repeated extraction and saves storage. Full encoder training requires revisiting image pixels; repeatedly deleting and re-downloading every batch each epoch can be slower. A bounded cache or carefully shuffled shards is a more suitable tradeoff. A **shard** is a manageable subset of a larger dataset.

## How this differs from the ETH/Google paper
Scaling Image Geo-Localization to Continent Level combines learned ground-view prototypes with aerial embeddings and fine geographic S2 cell codes. A **prototype** is a learned representative vector. **Aerial imagery** views the ground from above. **Cross-view** learning relates ground and aerial views. **S2** is a hierarchical geographic cell system.

Our fine pipeline uses two ground-image encoders and searches actual training photographs. It has no aerial encoder, fused ground-aerial cell database or geometric verification. It has not reproduced or surpassed ETH's accuracy. Their within-200-m result on a different region and dataset is not comparable to our within-25-km regional result. [ETH/Google project](https://scaling-geoloc.github.io/).

More reliable fine localization would need denser route coverage, reviewed coordinates, stronger route/date-separated evaluation and possibly local matching or licensed aligned aerial imagery. OCR, meaning optical character recognition, could extract text from signs, but it is not part of the current prediction path. Do not claim live web search, OCR or satellite fusion in the demo.

**Checkpoint:** Explain what is implemented before discussing possible extensions. Data acquisition, prototype research and installed inference are different things.

<!-- PAGE -->
# 18. Explain the complete project aloud

## A complete short explanation
KPlaceNet estimates where one photograph was taken. Our main Kerala-area method converts the image to a consistent 224-pixel RGB input, then extracts a 2,048-dimensional ResNet descriptor and a 384-dimensional DINOv2 descriptor. We normalize each, combine them with equal similarity weight and find the most similar entry among 15,939 training references. The output is that reference photograph's saved coordinate, with a visual reference and a Google Maps link.

The regional cell options use ResNet to select one of 300 geographic classes, optionally blending five-neighbor class votes. World mode uses a separate global ResNet classifier and adds temperature-scaled scores, a threshold decision and conformal-set machinery. Its installed calibration artifact is not checkpoint-verified, so its uncertainty output carries that limitation.

Our current regional fine result is 63.03% within 25 km and a 15.92-km median error on 357 reused official test photos. This is not exact-location accuracy. Sparse references, visually similar places, nearby-scene evaluation overlap and lack of outside-region rejection limit the system.

## Check your understanding
1. Why does a high similarity not prove a correct place?
2. Which methods return a centroid, and which returns reference GPS?
3. Why can a gallery example be a test photo while the displayed match is a training photo?
4. Does the fine method use the classifier's winning class to adjust its coordinate?
5. What changes when we select World instead of Kerala?
6. Why is the validation median much smaller than the official test median?
7. Can temperature scaling improve the winning global coordinate by itself?
8. Is a conformal set of 200 classes evidence of precise localization?
9. Why were the new Panoramax photos not added to deployment?
10. What must be rebuilt if the encoders change?

## Answers
1. It measures learned appearance; different places can look alike.

2. Cell options and World return centers; regional fine returns a reference coordinate.

3. Query and bank are separate roles and disjoint ID sets.

4. No: top-one combined-descriptor search supplies it.

5. Separate checkpoint, cell map and prediction workflow are used; regional DINO retrieval is not used.

6. Many validation places are near training references.

7. No: positive-temperature scaling preserves ranking.

8. No: broad class coverage and location precision differ.

9. The reviewed expansion did not meet the validation improvement condition.

10. Recompute matching reference descriptors and repeat the required validation/evaluation.

<!-- PAGE -->
# 19. Source map and essential vocabulary

## Primary project evidence
Use the repository paths below to trace the current behavior. The guide explains it; these files are the reproducible implementation and records.

[Fine inference](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/src/fine_inference.py) and [frontend](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/demo/app.py).
[Regional fine deployment verification](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/fine_deployment_20261021/verification.json).
[Method selection](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/l10_precision_20261021/selection_before_test.json).
[Uncertainty implementation](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/src/uncertainty.py) and [calibration identity checks](https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/src/demo_config.py).
[Official DINOv2](https://github.com/facebookresearch/dinov2), [Torchvision ResNet-50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), and [ETH/Google project](https://scaling-geoloc.github.io/).

## Vocabulary you should be able to explain
| Term | Meaning in this project |
|---|---|
| Training | Changing selected weights using labeled examples and a loss |
| Inference | Applying fixed models and lookup rules to an input |
| Descriptor / embedding | Learned numerical representation of image appearance |
| Retrieval | Selecting a stored reference by similarity |
| Classification | Selecting a discrete geographic class |
| Regression | Directly estimating numerical values; not our fine-output method |
| Centroid | Representative center coordinate of a class |
| Query / reference | Unknown-location input / known-location bank example |
| Calibration | Checking or adjusting score reliability on suitable labeled data |
| Abstention | Withholding an accepted decision under a specified rule |
| Coverage | Depending on context: geographic reference availability, or prediction-set inclusion rate; always state which |
| Generalization | Performance on genuinely new examples and conditions |
| Domain shift | A change in the kind of inputs compared with development data |
| Checkpoint / bank | Saved model parameters / saved reference descriptors and metadata |

You understand the project when you can trace a query to its coordinate, identify which information supplies that coordinate, and explain what the displayed score does and does not establish.
