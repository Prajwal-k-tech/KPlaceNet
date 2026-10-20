# KPlaceNet: last-minute cheat sheet

## Say this first
We estimate a photograph's geographic region from pixels. ResNet-50 produces 2,048 features, a classifier and optional training-only KNN vote choose one of 300 cells, and we display its centroid. Kerala-area mode assumes that domain; it cannot reject foreign scenes and its bounding rectangle includes neighboring territory.

## Procedure
RGB image; deterministic resize and 224 crop; ImageNet normalization; ResNet-50; 300 logits; softmax; optional 80/20 blend with five cosine-nearest training features; highest-scoring cell; saved centroid; map. Training GPS creates labels. Inference uses pixels. Gallery click loads an image; Locate photo runs inference.

## Final measured result
357 official test photos. L5 / L6 / final hybrid: within 25 km 41.18% / 47.62% / 53.50%; median error 42.08 / 28.41 / 21.84 km. Hybrid mean 80.65 km; p90 worsens versus L6. L7 reuses the L6 benchmark, so it is a follow-up, not fresh independent confirmation.

## What L6 changed
Freeze the L5 backbone and cells. Train only the 614,700-parameter head. Compare uniform and geographic smoothing. q = 0.9 one-hot + 0.1 softmax(-centroid distance / 15 km). AdamW, learning rate 0.0001, weight decay 0.01, 12 epochs. Geographic recipe wins validation at epoch 12 before test evaluation.

## The split problem
15,939 train; 1,000 random validation; 357 official test. IDs disjoint. In our recovered subset, 982 validation photos are within 1 km of training, versus zero official test photos. Random validation is optimistic.

## Do not confuse
K-means builds coordinate groups; KNN retrieves five training-only ResNet feature neighbors. Raw confidence is not measured correctness. Coverage is not top-1 accuracy. Bounds are not a confidence interval. Four decimal places are not GPS precision. Google Maps displays the estimate and does not correct it.

## Uncertainty and contribution
Kerala score is uncalibrated. Global legacy temperature scaling has unverified artifact identity. Separate 20k global study: ECE 0.4171 to 0.0453, coverage 90.8%, average set 159.7/300. Conformal rank: ceil((n+1)(1-alpha)); rank beyond n means infinity. My contributions include evaluation integrity, reproducibility, integration and L6/L7 follow-ups; credit the existing project separately.

## Demo and ETH comparison
Show Kottayam example, Locate photo, coordinates, raw score, Google Maps, then the Southern Kerala failure. ETH paper: prototypes plus aerial embeddings and geographic interpolation. Our L6 adapts boundary awareness only. The 350 Commons records are future-data candidates, not added training images.
