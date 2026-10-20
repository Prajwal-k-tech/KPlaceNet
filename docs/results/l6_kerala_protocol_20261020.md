# L6 Kerala classifier regularization: prespecified protocol

Written before candidate training or new test evaluation on 20 October 2026.

- Baseline: released L5 Kerala `last.pt`, epoch 10, with its unchanged 300-cell map.
- Data: restore the existing OSV-5M Kerala bounding-box slice. Reconstruct the exact seed-42 split: 15,939 train, 1,000 validation, 357 official test. This is not newly added training data.
- Features: frozen L5 ResNet-50, deterministic 256-short-edge resize and 224 center crop. No backbone gradients or new data augmentation in this head-only experiment.
- Two fixed head-training recipes, each initialized from the same original head: (a) uniform label smoothing 0.1; (b) 0.9 hard target plus 0.1 geographic target proportional to exp(-haversine_distance/15 km) over saved centroids. The geographic target uses training coordinates only.
- Both recipes: AdamW, learning rate 0.0001, weight decay 0.01, batch 256, 12 epochs, seed 42. The original classifier remains an eligible candidate.
- Select by highest validation within-25-km percentage, then smallest validation median distance. Record selection before test evaluation.
- Eligibility for deployment: at least 0.5 percentage-point validation within-25-km improvement with validation median no more than 2% worse. Report the selected candidate and baseline on the same official test images after selection. Do not tune again on the test result.
- Keep source weights intact. Any operational rejection after observing test regression must be stated explicitly; that test cannot then be called an untouched holdout for another tuning round.

## Relation to the ETH Zurich work

Lindenberger et al., *Scaling Image Geo-Localization to Continent Level*, NeurIPS 2025, combine learned ground prototypes, aerial embeddings and interpolation over geographic cells. This experiment adapts only the broad idea that nearby geographic targets should not be treated like distant alternatives. Our geographic soft targets are not their prototype interpolation, multi-similarity loss, S2 hierarchy or aerial system. No paper-level accuracy is claimed.

Sources: https://scaling-geoloc.github.io/ and https://arxiv.org/abs/2510.26795 . The repository's original stated inspiration is PlaNet, https://arxiv.org/abs/1602.05314 .

## Limits

The existing random validation holdout is spatially optimistic. The backbone has already seen the training subset in L5. This is a small adaptation of a pretrained regional classifier, not an independent from-scratch comparison, a new-data experiment, or a significance claim. The source coordinate filter is a bounding box, not an audited Kerala state polygon.
