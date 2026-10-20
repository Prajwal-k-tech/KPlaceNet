# Fine retrieval slide speaker notes /21 October2026

## Slide1

Earlier classifiers return one of300 centroids. L10 instead searches15939 training photos and returns the matched photo coordinate. Some reference coordinates may be repeated. No arbitrary GPS regression or outside-region rejection is claimed. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/src/fine_inference.py .

## Slide2

Resize short edge256, center crop224 and ImageNet normalization. L2 normalize each descriptor; concatenate sqrt(.5) times each. Cosine is the equally weighted encoder cosine. Top1 supplies its stored coordinate. ResNet2048 plus DINO384 gives2432 dimensions. DINO-only did not beat the prior validation baseline; equal mixture k1 won the six-recipe precision-first comparison. No end-to-end DINO fine-tuning was performed. Sources: https://github.com/facebookresearch/dinov2 and repository scripts/evaluate_l10_precision.py .

## Slide3

L7:191/357within25km(53.50%),median21.844km,within5km30/357(8.40%). L10:225/357within25km(63.025%),median15.9205km,within5km62/357(17.367%). L10 selected on validation within5km with25km guardrail; evaluated test after selection. Test already used across experiments, not fresh confirmation. None within1km because every test coordinate lies over1km from original reference bank. Source: https://github.com/Prajwal-k-tech/KPlaceNet/blob/main/docs/results/fine_deployment_20261021/verification.json .

## Slide4

Panoramax936candidates ->789uniqueencoded; first144reviewed contain menus, indoor views and repeated frames.118outdoorcandidates ->24spatiallyexcluded/thinnednewreferences; validation unchanged so not promoted. Raw photos streamed in memory, no full-image archive retained. ETH/Google combines ground prototypes with aerial embeddings and fine S2 codes, unlike our two-ground-encoder reference search. We have not reproduced or surpassed ETH. Need new routes, independent route/date evaluation, appropriate aerial imagery and local verification. Sources: https://docs.panoramax.fr/ and https://scaling-geoloc.github.io/ .