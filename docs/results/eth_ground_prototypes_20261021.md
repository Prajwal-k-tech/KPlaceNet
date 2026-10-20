# Ground geographic prototypes: feasibility experiment

21 October2026. **Experimental only; not installed in the live prediction path.**

We tested a small ground-only adaptation of components in [Scaling Image Geo-Localization to Continent Level](https://arxiv.org/html/2510.26795v1#S2), using the [released implementation](https://github.com/google-research/scaling-geoloc) to check the method structure. We did not reproduce the full paper or copy its trained model.

## Implemented

- Build a local regular geographic grid with2km or5km spacing, using a fixed regional equirectangular approximation. Only nodes supported by training interpolation memberships exist. This is not S2 and is not a Kerala administrative polygon.
- Assign each training position to four grid nodes with bilinear weights. The positive geographic representation is an interpolated, normalized mixture of node prototypes; this reduces abrupt assignment at cell boundaries.
- Initialize prototypes from weighted averages of frozen, equally combined ResNet-DINO training descriptors.
- Optimize the prototypes using the ground-prototype part of a multi-similarity-style objective: an interpolated positive and geographically separated negative prototypes. The image encoders stay frozen. AdamW,lr0.003,batch256,eight epochs,seed42.
- Test direct prototype-coordinate output and a contextual re-ranking of the top50 reference photos. Re-ranking adds0.1 or0.25 times query-to-interpolated-prototype similarity to original query-reference similarity. The selected reference still supplies the coordinate in re-ranking mode.

The ETH system jointly learns ground and aerial encoders with prototype-ground, prototype-aerial and ground-aerial constraints. We have no aerial data/encoder, no aerial-derived codes, no S2 hierarchy and no encoder optimization in this bounded adaptation. Its high-resolution cross-view capability should not be inferred from these experiments.

## Selection and transparency

Two grid spacings, three checkpoints(epoch0,4,8), and three output/rerank options yielded18 validation candidates. Epoch0 is the weighted-average initialization, **not a learned prototype model**.

The first selection implementation ranked candidates by within5km before applying the median safety condition. Its highest-within5km candidate, learned5km prototypes after8epochs with0.1reranking, achieved72.1% within5km but a1.122km median and failed the median condition. The code was corrected to filter candidate eligibility using all stated conditions before ranking. This amendment was made **after inspecting validation and before this test evaluation**, and is explicitly recorded in the protocol. These are exploratory results, not pristine preregistered or fresh independent confirmation.

With that correction, the selected candidate was **2km prototypes at epoch0, re-ranking weight0.1**. Validation within5km improved71.1% to71.8%, within25km83.3% to84.2%, median1.0722km to1.0379km. No validation/test images were used to initialize or fit the prototypes. No official test labels were used to choose the recipe.

A final deployment condition was saved before test: within5km and within25km may each lose no more than0.5percentage points, and median may not exceed1.02times the current result. This is an operational regression guard, not a statistical significance test.

## Reused official test,357 queries

| Metric | Current fine retrieval | Prototype-context re-ranking |
|---|---:|---:|
| Within5km |17.37%(62)|16.53%(59)|
| Within25km |63.03%(225)|65.83%(235)|
| Median error |15.92km|15.57km|
| Mean error |61.21km|57.70km|
| Within200km |90.76%|90.76%|

38 queries changed reference. The candidate loses three within5km successes while gaining ten within25km successes. It fails the stated fine-precision deployment guard, so the live model and guides' current-pipeline explanation remain unchanged. No top-one neural-prototype coordinate model was promoted.

The pure prototype-coordinate variants were weaker on validation: the strongest within5km value was60.5%, versus71.1% for the original retrieval. A geographic-grid output can be different from any training-photo coordinate, but this experiment demonstrates that novel output coordinates alone do not establish better accuracy.

## Feasibility conclusion

Prototype representations, boundary interpolation, negative-location separation and prototype-context re-ranking are feasible on the local GPU using cached descriptors. This ground-only experiment is not a consistent fine-precision improvement. Full ETH-style cross-view training remains a separate project requiring suitably licensed, aligned, high-resolution Kerala aerial imagery and ground views, as well as independent geographic/temporal evaluation. More GPU hardware alone does not supply that data.

## Reproduce

In the project's prepared GPU environment, run:

```bash
python scripts/evaluate_geographic_prototypes.py
python scripts/check_geographic_prototype_test.py
python -m unittest discover -s tests -q
```

These scripts require the existing frozen ResNet and DINO train/validation caches. The second script saves the selected recipe's test protocol before extracting test features and evaluating. It does not install model assets or modify demo/app.py. On this laptop use the launcher's isolated matching CUDA library setup. Trial outputs are under checkpoints/geographic_prototypes; text records and per-query test errors are committed in [the evidence directory](eth_ground_prototypes_20261021/).
