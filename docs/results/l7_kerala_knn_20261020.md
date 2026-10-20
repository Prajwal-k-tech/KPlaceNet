# L7 training-only retrieval follow-up

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
