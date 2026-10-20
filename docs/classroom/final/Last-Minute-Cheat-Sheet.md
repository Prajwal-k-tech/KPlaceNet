# KPlaceNet: last-minute cheat sheet
Current version: L10 fine retrieval / 21 October 2026

**Task:** estimate location from one street photograph. Main demo assumes Kerala-area input; the source rectangle includes neighboring states. No outside-region rejection.

**Pipeline:** RGB image -> resize/crop224x224 and ImageNet normalization -> ResNet50 descriptor(2048) + frozen DINOv2 ViT-S/14 descriptor(384) -> normalize each -> equal-weight concatenation -> cosine top-one among15,939 training references -> matched reference GPS. Similarity is not probability. Fine output is not a300-cell center and is not continuous regression.

**Results:** reused official test357 photos; median15.92km; within5km17.37%(62); within25km63.03%(225). Previous L7:21.84km and53.50% within25km. None within1km; all test locations are over1km from the original training bank. Random validation is much easier and not unseen-place evidence.

**Demo:** Kerala -> Fine coordinates -> Kottayam thumbnail -> Locate -> show reference and coordinate -> Google Maps -> compare Regional cells + KNN -> return fine -> Southern Kerala failure. Thumbnail selection only loads a query. Test query and retrieved training reference are different images.

**Earlier model:** ResNet softmax over300 geographic cells. L7 blends80% classifier +20% five-neighbor class votes and returns a center. L10 uses1-nearest-neighbor coordinates over combined features instead.

**Data:** train15,939; validation1,000; official test357. New Panoramax789 unique images encoded; not used in deployment because the reviewed subset failed validation promotion. Batch encoding saves raw-image storage; repeated downloading per epoch can be slower.

**ETH comparison:** ground prototypes + aerial encoder + fine S2 codes, vastly larger training scale. We have two ground encoders and no aerial fusion. We have neither reproduced nor beaten ETH.

**Run:** task workspace: `./outputs/Start-KPlaceNet.sh`. From repo: `../../outputs/Start-KPlaceNet.sh`. Open http://127.0.0.1:7861/ . Keep terminal open. Initial inference loads models. Weights installer: `python scripts/download_fine_retrieval.py`.

**Honesty:** curated demos are not average performance; reused test is not fresh confirmation; GPS decimals do not mean GPS precision; Great Wall can still match India; no claim of street-address accuracy.
