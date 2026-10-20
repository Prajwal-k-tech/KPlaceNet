# Final integration and verification, 20 October 2026

Prajwal's fork main and both contribution branches contain the integrated original history, evaluation corrections, responsive Atlas frontend, browser image-link drops, Google Maps links and the measured L6 Kerala follow-up. PR 1 into Asterisk-Hunter/KPlaceNet remains the maintainer merge route; direct writes to the original repository are unavailable. No original history or baseline checkpoint was overwritten.

## Current measured model

L6 freezes the existing regional backbone and cells and trains a geographic-smoothed classifier head. It improves the same 357-image official-test within-25-km rate from 41.18% to 47.62% and median error from 42.08 to 28.41 km. Validation chooses the recipe and epoch before test evaluation. One seed, spatially optimistic random validation and uncalibrated Kerala scores remain important limits. See docs/results/l6_kerala_20261020.md and its JSON/CSV evidence.

## Validation

- CPU environment: 80 tests run, 79 passed, one CUDA-only skip.
- Separate CUDA environment: all 80 tests passed.
- Full source images restored: 16,939 official-training records and 357 official-test records, with size and CRC checks on newly downloaded images.
- Reconstructed train/validation/test image IDs are disjoint. A coordinate audit measures 982/1,000 validation rows and zero of 357 test rows within 1 km of the recovered training rows.
- Both actual trained models infer on all ten configured gallery photos. The L6 browser prediction and Google Maps destination are verified.
- Published model assets match SHA-256 and size records. The L6 checkpoint loads with weights_only=True. Source model tensors are unchanged by the metadata compatibility repack.
- Frontend desktop and phone layouts were inspected. File and browser image-URL drop formats were exercised, with clear feedback for blocked or non-image links.

## Installed artifacts and fallback

The demo prefers checkpoints/l6_kerala_geographic/selected.pt and its cells.json when present, otherwise legacy L5. Install L6 using python scripts/download_kerala_l6.py. The verified release is https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l6-20261020-verified . The historical global calibration lacks an identity hash and is labeled unverified. The later corrected 20k global study is a separate model and protocol.

The task launcher prefers the isolated GPU environment and otherwise uses CPU. Matching workspace-local NVIDIA libraries fix the running 610.57.04 module's userspace mismatch without modifying system drivers. Kaggle authentication and available quota were checked, but the experiment ran locally.

## Preparation and data research

The updated report retains all 13 required sections. The study guide and live-demo playbook reflect L6, the current controls, limits and exact artifact paths. The additional-data collector produced 350 Commons candidate metadata records; these were not used for training or evaluation and still need image/coordinate/duplicate review. The confirmed ETH paper is a later improvement reference, not a claim of reproduction.


## Final L7 refinement
The optional live default uses a matching training-only bank with five cosine neighbors and fixed 80/20 classifier/retrieval blend. Validation selects before follow-up test reporting. The reused 357-image benchmark records 53.50% within 25 km and 21.84-km median error, with worse p90 than L6. There are now 82 tests (GPU all pass; CPU one CUDA-only skip). Release: https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l7-hybrid-20261020 . Regional mode assumes the Kerala-area bounding rectangle and cannot detect outside-region input.
