# Your final KPlaceNet preparation pack

Use this folder as the current source. The final live Kerala demo uses L6 geographic smoothing plus L7 training-only KNN when installed. Older slides and historical L5 figures describe earlier experiments.

1. Read Last-Minute-Cheat-Sheet.pdf for orientation.
2. Study Study-Guide.pdf in order. Explain the procedure aloud without notes.
3. Rehearse Demo-Playbook.pdf twice on http://127.0.0.1:7861/ .
4. Present Final-Presentation-L7-Final.pptx. Slides 1-13 form the main talk; slides 14-15 are optional viva backup. Its speaker notes contain explanations, sources and limitations. Final-Presentation.pdf is the offline backup.
5. Review Project-Report.pdf for the full 13-section submission. Fill the remaining institution, instructor and roll-number fields before submitting by 22 October.
6. Keep Experiment-and-Data-Notes.pdf for the measured final run, ETH comparison and additional-data findings.

The final hybrid result is 53.50% within 25 km and 21.84-km median error on 357 official test photos. Those numbers belong to L7 hybrid, not global mode. L6 alone scores 47.62% and 28.41 km. L7 is a follow-up on the already-used official benchmark. No new Commons image was used in training. All ten demo photos came from official test archives, but were deliberately selected.

Start this laptop's frontend from the task workspace with ./outputs/Start-KPlaceNet.sh . The launcher checks GPU availability and falls back to the existing CPU environment. Keep its terminal open.

The source is in work/KPlaceNet. The installed improved weights are checkpoints/l6_kerala_geographic/selected.pt with cells.json. The matching retrieval bank is checkpoints/l7_kerala_hybrid/train_bank.pt. Source code is pushed to Prajwal-k-tech/KPlaceNet and the existing PR into Asterisk-Hunter/KPlaceNet. Original main still needs a maintainer merge. Install the L6 model and L7 bank with python scripts/download_kerala_l7.py on another checkout; Git alone does not supply weights.

Do not claim GPS precision, 90% top-1 accuracy, that KNN can detect any foreign scene, that the ETH model was reproduced, that random validation proves generalization, or that a raw score is probability of correctness.

Model release: https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l6-20261020-verified

Matching neighbor-bank release: https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l7-hybrid-20261020


## Revised learning guides
The Study Guide has been rewritten as a 24-page course with definitions, numerical examples, chapter checkpoints and an answer key. Chapters 1-14 cover the regional core; later chapters cover global work, sources and viva practice. The 8-page Demo Playbook gives a Kerala-first five-minute action/speaking route, expected outputs, KNN comparison, a measured failure and recovery procedures. Use these revised versions instead of earlier copies.
