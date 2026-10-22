# Current preparation: 22 October 2026
The Study Guide and Demo Playbook teach only the current Kerala-area system, including the five-neighbor KNN option and its classifier-only comparison. They contain no World-mode chapters. Read Study-Guide.pdf first; Fine-Retrieval-Update.pdf is an optional short overview. The default live Kerala method is now L10, combining ResNet and DINOv2 and retrieving actual training-photo coordinates. L7 remains available as a comparison.

1. Last-Minute-Cheat-Sheet.pdf: one-page revision.
2. Study-Guide.pdf:17-page Kerala learning course with clear definitions, image preprocessing, ResNet/DINO features, fine retrieval, classifier cells, KNN voting, a numerical80/20 example, training and evaluation.
3. Demo-Playbook.pdf:5-page Kerala-only demo guide with startup commands, a timed walkthrough, KNN enabled/disabled comparison, viva answers and recovery.
4. Project-Report.pdf: 13-page Kerala-focused report following all 13 required sections, with the confirmed group, roll numbers and instructor. Project-Report.docx is the matching editable Word file.
5. Experiment-and-Data-Notes.pdf: measured comparisons, dataset acquisition, rejected experiment and ETH differences.

The existing Final-Presentation-L7-Final.pptx/PDF describes the historical L7 experiment. It has NOT been revised into an L10 deck in this run. Use the editable four-slide Fine-Retrieval-Slides.pptx as the current-method/results supplement, alongside the dated Fine-Retrieval-Update.pdf. Its old checkbox screenshots do not match the new method radio.

Run from this task workspace: ./outputs/Start-KPlaceNet.sh . From work/KPlaceNet: ../../outputs/Start-KPlaceNet.sh . Open http://127.0.0.1:7861/ . Keep terminal open.

Current reused-test result:15.92km median;17.37% within5km;63.03% within25km. Not street-address accuracy, not fresh independent confirmation, not a reproduction of ETH. New Panoramax images were acquired/encoded but not promoted into deployment.

Install assets in another checkout: python scripts/download_fine_retrieval.py . Fine-bank release:https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-fine-l10-20261021 . Git alone does not include weights or all reference photos.

Code is pushed to Prajwal-k-tech. Original Asterisk-Hunter main requires the maintainer to merge PR1; do not say the two repository mains are synchronized.
