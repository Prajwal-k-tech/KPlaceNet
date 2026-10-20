# Current preparation:21 October fine retrieval
Read Fine-Retrieval-Update.pdf first. The default live Kerala method is now L10, combining ResNet and DINOv2 and retrieving actual training-photo coordinates. L7 remains available as a comparison.

1. Last-Minute-Cheat-Sheet.pdf: one-page revision.
2. Study-Guide.pdf: current update first, then historical foundations with definitions and worked examples.
3. Demo-Playbook.pdf: current method controls, five-minute script, failure, recovery and viva answers.
4. Project-Report.pdf: current update plus original13-section report. Fill group/title-page details before submission.
5. Experiment-and-Data-Notes.pdf: measured comparisons, dataset acquisition, rejected experiment and ETH differences.

The existing Final-Presentation-L7-Final.pptx/PDF describes the historical L7 experiment. It has NOT been revised into an L10 deck in this run. Use the editable four-slide Fine-Retrieval-Slides.pptx as the current-method/results supplement, alongside the dated Fine-Retrieval-Update.pdf. Its old checkbox screenshots do not match the new method radio.

Run from this task workspace: ./outputs/Start-KPlaceNet.sh . From work/KPlaceNet: ../../outputs/Start-KPlaceNet.sh . Open http://127.0.0.1:7861/ . Keep terminal open.

Current reused-test result:15.92km median;17.37% within5km;63.03% within25km. Not street-address accuracy, not fresh independent confirmation, not a reproduction of ETH. New Panoramax images were acquired/encoded but not promoted into deployment.

Install assets in another checkout: python scripts/download_fine_retrieval.py . Fine-bank release:https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-fine-l10-20261021 . Git alone does not include weights or all reference photos.

Code is pushed to Prajwal-k-tech. Original Asterisk-Hunter main requires the maintainer to merge PR1; do not say the two repository mains are synchronized.
