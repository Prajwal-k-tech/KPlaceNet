# KPlaceNet recovery and reproduction

The local working copies were offloaded on 23 October 2026. Source branches and experiment artifacts are preserved in this fork; dataset photographs, local Python environments and caches are not stored in Git.

## Restore the latest integrated demo

```bash
git clone --branch recovery/20261023/integrated/integrate-prajwal-contributions \
  https://github.com/Prajwal-k-tech/KPlaceNet.git
cd KPlaceNet
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, use `.venv\Scripts\python.exe` instead. To match the previously working NVIDIA runtime, use Python 3.12 and the [official PyTorch 2.7.1 CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/):

```bash
.venv/bin/python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

The previous app actually imported `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`, Python 3.12.14. Its old environment also contained stale CPU-wheel metadata; do not copy that environment or treat an unfiltered package inventory as an install lock. A clean CPU installation is an alternative if GPU execution is not needed. A fresh full dependency installation was not repeated during this storage cleanup.

Download the deployed artifacts using the committed downloaders:

```bash
.venv/bin/python scripts/download_weights.py
.venv/bin/python scripts/download_kerala_l6.py
.venv/bin/python scripts/download_kerala_l7.py
.venv/bin/python scripts/download_fine_retrieval.py
.venv/bin/python scripts/check_demo_ready.py --mode all
.venv/bin/python -m demo.app
```

The existing downloaders identify artifact destinations. Deployed files are available in [upstream weights-v1](https://github.com/Asterisk-Hunter/KPlaceNet/releases/tag/weights-v1), [verified L6](https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l6-20261020-verified), [L7](https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-l7-hybrid-20261020), and [fine retrieval](https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/kerala-fine-l10-20261021). The additional research bundle below is optional for running that deployed demo. See the [classroom preparation index](https://github.com/Prajwal-k-tech/KPlaceNet/blob/recovery/20261023/integrated/integrate-prajwal-contributions/docs/classroom/final/START-HERE.md) and its recorded limitations.

## Restore all archived research checkpoints

The [offload recovery release](https://github.com/Prajwal-k-tech/KPlaceNet/releases/tag/offload-recovery-20261023) contains a SHA256 manifest and all additional checkpoint/feature-bank objects missing from the prior releases. Duplicate objects are stored only once. The zstd tar is split into three parts, each at most 1 GiB; the complete bundle download is about 2.22 GiB. Decoding needs `zstd` with a 512 MiB window. It contains models and derived features/metadata, not dataset photographs.

```bash
# Inspect without downloading the model bundle or starting training.
.venv/bin/python scripts/restore_offloaded_models.py --workspace integrated --plan-only
# Restore the integrated workspace's complete checkpoint tree.
.venv/bin/python scripts/restore_offloaded_models.py --workspace integrated
```

The helper verifies each downloaded part and each restored object by SHA256, preserves matching existing files, refuses to overwrite differing files, and removes temporary downloads after completion. It never starts training or loads pickle checkpoints. For the earlier evaluation worktree, clone `recovery/20261023/evaluation/prajwal/eval-integrity` and select `--workspace evaluation`. All six prior local source branches remain under `recovery/20261023/` in this fork. Raw source snapshot commits are `c02d3201d4a6249d63ba00f9b057578aa4c501c0` (evaluation) and `e588f469eb60888efee1b15a2e211766fe8698f7` (integrated).

Before offloading, all 306 additional unique objects were decompressed and checked against their original file hashes. GitHub's server-reported SHA256 for every uploaded part matched locally. An actual small-file restoration through the helper passed. A full new multi-GB download and every training/inference experiment were not repeated during cleanup.

## Datasets: download from their providers

[OSV-5M](https://huggingface.co/datasets/osv5m/osv5m) is distributed under CC BY-SA 4.0; retain attribution to [the dataset authors](https://github.com/gastruc/osv5m). The Kerala extraction script pins revision `cff33609b56b54d8743b7ee7a416eb8433e9a681`. The geographic index cache used revision `d2eb1a13eab1a5d3c1828e3db96db3063a607ccf` of `giahuy4205/osv5m-geographic-index`. Use the source's [data guide](https://github.com/Prajwal-k-tech/KPlaceNet/blob/recovery/20261023/integrated/integrate-prajwal-contributions/data/README.md) and the published sample/provenance manifests. Do not snapshot-download the whole roughly 259 GB dataset.

```bash
# Inspect the bounded regional extraction first.
.venv/bin/python scripts/extract_kerala_osv5m.py --dry-run
# Download/extract the pinned Kerala slice when needed.
.venv/bin/python scripts/extract_kerala_osv5m.py
# The earlier bounded 20k pool used four explicitly selected shards.
.venv/bin/python scripts/download_subset.py --dataset osv5m --split train \
  --output-dir data/osv5m_subset_20k --max-samples 20000 \
  --shards 00,01,02,03 --max-shards 4 --seed 42 --plan-only
# Replace --plan-only with --yes only after reviewing the download plan.
```

Official test data is evaluation-only. Recreate training/test selections with the committed manifests; do not mix test photos into training. IM2GPS3k is manual-only through its authors, as described in the data guide. Panoramax candidate manifests retain source URLs, authors, licenses, coordinates and hashes; the rejected expansion was not deployed and its raw photos were not kept. See [Experiment-and-Data-Notes.md](https://github.com/Prajwal-k-tech/KPlaceNet/blob/recovery/20261023/integrated/integrate-prajwal-contributions/docs/classroom/final/Experiment-and-Data-Notes.md).

Checkpoint recovery avoids retraining when the exact previous weights are needed. Re-running CUDA training from seeds/manifests can produce numerical differences and is not a promise of byte-identical new checkpoints. The recorded exploratory/reused test results are not a claim of global, metre-level, or independently confirmed accuracy.
