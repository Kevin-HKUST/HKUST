# MAIE 5101 — Programming for Artificial Intelligence

Semester 2, Spring 2026. A core course of the MSc(AIE) program, covering NumPy, Matplotlib, and Pandas alongside deep learning in PyTorch.

## Contents

| Path | Description |
|---|---|
| [`assignment-1/ai-platform/`](assignment-1/ai-platform) | Decorators, roles, and a model abstraction for a small AI-platform exercise |
| [`assignment-1/climate-analysis/`](assignment-1/climate-analysis) | Climate data analysis functions |
| [`assignment-2/`](assignment-2) | Pandas analysis of Australian weather data — `analysis.py` and `maie-pa2.ipynb` |
| [`assignment-3/`](assignment-3) | Image colorisation with PyTorch — `colorization.py`, `models.py`, the notebook, and the loss curves |
| `notes.pdf` | My course notes |

Each assignment-1 exercise keeps its `sample-output.txt` so the expected behaviour is visible without running the code.

## Datasets

Excluded from the repository because of their size; download before running:

- **Assignment 2** — `weatherAUS.csv` (28 MB), the Kaggle *Rain in Australia* dataset. Place it next to `analysis.py`.
- **Assignment 3** — CIFAR-10 (163 MB archive). Fetch via `torchvision.datasets.CIFAR10(root="dataset", download=True)`, which populates `dataset/cifar-10-batches-py/`.

The trained checkpoint `best_model.pth` is also excluded; rerun training to regenerate it.

## Not included

Eighteen lecture decks and the sample exam paper belong to the course staff and are not redistributed.
