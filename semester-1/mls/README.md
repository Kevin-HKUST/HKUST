# MLS — Machine Learning Systems

Semester 1. The course deliverable was the FedCough term project, documented below. The accompanying slide deck was a presentation PDF and is kept offline.

---

# FedCough

This project implements a Federated Learning (FL) system for cough detection (COVID-19 vs Healthy vs Symptomatic) using audio data. It includes both a Federated Learning simulation and a Centralized Baseline for comparison.

## Project Structure

- `data/`: Contains data processing scripts and storage.
  - `download_and_process.py`: Script to extract, process (Mel Spectrogram), and partition data for FL.
  - `dataset.py`: PyTorch Dataset definition.
  - `raw/`: Place your raw dataset (wav files or `archive.zip`) here.
  - `processed/`: Stores processed `.npy` files.
  - `clients/`: Stores client data partitions.
- `models/`: Neural network models (e.g., MobileNet).
- `system/`: Core FL system components (Server, Client).
- `utils/`: Helper utilities.
- `main.py`: Entry point for Federated Learning simulation.
- `baseline_train.py`: Entry point for Centralized Baseline training.
- `requirements.txt`: Python dependencies.

## Prerequisites

- Python 3.9
- CUDA-enabled GPU (recommended for faster training)

## Installation

1. Clone or download this repository.
2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data Preparation

Before running simulations, you must prepare the dataset. This project is designed to work with cough audio datasets (e.g., COUGHVID).

1. **Download Data**: Obtain the dataset (e.g., from Kaggle). You should have an `archive.zip` file containing `.wav` files and a metadata CSV.
2. **Place Data**: Move `archive.zip` to `data/raw/`:
   ```bash
   mkdir -p data/raw
   cp /path/to/archive.zip data/raw/
   ```
3. **Process Data**: Run the processing script. This will extract audio, convert to Mel Spectrograms, and partition data among clients.
   ```bash
   python data/download_and_process.py --num_clients 100 --alpha 0.7
   ```
   - `--num_clients`: Total number of simulated FL clients.
   - `--alpha`: Dirichlet distribution parameter for Non-IID data partition (lower = more heterogeneous).

## Running the Project

### 1. Federated Learning Simulation

To run the FL simulation:

```bash
python main.py --rounds 10 --clients 100 --fraction 0.1 --gpu
```

**Arguments:**
- `--rounds`: Number of Federated Learning rounds (default: 5).
- `--clients`: Total number of clients (default: 100).
- `--fraction`: Fraction of clients selected per round (e.g., 0.1 = 10%).
- `--epsilon`: Differential Privacy epsilon value (0.0 = disable privacy).
- `--gpu`: Flag to enable GPU usage.

### 2. Centralized Baseline

To run the centralized training baseline (standard deep learning training on all data):

```bash
python baseline_train.py
```

This will:
- Load data from all clients.
- Split into Train (80%) and Test (20%).
- Train a MobileNet model.
- Save results to `baseline_results.png` and `confusion_matrix.png`.

## Outputs

- **FL Simulation**: Prints round-wise accuracy and loss to the console.
- **Baseline**: Generates:
  - `baseline_results.png`: Training loss and accuracy plots.
  - `confusion_matrix.png`: Confusion matrix of the final model.

## Troubleshooting

- **Dataset Not Found**: Ensure `archive.zip` is in `data/raw/` or `.wav` files are extracted there.
- **CUDA/GPU**: If you have a GPU but scripts run on CPU, check your PyTorch installation with `torch.cuda.is_available()`.

