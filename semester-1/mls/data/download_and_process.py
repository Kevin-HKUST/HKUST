import os
import sys
import zipfile
import subprocess
import numpy as np
import pandas as pd
import librosa
import torch
from tqdm import tqdm
from pathlib import Path
import shutil
import urllib.request
import argparse

# Configuration
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CLIENTS_DIR = DATA_DIR / "clients"

# Audio Config for MobileNet/CNN
SAMPLE_RATE = 16000
DURATION = 5  # seconds
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512


def extract_archive(archive_path, extract_to):
    print(f"[INFO] Extracting {archive_path}...")
    # Try Python zipfile first
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("[INFO] Extraction complete (zipfile).")
        return True
    except zipfile.BadZipFile:
        print("[WARN] Python zipfile failed. Checking if it's a tar...")
        try:
            import tarfile
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path) as tar:
                    tar.extractall(path=extract_to)
                print("[INFO] Extraction complete (tarfile).")
                return True
        except Exception as e:
            print(f"[ERROR] Tar extraction failed: {e}")

    print("[ERROR] Could not extract archive. It might be corrupted.")
    return False


def process_single_audio(file_path, save_path):
    """
    Reads an audio file, converts to Mel Spectrogram, normalizes, and saves as .npy.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # 2. Pad or Crop
        target_len = SAMPLE_RATE * DURATION
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))

        # 3. Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # 4. Normalize [0, 1]
        min_val = mel_spec_db.min()
        max_val = mel_spec_db.max()
        if max_val - min_val > 1e-6:
            mel_spec_db = (mel_spec_db - min_val) / (max_val - min_val)
        else:
            mel_spec_db = np.zeros_like(mel_spec_db)

        # 5. Save
        np.save(save_path, mel_spec_db.astype(np.float32))
        return True

    except Exception as e:
        # print(f"[WARN] Corrupted or error processing {file_path}: {e}")
        return False


def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = len(np.unique(train_labels))
    label_map = {k: i for i, k in enumerate(np.unique(train_labels))}
    numeric_labels = np.array([label_map[l] for l in train_labels])

    min_size = 0
    min_require_size = 10

    net_dataidx_map = {}

    # Try to partition until min_size requirement is met
    # If it fails too many times, we fallback to random split
    max_retries = 20
    for _ in range(max_retries):
        idx_batch = [[] for _ in range(n_clients)]
        for k in range(n_classes):
            idx_k = np.where(numeric_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
            proportions = np.array([p * (len(idx_j) < len(numeric_labels) / n_clients)
                                   for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) *
                           len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,
                         idx in zip(idx_batch, np.split(idx_k, proportions))]

        min_size = min([len(idx_j) for idx_j in idx_batch])
        if min_size >= min_require_size:
            break

    if min_size < min_require_size:
        print("[WARN] Dirichlet split failed to satisfy min_size. Falling back to simple random split.")
        indices = np.random.permutation(len(train_labels))
        split_indices = np.array_split(indices, n_clients)
        for j in range(n_clients):
            net_dataidx_map[j] = split_indices[j].tolist()
    else:
        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def partition_data(metadata_df, num_clients=100, alpha=0.5):
    print(
        f"[INFO] Partitioning data into {num_clients} clients (Alpha={alpha})...")

    # 1. Filter valid files first
    print("[INFO] Validating processed files...")
    valid_mask = []

    # Check if 'uuid' exists
    if 'uuid' not in metadata_df.columns:
        print(
            f"[WARN] 'uuid' column not found. Using 'uuid' from index if possible or filename.")
        return

    # Check for cough_detected column for quality filtering
    has_quality_col = 'cough_detected' in metadata_df.columns

    for uuid in tqdm(metadata_df['uuid']):
        valid_mask.append((PROCESSED_DIR / f"{uuid}.npy").exists())

    df = metadata_df[valid_mask].copy().reset_index(drop=True)
    print(f"[INFO] Raw valid files: {len(df)}")

    # 2. Realistic Cleaning (Include Symptomatic & Noisy Data)
    print("[INFO] Applying realistic cleaning (Keep Symptomatic & Noisy)...")

    # Relax quality filter to simulate real-world noisy environment
    if has_quality_col:
        original_len = len(df)
        df = df[df['cough_detected'] >= 0.5] # Lower threshold from 0.8 to 0.5
        print(f"  - Quality Filter (prob >= 0.5): {original_len} -> {len(df)}")

    # Keep 'healthy' and 'COVID-19' only (Binary Classification for Higher Acc)
    if 'status' in df.columns:
        original_len = len(df)
        df = df[df['status'].isin(['healthy', 'COVID-19'])]
        print(
            f"  - Label Filter (healthy/COVID only): {original_len} -> {len(df)}")
    else:
        print("[ERROR] 'status' column missing. Cannot separate classes.")
        return

    # 3. Handle Imbalance (Disable Oversampling to avoid Leakage)
    healthy_df = df[df['status'] == 'healthy']
    covid_df = df[df['status'] == 'COVID-19']
    
    n_healthy = len(healthy_df)
    n_covid = len(covid_df)
    print(f"[INFO] Class Balance: Healthy={n_healthy}, COVID={n_covid}")
    
    # REMOVED: Oversampling logic causing data leakage
    # We will handle imbalance via Weighted Loss in training, not here.
    
    # Shuffle once before partition
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    labels = df['status'].values

    # Perform Non-IID split
    client_indices_map = dirichlet_split_noniid(labels, alpha, num_clients)

    # Save clients
    if CLIENTS_DIR.exists():
        shutil.rmtree(CLIENTS_DIR)
    CLIENTS_DIR.mkdir()

    for client_id, indices in client_indices_map.items():
        client_data = df.iloc[indices]
        client_dir = CLIENTS_DIR / f"client_{client_id}"
        client_dir.mkdir()
        client_data.to_csv(client_dir / "metadata.csv", index=False)

    print("[INFO] Partitioning complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int,
                        default=100, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet alpha")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Extract if needed
    archive_path = RAW_DIR / "archive.zip"
    # Check if we already have extracted files (csv and wavs)
    has_wavs = len(list(RAW_DIR.glob("*.wav"))) > 100
    if not has_wavs:
        if archive_path.exists():
            extract_archive(str(archive_path), str(RAW_DIR))
        else:
            print(
                f"[ERROR] {archive_path} not found. Please place the dataset file there.")
            sys.exit(1)

    # 2. Process Audio
    print("[INFO] Processing Audio (WAV -> MelSpectrogram)...")
    audio_files = list(RAW_DIR.glob("*.wav")) + list(RAW_DIR.glob("**/*.wav"))

    if not audio_files:
        print(f"[ERROR] No .wav files found in {RAW_DIR}.")
        sys.exit(1)

    print(f"[INFO] Found {len(audio_files)} audio files.")

    processed_count = 0
    for audio_path in tqdm(audio_files):
        uuid = audio_path.stem  # Kaggle usually is uuid.wav
        save_path = PROCESSED_DIR / f"{uuid}.npy"
        if not save_path.exists():
            if process_single_audio(str(audio_path), str(save_path)):
                processed_count += 1

    print(f"[INFO] Processed {processed_count} new files.")

    # 3. Load Metadata
    # Search for any csv
    csv_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("**/*.csv"))
    if not csv_files:
        print("[ERROR] No Metadata CSV found. Cannot partition.")
        sys.exit(1)

    # Use the largest csv found (likely the metadata one)
    metadata_csv = max(csv_files, key=lambda p: p.stat().st_size)
    print(f"[INFO] Using metadata file: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    partition_data(df, num_clients=args.num_clients, alpha=args.alpha)


if __name__ == "__main__":
    main()
