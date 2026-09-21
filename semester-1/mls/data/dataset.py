import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import os


class FedCoughDataset(Dataset):
    def __init__(self, client_id, data_dir, transform=None):
        """
        Args:
            client_id (int or str): ID of the client (e.g., 0, 'client_0')
            data_dir (str or Path): Root directory containing 'clients' and 'processed'
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Determine client folder
        if isinstance(client_id, int):
            self.client_name = f"client_{client_id}"
        else:
            self.client_name = client_id

        self.client_dir = self.data_dir / "clients" / self.client_name
        self.processed_dir = self.data_dir / "processed"

        # Load metadata
        self.metadata = pd.read_csv(self.client_dir / "metadata.csv")

        # Label mapping
        self.label_map = {
            'healthy': 0,
            'symptomatic': 1,
            'COVID-19': 2,
            'unknown': 0  # Treat unknown as healthy for now, or ignore
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        uuid = row['uuid']
        label_str = row['status']

        # Load Mel Spectrogram
        npy_path = self.processed_dir / f"{uuid}.npy"

        try:
            # Shape: [N_MELS, TIME_STEPS] e.g. [64, 157]
            spec = np.load(npy_path)

            # Add channel dimension: [1, 64, 157]
            spec = spec[np.newaxis, ...]

            # Apply transforms (Augmentation)
            if self.transform:
                spec = self.transform(spec)

            # Convert to tensor
            image = torch.from_numpy(spec).float()

            # Convert label
            label = self.label_map.get(label_str, 0)
            label = torch.tensor(label, dtype=torch.long)

            return image, label

        except Exception as e:
            # print(f"Error loading {npy_path}: {e}")
            # Return a dummy zero tensor in case of load failure to keep DataLoader alive
            return torch.zeros((1, 64, 157)), torch.tensor(0, dtype=torch.long)


class SpecAugment:
    """
    Simple SpecAugment implementation for data augmentation.
    """

    def __init__(self, freq_mask_param=10, time_mask_param=20):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def __call__(self, spec):
        # spec: numpy array [C, F, T]
        # Frequency masking
        F = spec.shape[1]
        f = np.random.randint(0, self.freq_mask_param)
        f0 = np.random.randint(0, F - f)
        spec[:, f0:f0+f, :] = 0

        # Time masking
        T = spec.shape[2]
        t = np.random.randint(0, self.time_mask_param)
        t0 = np.random.randint(0, T - t)
        spec[:, :, t0:t0+t] = 0

        return spec
