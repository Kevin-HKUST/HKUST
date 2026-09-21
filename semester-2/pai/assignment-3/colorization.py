import skimage.color
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from models import ColorizationCnn

def augment_transform():
    return v2.Compose([
        transforms.ToTensor(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(degrees=360),
    ])

def load_cifar_10(path: str):
    transform = transforms.ToTensor()
    full_train = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=augment_transform())
    test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform)
    return full_train, test_set

def rgb_to_lab_tensor(rgb_tensor):
    batch_size = rgb_tensor.shape[0]
    rgb_np = rgb_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    lab_list = []
    for i in range(batch_size):
        lab = skimage.color.rgb2lab(rgb_np[i])
        lab[:, :, 0] /= 100.0    # L: [0,100] -> [0,1]
        lab[:, :, 1] /= 128.0    # a: ~[-128,127] -> ~[-1,1]
        lab[:, :, 2] /= 128.0    # b: ~[-128,127] -> ~[-1,1]
        lab_list.append(lab)
    lab_np = np.array(lab_list)
    return torch.from_numpy(lab_np).permute(0, 3, 1, 2).float()

def lab_to_rgb_tensor(lab_tensor):
    batch_size = lab_tensor.shape[0]
    lab_np = lab_tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    rgb_list = []
    for i in range(batch_size):
        lab = lab_np[i].copy()
        lab[:, :, 0] *= 100.0
        lab[:, :, 1] *= 128.0
        lab[:, :, 2] *= 128.0
        rgb_list.append(skimage.color.lab2rgb(lab))
    rgb_np = np.array(rgb_list)
    return torch.from_numpy(rgb_np).permute(0, 3, 1, 2).float()

class LabDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        lab = rgb_to_lab_tensor(img.unsqueeze(0))
        return lab[0, 0:1], lab[0, 1:3]   # (1,H,W), (2,H,W)

def get_data_loaders(train_set, test_set, batch_size, train_val_split):
    gen = torch.Generator().manual_seed(5101)
    t_size = int(len(train_set) * train_val_split)
    v_size = len(train_set) - t_size
    t_sub, v_sub = torch.utils.data.random_split(train_set, [t_size, v_size], generator=gen)
    return (DataLoader(LabDataset(t_sub), batch_size=batch_size, shuffle=True),
            DataLoader(LabDataset(v_sub), batch_size=batch_size, shuffle=False),
            DataLoader(LabDataset(test_set), batch_size=batch_size, shuffle=False))

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cuda"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val = float('inf')
    for ep in range(epochs):
        model.train(); t_loss = 0.0
        for L, ab in train_loader:
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            loss = criterion(model(L), ab)
            loss.backward(); optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader); train_losses.append(t_loss)
        model.eval(); v_loss = 0.0
        with torch.no_grad():
            for L, ab in val_loader:
                L, ab = L.to(device), ab.to(device)
                v_loss += criterion(model(L), ab).item()
        v_loss /= len(val_loader); val_losses.append(v_loss)
        print(f"Epoch {ep+1:2d}/{epochs} | Train Loss: {t_loss:.5f} | Val Loss: {v_loss:.5f}")
        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  -> Saved new best model")
    return train_losses, val_losses

def visualize_predictions(model, val_loader, device='cuda', num_samples=5):
    if device == "cuda" and not torch.cuda.is_available():
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()
    samples_so_far = 0
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    with torch.no_grad():
        for L_batch, ab_batch in val_loader:
            L_batch = L_batch.to(device)
            ab_pred = model(L_batch)
            lab_pred = torch.cat([L_batch.cpu(), ab_pred.cpu()], dim=1)
            lab_gt = torch.cat([L_batch.cpu(), ab_batch], dim=1)
            rgb_pred = lab_to_rgb_tensor(lab_pred)
            rgb_gt = lab_to_rgb_tensor(lab_gt)
            gray_input = L_batch.repeat(1,3,1,1).cpu()
            for i in range(min(len(L_batch), num_samples - samples_so_far)):
                idx = samples_so_far + i
                axes[idx, 0].imshow(gray_input[i].permute(1,2,0).numpy(), cmap='gray')
                axes[idx, 0].set_title("Grayscale Input")
                axes[idx, 0].axis('off')
                axes[idx, 1].imshow(rgb_gt[i].permute(1,2,0).numpy())
                axes[idx, 1].set_title("Ground Truth")
                axes[idx, 1].axis('off')
                axes[idx, 2].imshow(rgb_pred[i].permute(1,2,0).numpy().clip(0,1))
                axes[idx, 2].set_title("Predicted Colorization")
                axes[idx, 2].axis('off')
            samples_so_far += len(L_batch)
            if samples_so_far >= num_samples:
                break
    plt.tight_layout()
    plt.show()

def compute_metrics(model, test_loader, device='cuda'):
    model.eval(); model.to(device)
    psnrs, ssims = [], []
    with torch.no_grad():
        for L, ab in test_loader:
            L = L.to(device)
            pred = lab_to_rgb_tensor(torch.cat([L.cpu(), model(L).cpu()], dim=1))
            gt = lab_to_rgb_tensor(torch.cat([L.cpu(), ab], dim=1))
            for i in range(len(L)):
                p = pred[i].permute(1,2,0).numpy().clip(0,1)
                g = gt[i].permute(1,2,0).numpy().clip(0,1)
                psnrs.append(peak_signal_noise_ratio(g, p, data_range=1.0))
                ssims.append(structural_similarity(g, p, data_range=1.0, channel_axis=2))
    return np.mean(psnrs), np.mean(ssims)
