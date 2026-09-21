import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from data.dataset import FedCoughDataset, SpecAugment
from models.mobilenet import get_model
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    # Config
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.001
    DATA_DIR = Path("FedCough_Project/data")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Helper function definition
    def print_class_dist(dataset, name="Dataset"):
        from collections import Counter
        labels = []
        for _, label in tqdm(dataset, desc=f"Scanning {name}"):
            labels.append(label.item())
        counts = Counter(labels)
        total = sum(counts.values())
        print(f"\n[{name}] Distribution:")
        for k, v in sorted(counts.items()):
            print(f"  Class {k}: {v} ({v/total*100:.2f}%)")

    # 1. Load All Data (Split by Clients)
    print("Loading data from all clients (User-based Split)...")
    # We just need to find all client folders
    client_dirs = sorted(list((DATA_DIR / "clients").glob("client_*")))
    
    # Shuffle clients to ensure random split
    import random
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(client_dirs)
    
    # Split clients: 80% Train, 20% Test
    split_idx = int(0.8 * len(client_dirs))
    train_clients = client_dirs[:split_idx]
    test_clients = client_dirs[split_idx:]
    
    print(f"Total Clients: {len(client_dirs)}")
    print(f"Train Clients: {len(train_clients)} | Test Clients: {len(test_clients)}")

    # Create Train Datasets (With Augmentation)
    train_datasets = []
    for client_dir in tqdm(train_clients, desc="Loading Train Data"):
        client_id = client_dir.name
        ds = FedCoughDataset(client_id, DATA_DIR, transform=SpecAugment())
        train_datasets.append(ds)
        
    # Create Test Datasets (No Augmentation - Clean Eval)
    test_datasets = []
    for client_dir in tqdm(test_clients, desc="Loading Test Data"):
        client_id = client_dir.name
        ds = FedCoughDataset(client_id, DATA_DIR, transform=None)
        test_datasets.append(ds)
        
    train_ds = ConcatDataset(train_datasets)
    test_ds = ConcatDataset(test_datasets)
    
    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    
    # 2. Dataloaders
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 3. Model
    # Diagnostic: Check Class Distribution & Compute Weights
    def get_class_weights(dataset):
        labels = []
        for _, label in tqdm(dataset, desc="Computing Weights"):
            labels.append(label.item())
        
        from collections import Counter
        counts = Counter(labels)
        n_samples = len(labels)
        n_classes = 3
        
        # Formula: N / (n_classes * count)
        weights = torch.zeros(n_classes)
        for i in range(n_classes):
            count = counts.get(i, 0)
            if count > 0:
                weights[i] = n_samples / (n_classes * count)
            else:
                weights[i] = 1.0
                
        print(f"\nClass Weights: {weights}")
        return weights.to(DEVICE)

    class_weights = get_class_weights(train_ds)
    
    print_class_dist(train_ds, "Train Set")
    print_class_dist(test_ds, "Test Set")

    model = get_model(num_classes=3).to(DEVICE)
    # Use Weighted Loss to handle imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Train Loop
    print("Starting Centralized Training...")
    history = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        avg_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        # Advanced metrics accumulators
        y_true_epoch = []
        y_pred_epoch = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true_epoch.extend(labels.cpu().numpy())
                y_pred_epoch.extend(predicted.cpu().numpy())
        
        acc = 100 * correct / total
        history['test_acc'].append(acc)
        
        # Calculate per-class recall for COVID (Class 2)
        from sklearn.metrics import confusion_matrix, f1_score
        try:
            cm = confusion_matrix(y_true_epoch, y_pred_epoch)
            # Check if class 2 exists in data
            unique_labels = sorted(list(set(y_true_epoch) | set(y_pred_epoch)))
            recall_covid = 0.0
            if 2 in unique_labels:
                covid_idx = unique_labels.index(2)
                if cm.shape[0] > covid_idx:
                    row_sum = cm[covid_idx].sum()
                    if row_sum > 0:
                        recall_covid = cm[covid_idx][covid_idx] / row_sum
            
            f1 = f1_score(y_true_epoch, y_pred_epoch, average='weighted')
        except:
            recall_covid = 0.0
            f1 = 0.0
            
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%, F1: {f1:.4f}, COVID Recall: {recall_covid:.4f}")
        
    print("Training Complete.")
    
    # 5. Detailed Evaluation (Confusion Matrix)
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    import numpy as np
    
    print("\nGenerating Detailed Report...")
    y_true = []
    y_pred = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final Eval"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    print("\nClassification Report:")
    # Fix: Updated for binary classification (Healthy vs COVID)
    # Note: Labels are 0 and 2. We need to tell sklearn to only look for these or map them.
    # But since model output is 3 classes, predicted might contain 1 (unlikely).
    # Safest way: map labels or just list the target names corresponding to present labels.
    
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    target_names = []
    if 0 in unique_labels: target_names.append('Healthy')
    if 1 in unique_labels: target_names.append('Symptomatic')
    if 2 in unique_labels: target_names.append('COVID-19')
    
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")
    
    # 6. Plot History
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.subplot(1, 2, 2)
    plt.plot(history['test_acc'])
    plt.title('Test Accuracy')
    plt.savefig('baseline_results.png')
    print("Results saved to baseline_results.png")

if __name__ == "__main__":
    main()

