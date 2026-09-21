import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import random
from models.mobilenet import get_model
from data.dataset import FedCoughDataset, SpecAugment

class Client:
    def __init__(self, client_id, data_dir, device='cpu', 
                 batch_size=16, local_epochs=3, learning_rate=0.01,
                 dp_epsilon=0.0):
        self.client_id = client_id
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.dp_epsilon = dp_epsilon # Differential Privacy Budget (Target Epsilon per round)
        
        # Privacy Budget Management (New Requirement)
        self.privacy_budget_spent = 0.0
        self.max_privacy_budget = 10.0 # Example limit, stops participating after this
        
        # Hardware Simulation
        self.battery = random.randint(20, 100) # Battery level 0-100
        self.compute_power = random.choice(['high', 'mid', 'low'])
        
        # Load Data
        # Use SpecAugment for training to improve robustness
        self.dataset = FedCoughDataset(client_id, data_dir, transform=SpecAugment())
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize local model (structure only)
        self.model = get_model(num_classes=3, device=device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, global_weights_float):
        """
        Local training loop.
        Args:
            global_weights_float: State dict of the global model (already float32).
        Returns:
            local_weights_float: State dict after training.
            train_loss: Average loss.
            num_samples: Number of samples used.
        """
        # If battery is low or compute is low, reduce epochs
        # Privacy Check: If budget exhausted, refuse to train
        if self.privacy_budget_spent >= self.max_privacy_budget and self.dp_epsilon > 0:
            print(f"[Client {self.client_id}] Privacy Budget Exhausted! ({self.privacy_budget_spent:.2f} >= {self.max_privacy_budget})")
            return global_weights_float, 0.0, 0 # Retire

        actual_epochs = self.local_epochs
        if self.battery < 30 or self.compute_power == 'low':
            actual_epochs = max(1, self.local_epochs // 2)
            
        # 1. Load global weights to model (CPU first)
        self.model.load_state_dict(global_weights_float)
        
        # 2. Move to GPU for training (Resource Acquired)
        self.model.to(self.device)
        self.model.train()
        
        # 3. Handle Class Imbalance (Simple Weighted Loss)
        # In FL, ideally we calculate this per-client or use global weights.
        # For simplicity/stability, we use a fixed weight similar to baseline.
        # [0.4, 2.0, 5.6] derived from baseline
        class_weights = torch.tensor([0.4, 2.0, 5.6]).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        try:
            for epoch in range(actual_epochs):
                batch_loss = 0.0
                for images, labels in self.dataloader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss += loss.item() * images.size(0)
                    
                    # Acc calc
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                epoch_loss += batch_loss
                
            avg_loss = epoch_loss / total if total > 0 else 0.0
            
            # Differential Privacy: Add noise to weights
            # Note: We do this on GPU or CPU? Let's do on CPU to save GPU time.
            self.model.cpu() # Release GPU immediately
            local_weights = copy.deepcopy(self.model.state_dict())
            
            if self.dp_epsilon > 0:
                self._add_dp_noise(local_weights)
                self.privacy_budget_spent += self.dp_epsilon
            
            # [Secure Aggregation] Server-aided Masking
            # Generate a secret seed and mask
            mask_seed = random.randint(0, 999999999)
            self._apply_mask(local_weights, mask_seed)
            
            # Return masked weights AND the seed (simulated encrypted upload)
            return local_weights, mask_seed, avg_loss, total
            
        except Exception as e:
            print(f"Client {self.client_id} Training Error: {e}")
            self.model.cpu() # Ensure cleanup
            return global_weights_float, 0, 0.0, 0 # Return original on failure
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _apply_mask(self, state_dict, seed):
        """Apply additive mask based on seed"""
        # Use a deterministic generator based on seed
        g = torch.Generator()
        g.manual_seed(seed)
        
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                # Generate mask of same shape
                mask = torch.randn(v.size(), generator=g)
                # Add mask: W_masked = W + Mask
                state_dict[k] += mask
    
    def _add_dp_noise(self, state_dict):
        """
        Simple Gaussian Noise mechanism for DP.
        Noise scale is inversely proportional to epsilon.
        """
        noise_scale = 0.01 / (self.dp_epsilon + 1e-6) 
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                noise = torch.randn_like(v) * noise_scale
                state_dict[k] += noise
                
    def get_test_loader(self):
        """Returns a dataloader for evaluation (no augmentation)"""
        test_ds = FedCoughDataset(self.client_id, self.dataset.data_dir, transform=None)
        return DataLoader(test_ds, batch_size=16, shuffle=False)
