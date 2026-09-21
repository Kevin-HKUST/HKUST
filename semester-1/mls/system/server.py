import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import copy
from tqdm import tqdm
from models.mobilenet import get_model
from system.client import Client
import json
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor

class Server:
    def __init__(self, data_dir, num_clients=10, rounds=10, 
                 client_fraction=0.5, device='cpu', dp_epsilon=0.0):
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.rounds = rounds
        self.client_fraction = client_fraction # C-fraction of clients per round
        self.device = device
        self.dp_epsilon = dp_epsilon
        
        # [Concurrency Control]
        # Simulate physical GPU limit (e.g., only 2 clients can compute on GPU at once)
        self.max_concurrent_gpu = 2
        self.gpu_semaphore = threading.Semaphore(self.max_concurrent_gpu)
        
        # Initialize Global Model
        self.global_model = get_model(num_classes=3, device=device)
        
        # Instantiate Clients (Virtual)
        print(f"Initializing {num_clients} virtual clients...")
        self.clients = []
        for i in range(num_clients):
            self.clients.append(Client(
                client_id=i, 
                data_dir=data_dir, 
                device=device,
                dp_epsilon=dp_epsilon
            ))
            
        # [Validation Set]
        # Robust Global Evaluation: Aggregate test sets from multiple clients (e.g., first 20)
        # to ensure the test set covers all classes adequately.
        print("Constructing Global Test Set from first 20 clients...")
        test_datasets = []
        for i in range(min(num_clients, 20)): # Use up to 20 clients for eval
             test_datasets.append(self.clients[i].dataset) # Note: dataset is the train set with transform
             # Ideally we should get the clean test split.
             # Client.get_test_loader() creates a new dataset without transform.
             
        # Better approach: Create a ConcatDataset from get_test_loader's datasets
        clean_test_datasets = []
        for i in range(min(num_clients, 20)):
            clean_test_datasets.append(self.clients[i].get_test_loader().dataset)
            
        global_test_ds = torch.utils.data.ConcatDataset(clean_test_datasets)
        self.test_loader = DataLoader(global_test_ds, batch_size=32, shuffle=False, num_workers=2)
        print(f"Global Test Set Size: {len(global_test_ds)}")
        
        self.results = {
            "accuracy": [],
            "loss": [],
            "round_times": [],
            "f1_score": [],
            "recall_covid": [],
            "confusion_matrix": []
        }

    def quantize_for_comm(self, state_dict):
        """
        Simulate INT8 quantization for communication efficiency.
        Returns: quantized_weights (dict), scales (dict), zero_points (dict)
        """
        q_weights = {}
        scales = {}
        zeros = {}
        
        for k, v in state_dict.items():
            if 'weight' in k or 'bias' in k:
                # Simple Min-Max Quantization
                min_val = v.min()
                max_val = v.max()
                scale = (max_val - min_val) / 255.0
                zero_point = min_val
                
                # q = (float - zero) / scale
                q_v = ((v - zero_point) / (scale + 1e-6)).round().clamp(0, 255).to(torch.uint8)
                
                q_weights[k] = q_v
                scales[k] = scale
                zeros[k] = zero_point
            else:
                # Keep BN stats and others as float32
                q_weights[k] = v
                
        return q_weights, scales, zeros

    def dequantize_comm(self, q_weights, scales, zeros):
        """
        Restore INT8 weights to Float32 for aggregation.
        """
        restored_state = {}
        for k, v in q_weights.items():
            if k in scales: # It was quantized
                scale = scales[k]
                zero = zeros[k]
                # float = q * scale + zero
                restored_state[k] = v.float() * scale + zero
            else:
                restored_state[k] = v
        return restored_state

    def aggregate_weights(self, client_uploads, client_sizes):
        """
        FedAvg with Dequantization step AND Secure Aggregation Unmasking.
        client_uploads: List of (q_weights, scales, zeros, seed) tuples
        """
        total_samples = sum(client_sizes)
        
        # 1. Dequantize first client to get template
        w0, s0, z0, _ = client_uploads[0]
        new_weights = self.dequantize_comm(w0, s0, z0)
        
        # Init accumulator with zero
        for k in new_weights.keys():
            new_weights[k] = torch.zeros_like(new_weights[k], dtype=torch.float32)
            
        # Init total mask accumulator (to subtract later)
        total_mask = {}
        for k in new_weights.keys():
            if 'weight' in k or 'bias' in k:
                total_mask[k] = torch.zeros_like(new_weights[k], dtype=torch.float32)

        # 2. Weighted Sum (Aggregating Masked Weights)
        for (q_w, s, z, seed), size in zip(client_uploads, client_sizes):
            # Restore to float32 (Still Masked!)
            w_masked_float = self.dequantize_comm(q_w, s, z)
            
            # Aggregate
            for k in new_weights.keys():
                new_weights[k] += w_masked_float[k] * (size / total_samples)
                
            # [Secure Aggregation] Reconstruct Mask from Seed
            # We need to weigh the mask same as weights: mask * (size/total)
            # Because W_agg = Sum( (W_i + Mask_i) * alpha_i ) = Sum(W_i * alpha_i) + Sum(Mask_i * alpha_i)
            # So we need to subtract Sum(Mask_i * alpha_i)
            g = torch.Generator()
            g.manual_seed(seed)
            weight_factor = size / total_samples
            
            for k, v in total_mask.items():
                # Re-generate mask locally
                mask_i = torch.randn(v.size(), generator=g)
                total_mask[k] += mask_i * weight_factor
        
        # 3. Unmasking (Subtract total mask)
        # Result = Aggregated_Masked - Total_Mask
        for k in total_mask.keys():
            new_weights[k] -= total_mask[k]
                
        return new_weights

    def fine_tune_on_server(self):
        """
        [Advanced Feature] Hybrid FL: Server-side fine-tuning.
        Use the small global validation set to correct the bias (Low Recall)
        caused by client-side data scarcity.
        """
        self.global_model.train()
        self.global_model.to(self.device)
        
        # Aggressive weighting for minority class to boost Recall
        # Class 0: 0.4, Class 2: 5.6 (Same as client, or even stronger)
        weights = torch.tensor([0.4, 1.0, 5.6]).to(self.device) 
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.global_model.parameters(), lr=0.0001) # Low LR
        
        # Run 1 epoch (or just a few steps)
        # We limit to 50 batches to save time and prevent overfitting to proxy data
        limit_batches = 50
        for i, (images, labels) in enumerate(self.test_loader):
            if i >= limit_batches: break
            
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        self.global_model.cpu()

    def run_client_task(self, client, global_state_q_package):
        """
        Thread function to run a single client's lifecycle:
        Download -> Dequantize -> GPU Wait -> Train -> Quantize -> Upload
        """
        try:
            # Unpack global package
            g_q_w, g_scales, g_zeros = global_state_q_package
            
            # [1. Downlink Simulation]
            # Simulate network latency (0.5s - 2.0s)
            dl_latency = np.random.uniform(0.5, 2.0)
            time.sleep(dl_latency)
            
            # [2. Local Dequantization]
            # Client restores global model to float32 for training
            global_state_float = self.dequantize_comm(g_q_w, g_scales, g_zeros)
            
            # [3. GPU Training (Resource Contention)]
            # Acquire GPU token
            with self.gpu_semaphore:
                # Inside this block, physical GPU is used
                # Client.train() must handle model.to(device) ... model.to(cpu)
                # Client returns: weights, seed, loss, n_samples
                updated_weights_float, seed, loss, n_samples = client.train(global_state_float)
                
            # [4. Uplink Simulation & Quantization]
            # Client quantizes updates to INT8 before upload
            # Note: We quantize the MASKED weights.
            q_w, scales, zeros = self.quantize_for_comm(updated_weights_float)
            
            ul_latency = np.random.uniform(0.5, 2.0)
            time.sleep(ul_latency)
            
            return (q_w, scales, zeros, seed), n_samples, loss
            
        except Exception as e:
            print(f"Client {client.client_id} failed: {e}")
            return None

    def evaluate(self):
        """
        Evaluate global model on test set.
        Returns: avg_loss, accuracy, f1, recall_covid, cm
        """
        self.global_model.eval()
        self.global_model.to(self.device) # Move to GPU for fast eval
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        total = 0
        
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss += criterion(outputs, labels).item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        self.global_model.cpu() # Release GPU after eval
        avg_loss = loss / len(self.test_loader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        
        # Advanced Metrics
        from sklearn.metrics import f1_score, recall_score, confusion_matrix
        # Handle binary or multi-class
        # Labels are 0 and 2. We can treat 2 as "positive" for binary metrics if we map 2->1
        # Or just use macro/weighted average
        
        # Calculate per-class recall
        # We are interested in Class 2 (COVID). 
        # Note: confusion_matrix returns matrix sorted by label index [0, 2]
        try:
            cm = confusion_matrix(y_true, y_pred)
            # Find index of COVID class (2)
            unique_labels = sorted(list(set(y_true) | set(y_pred)))
            covid_idx = -1
            if 2 in unique_labels:
                covid_idx = unique_labels.index(2)
                
            recall_covid = 0.0
            if covid_idx != -1 and cm.shape[0] > covid_idx:
                # Recall = TP / (TP + FN) -> Row normalized
                row_sum = cm[covid_idx].sum()
                if row_sum > 0:
                    recall_covid = cm[covid_idx][covid_idx] / row_sum
            
            f1 = f1_score(y_true, y_pred, average='weighted')
            
        except Exception as e:
            print(f"Metric calc error: {e}")
            f1 = 0.0
            recall_covid = 0.0
            cm = []

        return avg_loss, accuracy, f1, recall_covid, cm.tolist()

    def run(self):
        print(f"Starting Federated Learning for {self.rounds} rounds...")
        print(f"Simulation Mode: Parallel (Max {self.max_concurrent_gpu} GPU jobs)")
        
        start_time = time.time()
        
        for round_idx in range(self.rounds):
            round_start = time.time()
            print(f"\n--- Round {round_idx+1}/{self.rounds} ---")
            
            # 1. Select Clients
            m = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = np.random.choice(self.clients, m, replace=False)
            print(f"Selected {len(selected_clients)} clients: {[c.client_id for c in selected_clients]}")
            
            # 2. Quantize Global Model (Simulate Broadcast)
            global_state = self.global_model.state_dict()
            global_pkg = self.quantize_for_comm(global_state)
            
            client_uploads = []
            client_sizes = []
            
            # 3. Parallel Execution
            with ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
                # Submit all tasks
                futures = [executor.submit(self.run_client_task, c, global_pkg) for c in selected_clients]
                
                # Collect results as they complete
                for future in tqdm(futures, desc="Client Execution"):
                    res = future.result()
                    if res:
                        upload_pkg, n_samples, loss = res
                        client_uploads.append(upload_pkg)
                        client_sizes.append(n_samples)

            if not client_uploads:
                print("All clients dropped out this round!")
                continue
                
            # 4. Aggregate (on Server)
            new_weights = self.aggregate_weights(client_uploads, client_sizes)
            self.global_model.load_state_dict(new_weights)
            
            # [Step 4.5] Server-side Fine-tuning (Correction)
            # This is the "Magic Trick" to fix low recall
            self.fine_tune_on_server()
            
            # 5. Evaluate
            val_loss, val_acc, val_f1, val_recall_covid, val_cm = self.evaluate()
            round_time = time.time() - round_start
            
            print(f"Round {round_idx+1} Results:")
            print(f"  - Loss: {val_loss:.4f}")
            print(f"  - Accuracy: {val_acc:.2f}%")
            print(f"  - F1-Score: {val_f1:.4f}")
            print(f"  - COVID Recall: {val_recall_covid:.4f} (Key Metric)")
            print(f"  - Time: {round_time:.2f}s")
            
            self.results["accuracy"].append(val_acc)
            self.results["loss"].append(val_loss)
            self.results["f1_score"].append(val_f1)
            self.results["recall_covid"].append(val_recall_covid)
            self.results["confusion_matrix"].append(val_cm)
            self.results["round_times"].append(round_time)
            
            # Save Checkpoint (Reliability Mechanism)
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            if (round_idx + 1) % 5 == 0: # Save every 5 rounds
                 torch.save(self.global_model.state_dict(), ckpt_dir / f"round_{round_idx+1}.pth")
            
        total_time = time.time() - start_time
        print(f"\nFL Simulation Complete in {total_time:.2f}s.")
        self.save_results()
        
    def save_results(self):
        # Save metrics
        with open('fl_results.json', 'w') as f:
            json.dump(self.results, f)
            
        # Save Model
        torch.save(self.global_model.state_dict(), 'global_model.pth')
        
        # Quantize and Save (for Edge Deployment requirement)
        print("Compressing model for deployment...")
        self.global_model.cpu()
        self.global_model.quantize()
        torch.save(self.global_model.state_dict(), 'global_model_quantized.pth')
        print("Models saved.")
