import torch
import torch.nn as nn
import os
from pathlib import Path
import time

def export_to_onnx(model, save_path, input_shape=(1, 1, 64, 157)):
    """
    Export PyTorch model to ONNX format for deployment.
    """
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    # Export
    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"[INFO] Model exported to ONNX: {save_path}")
    return save_path

def auto_compress_pipeline(model, save_dir):
    """
    Automated pipeline to generate multiple deployment formats.
    1. FP32 (Original)
    2. INT8 (Quantized)
    3. ONNX (Interoperable)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Starting Model Compression Pipeline ---")
    
    # 1. Save FP32
    fp32_path = save_dir / "model_fp32.pth"
    torch.save(model.state_dict(), fp32_path)
    size_fp32 = os.path.getsize(fp32_path) / 1024 / 1024
    print(f"[1] Saved FP32 Model ({size_fp32:.2f} MB)")
    
    # 2. INT8 Quantization (Dynamic)
    print("[2] Applying Dynamic Quantization (INT8)...")
    model_int8 = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    int8_path = save_dir / "model_int8.pth"
    torch.save(model_int8.state_dict(), int8_path)
    size_int8 = os.path.getsize(int8_path) / 1024 / 1024
    print(f"    Saved INT8 Model ({size_int8:.2f} MB)")
    print(f"    Compression Ratio: {size_fp32/size_int8:.2f}x")
    
    # 3. Export ONNX (from FP32)
    print("[3] Exporting to ONNX...")
    onnx_path = save_dir / "model.onnx"
    try:
        export_to_onnx(model, onnx_path)
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")
        
    print("--- Compression Pipeline Complete ---\n")
    return fp32_path, int8_path, onnx_path

def evaluate_quality_loss(model_fp32, model_int8, test_loader, device='cpu'):
    """
    Compare accuracy of FP32 vs INT8 models to measure quality loss.
    """
    print("Evaluating Quantization Quality Loss...")
    
    def eval_model(m, name):
        m.eval()
        m.to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = m(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"  {name} Accuracy: {acc:.2f}%")
        return acc

    acc_fp32 = eval_model(model_fp32, "FP32")
    
    # INT8 model usually runs on CPU for PyTorch Dynamic Quantization
    acc_int8 = eval_model(model_int8.cpu(), "INT8") 
    
    loss = acc_fp32 - acc_int8
    print(f"Quality Loss: {loss:.2f}% (acceptable if < 1.0%)")
    return loss

