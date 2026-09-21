import torch
import torch.nn as nn
import torchvision.models as models

class FedCoughMobileNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(FedCoughMobileNet, self).__init__()
        
        # Load MobileNetV2
        # 'weights' parameter replaces 'pretrained' in newer torchvision versions
        try:
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            self.model = models.mobilenet_v2(weights=weights)
        except AttributeError:
            # Fallback for older torchvision
            self.model = models.mobilenet_v2(pretrained=pretrained)
            
        # Modify the classifier
        # MobileNetV2 classifier is:
        # (1): Linear(in_features=1280, out_features=1000, bias=True)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # x shape: [Batch, 1, H, W] (Mel Spectrogram)
        # MobileNet expects 3 channels.
        # We repeat the single channel 3 times to make it RGB-like.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        return self.model(x)

    def quantize(self):
        """
        Apply dynamic quantization to the model (Post-Training Quantization).
        This reduces model size for edge deployment.
        """
        self.model = torch.quantization.quantize_dynamic(
            self.model, {nn.Linear}, dtype=torch.qint8
        )
        return self

def get_model(num_classes=3, device='cpu'):
    model = FedCoughMobileNet(num_classes=num_classes)
    return model.to(device)

if __name__ == "__main__":
    # Test the model
    model = get_model()
    print("Model Architecture:")
    # print(model)
    
    # Create dummy input (Batch=1, Channel=1, H=64, W=157)
    # 5 seconds of audio @ 16kHz with hop=512 -> ~157 frames
    dummy_input = torch.randn(1, 1, 64, 157)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [1, 3]
    
    # Test Quantization
    print("\nTesting Quantization...")
    model.quantize()
    output_q = model(dummy_input)
    print("Quantization successful.")

