from models.nets import load_model

import torch

if __name__ == '__main__':
    model = load_model() #SegFormer-B0
    
    tensor = torch.randn((2, 3, 224, 224)) # [B, C, H, W]
    print(f"Input shape: {tensor.shape}")
    
    output = model(tensor)
    print(f"Output shape: {output.shape}")