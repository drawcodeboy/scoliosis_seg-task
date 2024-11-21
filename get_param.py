from models.nets import load_model
import argparse
import warnings

import torch
import torchprofile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--scale")
    
    args = parser.parse_args()
    
    model = load_model(model_name=args.model, scale=args.scale).to('cuda')
    
    p_num = 0
    for name, p in model.named_parameters():
        p_num += p.numel()
        
    print(f"Params: {round(p_num/1000000, 1)}M")
    
    input_t = torch.randn(1, 1, 640, 640).to('cuda')
    warnings.filterwarnings('ignore')
    macs = torchprofile.profile_macs(model, input_t)
    warnings.filterwarnings(action='default')
    print(f"GFLOPS: {macs/1e9:.4f}G")