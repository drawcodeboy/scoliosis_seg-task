from models.nets import load_model
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model")
    parser.add_argument("--scale")
    
    args = parser.parse_args()
    
    model = load_model(model_name=args.model, scale=args.scale)
    
    p_num = 0
    for name, p in model.named_parameters():
        p_num += p.numel()
        
    print(f"{round(p_num/1000000, 1)}M")