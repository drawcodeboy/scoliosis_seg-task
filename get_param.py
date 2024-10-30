from models.nets import load_model

if __name__ == '__main__':
    for i in range(0, 6):
        model = load_model(scale=f'b{i}')
        
        p_num = 0
        for name, p in model.named_parameters():
            p_num += p.numel()
            
        print(f"SegFormer-b{i}: {round(p_num/1000000, 1)}M")