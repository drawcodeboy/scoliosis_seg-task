from .SegFormer.segformer import SegFormer

def load_model(model_name='segformer', scale='b0', num_classes=1):
    if model_name == 'segformer':
        return SegFormer(phi=scale, num_classes=num_classes)