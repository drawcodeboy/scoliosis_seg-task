import torch
from torchsummary import summary

from models import *
from models.nets import SegFormer

model = SegFormer(num_classes=1, phi='b0')

print(model)
# summary(model, (1, 640, 640))

input_tensor = torch.randn(2, 1, 640, 640)
output_tensor = model(input_tensor)
print(output_tensor.shape)