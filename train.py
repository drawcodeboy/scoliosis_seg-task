from models import *
from models.nets import SegFormer

from torchsummary import summary

model = SegFormer(num_classes=1, phi='b5')

print(model)
summary(model, (1, 640, 640))