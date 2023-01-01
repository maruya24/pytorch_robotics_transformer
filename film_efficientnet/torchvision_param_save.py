from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchinfo import summary
import torch.nn as nn
import torch

net = efficientnet_b3(weights= EfficientNet_B3_Weights.DEFAULT)

torch.save(net.state_dict(), "./film_efficientnet/efficientnet_checkpoints/efficientnetb3.pth")

net_notop = nn.Sequential(*(list(net.children())[:-2]))

torch.save(net_notop.state_dict(), "./film_efficientnet/efficientnet_checkpoints/efficientnetb3_notop.pth")