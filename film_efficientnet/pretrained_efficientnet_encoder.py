"""Encoder based on Efficientnet."""


# film_efficientnet_encoder receive [bs, 3, 300, 300] and returns [bs, 1536, 10, 10]
# Here we use 1x1 conv. [bs, 1536, 10, 10] -> [bs, 512, 10, 10]
# then apply FiLM.

import torch
import torch.nn as nn
from typing import Optional

from pytorch_robotics_transformer.film_efficientnet.film_efficientnet_encoder import EfficientNetB3
from pytorch_robotics_transformer.film_efficientnet.film_conditioning_layer import FilmConditioning

class EfficientNetEncoder(nn.Module):
    def __init__(self, 
                 weights: Optional[str] = 'imagenet',
                 early_film: bool = True,
                 include_top: bool = False,
                 pooling: bool = True):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels=1536, # If we use EfficientNetB3 and input image has 3 channels, in_channels is 1536.
                                 out_channels=512,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False
                                 )
        self.net = EfficientNetB3(weights=weights, include_top=include_top, include_film=early_film)
        self.film_layer = FilmConditioning(num_channels=512, text_vector_size=512)

        self.early_film = early_film
        self._pooling = pooling

    def _encode(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run the image through the efficientnet encoder."""
        if self.early_film:
            return self.net(image, context)
        return self.net(image)


    def forward(self, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self._encode(image, context)
        features = self.conv1x1(features)
        features = self.film_layer(features, context)

        if not self._pooling:
            return features

        return torch.mean(features, dim=(2,3))