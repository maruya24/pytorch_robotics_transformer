import unittest
import torch
from torchinfo import summary
from skimage import data
from absl.testing import parameterized
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from pytorch_robotics_transformer.film_efficientnet.film_efficientnet_encoder import EfficientNetB3, ILSVRCPredictor

resize = 300
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# This transformation is for imagenet. Use another transformatioin for RT-1
img_trasnform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

class FilmEfficientnetTest(parameterized.TestCase, unittest.TestCase):
    # This test method will be implemented twice. One is implemented with include_film = True. Another is implemented with include_film = False.
    # To know further more, see a comment of https://github.com/abseil/abseil-py/blob/main/absl/testing/parameterized.py
    @parameterized.parameters([True, False])
    def test_film_efficientnet(self, include_film):
        fe =  EfficientNetB3(include_top=True, weights='imagenet', include_film=include_film)
        fe.eval()

        image = data.chelsea() # ndarray
        image_transformed = img_trasnform(image) # tensor

        # Show the image.
        # image_show = image_transformed.numpy().transpose((1, 2, 0))
        # image_show = np.clip(image_show, 0, 1)
        # plt.imshow(image_show)
        # plt.show()

        image_transformed = image_transformed.unsqueeze(0)
        context = torch.rand(1, 512)

        if include_film:
            eff_output = fe(image_transformed, context)
        else:
            eff_output = fe(image_transformed)
        
        predictor = ILSVRCPredictor(top=10)

        film_preds = predictor.predict_topk(eff_output)
        self.assertIn('tabby', film_preds)

if __name__ == '__main__':
    unittest.main()