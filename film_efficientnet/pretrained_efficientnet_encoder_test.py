from pytorch_robotics_transformer.film_efficientnet import film_efficientnet_encoder
from pytorch_robotics_transformer.film_efficientnet import pretrained_efficientnet_encoder as eff
import unittest
import numpy as np
from skimage import data
import torch
# import film_efficientnet_encoder
# import pretrained_efficientnet_encoder as eff
from torchvision import transforms

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


class PretrainedEfficientnetEncoderTest(unittest.TestCase):
    def test_encoding(self):
        """Test that we get a correctly shaped decoding."""
        # prepare context
        state = np.random.RandomState(0)
        context = state.uniform(-1, 1, (10, 512))
        context = torch.from_numpy(context).to(torch.float32)

        # prepare image
        image = data.chelsea() # ndarray
        image_transformed = img_trasnform(image) # tensor
        image_transformed = torch.tile(image_transformed.unsqueeze(0), (10,1,1,1))

        token_embedding_size = 512
        model = eff.EfficientNetEncoder(token_embedding_size = token_embedding_size)
        model.eval()
        preds = model(image_transformed, context)
        self.assertEqual(preds.shape, (10, token_embedding_size))

    def test_imagenet_classification(self):
        """Test that we can correctly classify an image of a cat."""
        # prepare context
        state = np.random.RandomState(0)
        context = state.uniform(-1, 1, (10, 512))
        context = torch.from_numpy(context).to(torch.float32)

        # prepare image
        image = data.chelsea() # ndarray
        image_transformed = img_trasnform(image) # tensor
        image_transformed = torch.tile(image_transformed.unsqueeze(0), (10,1,1,1))
        
        # prediction
        model = eff.EfficientNetEncoder(include_top=True)
        model.eval()
        preds = model._encode(image_transformed, context)
        preds = preds[0:1]

        predictor = film_efficientnet_encoder.ILSVRCPredictor()
        predicted_names = predictor.predict_topk(preds)


        self.assertIn('tabby', predicted_names)


if __name__ == '__main__':
    unittest.main()