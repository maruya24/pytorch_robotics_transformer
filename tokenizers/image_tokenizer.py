"""A FiLM Efficientnet contextual image tokenizer used in Robotics Transformer 1.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_robotics_transformer.film_efficientnet.pretrained_efficientnet_encoder import EfficientNetEncoder
from pytorch_robotics_transformer.tokenizers.token_learner import TokenLearnerModule

class RT1ImageTokenizer(nn.Module):
    def __init__(self,
               embedding_output_dim: int = 512,
               use_token_learner: bool = False,
               num_tokens: int = 8):
        super().__init__()
        self._tokenizer = EfficientNetEncoder(token_embedding_size=embedding_output_dim, early_film=True, pooling=False)

        self._use_token_learner = use_token_learner
        if self._use_token_learner:
            self._num_tokens = num_tokens
            self._token_learner = TokenLearnerModule(inputs_channels=512 , num_tokens=self._num_tokens)

    @property
    def tokens_per_context_image(self) -> int:
        if self._use_token_learner:
            num_tokens = self._num_tokens
        else:
            num_tokens = 100
        return num_tokens
            
    def forward(self, image: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Gets image tokens.

        Args:
        image: Images of shape (b, t, 3, h, w) to tokenize.
        context: An optional context vector (e.g., a natural language embedding).
            Expected to have shape (b, t, embedding_dim).
        training: Whether or not we are in training mode.

        Returns:
        tokens: has shape (batch, t, num_tokens_per_timestep, embedding_dim)
        """
        b, t, c , h, w = image.shape

        # Fold the time axis into the batch axis.
        image = image.view(b * t, c, h, w)
        if context is not None:
            context = context.view(b * t, -1)

        tokens = self._tokenizer(image, context=context) # [b * t, 512 , 10, 10]

        if self._use_token_learner:
            tokens = self._token_learner(tokens) # [b * t, num_token, 512]
            # Unflatten the time axis, which was previously flattened into the batch.
            tokens = tokens.view(b, t, tokens.shape[1], -1)
            return tokens # [b, t, num_token, 512]
        else:
            # Unflatten the time axis, which was previously flattened into the batch.
            tokens = tokens.view(b, t, 512, -1) # [b, t, 512 , 10 * 10]
            # If you don't use token learner, the number of token is 100.
            tokens = tokens.transpose(2, 3) # [b, t, 10 * 10, 512]
            return tokens