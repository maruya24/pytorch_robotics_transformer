# Robotics Transformer Network

from tokenizers import action_tokenizer
from tokenizers import image_tokenizer
import transformer

from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
from gym import spaces

class TransformerNetwork(nn.Module):
    """A transformer based actor network."""

    def __init__(
            self,
            input_tensor_space: spaces.Dict, # observatioin space like dict. keys are image, natural_language_embedding
            output_tensor_space: spaces.Dict, # action space like dict. keys are world_vector, rotation_delta, gripper_closedness_action, terminate_episode
            train_step_counter: int = 0, # ?
            vocab_size: int = 256, # Dimensionality of tokens from the output layer. This is also dimensionality of tokens from the input layer.
            token_embedding_size: int = 512, # RT1ImageTokenizerの引数のembedding_output_dim。どこにも使われていないからpytorchの方では消した。この値はEfficientNetの出力の後付け足した1x1convのチャネル数で決まる。現在は512. 1x1convはEfficientNetEncodeの実装にある。
            num_layers: int = 1,
            layer_size: int = 4096, # This corresponds to key_dim which is the size of each attention head for query, key and values.
            num_heads: int = 8,
            feed_forward_size: int = 512, # This corresponds to d_model which is embedding dimension of each token in transformer part.
            dropout_rate: float = 0.1,
            time_sequence_length: int = 1,
            crop_size: int = 236,
            # action_order: Optional[List[str]] = None,
            use_token_learner: Optional[bool] = True,
            return_attention_scores: bool = False):
        super().__init__()

        self._input_tensor_spec = input_tensor_space
        self._output_tensor_spec = output_tensor_space
        self._train_step_counter = train_step_counter
        self._actions = None
        self._returns = None
        self._vocab_size = vocab_size
        self._token_embedding_size = token_embedding_size
        self._time_sequence_length = time_sequence_length
        self._crop_size = crop_size

        # creatr transformer
        self._transformer = transformer.Transformer(
        num_layers=num_layers,
        layer_size=layer_size,
        num_heads=num_heads,
        feed_forward_size=feed_forward_size,
        dropout_rate=dropout_rate,
        vocab_size=self._vocab_size,
        return_attention_scores=return_attention_scores)

        # create tokenizers
        self._image_tokenizer = image_tokenizer.RT1ImageTokenizer(
            use_token_learner=use_token_learner,
            num_tokens=8)
        self._action_tokenizer = action_tokenizer.RT1ActionTokenizer(
            output_tensor_space, # action space
            vocab_size=self._vocab_size)

        # Get the number of tokens
        self._tokens_per_action = self._action_tokenizer.tokens_per_action
        self._tokens_per_context_image = self._image_tokenizer.tokens_per_context_image

        # generate loss mask and attention mask
        self._generate_masks()

    def _get_action_index_for_token(self, k):
        """Returns action associated with the token at given position `k`.

        If k is not an action token then it returns -1.
        If k is part of the first action in the sequence then returns 0 etc.

        Args:
            k: an int that represents the position in the sequence.

        Returns:
            The index of the action that this position belongs to, or if this
            position is part of an image token then returns -1.
        """
        if (k < 0 or k >= self._all_num_tokens):
            return -1

        n = k
        if n % self._single_time_step_num_tokens < self._tokens_per_context_image: # check whether k is context_image token
            return -1
        return int(n / self._single_time_step_num_tokens) # return which time index that k belongs to.

    # _action_tokens_mask is for loss computing. This has all indexes of action tokens in all tokens.
    # We can know which output tokens are action predictions by _action_tokens_mask - 1.
    # _default_attention_mask is modified causaul mask because we will only use observations tokens when predicting actions.
    # So we also have to mask action tokens.
    def _generate_masks(self):
        """Generate mask for action prediction loss and attention visualization."""
        # each time step = [image, action]
        self._single_time_step_num_tokens = (self._tokens_per_action + self._tokens_per_context_image)

        # full sequence = [prefix context + N x timestep + postfix context]
        self._all_num_tokens = self._time_sequence_length * self._single_time_step_num_tokens

        # create mask for action predition loss
        # self._action_tokens_mask has all indexes of action tokens in all tokens.
        self._action_tokens_mask = []
        for n in range(0, self._all_num_tokens, self._single_time_step_num_tokens):
            for x in range(0, self._tokens_per_action, 1):
                self._action_tokens_mask.append(x + n + self._tokens_per_context_image)

        # The look ahead mask ensures causality.
        # This is a lower triangular matrix. All elements other than 0 are 1. 
        # 0 means mask.
        self._default_attention_mask = torch.tril(torch.ones((self._all_num_tokens, self._all_num_tokens), dtype=torch.uint8))

        action_mask = np.ndarray(
            shape=(self._all_num_tokens, self._all_num_tokens), dtype=int)

        for i in range(self._all_num_tokens):
            for j in range(self._all_num_tokens):
                action_i = self._get_action_index_for_token(i)
                action_j = self._get_action_index_for_token(j)
                mask = 0
                if action_i != -1 and action_j != -1: # Check both of i and j are actions.
                    # Ignore actions of previous time steps.
                    if action_j < action_i:
                        mask = 1
                    # If we're not auto-regression, ignore action of current time step.
                    if (action_j == action_i and j <= i):
                        mask = 1
                action_mask[i, j] = mask
        self._default_attention_mask -= action_mask