# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows, 
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].


"""Tests for networks."""

import torch
import torch.nn.functional as F
from absl.testing import parameterized
import unittest
import numpy as np
from typing import Dict


from pytorch_robotics_transformer import transformer_network
from pytorch_robotics_transformer.transformer_network_test_set_up import BATCH_SIZE
from pytorch_robotics_transformer.transformer_network_test_set_up import NAME_TO_INF_OBSERVATIONS
from pytorch_robotics_transformer.transformer_network_test_set_up import NAME_TO_STATE_SPACES
from pytorch_robotics_transformer.transformer_network_test_set_up import observations_list
from pytorch_robotics_transformer.transformer_network_test_set_up import space_names_list
from pytorch_robotics_transformer.transformer_network_test_set_up import state_space_list
from pytorch_robotics_transformer.transformer_network_test_set_up import TIME_SEQUENCE_LENGTH
from pytorch_robotics_transformer.transformer_network_test_set_up import TransformerNetworkTestUtils
from pytorch_robotics_transformer.tokenizers.utils import batched_space_sampler
from pytorch_robotics_transformer.tokenizers.utils import np_to_tensor


class TransformerNetworkTest(TransformerNetworkTestUtils):
    # @parameterized.named_parameters([{
    #     'testcase_name': '_' + name,
    #     'state_space': spec,
    #     'train_observation': obs,
    # } for (name, spec, obs) in zip(space_names_list(), state_space_list(), observations_list())])
    # def testTransformerTrainLossCall(self, state_space, train_observation):
    #     network = transformer_network.TransformerNetwork(
    #     input_tensor_space=state_space,
    #     output_tensor_space=self._action_space,
    #     time_sequence_length=TIME_SEQUENCE_LENGTH)


    #     network.set_actions(self._train_action)

    #     network_state = batched_space_sampler(network._state_space, batch_size=BATCH_SIZE)
    #     network_state = np_to_tensor(network_state) # change np.ndarray type of sample values into tensor type

    #     output_actions, network_state = network(
    #         train_observation, network_state=network_state)

    #     expected_shape = [2, 3]

    #     self.assertEqual(list(network.get_actor_loss().shape), expected_shape)

    #     self.assertCountEqual(self._train_action.keys(), output_actions.keys())

    # @parameterized.named_parameters([{
    #     'testcase_name': '_' + name,
    #     'space_name': name,
    # } for name in space_names_list()])
    # def testTransformerInferenceLossCall(self, space_name):
    #     state_space = NAME_TO_STATE_SPACES[space_name]
    #     observation = NAME_TO_INF_OBSERVATIONS[space_name] #  observation has no time dimension unlike during training.

    #     network = transformer_network.TransformerNetwork(
    #     input_tensor_space=state_space,
    #     output_tensor_space=self._action_space,
    #     time_sequence_length=TIME_SEQUENCE_LENGTH)

    #     network.set_actions(self._inference_action) # self._inference_action has no time dimension unlike self._train_action.
    #     # inference currently only support batch size of 1
    #     network_state = batched_space_sampler(network._state_space, batch_size=1)
    #     network_state = np_to_tensor(network_state) # change np.ndarray type of sample values into tensor type

    #     output_actions, network_state = network(
    #         observation, network_state=network_state)

    #     self.assertEqual(network.get_actor_loss().item(), 0.0)
    #     self.assertCountEqual(self._inference_action.keys(), output_actions.keys())

    @parameterized.named_parameters([{
      'testcase_name': '_' + name,
      'state_space': spec,
    } for name, spec in zip(space_names_list(), state_space_list())])
    def testTransformerCausality(self, state_space):
        network = transformer_network.TransformerNetwork(
        input_tensor_space=state_space,
        output_tensor_space=self._action_space,
        time_sequence_length=TIME_SEQUENCE_LENGTH,
        dropout_rate=0.0)

        network.eval()

        time_sequence_length = network._time_sequence_length
        tokens_per_image = network._tokens_per_context_image
        tokens_per_action = network._tokens_per_action

        # size of all_tokens: (time_sequence_length * (tokens_per_image + tokens_per_action)) 
        def _split_image_and_action_tokens(all_tokens):
            image_start_indices = [(tokens_per_image + tokens_per_action) * k
                             for k in range(time_sequence_length)]

            image_tokens = torch.stack(
                [all_tokens[i:i + tokens_per_image] for i in image_start_indices],
                dim=0)
            action_start_indices = [i + tokens_per_image for i in image_start_indices]
            action_tokens = torch.stack([
                    all_tokens[i:i + tokens_per_action] for i in action_start_indices],
                    dim=0)

            image_tokens = F.one_hot(image_tokens, network._token_embedding_size)
            # Add batch dimension.
            image_tokens = image_tokens.unsqueeze(0) # image_tokens: (1, time_sequence_length, tokens_per_image, emb_dim)
            action_tokens = action_tokens.unsqueeze(0) # action: (1, time_sequence_length, tokens_per_action)

            return image_tokens, action_tokens

        # Generate some random tokens for image and actions.
        all_tokens = torch.randint(low=0, high=10, size=(time_sequence_length * (tokens_per_image + tokens_per_action),))
        context_image_tokens, action_tokens = _split_image_and_action_tokens(all_tokens)
        # Get the output tokens without any zeroed out input tokens.
        # output_tokens: (t*num_tokens, vocab_size)
        output_tokens = network._transformer_call(
            context_image_tokens=context_image_tokens,
            action_tokens=action_tokens,
            attention_mask=network._default_attention_mask,
            batch_size=1)[0]

        for t in range(time_sequence_length *
                   (tokens_per_image + tokens_per_action)):
            # Zero out future input tokens.
            all_tokens_at_t = torch.concat([all_tokens[:t + 1], torch.zeros_like(all_tokens[t + 1:])], 0)
            context_image_tokens, action_tokens = _split_image_and_action_tokens(all_tokens_at_t)

            # Get the output tokens with zeroed out input tokens after t.
            output_tokens_at_t = network._transformer_call(
                context_image_tokens=context_image_tokens,
                action_tokens=action_tokens,
                attention_mask=network._default_attention_mask,
                batch_size=1)[0]

            # The output token is unchanged if future input tokens are zeroed out.
            np.testing.assert_array_equal(output_tokens[:t + 1].detach().numpy(), output_tokens_at_t[:t + 1].detach().numpy())


if __name__ == '__main__':
    unittest.main()
