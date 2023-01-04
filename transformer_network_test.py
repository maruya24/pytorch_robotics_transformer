"""Tests for networks."""

import torch
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

torch.manual_seed(1234) # this is for random behavior of preprocessors

class TransformerNetworkTest(TransformerNetworkTestUtils):
    @parameterized.named_parameters([{
        'testcase_name': '_' + name,
        'state_space': spec,
        'train_observation': obs,
    } for (name, spec, obs) in zip(space_names_list(), state_space_list(), observations_list())])
    def testTransformerTrainLossCall(self, state_space, train_observation):
        network = transformer_network.TransformerNetwork(
        input_tensor_space=state_space,
        output_tensor_space=self._action_space,
        time_sequence_length=TIME_SEQUENCE_LENGTH)


        network.set_actions(self._train_action)

        network_state = batched_space_sampler(network._state_space, BATCH_SIZE)
        network_state = np_to_tensor(network_state) # change np.ndarray type of sample value into tensor type

        output_actions, network_state = network(
            train_observation, network_state=network_state)

        expected_shape = [2, 3]

        self.assertEqual(list(network.get_actor_loss().shape), expected_shape)

        self.assertCountEqual(self._train_action.keys(), output_actions.keys())

if __name__ == '__main__':
    unittest.main()