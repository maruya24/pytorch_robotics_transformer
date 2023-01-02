from typing import Optional, Tuple, Union, Any, Dict, List
import numpy as np
from gym import spaces
import unittest
from tokenizers.batched_action_sample import batched_action_sampler
import torch
from absl.testing import parameterized

def get_outer_rank(observations: Dict[str, torch.Tensor], input_tensor_space) -> int:
        # used to determine training vs inference call
        # outer_rank will be 2 -> [b, t] during training and
        # outer_rank will be 1 -> [b] during inference

        for k in observations.keys():
            obs_value = observations[k]
            obs_value_shape = obs_value.shape

            obs_space = input_tensor_space[k]
            obs_space_shape = obs_space.shape
            break
        return len(obs_value_shape) - len(obs_space_shape)


obs_space_1 = spaces.Dict(
            {
                'timestep': spaces.Discrete(100)
            }
        )

obs_space_2 = spaces.Dict(
            {
                'world_vector': spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)
            }
        )

class TestGetOuterRank_1(unittest.TestCase):
    def test_get_outer_rank(self):
        batch_size = 2

        obs_sample = batched_action_sampler(obs_space_1, batch_size)
        outer_rank = get_outer_rank(obs_sample, obs_space_1)
        self.assertEqual(1, outer_rank)


        obs_sample = batched_action_sampler(obs_space_2, batch_size)
        outer_rank = get_outer_rank(obs_sample, obs_space_2)
        self.assertEqual(1, outer_rank)

if __name__ == '__main__':
    unittest.main()
