from typing import Dict
import numpy as np
from gym import spaces

# This function will turn the action_space into the batched action_space and return a batched action sample.
# The output format is compatible with OpenAI gym's Vectorized Environments.
def batched_action_sampler(action_space: spaces.Dict, batch_size: int):
    batched_sample : Dict[str, np.ndarray] = {}
    samples = [action_space.sample() for _ in range(batch_size)] # 辞書のリスト
    for key in samples[0].keys():
        value_list = []
        for i in range(batch_size):
            value_list.append(samples[i][key])
        value_list = np.stack(value_list, axis=0)
        batched_sample[key] = value_list
    return batched_sample