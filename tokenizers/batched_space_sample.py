from typing import Dict
import numpy as np
from gym import spaces

# This function will turn the space into the batched space and return a batched action sample.
# The output format is compatible with OpenAI gym's Vectorized Environments.
def batched_space_sampler(space: spaces.Dict, batch_size: int):
    batched_sample : Dict[str, np.ndarray] = {}
    samples = [space.sample() for _ in range(batch_size)] # 辞書のリスト
    for key in samples[0].keys():
        value_list = []
        for i in range(batch_size):
            value_list.append(samples[i][key])
        value_list = np.stack(value_list, axis=0)
        batched_sample[key] = value_list
    return batched_sample