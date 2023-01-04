# Robotics Transformer (RT-1), in PyTorch
PyTorch re-implementation of [RT-1](https://github.com/google-research/robotics_transformer). This model is based on 2022 paper ["RT-1: Robotics Transformer for Real-World Control at Scale"](https://arxiv.org/abs/2212.06817).
This implementation mostly follows the original codes but I made some changes for PyTorch compatibility or simplification.

## Features
* Film efficient net based image tokenizer backbone
* Token learner based compression of input tokens
* Transformer for end to end robotic control
* Testing utilities


## Main Changes
- Translated TensorFlow implementation into PyTorch implementation.
- Used spaces of OpenAI gym to define observation and action variables instead of using tensor_spec of TF-Agents.
- Took actions that are before the current timestep into consideration.
- Added extra comments.
- Abolished 4-d constraint of some specs.
- Didn't implement the add_summaries function of transformer_network.py which is for tensorboard logging.
- Added useful functions and store those in utils.py
- Omitted some variables, functions, classes, and tests including those that are no longer necessary for this repository.

## Note
I didn't implement squence_agent.py and its test codes for now. 
I'm going to upload those codes in a few days. And I may add tensorboard logging.


## License
This library is licensed under the terms of the Apache license.