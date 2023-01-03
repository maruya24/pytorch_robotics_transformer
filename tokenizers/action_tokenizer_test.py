import unittest
from gym import spaces
from pytorch_robotics_transformer.tokenizers.action_tokenizer import RT1ActionTokenizer
import numpy as np
from collections import OrderedDict
from typing import List, Dict
from pytorch_robotics_transformer.tokenizers.batched_space_sample import batched_space_sampler

class ActionTokenizerTest(unittest.TestCase):
    
    # Use one Discrete action.
    # Check tokens_per_action and token.
    def testTokenize_Discrete(self):
        action_space = spaces.Dict(
            {
                'terminate': spaces.Discrete(2) 
            }
        )
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=10)
        self.assertEqual(1, tokenizer.tokens_per_action)

        action = {
            'terminate': 1
        }
        action_tokens = tokenizer.tokenize(action)
        self.assertEqual(np.array([1]), action_tokens)

    def testDetokenize_Discrete(self):
        action_space = spaces.Dict(
            {
                'terminate': spaces.Discrete(2)
            }
        )
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=10)
        # 0 token should become 0
        action = tokenizer.detokenize(np.array([0])) # tokenized action is ndarray.
        self.assertEqual(0, action['terminate'])
        # 1 token should become 1
        action = tokenizer.detokenize(np.array([1]))
        self.assertEqual(1, action['terminate'])
        # OOV(Out of vocabulary) 3 token should become a default 0
        action = tokenizer.detokenize(np.array([3]))
        self.assertEqual(0, action['terminate'])


    # Use one Box action.
    # Check tokens_per_action and token.
    def testTokenize_Box(self):
        action_space = spaces.Dict(
            {
                'world_vector': spaces.Box(low= -1., high= 1., shape=(3,), dtype=np.float32)
            }
        )
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=10)
        self.assertEqual(3, tokenizer.tokens_per_action)

        action = {
            'world_vector': np.array([0.1, 0.5, -0.8])
        }
        action_tokens = tokenizer.tokenize(action)
        self.assertSequenceEqual([4, 6, 0], list(action_tokens))


    def testTokenize_Box_with_time_dimension(self):
        action_space = spaces.Dict(
            {
                'world_vector': spaces.Box(low= -1., high= 1., shape=(3,), dtype=np.float32)
            }
        )
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=10)
        self.assertEqual(3, tokenizer.tokens_per_action)

        # We check if tokenizer works correctly when action vector has dimensions of batch size and time.
        batch_size = 2
        time_dimension = 3
        world_vec = np.array([[0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8], [0.1, 0.5, -0.8]])
        world_vec = np.reshape(world_vec, (batch_size, time_dimension, tokenizer.tokens_per_action))
        action = {
            'world_vector': world_vec
        }
        action_tokens = tokenizer.tokenize(action)
        self.assertSequenceEqual(
        [batch_size, time_dimension, tokenizer.tokens_per_action], list(action_tokens.shape))


    def testTokenize_Box_at_limits(self):
        minimum = -1.
        maximum = 1.
        vocab_size = 10
        action_space = spaces.Dict(
            {
                'world_vector': spaces.Box(low=minimum, high= maximum, shape=(2,), dtype=np.float32)
            }
        )
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=vocab_size)
        self.assertEqual(2, tokenizer.tokens_per_action)
        action = {
            'world_vector': [minimum, maximum]
        }
        action_tokens = tokenizer.tokenize(action)
        # Minimum value will go to 0
        # Maximum value witll go to vocab_size-1
        self.assertSequenceEqual([0, vocab_size - 1], list(action_tokens))


    def testTokenize_invalid_action_spec_shape(self):
        action_space = spaces.Dict(
            {
                'world_vector': spaces.Box(low=-1., high=1., shape=(2,2), dtype=np.float32)
            }
        )
        with self.assertRaises(ValueError):
            RT1ActionTokenizer(action_space, vocab_size=10)


    # This test is the closest to a real situation.
    def testTokenizeAndDetokenizeIsEqual(self):
        action_space_dict = OrderedDict([('terminate', spaces.Discrete(2)), 
                                         ('world_vector', spaces.Box(low= -1.0, high= 1.0, shape=(3,), dtype=np.float32)),
                                         ('rotation_delta', spaces.Box(low= -np.pi / 2., high= np.pi / 2., shape=(3,), dtype=np.float32)),
                                         ('gripper_closedness_action', spaces.Box(low= -1.0  , high= 1.0, shape=(1,), dtype=np.float32))
                                         ])
        action_space = spaces.Dict(action_space_dict)
        tokenizer = RT1ActionTokenizer(action_space, vocab_size=1024)
        self.assertEqual(8, tokenizer.tokens_per_action)

        # Repeat the following test N times with fuzzy inputs.
        n_repeat = 10
        for _ in range(n_repeat):
            action = action_space.sample()
            action_tokens = tokenizer.tokenize(action)
            policy_action = tokenizer.detokenize(action_tokens)
            # print(action)
            # print(action_tokens)
            # print(policy_action)
            for k in action:
                np.testing.assert_allclose(action[k], policy_action[k], 2)
        
        # Repeat the test with batched actions
        batch_size = 2
        batched_action = batched_space_sampler(action_space, batch_size)
        action_tokens = tokenizer.tokenize(batched_action)
        policy_action = tokenizer.detokenize(action_tokens)

        # print(batched_action)
        # print(action_tokens)
        # print(policy_action)

        for k in batched_action:
            for a, policy_a in zip(batched_action[k], policy_action[k]):
                np.testing.assert_almost_equal(a, policy_a, decimal=2)



if __name__ == '__main__':
    unittest.main()