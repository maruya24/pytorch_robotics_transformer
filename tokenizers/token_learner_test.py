import unittest
import torch
from pytorch_robotics_transformer.tokenizers.token_learner import TokenLearnerModule

# Type this command on your terminal aboved robotics_transformer_pytorch directory.
# python -m robotics_transformer_pytorch.tokenizer.token_learnerr_test

class TokenLearnerTest(unittest.TestCase):
    def testTokenLearner(self, embedding_dim=512, num_tokens=8):
        batch = 1
        seq = 2
        token_learner_layer = TokenLearnerModule(
            inputs_channels=embedding_dim, 
            num_tokens=num_tokens)
        # seq is time-series length
        # embedding_dim = channels
        inputvec = torch.randn((batch * seq, embedding_dim, 10, 10))

        learnedtokens = token_learner_layer(inputvec)
        self.assertEquals(list(learnedtokens.shape), [batch * seq, num_tokens, embedding_dim])

if __name__ == '__main__':
    unittest.main()