import unittest
import torch

# from robotics_transformer_pytorch.transformer import Transformer
from transformer import Transformer
from absl.testing import parameterized

class TransformerTest(parameterized.TestCase, unittest.TestCase):

    def setUp(self):
        self._vocab_size = 10
        batch_size = 8
        sequence_len = 12
        self._tokens = torch.rand((batch_size, sequence_len, self._vocab_size))

    @parameterized.parameters(True, False)
    def test_transformer_forwardpass(self, return_attention_scores):
        network = Transformer(
            num_layers=2,
            layer_size=512,
            num_heads=4,
            feed_forward_size=256,
            dropout_rate=0.1,
            vocab_size=self._vocab_size,
            return_attention_scores=return_attention_scores,
            max_seq_len=15)

        output_tokens, attention_scores = network(self._tokens, attention_mask=None)
        self.assertSequenceEqual(self._tokens.shape, output_tokens.shape)
        if return_attention_scores:
            self.assertNotEmpty(attention_scores)
        else:
            self.assertEmpty(attention_scores)



if __name__ == '__main__':
    unittest.main()