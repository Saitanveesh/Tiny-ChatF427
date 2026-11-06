"""Small, checkpoint-free contract checks for the public PyTorch reference model."""
import unittest

import torch

from tinychat.model import TinyChatF427, count_parameters


class TinyChatModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        cls.model = TinyChatF427().eval()

    def test_locked_parameter_count(self):
        self.assertEqual(count_parameters(self.model), 1_206_864)

    def test_output_dimensions(self):
        with torch.no_grad():
            out = self.model(torch.tensor([[0, 1, 2], [2, 3, 4]]))
        self.assertEqual(tuple(out.shape), (2, 3, 1024))

    def test_causal_prefix_is_unaffected_by_future_tokens(self):
        prefix = torch.tensor([[1, 17, 6]])
        extended = torch.tensor([[1, 17, 6, 21, 7]])
        with torch.no_grad():
            prefix_logits = self.model(prefix)
            extended_logits = self.model(extended)[:, :3, :]
        torch.testing.assert_close(prefix_logits, extended_logits, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
