"""Encoder-only context ablation must preserve the completion history window."""

import unittest
import torch

from test_bcrbc_masked_current import make_args
from modules.bcrbc.bcrbc_model import BCRBCModel


class EncoderWindowTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.args = make_args()
        self.obs = torch.randn(2, 5, 2, 7)

    def test_default_matches_explicit_shared_window(self):
        self.args.bcrbc_encoder_context_window = None
        torch.manual_seed(19)
        inherited = BCRBCModel(7, 3, 2, self.args)
        self.args.bcrbc_encoder_context_window = self.args.bcrbc_context_window
        torch.manual_seed(19)
        explicit = BCRBCModel(7, 3, 2, self.args)
        torch.testing.assert_close(inherited.encode_observations(self.obs),
                                   explicit.encode_observations(self.obs), rtol=0, atol=0)

    def test_per_step_encoder_ignores_past_but_reads_current(self):
        self.args.bcrbc_encoder_context_window = 1
        model = BCRBCModel(7, 3, 2, self.args)
        original = model.encode_observations(self.obs)
        changed = self.obs.clone()
        changed[:, :-1] += 10
        torch.testing.assert_close(original[:, -1], model.encode_observations(changed)[:, -1])
        changed[:, -1] += 10
        self.assertFalse(torch.allclose(original[:, -1], model.encode_observations(changed)[:, -1]))
        self.assertEqual(model.observation_decoder.transformer.context_window, 3)
        self.assertEqual(model.transformer.transformer.context_window, 3)
        self.assertEqual(model.context_window, 3)

    def test_per_step_encoder_cached_equals_dense(self):
        self.args.bcrbc_encoder_context_window = 1
        model = BCRBCModel(7, 3, 2, self.args)
        dense = model.encode_observations(self.obs)
        cache, outputs = None, []
        for t in range(self.obs.shape[1]):
            z, cache = model.encode_observations(self.obs[:, t:t+1], kv_cache=cache,
                                               use_kv_cache=True, rope_offset=t)
            outputs.append(z)
        torch.testing.assert_close(dense, torch.cat(outputs, dim=1))


if __name__ == "__main__":
    unittest.main()
