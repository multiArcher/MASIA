"""CACOM integration and optional numerical parity with the pinned upstream."""

import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch as th
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from components.episode_buffer import EpisodeBatch
from components.transforms import OneHot
from controllers.cacom_controller import CACOMMAC
from modules.agents.cacom_agent import CACOMAgent, ExpGate, LsqQuan
from utils.maker import MACMaker, LearnerMaker


def make_args(**overrides):
    config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text(encoding="utf-8"))
    config.update(yaml.safe_load((ROOT / "src/config/algs/cacom.yaml").read_text(encoding="utf-8")))
    config.update(n_agents=3, n_actions=5, state_shape=10, device="cpu", use_cuda=False,
                  hidden_dim=16, encode_dim=4, request_dim=3, response_dim=4,
                  env_info={"obs_components": [2, (2, 3), 4]},
                  start_train_gate=10, train_gate_intervel=1, target_update_interval=2)
    config.update(overrides)
    return SimpleNamespace(**config)


def make_batch(args, length=5):
    scheme = {"obs": {"vshape": 12, "group": "agents"},
              "state": {"vshape": 10},
              "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
              "avail_actions": {"vshape": (5,), "group": "agents", "dtype": th.int},
              "reward": {"vshape": (1,)},
              "terminated": {"vshape": (1,), "dtype": th.uint8}}
    batch = EpisodeBatch(scheme, {"agents": 3}, 2, length,
                         preprocess={"actions": ("actions_onehot", [OneHot(5)])}, device=args.device)
    batch.update({"obs": th.randn(2, length, 3, 12), "state": th.randn(2, length, 10),
                  "actions": th.randint(1, 5, (2, length, 3, 1)),
                  "avail_actions": th.ones(2, length, 3, 5),
                  "reward": th.randn(2, length, 1), "terminated": th.zeros(2, length, 1)})
    batch["avail_actions"][..., 0] = 0
    # First episode terminates at transition 1; t=2 is the final state.
    batch["terminated"][0, 1] = 1
    batch["filled"][0, 3:] = 0
    batch["terminated"][1, length - 2] = 1
    return batch


class Logger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, key, value, t):
        self.stats[key] = value


class CACOMTest(unittest.TestCase):
    def setUp(self):
        th.set_num_threads(1)
        th.manual_seed(7)
        random.seed(7)
        self.args = make_args()
        self.batch = make_batch(self.args)
        self.mac = MACMaker.make("cacom_mac", self.batch.scheme, {"agents": 3}, self.args)

    def learner(self, mac=None, args=None):
        return LearnerMaker.make("cacom_learner", mac or self.mac, self.batch.scheme,
                                 Logger(), args or self.args)

    def test_segments_actions_and_observation_delay(self):
        self.assertEqual(self.mac.args.obs_segs, [(1, 2), (2, 3), (1, 4), (1, 5), (1, 3)])
        self.assertIsNone(self.args.obs_segs)
        self.mac.init_hidden(2)
        selected = self.mac.select_actions(self.batch, 0, 0, bs=[1], test_mode=True)
        self.assertEqual(tuple(selected.shape), (1, 3))
        self.assertTrue((selected != 0).all())
        self.assertFalse(self.mac.hidden_states.requires_grad)
        self.mac.observation_delay_model.enabled = True
        self.mac.observation_delay_model.apply_train = True
        self.mac.observation_delay_model.delay_mean = 2
        self.mac.train()
        inputs = self.mac._build_inputs(self.batch, 3).reshape(2, 3, -1)
        th.testing.assert_close(inputs[..., :12], self.batch["obs"][:, 1])
        self.mac.eval()
        inputs = self.mac._build_inputs(self.batch, 3).reshape(2, 3, -1)
        th.testing.assert_close(inputs[..., :12], self.batch["obs"][:, 3])

    def test_smacv2_and_explicit_segmentation(self):
        self.args.env_info["obs_components"] = {"move": (1, 2), "enemy": (2, 3), "own": (1, 4)}
        mac = CACOMMAC(self.batch.scheme, {}, self.args)
        self.assertEqual(mac.args.obs_segs, self.mac.args.obs_segs)
        self.args.env_info = {}
        with self.assertRaisesRegex(ValueError, "needs obs_segs"):
            CACOMMAC(self.batch.scheme, {}, self.args)
        self.args.obs_segs = [(1, 12), (1, 5), (1, 3)]
        CACOMMAC(self.batch.scheme, {}, self.args)
        self.args.obs_segs = [(1, 12)]
        with self.assertRaisesRegex(ValueError, "input has"):
            CACOMMAC(self.batch.scheme, {}, self.args)
        with self.assertRaises(ValueError):
            LsqQuan(bit=1)

    def test_train_gate_targets_and_checkpoint(self):
        learner = self.learner()
        self.assertFalse({id(p) for p in learner.params} & {id(p) for p in learner.gate_params})
        old_agent = copy.deepcopy(self.mac.agent.state_dict())
        old_gate = copy.deepcopy(self.mac.gate.state_dict())
        learner.train(self.batch, 0, 0)
        self.assertTrue(any(not th.equal(v, old_agent[k]) for k, v in self.mac.agent.state_dict().items()))
        self.assertTrue(any(not th.equal(v, old_gate[k]) for k, v in self.mac.gate.state_dict().items()))
        # Also exercise counterfactual-label phase and hard target synchronization.
        learner.train(self.batch, 20, 2)
        for key, value in self.mac.state_dict().items():
            th.testing.assert_close(value, learner.target_mac.state_dict()[key])
        self.assertTrue(all(th.isfinite(th.tensor(v)) for v in learner.logger.stats.values()))
        self.assertIn("loss/gate_loss", learner.logger.stats)
        with tempfile.TemporaryDirectory() as path:
            learner.save_models(path)
            loaded = self.learner(copy.deepcopy(self.mac))
            loaded.load_models(path)
            for original, restored in ((learner.mac, loaded.mac), (learner.target_mac, loaded.target_mac),
                                       (learner.mixer, loaded.mixer), (learner.target_mixer, loaded.target_mixer)):
                for key, value in original.state_dict().items():
                    th.testing.assert_close(value, restored.state_dict()[key])
            self.assertEqual(len(loaded.gate_optimizer.state), len(learner.gate_optimizer.state))
            random.seed(9)
            learner.train(self.batch, 40, 4)
            random.seed(9)
            loaded.train(self.batch, 40, 4)
            for key, value in learner.mac.state_dict().items():
                th.testing.assert_close(value, loaded.mac.state_dict()[key])

    def test_padded_and_post_terminal_values_do_not_train(self):
        original = self.learner()
        changed = copy.deepcopy(original)
        other = copy.deepcopy(self.batch)
        other["obs"][0, 2:] += 30
        other["state"][0, 2:] += 30
        other["reward"][0, 2:] += 30
        random.seed(17)
        original.train(self.batch, 20, 0)
        random.seed(17)
        changed.train(other, 20, 0)
        for key, value in original.mac.state_dict().items():
            th.testing.assert_close(value, changed.mac.state_dict()[key], atol=1e-6, rtol=1e-5)
        for key, value in original.logger.stats.items():
            if key != "communication/reply_freq":
                self.assertAlmostEqual(value, changed.logger.stats[key], places=5)

    def test_no_quantization_or_auxiliary_and_vdn(self):
        for mixer in ("vdn", None):
            args = make_args(discrete_bits=None, pred_weight=0, mixer=mixer)
            mac = CACOMMAC(self.batch.scheme, {}, args)
            learner = self.learner(mac, args)
            learner.train(self.batch, 20, 2)
            self.assertEqual(learner.logger.stats["loss/aux_loss"], 0)

    @unittest.skipUnless(th.cuda.is_available(), "CUDA unavailable")
    def test_cuda_train_and_move_to_cpu(self):
        learner = self.learner().to("cuda")
        self.batch.to("cuda")
        learner.train(self.batch, 20, 2)
        learner.to("cpu")
        self.batch.to("cpu")
        learner.train(self.batch, 30, 4)

    @unittest.skipUnless(os.environ.get("CACOM_UPSTREAM_PATH"), "Set CACOM_UPSTREAM_PATH for reference parity")
    def test_upstream_forward_gradients_and_counterfactual_gate(self):
        path = Path(os.environ["CACOM_UPSTREAM_PATH"]) / "src/modules/agents"
        # Import the original, unmodified files as a separate temporary package.
        import types
        package = types.ModuleType("cacom_reference")
        package.__path__ = [str(path)]
        sys.modules["cacom_reference"] = package
        spec = importlib.util.spec_from_file_location("cacom_reference.cacom_agent", path / "cacom_agent.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = copy.copy(self.mac.args)
        args.rnn_hidden_dim = args.hidden_dim
        args.nn_hidden_size = 64
        reference = module.CACOM_Agent(20, args)
        gate = module.ExpGate(args)
        reference.load_state_dict(self.mac.agent.state_dict())
        gate.load_state_dict(self.mac.gate.state_dict())
        inputs = th.randn(6, 20)
        hidden = th.randn(2, 3, args.hidden_dim)
        for all_through in (True, False):
            expected = reference(inputs, hidden, 2, gate, all_through=all_through, train_mode=True)
            actual = self.mac.agent(inputs, hidden, 2, self.mac.gate, all_through=all_through, train_mode=True)
            for i in (0, 1, 3):
                th.testing.assert_close(actual[i], expected[i])
            th.testing.assert_close(actual[2]["aux_loss"].mean(), expected[2]["aux_loss"])
            reference.zero_grad()
            self.mac.agent.zero_grad()
            (expected[0].square().mean() + expected[2]["aux_loss"]).backward()
            (actual[0].square().mean() + actual[2]["aux_loss"].mean()).backward()
            for (name, p), (_, ref) in zip(self.mac.agent.named_parameters(), reference.named_parameters()):
                th.testing.assert_close(p.grad, ref.grad, msg=name)
        random.seed(42)
        expected = reference.cal_gate_labels(inputs, hidden, 2, gate)
        random.seed(42)
        actual = self.mac.agent.cal_gate_labels(inputs, hidden, 2, self.mac.gate)
        self.assertEqual(actual[-1], expected[-1])
        for a, b in zip(actual[:-1], expected[:-1]):
            th.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
