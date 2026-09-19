import logging
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from components.episode_buffer import EpisodeBatch
from controllers.masia_controller import MASIAMAC
from learners.masia_learner import MASIALearner
from modules.agents.masia_agent import MASIAAgent
from utils.maker import AgentMaker, LearnerMaker, MACMaker


class DummyLogger:
    def __init__(self):
        self.stats = []
        self.console_logger = logging.getLogger("masia_test")

    def log_stat(self, key, value, t):
        self.stats.append((key, value, t))

    def info(self, *args, **kwargs):
        pass


def _base_args(**overrides):
    args = SimpleNamespace(
        n_agents=3,
        n_actions=5,
        state_shape=12,
        hidden_dim=16,
        agent_hidden_dim=16,
        use_rnn=True,
        obs_agent_id=True,
        obs_last_action=False,
        concat_obs=True,
        ob_embed_dim=8,
        encoder_use_rnn=True,
        encoder_hidden_dim=8,
        ae_enc_hidden_dims=[],
        ae_dec_hidden_dims=[],
        attn_embed_dim=4,
        state_encoder="ob_attn_ae",
        state_repre_dim=4,
        use_latent_model=True,
        use_rew_pred=True,
        use_momentum_encoder=True,
        use_residual=True,
        momentum_tau=1,
        pred_len=2,
        latent_model="mlp",
        model_hidden_dim=16,
        action_embed_dim=4,
        spr_dim=8,
        rl_signal=True,
        spr_coef=1.0,
        rew_pred_coef=1.0,
        repr_coef=1.0,
        mixer="qmix",
        mixing_embed_dim=8,
        hypernet_layers=2,
        hypernet_embed=16,
        agent="masia",
        mac="masia_mac",
        learner="masia_learner",
        agent_output_type="q",
        action_selector="epsilon_greedy",
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=100,
        evaluation_epsilon=0.0,
        lr=0.0005,
        gamma=0.99,
        grad_norm_clip=10,
        double_q=True,
        target_update_interval_or_tau=200,
        standardise_returns=False,
        standardise_rewards=False,
        common_reward=True,
        learner_log_interval=1,
        noise_env=False,
        noise_type=0,
        obs_delay_enabled=False,
        obs_delay_apply_train=False,
        obs_delay_apply_test=False,
        obs_gaussian_delay_mean=0.0,
        obs_gaussian_delay_std=0.0,
        obs_delay_discretization="round",
        device=torch.device("cpu"),
        use_cuda=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _scheme(n_agents, n_actions, obs_dim, state_dim):
    return {
        "state": {"vshape": state_dim, "dtype": torch.float32},
        "obs": {"vshape": obs_dim, "group": "agents", "dtype": torch.float32},
        "actions": {"vshape": (1,), "group": "agents", "dtype": torch.long},
        "actions_onehot": {"vshape": (n_actions,), "group": "agents", "dtype": torch.float32},
        "avail_actions": {"vshape": (n_actions,), "group": "agents", "dtype": torch.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": torch.uint8},
    }


def _make_batch(args, batch_size=2, seq_len=5, obs_dim=7):
    scheme = _scheme(args.n_agents, args.n_actions, obs_dim, int(args.state_shape))
    groups = {"agents": args.n_agents}
    batch = EpisodeBatch(scheme, groups, batch_size, seq_len, device=args.device)
    filled = torch.ones(batch_size, seq_len, 1)
    obs = torch.randn(batch_size, seq_len, args.n_agents, obs_dim)
    state = torch.randn(batch_size, seq_len, int(args.state_shape))
    actions = torch.randint(0, args.n_actions, (batch_size, seq_len, args.n_agents, 1))
    actions_onehot = torch.zeros(batch_size, seq_len, args.n_agents, args.n_actions)
    actions_onehot.scatter_(-1, actions, 1)
    avail = torch.ones(batch_size, seq_len, args.n_agents, args.n_actions, dtype=torch.int)
    reward = torch.randn(batch_size, seq_len, 1)
    terminated = torch.zeros(batch_size, seq_len, 1, dtype=torch.uint8)
    terminated[:, -1] = 1
    batch.update(
        {
            "obs": obs,
            "state": state,
            "actions": actions,
            "actions_onehot": actions_onehot,
            "avail_actions": avail,
            "reward": reward,
            "terminated": terminated,
            "filled": filled,
        },
        mark_filled=False,
    )
    return batch, scheme, groups


class MASIATests(unittest.TestCase):
    def test_yaml_files_parse(self):
        import yaml

        for name in ("masia.yaml", "masia_vdn.yaml"):
            path = Path(__file__).resolve().parents[1] / "src" / "config" / "algs" / name
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["mac"], "masia_mac")
            self.assertEqual(cfg["agent"], "masia")
            self.assertEqual(cfg["learner"], "masia_learner")
            self.assertEqual(cfg["state_encoder"], "ob_attn_ae")
            self.assertIsInstance(cfg["ae_enc_hidden_dims"], list)

    def test_makers_register_masia(self):
        args = _base_args()
        obs_dim = 7
        input_shape = obs_dim + args.n_agents
        agent = AgentMaker.make("masia", input_shape, args)
        self.assertIsInstance(agent, MASIAAgent)

        batch, scheme, groups = _make_batch(args, obs_dim=obs_dim)
        mac = MACMaker.make("masia_mac", scheme, groups, args)
        self.assertIsInstance(mac, MASIAMAC)
        learner = LearnerMaker.make("masia_learner", mac, scheme, DummyLogger(), args)
        self.assertIsInstance(learner, MASIALearner)
        self.assertIsNotNone(batch)

    def test_select_actions_and_train_step(self):
        args = _base_args()
        obs_dim = 7
        batch, scheme, groups = _make_batch(args, obs_dim=obs_dim)
        mac = MASIAMAC(scheme, groups, args)
        learner = MASIALearner(mac, scheme, DummyLogger(), args)
        mac.init_hidden(batch.batch_size)
        actions = mac.select_actions(batch, t_ep=0, t_env=0, test_mode=True)
        self.assertEqual(tuple(actions.shape), (batch.batch_size, args.n_agents))
        learner.train(batch, t_env=0, episode_num=0)

        with tempfile.TemporaryDirectory() as tmp:
            learner.save_models(tmp)
            learner.load_models(tmp)

    def test_skip_encoders_forward(self):
        for encoder in ("ob_attn_skipcat_ae", "ob_attn_skipsum_ae"):
            with self.subTest(encoder=encoder):
                args = _base_args(state_encoder=encoder)
                batch, scheme, groups = _make_batch(args)
                mac = MASIAMAC(scheme, groups, args)
                mac.init_hidden(batch.batch_size)
                q = mac.forward(batch, t=0, test_mode=True)
                self.assertEqual(
                    tuple(q.shape),
                    (batch.batch_size, args.n_agents, args.n_actions),
                )


if __name__ == "__main__":
    unittest.main()
