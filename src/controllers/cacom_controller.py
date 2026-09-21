"""CACOM adapter for the repository's existing runner and EpisodeBatch API."""

from collections.abc import Mapping
from copy import copy
from numbers import Integral

import torch as th

from controllers.basic_controller import BasicMAC
from modules.agents.cacom_agent import ExpGate


class CACOMMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        # Resolve segmentation locally; never mutate the shared experiment config.
        args = copy(args)
        if args.n_agents < 2:
            raise ValueError("CACOM requires at least two agents")
        if args.agent_output_type != "q":
            raise ValueError("This adapter implements the authors' released Q-learning CACOM")
        args.obs_segs = self._resolve_obs_segs(scheme, args)
        super().__init__(scheme, groups, args)
        self.gate = ExpGate(args)

    @staticmethod
    def _resolve_obs_segs(scheme, args):
        # Explicit upstream obs_segs includes last action and agent ID already.
        segments = getattr(args, "obs_segs", None)
        if segments is None:
            components = getattr(args, "env_info", {}).get("obs_components")
            if components is None:
                raise ValueError(
                    "CACOM needs obs_segs for this environment: ordered [count, width] "
                    "pairs covering observation, then optional last action and agent ID. "
                    "SMAC/SMACv2 supply obs_components automatically."
                )
            if isinstance(components, Mapping):
                components = components.values()
            segments = [(1, c) if isinstance(c, Integral) else tuple(c) for c in components]
            segments = [(n, d) for n, d in segments if n and d]
            if sum(n * d for n, d in segments) != scheme["obs"]["vshape"]:
                raise ValueError("Environment obs_components does not cover obs; set obs_segs explicitly")
            if args.obs_last_action:
                segments.append((1, scheme["actions_onehot"]["vshape"][0]))
            if args.obs_agent_id:
                segments.append((1, args.n_agents))
        if not segments or any(
            len(seg) != 2 or any(not isinstance(v, Integral) or isinstance(v, bool) or v <= 0 for v in seg)
            for seg in segments
        ):
            raise ValueError("obs_segs must be nonempty positive integer [count, width] pairs")
        return [tuple(seg) for seg in segments]

    @th.no_grad()
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Preserve upstream rollout semantics: the learned gate is always used.
        outputs, _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            outputs[bs], ep_batch["avail_actions"][:, t_ep][bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False, all_through=False, **kwargs):
        q, self.hidden_states, losses, reply_freq = self.agent(
            self._build_inputs(ep_batch, t), self.hidden_states, ep_batch.batch_size,
            self.gate, all_through=all_through, test_mode=test_mode, **kwargs
        )
        return q.view(ep_batch.batch_size, self.n_agents, -1), losses, reply_freq

    def forward_gate(self, ep_batch, t):
        self.hidden_states, probs, q_pos, q_neg, sender = self.agent.cal_gate_labels(
            self._build_inputs(ep_batch, t), self.hidden_states, ep_batch.batch_size, self.gate
        )
        peers = [i for i in range(self.n_agents) if i != sender]
        # The released Q-learning implementation compares unmasked max-Q values.
        q_diff = q_pos[:, peers].max(-1).values - q_neg[:, peers].max(-1).values
        return q_diff, probs

    def gate_parameters(self):
        return self.gate.parameters()

    def load_state(self, other_mac):
        super().load_state(other_mac)
        self.gate.load_state_dict(other_mac.gate.state_dict())

    def save_models(self, path):
        super().save_models(path)
        th.save(self.gate.state_dict(), f"{path}/gate.th")

    def load_models(self, path):
        super().load_models(path)
        self.gate.load_state_dict(th.load(f"{path}/gate.th", map_location="cpu", weights_only=True))
