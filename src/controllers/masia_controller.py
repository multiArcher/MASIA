"""Multi-agent controller for MASIA.

Ported from https://github.com/chenf-ai/MASIA without changing run/main.
"""

from types import SimpleNamespace

import torch as th

from components.observation_delay_model import ObservationDelayModel
from controllers.mac import MAC
from utils.maker import ActionSelectorMaker, AgentMaker


class MASIAMAC(MAC):
    def __init__(self, scheme: dict, groups: dict, args: SimpleNamespace):
        super(MASIAMAC, self).__init__(scheme, groups, args)
        self.n_agents = args.n_agents
        self.args = args
        self.device = args.device
        self.agent_output_type = args.agent_output_type

        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.action_selector = ActionSelectorMaker.make(args.action_selector, args)
        self.observation_delay_model = ObservationDelayModel(args)

        self.hidden_states = None
        self.encoder_hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, **kwargs):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if test_mode:
            self.agent.eval()
        agent_outs, self.hidden_states, self.encoder_hidden_states = self.agent(
            agent_inputs, self.hidden_states, self.encoder_hidden_states
        )

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(
                    ep_batch.batch_size * self.n_agents, -1
                )
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def rl_forward(self, ep_batch, state_repr, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        agent_outs, self.hidden_states = self.agent.rl_forward(
            agent_inputs, state_repr, self.hidden_states
        )
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def enc_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        state_repr, self.encoder_hidden_states = self.agent.enc_forward(
            agent_inputs, self.encoder_hidden_states
        )
        if self.args.state_encoder in [
            "ob_attn_ae",
            "ob_attn_skipsum_ae",
            "ob_attn_skipcat_ae",
        ]:
            return state_repr.view(ep_batch.batch_size, -1)
        return state_repr.view(ep_batch.batch_size, self.n_agents, -1)

    def vae_forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if "vae" in self.args.state_encoder:
            recons, inputs, mu, log_var, self.encoder_hidden_states = self.agent.vae_forward(
                agent_inputs, self.encoder_hidden_states
            )
            return recons, inputs, mu, log_var
        if "ae" in self.args.state_encoder:
            recons, inputs, z, self.encoder_hidden_states = self.agent.vae_forward(
                agent_inputs, self.encoder_hidden_states
            )
            return recons, inputs, z
        raise ValueError("Unsupported state encoder type!")

    def target_transform(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        if "vae" in self.args.state_encoder:
            raise AssertionError("Shouldn't use vae.")
        target_projected, self.encoder_hidden_states = self.agent.target_transform(
            agent_inputs, self.encoder_hidden_states
        )
        return target_projected

    def init_hidden(self, batch_size, fat=False):
        self.hidden_states = (
            self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        )
        if not fat:
            self.encoder_hidden_states = (
                self.agent.encoder_init_hidden()
                .unsqueeze(0)
                .expand(batch_size, self.n_agents, -1)
            )
        else:
            self.encoder_hidden_states = (
                self.agent.encoder_init_hidden()
                .unsqueeze(0)
                .expand(batch_size * self.n_agents, self.n_agents, -1)
            )

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        self.agent = AgentMaker.make(self.args.agent, input_shape, self.args)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        if isinstance(t, int):
            obs_t = self.observation_delay_model.apply(
                batch["obs"], slice(t, t + 1), training=self.training
            ).squeeze(1)
        else:
            obs_t = self.observation_delay_model.apply(
                batch["obs"], t, training=self.training
            )

        inputs = [obs_t]
        if self.args.obs_last_action:
            if isinstance(t, int):
                last_t = t
            else:
                last_t = t.start
            if last_t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, last_t]))
            else:
                inputs.append(batch["actions_onehot"][:, last_t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        return th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
