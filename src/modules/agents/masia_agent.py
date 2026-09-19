"""MASIA agent, permutation-invariant message encoders, and latent transition model.

Ported from the official EPyMARL implementation:
https://github.com/chenf-ai/MASIA

Guan et al., "Efficient Multi-agent Communication via Self-supervised
Information Aggregation", NeurIPS 2022 / arXiv:2302.09605.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _message_self_attention(query, key, value, attn_dim):
    """Permutation-invariant message aggregation used by MASIA encoders.

    query: [bs, n_agents, attn_embed_dim]
    key:   [bs, attn_embed_dim, n_agents]
    value: [bs, n_agents, hidden_dim]
    """
    energy = torch.bmm(query, key / (attn_dim ** 0.5))
    score = F.softmax(energy, dim=-1)
    return torch.bmm(score, value)


def _build_mlp(last_h_dim, hidden_dims):
    modules = []
    for i, h_dim in enumerate(hidden_dims):
        if i == len(hidden_dims) - 1:
            modules.append(nn.Sequential(nn.Linear(last_h_dim, h_dim)))
        else:
            modules.append(nn.Sequential(nn.Linear(last_h_dim, h_dim), nn.ReLU()))
        last_h_dim = h_dim
    return nn.Sequential(*modules)


def _mse_reconstruction_loss(recons, target):
    recons_loss = torch.mean((recons - target) ** 2, dim=-1)
    return {"loss": recons_loss}


class ObAttnAEEnc(nn.Module):
    """Attention autoencoder that aggregates teammate messages into a shared z."""

    def __init__(
        self,
        input_shape,
        output_shape,
        n_agents,
        latent_dim,
        args,
        enc_hidden_dims=None,
        dec_hidden_dims=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_agents = n_agents
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = list(getattr(args, "ae_enc_hidden_dims", []) or []) + [self.latent_dim]
        if dec_hidden_dims is None:
            dec_hidden_dims = list(getattr(args, "ae_dec_hidden_dims", []) or []) + [output_shape]

        self.query = nn.Linear(input_shape, args.attn_embed_dim)
        self.key = nn.Linear(input_shape, args.attn_embed_dim)
        self.value = nn.Linear(input_shape, args.encoder_hidden_dim)

        if args.encoder_use_rnn:
            self.encoder_rnn = nn.GRUCell(args.encoder_hidden_dim, args.encoder_hidden_dim)
        else:
            self.encoder_rnn = nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim)

        self.encoder = _build_mlp(args.encoder_hidden_dim, enc_hidden_dims)
        self.decoder = _build_mlp(self.latent_dim * self.n_agents, dec_hidden_dims)

    def encode(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.n_agents
        query = self.query(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim)
        key = self.key(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim).permute(0, 2, 1)
        value = self.value(inputs).reshape(bs, self.n_agents, self.args.encoder_hidden_dim)
        attn_out = _message_self_attention(
            query, key, value, self.args.attn_embed_dim
        ).reshape(bs * self.n_agents, self.args.encoder_hidden_dim)

        h_in = encoder_hidden_state.reshape(-1, self.args.encoder_hidden_dim)
        if self.args.encoder_use_rnn:
            h = self.encoder_rnn(attn_out, h_in)
        else:
            h = F.relu(self.encoder_rnn(attn_out))

        z = self.encoder(h).reshape(bs, self.n_agents * self.latent_dim)
        return z, h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, inputs, encoder_hidden_state, **kwargs):
        z, h = self.encode(inputs, encoder_hidden_state)
        return self.decode(z), inputs, z, h

    def loss_function(self, *args, **kwargs):
        return _mse_reconstruction_loss(args[0], args[1])


class ObAttnSkipCatAEEnc(nn.Module):
    """Attention AE with a skip connection concatenated into the encoder MLP."""

    def __init__(
        self,
        input_shape,
        output_shape,
        n_agents,
        latent_dim,
        args,
        enc_hidden_dims=None,
        dec_hidden_dims=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_agents = n_agents
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = list(getattr(args, "ae_enc_hidden_dims", []) or []) + [self.latent_dim]
        if dec_hidden_dims is None:
            dec_hidden_dims = list(getattr(args, "ae_dec_hidden_dims", []) or []) + [output_shape]

        self.query = nn.Linear(input_shape, args.attn_embed_dim)
        self.key = nn.Linear(input_shape, args.attn_embed_dim)
        self.value = nn.Linear(input_shape, args.encoder_hidden_dim)
        self.skip_layer = nn.Linear(input_shape, args.encoder_hidden_dim)

        if args.encoder_use_rnn:
            self.encoder_rnn = nn.GRUCell(args.encoder_hidden_dim, args.encoder_hidden_dim)
        else:
            self.encoder_rnn = nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim)

        self.encoder = _build_mlp(args.encoder_hidden_dim * 2, enc_hidden_dims)
        self.decoder = _build_mlp(self.latent_dim * self.n_agents, dec_hidden_dims)

    def encode(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.n_agents
        query = self.query(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim)
        key = self.key(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim).permute(0, 2, 1)
        value = self.value(inputs).reshape(bs, self.n_agents, self.args.encoder_hidden_dim)
        attn_out = _message_self_attention(
            query, key, value, self.args.attn_embed_dim
        ).reshape(bs * self.n_agents, self.args.encoder_hidden_dim)

        h_in = encoder_hidden_state.reshape(-1, self.args.encoder_hidden_dim)
        if self.args.encoder_use_rnn:
            h = self.encoder_rnn(attn_out, h_in)
        else:
            h = F.relu(self.encoder_rnn(attn_out))

        skip_out = self.skip_layer(inputs)
        encoder_input = torch.cat([h, skip_out], dim=-1)
        z = self.encoder(encoder_input).reshape(bs, self.n_agents * self.latent_dim)
        return z, h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, inputs, encoder_hidden_state, **kwargs):
        z, h = self.encode(inputs, encoder_hidden_state)
        return self.decode(z), inputs, z, h

    def loss_function(self, *args, **kwargs):
        return _mse_reconstruction_loss(args[0], args[1])


class ObAttnSkipSumAEEnc(nn.Module):
    """Attention AE with a residual skip added onto the encoder RNN output."""

    def __init__(
        self,
        input_shape,
        output_shape,
        n_agents,
        latent_dim,
        args,
        enc_hidden_dims=None,
        dec_hidden_dims=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_agents = n_agents
        self.args = args

        if enc_hidden_dims is None:
            enc_hidden_dims = list(getattr(args, "ae_enc_hidden_dims", []) or []) + [self.latent_dim]
        if dec_hidden_dims is None:
            dec_hidden_dims = list(getattr(args, "ae_dec_hidden_dims", []) or []) + [output_shape]

        self.query = nn.Linear(input_shape, args.attn_embed_dim)
        self.key = nn.Linear(input_shape, args.attn_embed_dim)
        self.value = nn.Linear(input_shape, args.encoder_hidden_dim)
        self.skip_layer = nn.Linear(input_shape, args.encoder_hidden_dim)

        if args.encoder_use_rnn:
            self.encoder_rnn = nn.GRUCell(args.encoder_hidden_dim, args.encoder_hidden_dim)
        else:
            self.encoder_rnn = nn.Linear(args.encoder_hidden_dim, args.encoder_hidden_dim)

        self.encoder = _build_mlp(args.encoder_hidden_dim, enc_hidden_dims)
        self.decoder = _build_mlp(self.latent_dim * self.n_agents, dec_hidden_dims)

    def encode(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.n_agents
        query = self.query(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim)
        key = self.key(inputs).reshape(bs, self.n_agents, self.args.attn_embed_dim).permute(0, 2, 1)
        value = self.value(inputs).reshape(bs, self.n_agents, self.args.encoder_hidden_dim)
        attn_out = _message_self_attention(
            query, key, value, self.args.attn_embed_dim
        ).reshape(bs * self.n_agents, self.args.encoder_hidden_dim)

        h_in = encoder_hidden_state.reshape(-1, self.args.encoder_hidden_dim)
        if self.args.encoder_use_rnn:
            h = self.encoder_rnn(attn_out, h_in)
        else:
            h = F.relu(self.encoder_rnn(attn_out))

        h = h + self.skip_layer(inputs)
        z = self.encoder(h).reshape(bs, self.n_agents * self.latent_dim)
        return z, h

    def decode(self, z):
        return self.decoder(z)

    def forward(self, inputs, encoder_hidden_state, **kwargs):
        z, h = self.encode(inputs, encoder_hidden_state)
        return self.decode(z), inputs, z, h

    def loss_function(self, *args, **kwargs):
        return _mse_reconstruction_loss(args[0], args[1])


STATE_ENCODER_REGISTRY = {
    "ob_attn_ae": ObAttnAEEnc,
    "ob_attn_skipcat_ae": ObAttnSkipCatAEEnc,
    "ob_attn_skipsum_ae": ObAttnSkipSumAEEnc,
}


class TransitionModel(nn.Module):
    """Latent dynamics model used for self-supervised future prediction (SPR)."""

    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.state_encoder in [
            "ob_ind_ae",
            "ob_attn_ae",
            "ob_attn_skipsum_ae",
            "ob_attn_skipcat_ae",
        ]:
            state_repre_dim = args.state_repre_dim * args.n_agents
        else:
            state_repre_dim = args.state_repre_dim

        self.action_embed = nn.Linear(args.n_actions, args.action_embed_dim)
        self.joint_action_embed = nn.Sequential(
            nn.Linear(args.action_embed_dim * args.n_agents, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        self.state_repre_embed = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, args.model_hidden_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(args.model_hidden_dim * 2, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, state_repre_dim),
        )
        self.reward_predictor = nn.Sequential(
            nn.Linear(state_repre_dim, args.model_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.model_hidden_dim, 1),
        )

    def forward(self, state_repre, actions):
        origin_shape = state_repre.shape
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)

        batch_size, seq_len, n_agents, _ = actions.shape
        action_embed = self.action_embed(actions).reshape(
            batch_size, seq_len, n_agents * self.args.action_embed_dim
        )
        joint_action_embed = self.joint_action_embed(F.relu(action_embed))
        z_embed = self.state_repre_embed(state_repre)
        next_state = self.network(torch.cat([z_embed, joint_action_embed], dim=-1))
        if self.args.use_residual:
            next_state = next_state + state_repre
        return next_state.reshape(*origin_shape)

    def predict_reward(self, state_repre):
        if self.args.state_encoder == "ob_ind_ae":
            state_repre = state_repre.flatten(-2, -1)
        return self.reward_predictor(state_repre)


def _update_state_dict(model, state_dict, tau=1):
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {
            k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()
        }
        model.load_state_dict(update_sd)


class MASIAAgent(nn.Module):
    """Each agent extracts a gated slice of the aggregated representation z."""

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.raw_input_shape = self._get_input_shape(input_shape)
        self.hidden_dim = int(getattr(args, "hidden_dim", getattr(args, "agent_hidden_dim", 64)))

        state_dim = int(np.prod(args.state_shape))
        encoder_cls = STATE_ENCODER_REGISTRY.get(args.state_encoder)
        if encoder_cls is None:
            raise ValueError(
                f"Unknown MASIA state_encoder '{args.state_encoder}'. "
                f"Supported: {sorted(STATE_ENCODER_REGISTRY)}"
            )
        self.encoder = encoder_cls(
            input_shape=input_shape,
            output_shape=state_dim,
            n_agents=args.n_agents,
            latent_dim=args.state_repre_dim,
            args=args,
        )

        self.latent_dim = args.state_repre_dim * args.n_agents

        if self.args.use_latent_model:
            self.projection = nn.Sequential(
                nn.Linear(self.latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.args.spr_dim),
            )
            self.final_classifier = nn.Sequential(
                nn.Linear(self.args.spr_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.args.spr_dim),
            )
            if self.args.use_momentum_encoder:
                self.target_encoder = copy.deepcopy(self.encoder)
                self.target_projection = copy.deepcopy(self.projection)
                for param in list(self.target_encoder.parameters()) + list(
                    self.target_projection.parameters()
                ):
                    param.requires_grad = False
            else:
                self.target_encoder = self.encoder
                self.target_projection = self.projection

        extra_dim = 0
        if self.args.obs_last_action:
            extra_dim += self.args.n_actions
        if self.args.obs_agent_id:
            extra_dim += self.args.n_agents

        self.gate = nn.Linear(input_shape, self.latent_dim)
        if self.args.concat_obs:
            self.ob_fc = nn.Linear(self.raw_input_shape, args.ob_embed_dim)
            self.fc1 = nn.Linear(
                args.ob_embed_dim + self.latent_dim + extra_dim, self.hidden_dim
            )
        else:
            self.ob_fc = None
            self.fc1 = nn.Linear(self.latent_dim + extra_dim, self.hidden_dim)

        if self.args.use_rnn:
            self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        else:
            self.rnn = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def encoder_init_hidden(self):
        return self.fc1.weight.new(1, self.args.encoder_hidden_dim).zero_()

    def _encode_messages(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.args.n_agents
        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        if "ae" not in self.args.state_encoder:
            raise ValueError("Unknown encoder!!!")

        if self.args.noise_env and self.args.noise_type == 0:
            noise = torch.randn(*inputs.shape, device=inputs.device)
            inputs = inputs + noise
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        elif self.args.noise_env and self.args.noise_type == 1:
            inputs = (
                inputs.reshape(bs, self.args.n_agents, inputs.shape[-1])
                .unsqueeze(1)
                .repeat(1, self.args.n_agents, 1, 1)
            )
            noise = torch.randn(
                bs,
                self.args.n_agents,
                self.args.n_agents,
                inputs.shape[-1],
                device=inputs.device,
            )
            mask = (
                (1 - torch.eye(self.args.n_agents, self.args.n_agents, device=inputs.device))
                .unsqueeze(0)
                .repeat(bs, 1, 1)
                .unsqueeze(-1)
            )
            inputs = (inputs + noise * mask).flatten(0, 2)
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        elif not self.args.noise_env:
            z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        else:
            raise ValueError("Don't get here!!!")
        return z, encoder_h, inputs, bs

    def _policy_from_z(self, inputs, z, hidden_state, bs, per_agent_z=False):
        raw_inputs, extra_inputs = self._build_inputs(inputs)
        weighted = torch.sigmoid(self.gate(inputs))
        if per_agent_z:
            repeated_z = z
        else:
            repeated_z = z.unsqueeze(1).repeat(1, self.args.n_agents, 1).reshape(
                bs * self.args.n_agents, -1
            )
        weighted_z = weighted * repeated_z

        if self.args.concat_obs:
            ob_embed = self.ob_fc(raw_inputs)
            action_inputs = torch.cat([ob_embed, weighted_z, extra_inputs], dim=-1)
        else:
            action_inputs = torch.cat([weighted_z, extra_inputs], dim=-1)

        x = F.relu(self.fc1(action_inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h

    def forward(self, inputs, hidden_state, encoder_hidden_state):
        z, encoder_h, gated_inputs, bs = self._encode_messages(inputs, encoder_hidden_state)
        per_agent_z = bool(self.args.noise_env and self.args.noise_type == 1)
        q, h = self._policy_from_z(
            gated_inputs, z, hidden_state, bs, per_agent_z=per_agent_z
        )
        return q, h, encoder_h

    def enc_forward(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.args.n_agents
        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        if "ae" not in self.args.state_encoder:
            raise ValueError("Unknown encoder!!!")

        if self.args.noise_env:
            noise = torch.randn(bs * self.args.n_agents, inputs.shape[-1], device=inputs.device)
            inputs = inputs + noise
        z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        return z, encoder_h

    def vae_forward(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.args.n_agents
        if self.args.noise_env:
            noise = torch.randn(bs * self.args.n_agents, inputs.shape[-1], device=inputs.device)
            inputs = inputs + noise
        return self.encoder(inputs, encoder_hidden_state)

    def rl_forward(self, inputs, state_repr, hidden_state):
        bs = inputs.shape[0] // self.args.n_agents
        q, h = self._policy_from_z(inputs, state_repr, hidden_state, bs, per_agent_z=False)
        return q, h

    def online_transform(self, inputs, encoder_hidden_state):
        bs = inputs.shape[0] // self.args.n_agents
        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        if "ae" not in self.args.state_encoder:
            raise ValueError("Unknown encoder!!!")
        if self.args.noise_env:
            noise = torch.randn(bs * self.args.n_agents, inputs.shape[-1], device=inputs.device)
            inputs = inputs + noise
        z, encoder_h = self.encoder.encode(inputs, encoder_hidden_state)
        projected = self.projection(z)
        predicted = self.final_classifier(projected)
        return predicted, encoder_h

    def online_projection(self, z):
        projected = self.projection(z)
        return self.final_classifier(projected)

    def target_transform(self, inputs, encoder_hidden_state):
        if "vae" in self.args.state_encoder:
            raise NotImplementedError
        if "ae" not in self.args.state_encoder:
            raise ValueError("Unknown encoder!!!")
        if self.args.noise_env:
            noise = torch.randn(*inputs.shape, device=inputs.device)
            inputs = inputs + noise
        z, encoder_h = self.target_encoder.encode(inputs, encoder_hidden_state)
        return self.target_projection(z), encoder_h

    def momentum_update(self):
        if not self.args.use_momentum_encoder:
            return
        _update_state_dict(
            self.target_encoder, self.encoder.state_dict(), self.args.momentum_tau
        )
        _update_state_dict(
            self.target_projection, self.projection.state_dict(), self.args.momentum_tau
        )

    def _build_inputs(self, inputs):
        base_inputs = inputs[:, : self.raw_input_shape]
        extra_inputs = inputs[:, self.raw_input_shape :]
        return base_inputs, extra_inputs

    def _get_input_shape(self, input_shape):
        if self.args.obs_last_action:
            input_shape -= self.args.n_actions
        if self.args.obs_agent_id:
            input_shape -= self.args.n_agents
        return input_shape
