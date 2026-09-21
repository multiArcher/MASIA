"""CACOM networks adapted from LXXXXR/CACOM (Apache-2.0).
Upstream commit: 97493a0b2c402e88a06d4e0d21327c41bbd21709.
Two-stage attention, auxiliary Q prediction, counterfactual gate and LSQ retain
upstream equations. See docs/cacom.md for integration differences.
"""

import random

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .agent import Agent


class CACOMAgent(Agent):
    """Encode observation entities, broadcast requests, then personalize replies.

    Features: [B, N, segments, encode_dim]. Requests/replies before transposition:
    [B, sender, receiver, message_dim]. Only non-self replies enter the actor.
    """
    def __init__(self, input_shape, args):
        super(CACOMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.encode_dim = args.encode_dim
        self.request_dim = args.request_dim
        self.response_dim = args.response_dim
        self.obs_segs = args.obs_segs

        NN_HIDDEN_MULTI = args.nn_hidden_multi
        activation_func = nn.LeakyReLU()

        self.input_encoders = nn.ModuleList()
        obs_seg_dim = 0
        obs_seg_num = 0
        for seg_num, seg_len in self.obs_segs:
            obs_seg_dim = obs_seg_dim + seg_num * seg_len
            obs_seg_num = obs_seg_num + seg_num
            self.input_encoders.append(nn.Linear(seg_len, self.encode_dim))

        if obs_seg_dim != input_shape:
            raise ValueError(f"obs_segs covers {obs_seg_dim} features, input has {input_shape}")
        self.input_kqv = nn.Linear(self.encode_dim, self.encode_dim * 3)
        self.input_fc = nn.Sequential(
            nn.Linear(self.encode_dim, NN_HIDDEN_MULTI * self.encode_dim),
            activation_func,
            nn.Linear(NN_HIDDEN_MULTI * self.encode_dim, self.encode_dim),
        )
        self.request_generator = nn.Linear(
            args.hidden_dim + obs_seg_num * self.encode_dim, args.request_dim
        )

        self.response_kv = nn.Linear(self.encode_dim, 2 * self.encode_dim)
        self.response_q = nn.Linear(args.request_dim, args.encode_dim)
        self.response_fc = nn.Sequential(
            nn.Linear(self.encode_dim, NN_HIDDEN_MULTI * self.encode_dim),
            activation_func,
            nn.Linear(NN_HIDDEN_MULTI * self.encode_dim, args.response_dim),
        )

        self.actor_feat_kqv = nn.Sequential(
            activation_func, nn.Linear(self.encode_dim, 3 * self.encode_dim)
        )
        self.actor_msg_kqv = nn.Linear(args.response_dim, 3 * self.encode_dim)
        self.actor_linear = nn.Linear(
            (self.n_agents - 1 + obs_seg_num) * self.encode_dim, args.hidden_dim
        )
        self.actor_rnn = nn.GRUCell(
            input_size=args.hidden_dim, hidden_size=args.hidden_dim
        )
        self.actor_linear_2 = nn.Linear(args.hidden_dim, self.n_actions)

        self.pred_feat_kv = nn.Linear(self.encode_dim, 2 * self.encode_dim)
        self.pred_msg_kqv = nn.Linear(args.response_dim, 3 * self.encode_dim)
        self.pred_linear = nn.Linear(
            (self.n_agents - 1) * self.encode_dim, (self.n_agents - 1) * self.n_actions
        )

        if getattr(self.args, "discrete_bits", None) is not None:
            self.req_quan = LsqQuan(bit=self.args.discrete_bits)
            self.res_quan = LsqQuan(bit=self.args.discrete_bits)

    def init_hidden(self):
        return self.actor_linear.weight.new(1, self.args.hidden_dim).zero_()

    def forward(
        self,
        inputs,
        hidden_state,
        bs,
        exp_gate,
        all_through=False,
        test_mode=False,
        **kwargs
    ):
        feat, requests = self.pre_comm(inputs, hidden_state, bs)

        response = self.comm_response(feat, requests)
        _, response_mask = exp_gate(
            requests, feat, test_mode=True, all_through=all_through
        )
        response = response * response_mask

        response = response.permute((0, 2, 1, 3))
        mask = th.eye(self.n_agents, dtype=bool, device=feat.device)[
            None, :, :, None
        ].repeat(bs, 1, 1, self.response_dim)
        response = response.masked_select(~mask).reshape(
            (bs, self.n_agents, -1, self.response_dim)
        )

        if getattr(self.args, "discrete_bits", None) is not None:
            response = self.res_quan(response)

        h, return_q = self.after_comm(feat, response, hidden_state, bs)
        returns = {}
        if "train_mode" in kwargs and kwargs["train_mode"]:
            if hasattr(self.args, "pred_weight") and self.args.pred_weight > 0:
                returns["aux_loss"] = self.args.pred_weight * self.calculate_aux_loss(
                    bs, feat, response, return_q, reduction="none"
                )

        return return_q, h, returns, response_mask.float().mean().detach()

    def pre_comm(self, inputs, hidden_state, bs):
        obs_seg_dim = 0
        obs_encoded = []
        for i, (seg_num, seg_len) in enumerate(self.obs_segs):
            obs_seg = inputs[:, obs_seg_dim : obs_seg_dim + seg_num * seg_len].clone()
            obs_seg = obs_seg.reshape((bs, self.n_agents, seg_num, seg_len))
            obs_seg = self.input_encoders[i](obs_seg)

            obs_encoded.append(obs_seg)
            obs_seg_dim = obs_seg_dim + seg_num * seg_len

        obs_encoded = th.cat(obs_encoded, dim=2)
        obs_encoded = obs_encoded.reshape((bs, self.n_agents, -1, self.encode_dim))
        kqv = self.input_kqv(obs_encoded)
        k, q, v = th.chunk(kqv, 3, dim=-1)
        feat_scores = th.matmul(q, k.permute((0, 1, 3, 2))) / np.sqrt(self.encode_dim)
        feat_weights = F.softmax(feat_scores, dim=-1)[:, :, :, :, None]
        feat = (v[:, :, None, :, :] * feat_weights).sum(dim=-2)
        feat = obs_encoded + feat
        feat = feat + self.input_fc(feat)

        requests = self.request_generator(
            th.cat([hidden_state, feat.reshape((bs, self.n_agents, -1))], dim=-1)
        )
        if getattr(self.args, "discrete_bits", None) is not None:
            requests = self.req_quan(requests)
        requests = requests[:, None, :, :].repeat(1, self.n_agents, 1, 1)

        return feat, requests

    def comm_response(self, feat, requests):
        kv = self.response_kv(feat)
        k, v = th.chunk(kv, 2, dim=-1)
        q = self.response_q(requests).permute((0, 1, 3, 2))
        response_scores = th.matmul(k, q) / np.sqrt(self.encode_dim)
        response_weights = F.softmax(response_scores, dim=-2)
        response = (v[:, :, :, None, :] * response_weights[:, :, :, :, None]).sum(
            dim=-3
        )
        response = self.response_fc(response)

        return response

    def after_comm(self, feat, response, hidden_state, bs):
        msg_kqv = self.actor_msg_kqv(response)
        msg_k, msg_q, msg_v = th.chunk(msg_kqv, 3, dim=-1)
        kqv = self.actor_feat_kqv(feat)
        k, q, v = th.chunk(kqv, 3, dim=-1)
        k = th.cat([k, msg_k], dim=2).permute((0, 1, 3, 2))
        q = th.cat([q, msg_q], dim=2)
        v = th.cat([v, msg_v], dim=2)[:, :, None, :, :]
        scores = th.matmul(q, k) / np.sqrt(self.encode_dim)
        soft_weights = F.softmax(scores, dim=-1)[:, :, :, :, None]
        x = (
            (v * soft_weights)
            .sum(dim=-2)
            .reshape((bs, self.n_agents, -1, self.encode_dim))
        )
        # residual connection for feature
        x[:, :, : feat.shape[2], :] = feat + x[:, :, : feat.shape[2], :]

        x = self.actor_linear(x.reshape((bs * self.n_agents, -1)))
        hidden_state = hidden_state.reshape((bs * self.n_agents, -1))
        h = self.actor_rnn(x, hidden_state)
        h = h.reshape((bs, self.n_agents, -1))

        return_q = self.actor_linear_2(h).reshape((-1, self.n_actions))

        return h, return_q

    def cal_gate_labels(self, inputs, hidden_state, bs, exp_gate):
        with th.no_grad():
            feat, requests = self.pre_comm(inputs, hidden_state, bs)
            response_ori = self.comm_response(feat, requests)

        response_probs, response_mask = exp_gate(requests, feat, test_mode=False)
        response_mask = response_mask.detach()

        idx = random.randrange(self.n_agents)
        probs = response_probs[:, idx, [i for i in range(self.n_agents) if i != idx], :]

        with th.no_grad():
            response = response_ori * response_mask
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=feat.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if getattr(self.args, "discrete_bits", None) is not None:
                response = self.res_quan(response)
            h, _ = self.after_comm(feat, response, hidden_state, bs)

            response_mask[:, idx, :, :] = 1
            response = response_ori * response_mask
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=feat.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if getattr(self.args, "discrete_bits", None) is not None:
                response = self.res_quan(response)
            _, q_pos = self.after_comm(feat, response, hidden_state, bs)
            q_pos = q_pos.reshape((bs, self.n_agents, -1))

            response_mask[:, idx, :, :] = 0
            response = response_ori * response_mask
            response = response.permute((0, 2, 1, 3))
            mask = th.eye(self.n_agents, dtype=bool, device=feat.device)[
                None, :, :, None
            ].repeat(bs, 1, 1, self.response_dim)
            response = response.masked_select(~mask).reshape(
                (bs, self.n_agents, -1, self.response_dim)
            )
            if getattr(self.args, "discrete_bits", None) is not None:
                response = self.res_quan(response)
            _, q_neg = self.after_comm(feat, response, hidden_state, bs)
            q_neg = q_neg.reshape((bs, self.n_agents, -1))

        return h, probs, q_pos, q_neg, idx

    def calculate_aux_loss(self, bs, feat, response, return_q, reduction="mean"):
        """Predict other agents' detached Q vectors; optionally return [B, N]."""
        msg_kqv = self.pred_msg_kqv(response)
        msg_k, q, msg_v = th.chunk(msg_kqv, 3, dim=-1)
        kv = self.pred_feat_kv(feat)
        k, v = th.chunk(kv, 2, dim=-1)
        k = th.cat([k, msg_k], dim=2).permute((0, 1, 3, 2))
        v = th.cat([v, msg_v], dim=2)[:, :, None, :, :]
        scores = th.matmul(q, k) / np.sqrt(self.encode_dim)
        soft_weights = F.softmax(scores, dim=-1)[:, :, :, :, None]
        x = (v * soft_weights).sum(dim=-2).reshape((bs * self.n_agents, -1))
        x = self.pred_linear(x).reshape((bs, self.n_agents, -1, self.n_actions))
        return_q = return_q.reshape((bs, self.n_agents, self.n_actions))[
            :, None, :, :
        ].repeat(1, self.n_agents, 1, 1)
        mask = th.eye(self.n_agents, dtype=bool, device=feat.device)[
            None, :, :, None
        ].repeat(bs, 1, 1, self.n_actions)
        return_q = return_q.masked_select(~mask).reshape(
            (bs, self.n_agents, -1, self.n_actions)
        )
        loss = F.mse_loss(x, return_q.detach(), reduction="none").mean(dim=(-1, -2))
        if reduction == "mean":
            loss = loss.mean()

        return loss


class ExpGate(nn.Module):
    """Binary reply gate; class zero sends, class one suppresses communication."""
    def __init__(self, args):
        super(ExpGate, self).__init__()
        self.encode_dim = args.encode_dim
        self.obs_segs = args.obs_segs
        seq_len = 0
        for seg_num, _ in self.obs_segs:
            seq_len = seq_len + seg_num

        self.k = nn.Linear(self.encode_dim, self.encode_dim)
        self.q = nn.Linear(args.request_dim, self.encode_dim)
        self.gate = nn.Linear(seq_len, 2)

    def forward(self, requests, feat, test_mode, all_through=False):

        if all_through:
            probs = None
            masks = th.ones_like(requests[:, :, :, [0]])

        else:
            if test_mode:
                with th.no_grad():
                    k = self.k(feat).permute((0, 1, 3, 2))
                    q = self.q(requests)
                    weights = th.matmul(q, k)
                    probs = self.gate(weights)
                    probs = F.softmax(probs, dim=-1)[:, :, :, [0]]
                    masks = probs > 0.5
            else:
                k = self.k(feat).permute((0, 1, 3, 2))
                q = self.q(requests)
                weights = th.matmul(q, k)
                probs = self.gate(weights)
                masks = F.softmax(probs, dim=-1)[:, :, :, [0]] > 0.5

        return probs, masks


def grad_scale(s, scale):
    y = s
    y_grad = s * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad

class LsqQuan(nn.Module):
    """Upstream LSQ straight-through quantizer, adapted from zhutmost/lsq-net."""

    def __init__(self, bit=8, symmetric=True, per_channel=False):
        super().__init__()
        if not isinstance(bit, int) or isinstance(bit, bool) or bit < 2:
            raise ValueError("CACOM LSQ requires discrete_bits >= 2 (or null to disable)")
        if per_channel:
            raise ValueError("CACOM uses per-tensor LSQ")
        if symmetric:
            # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
            self.thd_neg = -(2 ** (bit - 1)) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
            self.thd_neg = -(2 ** (bit - 1))
            self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s = nn.Parameter(th.ones(1))

    def forward(self, x):
        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = grad_scale(self.s, s_grad_scale)

        x = x / s_scale
        x = th.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x

