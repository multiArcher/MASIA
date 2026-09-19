"""MASIA learner: self-supervised aggregation loss plus QMIX/VDN TD learning.

The original official constructor took an extra ``latent_model`` argument, but
that repo's ``run.py`` still called ``Learner(mac, scheme, logger, args)``.
This learner matches this repository's maker/run signature and builds the
latent model internally so ``run`` / ``main`` stay unchanged.
"""

from __future__ import annotations

import copy
import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from learners.learner import Learner
from modules.agents.masia_agent import TransitionModel
from utils.maker import MixerMaker


class MASIALearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger
        self.device = args.device

        if not self.args.rl_signal:
            raise AssertionError("Must use rl signal in this method !!!")

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = MixerMaker.make(args.mixer, args)
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.latent_model = None
        if self.args.use_latent_model:
            self.latent_model = TransitionModel(args)
            self.params += list(self.latent_model.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=self.device)
        if self.args.standardise_rewards:
            rew_shape = (1,) if self.args.common_reward else (self.n_agents,)
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)

    def repr_train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        states = batch["state"]
        actions_onehot = batch["actions_onehot"]
        rewards = batch["reward"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)
        recons = th.stack(recons, dim=1)
        z = th.stack(z, dim=1)

        bs, seq_len = states.shape[0], states.shape[1]
        loss_dict = self.mac.agent.encoder.loss_function(
            recons.reshape(bs * seq_len, -1), states.reshape(bs * seq_len, -1)
        )
        vae_loss = loss_dict["loss"].reshape(bs, seq_len, 1)
        mask = mask.expand_as(vae_loss)
        masked_vae_loss = (vae_loss * mask).sum() / mask.sum()

        tot_spr_loss = None
        tot_rew_loss = None
        if self.args.use_latent_model:
            target_projected = []
            with th.no_grad():
                self.mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_projected.append(self.mac.target_transform(batch, t))
            target_projected = th.stack(target_projected, dim=1)

            curr_z = z
            predicted_f = self.mac.agent.online_projection(curr_z)
            tot_spr_loss = self.compute_spr_loss(predicted_f, target_projected, mask)
            if self.args.use_rew_pred:
                predicted_rew = self.latent_model.predict_reward(curr_z)
                tot_rew_loss = self.compute_rew_loss(predicted_rew, rewards, mask)
            for t in range(self.args.pred_len):
                curr_z = self.latent_model(curr_z, actions_onehot[:, t:])[:, :-1]
                predicted_f = self.mac.agent.online_projection(curr_z)
                tot_spr_loss = tot_spr_loss + self.compute_spr_loss(
                    predicted_f, target_projected[:, t + 1 :], mask[:, t + 1 :]
                )
                if self.args.use_rew_pred:
                    predicted_rew = self.latent_model.predict_reward(curr_z)
                    tot_rew_loss = tot_rew_loss + self.compute_rew_loss(
                        predicted_rew, rewards[:, t + 1 :], mask[:, t + 1 :]
                    )

            if self.args.use_rew_pred:
                repr_loss = (
                    masked_vae_loss
                    + self.args.spr_coef * tot_spr_loss
                    + self.args.rew_pred_coef * tot_rew_loss
                )
            else:
                repr_loss = masked_vae_loss + self.args.spr_coef * tot_spr_loss
        else:
            repr_loss = masked_vae_loss

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss/repr_loss", repr_loss.item(), t_env)
            self.logger.log_stat("loss/vae_loss", masked_vae_loss.item(), t_env)
            if self.args.use_latent_model and tot_spr_loss is not None:
                self.logger.log_stat("loss/model_loss", tot_spr_loss.item(), t_env)
                if self.args.use_rew_pred and tot_rew_loss is not None:
                    self.logger.log_stat("loss/rew_pred_loss", tot_rew_loss.item(), t_env)

        return repr_loss

    def compute_rew_loss(self, pred_rew, env_rew, mask):
        mask = mask.squeeze(-1)
        rew_loss = F.mse_loss(pred_rew, env_rew, reduction="none").sum(-1)
        return (rew_loss * mask).sum() / mask.sum()

    def compute_spr_loss(self, pred_f, target_f, mask):
        mask = mask.squeeze(-1)
        spr_loss = F.mse_loss(pred_f, target_f, reduction="none").sum(-1)
        return (spr_loss * mask).sum() / mask.sum()

    def rl_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, repr_loss):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.mac.enc_forward(batch, t=t)
            if not self.args.rl_signal:
                state_repr_t = state_repr_t.detach()
            agent_outs = self.mac.rl_forward(batch, state_repr_t, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            state_repr_t = self.target_mac.enc_forward(batch, t=t)
            target_agent_outs = self.target_mac.rl_forward(batch, state_repr_t, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals.detach()

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        td_error = chosen_action_qvals - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        rl_loss = (masked_td_error ** 2).sum() / mask.sum()
        tot_loss = rl_loss + self.args.repr_coef * repr_loss

        self.optimiser.zero_grad()
        tot_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if (
            self.args.target_update_interval_or_tau > 1
            and (self.training_steps - self.last_target_update_step)
            / self.args.target_update_interval_or_tau
            >= 1.0
        ):
            self._update_targets_hard()
            self.mac.agent.momentum_update()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)
            self.mac.agent.momentum_update()

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss/rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("loss/tot_loss", tot_loss.item(), t_env)
            self.logger.log_stat(
                "running/grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
                t_env,
            )
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "q_values/td_error_abs",
                masked_td_error.abs().sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat(
                "q_values/q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.logger.log_stat(
                "q_values/target_mean",
                (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                t_env,
            )
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        repr_loss = self.repr_train(batch, t_env, episode_num)
        self.rl_train(batch, t_env, episode_num, repr_loss)

    def test_encoder(self, batch: EpisodeBatch):
        states = batch["state"]
        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        recons, z = [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            recons_t, _, z_t = self.mac.vae_forward(batch, t)
            recons.append(recons_t)
            z.append(z_t)

        encoder_result = {
            "recons": th.stack(recons, dim=1),
            "z": th.stack(z, dim=1),
            "states": states,
            "mask": mask,
        }
        th.save(encoder_result, os.path.join(self.args.encoder_result_direc, "result.pth"))

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(
                self.target_mixer.parameters(), self.mixer.parameters()
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + param.data * tau
                )

    def cuda(self):
        self.mac.to(self.args.device)
        self.target_mac.to(self.args.device)
        if self.latent_model is not None:
            self.latent_model.to(self.args.device)
        if self.mixer is not None:
            self.mixer.to(self.args.device)
            self.target_mixer.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        if self.latent_model is not None:
            th.save(self.latent_model.state_dict(), "{}/latent_model.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage)
            )
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage)
        )
        if self.latent_model is not None:
            self.latent_model.load_state_dict(
                th.load(
                    "{}/latent_model.th".format(path),
                    map_location=lambda storage, loc: storage,
                )
            )
