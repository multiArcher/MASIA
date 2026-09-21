"""CACOM's one-step QMIX loss, auxiliary prediction and separate gate training.

Adapted from LXXXXR/CACOM, commit 97493a0b2c402e88a06d4e0d21327c41bbd21709
(Apache-2.0). Integration fixes are documented in docs/cacom.md.
"""

import copy
from pathlib import Path

import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop

from learners.learner import Learner
from utils.maker import MixerMaker


class CACOMLearner(Learner):
    def __init__(self, mac, scheme, logger, args):
        if not args.common_reward:
            raise ValueError("CACOM QMIX requires common_reward=True")
        if getattr(args, "standardise_rewards", False) or getattr(args, "standardise_returns", False):
            raise ValueError("Released CACOM uses unstandardised rewards and returns")
        if args.mixer not in (None, "qmix", "vdn"):
            raise ValueError("CACOM supports qmix, vdn, or null mixer")
        if args.train_gate_intervel <= 0 or args.target_update_interval <= 0:
            raise ValueError("CACOM training intervals must be positive")
        self.args, self.mac, self.logger = args, mac, logger
        self.device = args.device
        self.target_mac = copy.deepcopy(mac)
        self.mixer = MixerMaker.make(args.mixer, args) if args.mixer else None
        self.target_mixer = copy.deepcopy(self.mixer)
        # MAC is now nn.Module: explicitly exclude the independently trained gate.
        self.params = list(mac.agent.parameters())
        if self.mixer is not None:
            self.params += list(self.mixer.parameters())
        self.gate_params = list(mac.gate_parameters())
        self.optimiser = RMSprop(self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.gate_optimizer = RMSprop(self.gate_params, lr=args.gate_lr,
                                     alpha=args.optim_alpha, eps=args.optim_eps)
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1
        self.train_gate_t = -args.train_gate_intervel - 1

    def train(self, batch, t_env, episode_num):
        self.mac.train()
        # Replay observation delay follows training settings, also for targets.
        self.target_mac.train()
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - terminated[:, :-1]
        if mask.sum() == 0:
            return
        all_through = t_env < self.args.start_train_gate
        outputs, aux, freqs = [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q, losses, freq = self.mac.forward(batch, t, all_through=all_through,
                                              train_mode=t < batch.max_seq_length - 1)
            outputs.append(q)
            if "aux_loss" in losses:
                aux.append(losses["aux_loss"])
            freqs.append(freq)
        mac_out = th.stack(outputs, dim=1)
        chosen = th.gather(mac_out[:, :-1], 3, actions).squeeze(3)
        with th.no_grad():
            self.target_mac.init_hidden(batch.batch_size)
            # Match upstream: target/rollout paths use the gate even in warm-up.
            target_out = th.stack([
                self.target_mac.forward(batch, t)[0] for t in range(batch.max_seq_length)
            ], dim=1)[:, 1:]
            avail = batch["avail_actions"][:, 1:]
            target_out = target_out.masked_fill(avail == 0, -9999999)
            if self.args.double_q:
                greedy = mac_out[:, 1:].detach().masked_fill(avail == 0, -9999999).argmax(-1, keepdim=True)
                target_max = target_out.gather(3, greedy).squeeze(3)
            else:
                target_max = target_out.max(-1).values
            if self.target_mixer is not None:
                target_max = self.target_mixer(target_max, batch["state"][:, 1:])
            targets = rewards + self.args.gamma * (1 - terminated) * target_max
        if self.mixer is not None:
            chosen = self.mixer(chosen, batch["state"][:, :-1])
        td_error = chosen - targets
        td_mask = mask.expand_as(td_error)
        td_loss = (td_error * td_mask).square().sum() / td_mask.sum()
        aux_loss = td_loss.new_zeros(())
        if aux:
            aux_values = th.stack(aux, dim=1)
            aux_mask = mask.expand_as(aux_values)
            aux_loss = (aux_values * aux_mask).sum() / aux_mask.sum()
        loss = td_loss + aux_loss
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        gate_loss = None
        if t_env - self.train_gate_t >= self.args.train_gate_intervel:
            gate_loss = self._train_gate(batch, mask, all_through)
            self.train_gate_t = t_env
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            stats = {"loss/total_loss": loss, "loss/td_loss": td_loss,
                     "loss/aux_loss": aux_loss, "running/grad_norm": grad_norm,
                     "communication/reply_freq": th.stack(freqs).mean(),
                     "q_values/td_error_abs": (td_error.abs() * td_mask).sum() / td_mask.sum()}
            if gate_loss is not None:
                stats["loss/gate_loss"] = gate_loss
            for key, value in stats.items():
                self.logger.log_stat(key, value.item(), t_env)
            self.log_stats_t = t_env
        # No graph needs to survive a replay update or enter a whole-MAC snapshot.
        self.mac.hidden_states = None
        self.target_mac.hidden_states = None

    def _train_gate(self, batch, mask, all_through):
        self.mac.init_hidden(batch.batch_size)
        logits, differences = [], []
        for t in range(batch.max_seq_length - 1):
            diff, prob = self.mac.forward_gate(batch, t)
            differences.append(diff)
            logits.append(prob)
        differences = th.stack(differences, dim=1)
        logits = th.stack(logits, dim=1)
        labels = (th.zeros_like(differences, dtype=th.long) if all_through else
                  (differences <= self.args.cut_off_threshold).long())
        errors = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1), reduction="none")
        gate_mask = mask.expand_as(differences).reshape(-1)
        loss = (errors * gate_mask).sum() / gate_mask.sum()
        self.gate_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.gate_params, self.args.grad_norm_clip)
        self.gate_optimizer.step()
        return loss.detach()

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        with th.no_grad():
            for target, live in zip(self.target_mac.parameters(), self.mac.parameters()):
                target.lerp_(live, tau)
            if self.mixer is not None:
                for target, live in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                    target.lerp_(live, tau)

    def to(self, device):
        self.device = device
        for module in (self.mac, self.target_mac, self.mixer, self.target_mixer):
            if module is not None:
                module.to(device)
        for optimiser in (self.optimiser, self.gate_optimizer):
            for state in optimiser.state.values():
                for key, value in state.items():
                    if th.is_tensor(value):
                        state[key] = value.to(device)
        return self

    def cuda(self):
        return self.to(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), f"{path}/mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}/opt.th")
        th.save(self.gate_optimizer.state_dict(), f"{path}/gate_opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load(f"{path}/mixer.th", map_location="cpu", weights_only=True))
        self.optimiser.load_state_dict(th.load(f"{path}/opt.th", map_location="cpu", weights_only=True))
        # Old upstream checkpoints contain no gate optimizer state.
        if (Path(path) / "gate_opt.th").exists():
            self.gate_optimizer.load_state_dict(th.load(f"{path}/gate_opt.th", map_location="cpu", weights_only=True))
        self._update_targets_hard()
        self.to(self.device)
