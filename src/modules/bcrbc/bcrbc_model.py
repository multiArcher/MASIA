import torch
import torch.nn as nn

from modules.bcrbc.agent_readout import AgentReadout
from modules.bcrbc.block_causal_transformer import BlockCausalTransformer
from modules.bcrbc.dynamics_tokenizer import DynamicsTokenizer
from modules.bcrbc.observation_tokenizer import ObservationDecoder, ObservationEncoder
from modules.bcrbc.q_head import BCRBCQHead


class BCRBCModel(nn.Module):
    def __init__(self, observation_dim, n_actions, n_agents, args):
        super().__init__()

        model_hidden_dim = args.bcrbc_d_model
        agent_output_dim = args.bcrbc_agent_output_dim
        transformer_depth = args.bcrbc_depth
        time_block_every = args.bcrbc_time_block_every
        attention_heads = args.bcrbc_heads
        z_dim = args.bcrbc_z_dim
        num_z_tokens = args.bcrbc_num_z_tokens

        self.z_dim = z_dim
        self.num_z_tokens = num_z_tokens
        self.observation_dim = observation_dim
        self.context_window = args.bcrbc_context_window
        self.encoder_context_window = getattr(args, "bcrbc_encoder_context_window", None)
        if self.encoder_context_window is None:
            self.encoder_context_window = self.context_window
        self.flow_steps = args.bcrbc_flow_steps
        self.generation_horizon = args.bcrbc_generation_horizon
        self.rec_loss_enabled = args.rec_loss_weight > 0
        # N=0 is the pure-MASK ablation, including its auxiliary objectives.
        self.flow_loss_enabled = self.generation_horizon != 0 and args.flow_loss_weight > 0
        self.generated_rec_loss_enabled = self.generation_horizon != 0 and args.generated_rec_loss_weight > 0

        self.observation_encoder = ObservationEncoder(
            observation_dim,
            model_hidden_dim,
            z_dim,
            num_z_tokens,
            transformer_depth,
            attention_heads,
            context_window=self.encoder_context_window,
            time_block_every=time_block_every,
        )
        self.observation_decoder = ObservationDecoder(
            observation_dim,
            model_hidden_dim,
            z_dim,
            num_z_tokens,
            transformer_depth,
            attention_heads,
            context_window=self.context_window,
            time_block_every=time_block_every,
        )

        max_time_steps = args.env_info["episode_limit"] + 2
        self.dynamics_tokenizer = DynamicsTokenizer(
            z_dim,
            n_actions,
            n_agents,
            model_hidden_dim,
            max_t=max_time_steps,
            num_z_tokens=num_z_tokens,
        )
        self.transformer = BlockCausalTransformer(
            model_hidden_dim,
            transformer_depth,
            attention_heads,
            dropout=args.bcrbc_dropout,
            agent_slice=self.dynamics_tokenizer.query_slice,
            context_window=self.context_window,
            time_block_every=time_block_every,
        )
        self.z_output_norm = nn.LayerNorm(model_hidden_dim)
        self.z_predictor = nn.Linear(model_hidden_dim, z_dim)
        self.agent_readout = AgentReadout(model_hidden_dim, agent_output_dim)
        self.q_head = BCRBCQHead(agent_output_dim, n_actions, args.bcrbc_q_hidden_dim)

    def encode_observations(self, observations, **kwargs):
        return self.observation_encoder(observations, **kwargs)

    def decode_observations(self, z, **kwargs):
        return self.observation_decoder(z, **kwargs)

    @torch.no_grad()
    def prepare_condition(self, history_z, previous_actions, start_t=0, kv_cache=None):
        """Build detached MASK-history KV once for all current queries."""
        tokens = self.dynamics_tokenizer(
            history_z, previous_actions, torch.ones_like(history_z[..., :1, :1]),
            start_t=start_t,
        )
        return self.transformer.prepare_condition(
            tokens, start_t=start_t, kv_cache=kv_cache, detach=True,
        )

    def estimate_clean_z(
        self,
        noisy_z,
        previous_actions,
        signal_levels,
        start_t=0,
        kv_cache=None,
        use_kv_cache=False,
        rope_offset=None,
        condition=None,
        compute_q=True,
    ):
        tokens = self.dynamics_tokenizer(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
        )
        if condition is not None:
            transformer_outputs = self.transformer.query_condition(
                tokens, condition, start_t=start_t, return_cache=use_kv_cache,
            )
            if use_kv_cache:
                transformer_outputs, new_kv_cache = transformer_outputs
            else:
                new_kv_cache = None
        elif use_kv_cache:
            transformer_outputs, new_kv_cache = self.transformer(
                tokens, kv_cache=kv_cache, use_kv_cache=True, rope_offset=rope_offset,
            )
        else:
            transformer_outputs = self.transformer(tokens)
            new_kv_cache = None

        z_outputs = transformer_outputs[..., self.dynamics_tokenizer.z_slice, :]
        normalized_z_outputs = self.z_output_norm(z_outputs)
        predicted_z = self.z_predictor(normalized_z_outputs)
        output = {
            "predicted_z": predicted_z,
            "kv_cache": new_kv_cache,
        }
        if compute_q:
            agent_outputs = self.agent_readout(
                transformer_outputs, self.dynamics_tokenizer.query_slice,
            )
            output["agent_outputs"] = agent_outputs
            output["q_values"] = self.q_head(agent_outputs)
        return output

    def complete_current(self, encoded_z, previous_actions, missing_mask,
                         noise, condition, start_t=0, use_kv_cache=False):
        """Generate current blocks against fixed conditions for this solver chain.

        Linear flow: x(s) = (1-s) noise + s target. The network predicts the
        clean endpoint, giving velocity (predicted_z - x) / (1-s).
        Solver iterations and Q share the same historical condition.
        """
        current_z = torch.where(missing_mask, noise, encoded_z)
        sampling = self.flow_steps > 0 and self.generation_horizon != 0
        if not sampling:
            current_z = encoded_z  # Pure-MASK ablation; no sampling noise enters Q.
        if sampling and missing_mask.any():
            for index in range(self.flow_steps):
                signal = torch.full_like(
                    encoded_z[..., :1, :1], index / self.flow_steps,
                )
                signal = torch.where(missing_mask, signal, 1.0)
                estimate = self.estimate_clean_z(
                    current_z, previous_actions, signal,
                    start_t=start_t, condition=condition, compute_q=False,
                )["predicted_z"]
                # dt / (1-s) = 1 / (K-index); last step reaches the endpoint.
                velocity_step = (estimate - current_z) / (self.flow_steps - index)
                updated_z = current_z + velocity_step
                current_z = torch.where(missing_mask, updated_z, encoded_z)
        output = self.estimate_clean_z(
            current_z, previous_actions,
            torch.ones_like(encoded_z[..., :1, :1]), start_t=start_t,
            condition=condition, use_kv_cache=use_kv_cache,
        )
        output["z"] = current_z
        return output

    def completion_horizon(self, time_steps):
        """Resolve Full and retain compatibility with the zero-solver MASK control."""
        if self.flow_steps == 0:
            return 0
        available = min(time_steps, self.context_window or time_steps)
        if self.generation_horizon < 0:
            return available
        return min(self.generation_horizon, available)

    def complete_horizon(self, encoded_z, previous_actions, missing_mask,
                         noise, condition, start_t=0, last_only=False):
        """Batch decision windows by completion depth; execute one window sequentially.

        At depth N, query t reads depth d at t-(N-d), for d=1..N-1.
        These blocks share the same MASK prefix ending at t-N. Intermediate
        completion is detached, matching truncated gradients through history.
        """
        horizon = self.completion_horizon(encoded_z.shape[1])
        history_condition = condition
        generated = []
        for depth in range(1, horizon):
            current = slice(depth - 1, depth) if last_only else slice(None)
            query_start = start_t + depth - 1 if last_only else start_t
            with torch.no_grad():
                support = self.complete_current(
                    encoded_z[:, current], previous_actions[:, current],
                    missing_mask[:, current], noise[:, current], condition,
                    start_t=query_start, use_kv_cache=True,
                )
            positions = torch.arange(
                query_start, query_start + support["z"].shape[1], device=encoded_z.device,
            )
            generated.append({"kv": support["kv_cache"], "positions": positions})
            sources = [history_condition, *generated]
            combined_kv = []
            for layer in range(len(history_condition["kv"])):
                keys = torch.cat([source["kv"][layer][0] for source in sources], dim=-2)
                values = torch.cat([source["kv"][layer][1] for source in sources], dim=-2)
                combined_kv.append((keys, values))
            condition = {
                "kv": combined_kv,
                "positions": torch.cat([source["positions"] for source in sources]),
                "generation_depth": torch.cat([
                    torch.full_like(source["positions"], index)
                    for index, source in enumerate(sources)
                ]),
                "generation_horizon": depth + 1,
            }
        current = slice(-1, None) if last_only else slice(None)
        query_start = start_t + encoded_z.shape[1] - 1 if last_only else start_t
        output = self.complete_current(
            encoded_z[:, current], previous_actions[:, current], missing_mask[:, current],
            noise[:, current], condition, start_t=query_start,
        )
        return output, condition

    def forward_training(self, observations, previous_actions,
                         missing_mask, start_t=0, completion_noise=None,
                         compute_aux=True, flow_signal=None, flow_noise=None):
        """Encode MASK history once and batch decisions over N-step completion windows."""
        history_z = self.encode_observations(
            observations, missing_mask=missing_mask, rope_offset=start_t
        )
        if completion_noise is None:
            completion_noise = torch.randn_like(history_z)
        condition = self.prepare_condition(history_z, previous_actions, start_t=start_t)
        output, condition = self.complete_horizon(
            history_z, previous_actions, missing_mask, completion_noise,
            condition, start_t=start_t,
        )
        output.update({
            "history_z": history_z,
            "missing_mask": missing_mask,
            "completion_noise": completion_noise,
        })
        if compute_aux:
            if self.rec_loss_enabled or self.flow_loss_enabled:
                with torch.set_grad_enabled(torch.is_grad_enabled() and self.rec_loss_enabled):
                    target_z = self.encode_observations(observations, rope_offset=start_t)
                output["target_z"] = target_z.detach()
            # Keep the random stream unchanged when the flow loss is disabled.
            if flow_signal is None:
                flow_signal = torch.rand_like(history_z[..., :1, :1])
            if flow_noise is None:
                flow_noise = torch.randn_like(history_z)
            if self.flow_loss_enabled:
                # Clean targets enter only this auxiliary query, never Q or history.
                noisy = torch.lerp(flow_noise, target_z.detach(), flow_signal)
                noisy = torch.where(missing_mask, noisy, history_z.detach())
                flow = self.estimate_clean_z(
                    noisy, previous_actions, torch.where(missing_mask, flow_signal, 1.0),
                    start_t=start_t, condition=condition, compute_q=False,
                )
                output["predicted_z"] = flow["predicted_z"]
            decoder_condition = None
            if self.rec_loss_enabled:
                output["reconstructed_observations"] = self.decode_observations(
                    target_z, rope_offset=start_t,
                )
                masked_rec = self.decode_observations(
                    history_z, rope_offset=start_t,
                    use_kv_cache=self.generated_rec_loss_enabled,
                )
                if self.generated_rec_loss_enabled:
                    masked_rec, decoder_cache = masked_rec
                    decoder_condition = {
                        "kv": [(key.detach(), value.detach()) for key, value in decoder_cache],
                        "positions": torch.arange(
                            start_t, start_t + history_z.shape[1], device=history_z.device,
                        ),
                    }
                output["masked_reconstructed_observations"] = masked_rec
            if self.generated_rec_loss_enabled:
                output["generated_reconstructed_observations"] = self.decode_observations(
                    output["z"], history_z=history_z, rope_offset=start_t,
                    condition=decoder_condition,
                )
        return output

    def forward(
        self,
        observations,
        previous_actions,
        start_t=0,
        z_override=None,
        noisy_z=None,
        signal_levels=None,
        kv_cache=None,
        use_kv_cache=False,
        rope_offset=None,
        reconstruct=True,
    ):
        z = (
            self.encode_observations(observations)
            if z_override is None
            else z_override
        )
        if noisy_z is None:
            noisy_z = z
        if signal_levels is None:
            signal_levels = z.new_ones(*z.shape[:3], 1, 1)

        dynamics_output = self.estimate_clean_z(
            noisy_z,
            previous_actions,
            signal_levels,
            start_t=start_t,
            kv_cache=kv_cache,
            use_kv_cache=use_kv_cache,
            rope_offset=rope_offset,
        )
        dynamics_output["z"] = z
        if reconstruct:
            dynamics_output["reconstructed_observations"] = self.decode_observations(z)
        return dynamics_output
