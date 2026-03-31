"""
Voice: Convergent Latent Deliberation

For use with RoPE-based models (Llama, Mistral, Qwen, etc.).
RoPE applies positional encoding inside attention layers, not to inputs_embeds,
so recycled hidden states don't get double-encoded with positional information.

Fixed version addressing:
1. KV-cache decoding instead of hidden/embedding concatenation
2. Momentum variable reference-before-assignment bug
3. Mean pooling for convergence measurement instead of last-token only
4. Layer norm on recycling for numerical stability
5. think() returns inputs_embeds that produced final_hidden (no off-by-one)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from typing import Optional, List, Tuple

Outputs = namedtuple("Outputs", ["loss", "final_hidden", "logits", "n_iters", "converged"])


class Voice(nn.Module):
    """
    A single voice: recycles its hidden state until convergence,
    then decodes to tokens via KV cache.
    """

    def __init__(
        self,
        base_causallm,
        eos_token_id: int,
        max_iters: int = 32,
        convergence_threshold: float = 0.999,
        momentum: float = 0.0,
    ):
        super().__init__()
        self.base_causallm = base_causallm
        self.eos_token_id = eos_token_id
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.momentum = momentum

        self.hidden_size = base_causallm.config.hidden_size

        self.embedding = self._get_embedding()

        # Learned projection: hidden space → embedding space
        self.recycle_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.eye_(self.recycle_proj.weight)

        # Layer norm after projection — critical stability fix
        self.recycle_norm = nn.LayerNorm(self.hidden_size)

        # Convergence probe: learned "EOS in latent space"
        # Start with this disabled (set use_probe=False) for H1 experiments
        self.convergence_probe = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.use_probe = False  # off by default; enable after H1

    def _get_embedding(self):
        if hasattr(self.base_causallm, "model") and hasattr(
            self.base_causallm.model, "embed_tokens"
        ):
            return self.base_causallm.model.embed_tokens  # Llama, Mistral, Qwen
        return self.base_causallm.get_input_embeddings()

    def _recycle(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Project hidden state back to embedding space.
        For RoPE models, no positional correction is needed — PE is applied
        inside attention layers via rotary embeddings, not to inputs_embeds.
        """
        projected = self.recycle_proj(hidden)
        return self.recycle_norm(projected)

    def _forward_pass(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ):
        """Single forward pass. Returns hidden states, logits, optional KV cache."""
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        hidden = outputs.hidden_states[-1]
        past_kv = outputs.past_key_values if use_cache else None
        return hidden, outputs.logits, past_kv

    def think(
        self,
        prompt_embeds: torch.Tensor,
        voice_embed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, bool, List[torch.Tensor]]:
        """
        Core primitive: recycle hidden states until convergence.

        Uses mean pooling for convergence measurement.
        Uses cosine similarity + optional learned probe for stopping.

        Returns:
            final_hidden: the converged hidden state
            final_input: the inputs_embeds that PRODUCED final_hidden
                         (use this for KV-cache decoding, not _recycle(final_hidden))
            n_iters: how many iterations it took
            converged: whether convergence criterion was met
            trajectory: list of mean-pooled hidden states (for analysis)
        """
        # Apply voice as additive bias
        if voice_embed is not None:
            inputs_embeds = prompt_embeds + voice_embed
        else:
            inputs_embeds = prompt_embeds

        prev_pooled = None
        prev_hidden = None  # for momentum — initialized before use
        trajectory = []

        for i in range(self.max_iters):
            hidden, _, _ = self._forward_pass(inputs_embeds, attention_mask)

            # Mean pool for convergence (not last-token — avoids positional bias)
            pooled = hidden.mean(dim=1, keepdim=True)  # (1, 1, hidden_size)
            trajectory.append(pooled.detach())

            # Check convergence
            if prev_pooled is not None:
                cos_sim = F.cosine_similarity(
                    pooled.view(1, -1),
                    prev_pooled.view(1, -1),
                )

                converged = cos_sim.item() > self.convergence_threshold

                # Optional learned probe
                if self.use_probe and converged:
                    probe_input = torch.cat(
                        [pooled.view(1, -1), prev_pooled.view(1, -1)], dim=-1
                    )
                    probe_score = self.convergence_probe(probe_input)
                    converged = converged and probe_score.item() > 0.5

                if converged:
                    # Return inputs_embeds that produced this hidden state
                    return hidden, inputs_embeds, i + 1, True, trajectory

            # Recycle: project back to embedding space
            recycled = self._recycle(hidden)

            # Momentum: blend with previous recycled state
            # Note: prev_hidden is assigned BEFORE this block uses it
            if prev_hidden is not None and self.momentum > 0:
                prev_recycled = self._recycle(prev_hidden)
                recycled = (1 - self.momentum) * recycled + self.momentum * prev_recycled

            prev_pooled = pooled
            prev_hidden = hidden  # save for next iteration's momentum

            # Re-apply voice identity each iteration
            if voice_embed is not None:
                inputs_embeds = recycled + voice_embed
            else:
                inputs_embeds = recycled

        return hidden, inputs_embeds, self.max_iters, False, trajectory

    def generate(
        self,
        input_ids: torch.Tensor,
        voice_embed: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
    ) -> Tuple[List[int], int]:
        """
        Think in latent space until converged, then decode via KV cache.
        Only tokenizes ONCE — at the end.
        """
        prompt_embeds = self.embedding(input_ids)

        # Think until converged
        # final_input is the inputs_embeds that PRODUCED final_hidden —
        # decode from this, not from _recycle(final_hidden) which would
        # be one extra iteration past convergence.
        final_hidden, final_input, n_iters, converged, trajectory = self.think(
            prompt_embeds, voice_embed
        )

        # Forward pass with KV cache for decoding, using the exact input
        # that produced the converged state
        _, logits, past_kv = self._forward_pass(final_input, use_cache=True)

        # Decode autoregressively from KV cache
        tokens = []
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        for _ in range(max_new_tokens):
            if next_token.item() == self.eos_token_id:
                break
            tokens.append(next_token.item())

            next_embed = self.embedding(next_token).unsqueeze(1)
            outputs = self.base_causallm(
                inputs_embeds=next_embed,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        return tokens, n_iters


class Quorum(nn.Module):
    """
    Multiple voices deliberating over shared state.
    Each voice thinks, produces a delta, deltas are aggregated
    weighted by magnitude (louder voices count more).
    """

    def __init__(
        self,
        voice: Voice,
        n_rounds: int = 4,
        convergence_threshold: float = 0.999,
    ):
        super().__init__()
        self.voice = voice
        self.n_rounds = n_rounds
        self.convergence_threshold = convergence_threshold

    def deliberate(
        self,
        prompt_embeds: torch.Tensor,
        voice_embeds: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """
        Voices take turns thinking, sharing deltas weighted by magnitude.

        Returns:
            consensus_hidden: the converged shared state
            individual_hiddens: each voice's final state (for dissent report)
            rounds_taken: how many rounds of deliberation
        """
        n_voices = len(voice_embeds)
        shared_state = prompt_embeds

        for round_idx in range(self.n_rounds):
            s0 = shared_state
            deltas = []
            individual_hiddens = []

            # Each voice thinks from current shared state
            for voice_embed in voice_embeds:
                hidden, _, _, _, _ = self.voice.think(
                    shared_state, voice_embed, attention_mask
                )
                individual_hiddens.append(hidden)
                delta = hidden - s0
                deltas.append(delta)

            # Weighted mean: voices that moved more count more
            delta_magnitudes = torch.tensor(
                [d.norm().item() for d in deltas], device=prompt_embeds.device
            )
            if delta_magnitudes.sum() > 0:
                weights = delta_magnitudes / delta_magnitudes.sum()
            else:
                weights = torch.ones(n_voices, device=prompt_embeds.device) / n_voices

            weighted_delta = sum(w * d for w, d in zip(weights, deltas))
            shared_state = s0 + weighted_delta

            # Check inter-voice agreement
            if len(deltas) > 1:
                delta_flat = [d.mean(dim=1).view(1, -1) for d in deltas]
                agreements = []
                for i in range(n_voices):
                    for j in range(i + 1, n_voices):
                        sim = F.cosine_similarity(delta_flat[i], delta_flat[j])
                        agreements.append(sim.item())

                mean_agreement = sum(agreements) / len(agreements)
                max_delta_mag = delta_magnitudes.max().item()

                if mean_agreement > self.convergence_threshold or max_delta_mag < 1e-3:
                    return shared_state, individual_hiddens, round_idx + 1

        return shared_state, individual_hiddens, self.n_rounds

    def deliberate_active(
        self,
        prompt_embeds: torch.Tensor,
        voice_embeds: List[torch.Tensor],
        predicted_perturbations: Optional[List[float]] = None,
        top_k: int = 5,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[int], int]:
        """
        Active voice selection: only run the most perturbed voices.
        """
        if predicted_perturbations is not None:
            ranked = sorted(
                enumerate(predicted_perturbations),
                key=lambda x: x[1],
                reverse=True,
            )
            selected_indices = [idx for idx, _ in ranked[:top_k]]
            selected_embeds = [voice_embeds[i] for i in selected_indices]
        else:
            selected_indices = list(range(len(voice_embeds)))
            selected_embeds = voice_embeds

        consensus, individuals, rounds = self.deliberate(
            prompt_embeds, selected_embeds, attention_mask
        )
        return consensus, selected_indices, rounds


class VoiceRegistry:
    """
    Store and compose voice embeddings.
    Voices are vectors. Composition is addition. Storage is cheap.
    """

    def __init__(self, hidden_size: int, device: str = "cpu"):
        self.hidden_size = hidden_size
        self.device = device
        self.voices = {}
        self.impulse_cache = {}

    def register(self, name: str, embedding: torch.Tensor):
        assert embedding.shape[-1] == self.hidden_size
        self.voices[name] = embedding.to(self.device)

    def get(self, name: str) -> torch.Tensor:
        return self.voices[name]

    def compose(self, *names: str, weights: Optional[List[float]] = None) -> torch.Tensor:
        if weights is None:
            weights = [1.0] * len(names)
        result = torch.zeros(1, 1, self.hidden_size, device=self.device)
        for name, w in zip(names, weights):
            result = result + w * self.voices[name]
        return result

    def centroid(self) -> torch.Tensor:
        all_vecs = torch.stack(list(self.voices.values()))
        return all_vecs.mean(dim=0, keepdim=True)

    def sample_on_manifold(self, n_samples: int) -> List[torch.Tensor]:
        """
        Sample n voices on the subspace defined by registered voices.
        Center = centroid, radius = mean distance from centroid.
        Samples on the hypersphere within the spanned subspace,
        not the full embedding space.
        """
        if len(self.voices) < 2:
            # Cannot define a subspace with < 2 voices; perturb the single voice
            center = list(self.voices.values())[0]
            return [center + torch.randn_like(center) * 1e-4 for _ in range(n_samples)]

        all_vecs = torch.stack([v.view(-1) for v in self.voices.values()])
        center = all_vecs.mean(dim=0)
        radius = (all_vecs - center).norm(dim=-1).mean()

        # Compute the subspace spanned by the voices (relative to centroid)
        centered = all_vecs - center
        # SVD gives us the principal directions of the voice manifold
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        # Vt rows are the basis vectors of the subspace
        n_dims = min(len(self.voices) - 1, Vt.shape[0])  # rank of subspace
        basis = Vt[:n_dims]  # (n_dims, hidden_size)

        samples = []
        for _ in range(n_samples):
            # Random direction in the subspace
            coeffs = torch.randn(n_dims, device=self.device)
            coeffs = coeffs / coeffs.norm()
            direction = (coeffs @ basis)  # project into full space
            direction = direction / direction.norm()

            point = center + radius * direction
            samples.append(point.view(1, 1, -1))

        return samples

    def cache_impulse(
        self, voice_name: str, prompt_hash: int, delta: torch.Tensor
    ):
        self.impulse_cache[(voice_name, prompt_hash)] = delta.detach()

    def predict_perturbation(
        self, voice_name: str, prompt_hash: int
    ) -> float:
        key = (voice_name, prompt_hash)
        if key in self.impulse_cache:
            return self.impulse_cache[key].norm().item()
        return float("inf")
