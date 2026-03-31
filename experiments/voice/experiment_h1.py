"""
Voice Experiment 0: Does self-recursive hidden-state recycling preserve meaning?

Target model: Llama 3.2 1B (or any RoPE-based model)
RoPE models apply positional encoding inside attention layers via rotary
embeddings on Q and K, NOT added to inputs_embeds. This eliminates the
positional double-encoding confound entirely, giving a clean test of H1.

Hypothesis (H1): When a pretrained causal LM's hidden states are projected back
to embedding space and re-fed as input, the model converges to a semantically
stable state that remains usable for decoding.

Design:
- Frozen pretrained base LM (no fine-tuning)
- No quorum, no probe, no voice embeddings
- Recycle with identity-init projection + layer norm
- Convergence measured by cosine similarity + delta norm (mean-pooled)
- Decode at 0, 1, 2, 4, 8, 16 iterations via KV cache
- Compare semantic consistency and task loss against non-recycled baseline

Failure modes:
1. NUMERICAL_INSTABILITY — norms explode or collapse
2. SEMANTIC_COLLAPSE — output degenerates (empty, extreme entropy)
3. SEMANTIC_STABLE_NO_BENEFIT — coherent but identical to baseline
4. PREDICTION_SHIFTED — coherent and different (needs judge eval)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Tuple, Optional
import json


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class IterationSnapshot:
    """Metrics captured at a single recycle iteration."""
    n_iters: int
    decoded_text: str
    next_token: str
    next_token_prob: float
    mean_hidden_norm: float
    cosine_vs_previous: float  # -1 for iter 0
    cosine_vs_original: float
    logit_entropy: float
    top1_prob: float


@dataclass
class ExperimentResult:
    """Full result for one prompt across all iteration counts."""
    prompt: str
    baseline_text: str
    snapshots: List[IterationSnapshot]
    hidden_norms: List[float]
    cosine_trajectory: List[float]


# ============================================================================
# Core recycling mechanism
# ============================================================================

class RecycleCore(nn.Module):
    """
    Minimal hidden-state recycling for RoPE models.

    RoPE (Rotary Position Embeddings) applies positional information inside
    each attention layer as rotations on Q and K vectors, based on position_ids.
    Nothing is added to inputs_embeds. This means recycled hidden states
    passed as inputs_embeds do NOT get double-encoded with positional info,
    eliminating the confound that makes absolute-PE models (GPT-2) problematic.

    The recycling operation is:
        recycled = LayerNorm(Proj(hidden_state))
    where Proj is initialized as identity and LayerNorm prevents drift.
    """

    def __init__(self, base_model: AutoModelForCausalLM):
        super().__init__()
        self.base_model = base_model

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.hidden_size = base_model.config.hidden_size

        # The bridge: hidden space → embedding space
        # Identity init: first recycle ≈ identity, deviations are learned
        self.recycle_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.eye_(self.recycle_proj.weight)

        # Layer norm after projection — prevents magnitude drift across iterations
        self.recycle_norm = nn.LayerNorm(self.hidden_size)

    def _get_embedding_layer(self):
        if hasattr(self.base_model, "model") and hasattr(self.base_model.model, "embed_tokens"):
            return self.base_model.model.embed_tokens  # Llama, Mistral, Qwen
        return self.base_model.get_input_embeddings()

    def recycle(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden state back to embedding space. No PE correction needed for RoPE."""
        projected = self.recycle_proj(hidden)
        return self.recycle_norm(projected)

    def forward_pass(
        self,
        inputs_embeds: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Single forward pass. Returns (hidden_states, logits, past_kv)."""
        outputs = self.base_model(
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        hidden = outputs.hidden_states[-1]
        past_kv = outputs.past_key_values if use_cache else None
        return hidden, outputs.logits, past_kv

    def run_recycle_loop(
        self,
        input_ids: torch.Tensor,
        max_iters: int = 16,
    ) -> Tuple[
        List[torch.Tensor],  # hidden states at each iteration
        List[torch.Tensor],  # inputs_embeds used at each iteration
        List[torch.Tensor],  # logits at each iteration
        List[float],         # mean-pooled hidden norms
        List[float],         # cosine similarity trajectory
    ]:
        """
        Run the recycle loop: forward → project → norm → forward → ...

        Iteration 0 is the baseline (normal token embeddings, no recycling).
        """
        embed_layer = self._get_embedding_layer()
        initial_embeds = embed_layer(input_ids)

        hiddens = []
        inputs_used = []
        all_logits = []
        norms = []
        cosines = []
        prev_pooled = None

        current_input = initial_embeds

        for i in range(max_iters + 1):
            inputs_used.append(current_input.detach().clone())

            hidden, logits, _ = self.forward_pass(current_input)
            hiddens.append(hidden.detach())
            all_logits.append(logits.detach())

            # Mean-pooled hidden state for convergence metrics
            pooled = hidden.mean(dim=1)
            norms.append(pooled.norm(dim=-1).mean().item())

            if prev_pooled is not None:
                cos = F.cosine_similarity(pooled, prev_pooled, dim=-1).mean().item()
                cosines.append(cos)
            else:
                cosines.append(-1.0)
            prev_pooled = pooled.detach()

            # Recycle for next iteration
            if i < max_iters:
                current_input = self.recycle(hidden)

        return hiddens, inputs_used, all_logits, norms, cosines

    def decode_from(
        self,
        inputs_embeds: torch.Tensor,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 50,
    ) -> Tuple[str, str, float, float, float]:
        """
        Decode tokens from input embeddings using KV cache.
        Re-runs forward pass with use_cache=True, then steps autoregressively.

        Returns: (decoded_text, next_token_str, next_token_prob, entropy, top1_prob)
        """
        _, logits, past_kv = self.forward_pass(inputs_embeds, use_cache=True)

        # Metrics from last position
        probs = F.softmax(logits[:, -1, :], dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum().item()
        top1_prob = probs.max().item()
        next_token_id = torch.argmax(probs, dim=-1)
        next_token_str = tokenizer.decode([next_token_id.item()])
        next_token_prob = probs[0, next_token_id.item()].item()

        # Autoregressive decoding from KV cache
        embed_layer = self._get_embedding_layer()
        tokens = []
        current_token = next_token_id

        for _ in range(max_new_tokens):
            if current_token.item() == tokenizer.eos_token_id:
                break
            tokens.append(current_token.item())

            new_embed = embed_layer(current_token).unsqueeze(1)
            outputs = self.base_model(
                inputs_embeds=new_embed,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            current_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return text, next_token_str, next_token_prob, entropy, top1_prob


# ============================================================================
# Experiment runner
# ============================================================================

def run_experiment(
    model_name: str = "meta-llama/Llama-3.2-1B",
    prompts: Optional[List[str]] = None,
    iteration_checkpoints: Optional[List[int]] = None,
    max_new_tokens: int = 50,
    device: str = None,
) -> List[ExperimentResult]:
    """
    H1 Experiment: Does self-recursive hidden-state recycling preserve meaning?
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if prompts is None:
        prompts = [
            # Factual — clear correct answer, should be stable
            "The capital of France is",
            "Water freezes at a temperature of",

            # Arithmetic — tests whether recycling helps reasoning
            "To solve 7 times 8, the answer is",
            "If I have 3 apples and buy 5 more, I have",

            # Knowledge — single correct continuation
            "The largest planet in our solar system is",
            "Albert Einstein developed the theory of",

            # Creative — tests whether recycling kills diversity
            "Once upon a time, in a kingdom far away,",
            "The old lighthouse keeper watched as",

            # Ambiguous — multiple valid paths
            "The best way to handle this situation is to",
            "In my opinion, the most important thing in life is",

            # Technical
            "In Python, a list comprehension is",
            "The time complexity of binary search is O(",
        ]

    if iteration_checkpoints is None:
        iteration_checkpoints = [0, 1, 2, 4, 8, 16]

    max_iters = max(iteration_checkpoints)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    )
    base_model.to(device)
    base_model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    recycler = RecycleCore(base_model)
    recycler.to(device)
    recycler.eval()

    print(f"Hidden size: {recycler.hidden_size}")
    print(f"Iteration checkpoints: {iteration_checkpoints}")
    print()

    results = []

    for prompt_idx, prompt in enumerate(prompts):
        print(f"{'='*70}")
        print(f"[{prompt_idx+1}/{len(prompts)}] {prompt}")
        print(f"{'='*70}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Baseline: normal autoregressive generation
        with torch.no_grad():
            baseline_ids = base_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            baseline_text = tokenizer.decode(
                baseline_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )
        print(f"  BASELINE: {baseline_text[:80]}")

        # Run recycle loop
        with torch.no_grad():
            hiddens, inputs_used, all_logits, norms, cosines = \
                recycler.run_recycle_loop(input_ids, max_iters)

        # Decode at each checkpoint
        snapshots = []
        for n_iter in iteration_checkpoints:
            with torch.no_grad():
                text, next_tok, next_prob, entropy, top1 = recycler.decode_from(
                    inputs_used[n_iter], tokenizer, max_new_tokens
                )

            # Cosine vs original (iteration 0)
            pooled_now = hiddens[n_iter].mean(dim=1)
            pooled_orig = hiddens[0].mean(dim=1)
            cos_vs_orig = F.cosine_similarity(
                pooled_now, pooled_orig, dim=-1
            ).mean().item()

            snap = IterationSnapshot(
                n_iters=n_iter,
                decoded_text=text,
                next_token=next_tok.strip(),
                next_token_prob=next_prob,
                mean_hidden_norm=norms[n_iter],
                cosine_vs_previous=cosines[n_iter],
                cosine_vs_original=cos_vs_orig,
                logit_entropy=entropy,
                top1_prob=top1,
            )
            snapshots.append(snap)

            tag = "BASE " if n_iter == 0 else f"i={n_iter:2d}  "
            print(f"  {tag} | "
                  f"next='{next_tok.strip()}'({next_prob:.3f}) | "
                  f"norm={norms[n_iter]:.1f} | "
                  f"cos_p={cosines[n_iter]:+.4f} | "
                  f"cos_0={cos_vs_orig:+.4f} | "
                  f"H={entropy:.1f} | "
                  f"text: {text[:60]}")

        results.append(ExperimentResult(
            prompt=prompt,
            baseline_text=baseline_text,
            snapshots=snapshots,
            hidden_norms=norms,
            cosine_trajectory=cosines,
        ))
        print()

    return results


# ============================================================================
# Analysis
# ============================================================================

def classify_result(r: ExperimentResult) -> dict:
    """
    Classify into failure modes:
    1. NUMERICAL_INSTABILITY — norms explode or collapse
    2. SEMANTIC_COLLAPSE — degenerate output (empty or extreme entropy)
    3. SEMANTIC_STABLE_NO_BENEFIT — coherent but same as baseline
    4. PREDICTION_SHIFTED — coherent and different (needs judge eval)
    """
    norms = r.hidden_norms
    cosines = r.cosine_trajectory

    norm_ratio = norms[-1] / (norms[0] + 1e-10)
    norm_stable = 0.1 < norm_ratio < 10.0

    late_cosines = [c for c in cosines[-4:] if c > -1]
    geo_converged = len(late_cosines) > 0 and min(late_cosines) > 0.95

    iter0 = r.snapshots[0]
    iter_last = r.snapshots[-1]
    text_nonempty = len(iter_last.decoded_text.strip()) > 0
    entropy_reasonable = 0.1 < iter_last.logit_entropy < 15.0

    next_tokens = [s.next_token for s in r.snapshots]
    prediction_stable = next_tokens[-1] == next_tokens[0]

    if not norm_stable:
        classification = "1_NUMERICAL_INSTABILITY"
    elif not text_nonempty or not entropy_reasonable:
        classification = "2_SEMANTIC_COLLAPSE"
    elif geo_converged and text_nonempty:
        if prediction_stable:
            classification = "3_SEMANTIC_STABLE_NO_BENEFIT"
        else:
            classification = "4_PREDICTION_SHIFTED"
    else:
        classification = "UNCLEAR"

    return {
        "prompt": r.prompt,
        "classification": classification,
        "norm_first": round(norms[0], 2),
        "norm_last": round(norms[-1], 2),
        "norm_ratio": round(norm_ratio, 3),
        "norm_stable": norm_stable,
        "geo_converged": geo_converged,
        "late_cosines": [round(c, 4) for c in late_cosines],
        "prediction_stable": prediction_stable,
        "next_tokens_over_iters": next_tokens,
        "baseline_text": r.baseline_text[:100],
        "iter0_text": iter0.decoded_text[:100],
        "iter_last_text": iter_last.decoded_text[:100],
    }


def print_summary(results: List[ExperimentResult]):
    """Print experimental summary."""
    print("\n" + "=" * 70)
    print("H1 EXPERIMENT RESULTS")
    print("Does self-recursive hidden-state recycling preserve meaning?")
    print("=" * 70)

    classifications = {}

    for r in results:
        a = classify_result(r)
        c = a["classification"]
        classifications[c] = classifications.get(c, 0) + 1

        print(f"\n[{c}]")
        print(f"  Prompt:     {a['prompt']}")
        print(f"  Norm:       {a['norm_first']} → {a['norm_last']} (ratio: {a['norm_ratio']})")
        print(f"  Converged:  {a['geo_converged']} (late cosines: {a['late_cosines']})")
        print(f"  Next token: {a['next_tokens_over_iters']}")
        print(f"  Baseline:   {a['baseline_text'][:70]}")
        print(f"  Iter 0:     {a['iter0_text'][:70]}")
        print(f"  Iter last:  {a['iter_last_text'][:70]}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for c, n in sorted(classifications.items()):
        print(f"  {c}: {n}")

    total = len(results)
    stable = classifications.get("3_SEMANTIC_STABLE_NO_BENEFIT", 0)
    shifted = classifications.get("4_PREDICTION_SHIFTED", 0)
    collapsed = classifications.get("2_SEMANTIC_COLLAPSE", 0)
    unstable = classifications.get("1_NUMERICAL_INSTABILITY", 0)

    print()
    if unstable > 0:
        print(f"  ⚠ {unstable}/{total} numerically unstable — recycling loop diverges.")
    if collapsed > 0:
        print(f"  ⚠ {collapsed}/{total} semantic collapse — converges to nonsense.")
    if stable > 0:
        print(f"  ✓ {stable}/{total} semantically stable — recycling preserves meaning.")
    if shifted > 0:
        print(f"  ? {shifted}/{total} predictions shifted — could be refinement or drift.")
        print(f"    Manual inspection or judge-model evaluation needed.")

    print()
    if unstable + collapsed == 0:
        print("  → H1 PROVISIONALLY SUPPORTED at the level of automated heuristics.")
        print("    Semantic preservation looks intact, but prediction shifts (if any)")
        print("    require human or judge-model evaluation to distinguish refinement")
        print("    from drift. Not yet publication-grade evidence without that step.")
    elif unstable + collapsed < total / 2:
        print("  → H1 PARTIALLY SUPPORTED: Some prompts survive, others don't.")
        print("    Investigate which prompt types fail and why.")
    else:
        print("  → H1 NOT SUPPORTED: Recycling does not preserve meaning.")
        print("    Consider training the bridge (recycle_proj) before retesting.")

    print(f"{'='*70}")


def save_results(results: List[ExperimentResult], path: str = "experiment_results.json"):
    """Save results as JSON."""
    serializable = []
    for r in results:
        analysis = classify_result(r)
        serializable.append({
            "prompt": r.prompt,
            "baseline": r.baseline_text,
            "classification": analysis["classification"],
            "hidden_norms": [round(n, 4) for n in r.hidden_norms],
            "cosine_trajectory": [round(c, 4) for c in r.cosine_trajectory],
            "snapshots": [
                {
                    "n_iters": s.n_iters,
                    "text": s.decoded_text,
                    "next_token": s.next_token,
                    "next_token_prob": round(s.next_token_prob, 4),
                    "norm": round(s.mean_hidden_norm, 4),
                    "cos_prev": round(s.cosine_vs_previous, 4),
                    "cos_orig": round(s.cosine_vs_original, 4),
                    "entropy": round(s.logit_entropy, 2),
                    "top1_prob": round(s.top1_prob, 4),
                }
                for s in r.snapshots
            ],
        })

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {path}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice H1 Experiment")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name (must be RoPE-based: Llama, Mistral, Qwen, etc.)",
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to decode")
    parser.add_argument("--max-iters", type=int, default=16, help="Max recycle iterations")
    args = parser.parse_args()

    checkpoints = [0, 1, 2, 4, 8]
    if args.max_iters >= 16:
        checkpoints.append(16)

    results = run_experiment(
        model_name=args.model,
        iteration_checkpoints=checkpoints,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )

    print_summary(results)
    save_results(results)
