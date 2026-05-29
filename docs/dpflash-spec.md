# DPFlash: Block-Diffusion Speculative Decoding on Crux CPU

*Specification and Implementation Plan — v1.0*

Porting Lucebox's CUDA PFlash (speculative prefill) and DFlash (block-diffusion
speculative decode) to the Crux compute substrate for CPU execution. Weight
format: INT4 GPTQ in safetensors. Target models: Qwen3-0.6B (PFlash scorer),
Qwen3.5-DFlash draft (5-layer block-diffusion), Qwen3.6-27B (target).

---

## 1. Project Goals

1. **Primary**: Run the full PFlash + DFlash inference pipeline on CPU through
   Crux, end-to-end: tokenize prompt, optionally compress long prompts via
   PFlash, speculative-decode via DFlash, produce output tokens.

2. **Secondary**: Use this workload to build out Crux's infrastructure —
   safetensors loading, INT4 quantized matmul, transformer primitives, and
   multi-program orchestration patterns. Every piece built should be reusable
   for other models.

3. **Non-goals**: Match GPU throughput. The CPU path is a correctness reference
   and a substrate-validation vehicle, not a speed target. SIMD and threading
   are future optimizations layered onto working scalar code.

---

## 2. Source System Summary

### 2.1 Algorithms (not original to Lucebox)

| Component | Paper | Reference Impl |
|---|---|---|
| DFlash (block-diffusion spec decode) | Wang et al., arXiv:2602.06036 | z-lab/dflash (Python/PyTorch) |
| DDTree (tree-structured verification) | Ringel & Romano, arXiv:2604.12989 | — |
| Speculative Prefill | Liu et al., arXiv:2502.02789 | Jingyu6/speculative_prefill |
| Cross-Family Speculative Prefill | SambaNova, arXiv:2603.02631 | — |
| FlashPrefill (block-sparse attention) | Fan et al., arXiv:2603.06199 | qhfan/FlashPrefill (Triton) |

### 2.2 Lucebox Implementation

C++/CUDA on ggml (pinned llama.cpp fork). Single RTX 3090 (24 GB). Three
models in one process:

- **Target**: Qwen3.6-27B Q4_K_M GGUF (~16 GB)
- **Draft**: Qwen3.5-DFlash 5-layer BF16 safetensors (~3.46 GB)
- **PFlash scorer**: Qwen3-0.6B BF16 GGUF (~1.5 GB)

VRAM budget forces park/unpark choreography (scorer and target cannot coexist).
On CPU with 64+ GB RAM, all three models coexist trivially.

---

## 3. Model Architectures

### 3.1 Qwen3.6-27B Target (Hybrid: Full-Attention + Gated DeltaNet)

This is NOT a standard transformer. 64 layers total:
- Every 4th layer (0, 4, 8, ..., 60) = **full softmax attention** (16 layers)
- All other layers = **Gated DeltaNet** linear attention with learned recurrence (48 layers)

#### Full-Attention Layers

| Parameter | Value |
|---|---:|
| n_embd | 5120 |
| n_head (Q) | 24 |
| n_head_kv | 4 |
| head_dim | 256 |
| QKV projection | Q is packed `[2*q_dim]` = Q concat gate; gate is sigmoid-applied post-attention |
| RoPE | Multi-RoPE with sections `[11, 11, 10, 0]`, theta = varies per model |
| K/V norm | Per-head RMSNorm on Q and K before RoPE |
| Attention | Causal, with optional sliding window (default 2048) |
| FFN | SwiGLU: `down(silu(gate(x)) * up(x))` |
| FFN intermediate | Inferred from weights |
| Norm | RMSNorm (eps = 1e-6) |

#### Gated DeltaNet Layers (Linear Attention with Recurrence)

| Parameter | Value |
|---|---:|
| d_inner | 6144 |
| d_state | 128 |
| d_conv | 4 |
| dt_rank | 48 |
| n_group | 16 |

Per-layer computation:
1. `QKV = wqkv @ x` (fused projection)
2. `z = wqkv_gate @ x` (gate for output)
3. `beta = sigmoid(ssm_beta @ x)`
4. `alpha_raw = softplus(ssm_alpha @ x + dt_bias)`
5. `g = alpha_raw * ssm_a` (decay gate)
6. Prepend conv_state window to QKV, run 1D convolution (d_conv=4 kernel)
7. Apply SiLU activation
8. Slice into Q, K, V components
9. L2-normalize Q and K
10. Repeat K, V for group-query pattern (n_group=16)
11. **Recurrent update**: `S_new = g * S_old + beta * (K^T @ V)` where S is `[d_state, d_state]` per head
12. **Output**: `O = Q @ S_new`
13. Apply `rms_norm(O) * silu(z)`, then output projection

**State management**: Each DeltaNet layer maintains:
- `ssm_state`: `[d_state=128, d_state=128]` f32 recurrent matrix per layer (48 layers)
- `conv_state`: `[d_conv-1=3, d_inner]` f32 sliding convolution window per layer

**Rollback requirement**: Unlike KV cache (which can be truncated), DeltaNet
recurrent state is a running product. Rollback requires either:
- Snapshotting state before verification and restoring on mismatch
- Storing per-token intermediate states and reading back the correct one

Lucebox uses the latter: `ssm_intermediate[128, 128, 48, max_verify_tokens]` in
f16, with custom CUDA kernels for tree-aware state persistence.

#### Full-Attention Layer Interval

`full_attention_interval = 4`. Layer indices `{0, 4, 8, 12, ..., 60}` are
full-attention. All others are DeltaNet. This is read from GGUF metadata.

#### Hidden State Capture for Draft Conditioning

5 target layers are selected as feature-capture points. Selection rule (from
`build_target_layer_ids`): evenly spaced from layer 1 to layer N-3. For 64
layers: `{1, 16, 31, 46, 61}`.

At each capture layer, the post-FFN activation (5120-dim) is written to a ring
buffer `target_feat[25600, capacity]` in BF16 (or F32 on CPU). The 5 layers'
features are stacked along the feature dimension: 5 * 5120 = 25600.

### 3.2 Qwen3.5-DFlash Draft (5-Layer Block-Diffusion Transformer)

| Parameter | Value |
|---|---:|
| n_layer | 5 |
| hidden_size | 5120 |
| n_head (Q) | 32 |
| n_head_kv | 8 (GQA 4:1 ratio) |
| head_dim | 128 |
| intermediate_size | 17408 |
| vocab_size | 248320 |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-6 |
| block_size | 16 |
| mask_token_id | 248070 |
| n_target_layers | 5 |

#### Architecture Differences from a Standard Transformer

1. **Non-causal attention** (`is_causal = False`): All 16 draft positions
   attend to all other positions bidirectionally. This is the "block diffusion"
   property.

2. **Cross+self attention**: Each layer's K and V are concatenated from TWO
   sources:
   - **Context** K/V: projected from target's captured hidden states
   - **Proposal** K/V: projected from the draft's own hidden states (the
     denoising block)
   - Final K = `concat(K_ctx, K_proposal)`, V = `concat(V_ctx, V_proposal)`

3. **Feature fusion layer** (`fc`): A single linear projection
   `[5 * 5120, 5120]` = `[25600, 5120]` (131M params, no bias) followed by
   RMSNorm. This compresses the multi-layer target features into the draft's
   hidden dimension.

4. **No embed_tokens or lm_head**: The draft borrows these from the target
   model. Only the transformer body + fc + hidden_norm are draft-specific.

5. **QKNorm**: Both Q and K have per-head RMSNorm applied before RoPE
   (following Qwen3 convention).

6. **RoPE position assignment**:
   - Queries: positions `[ctx_len .. ctx_len + block_size - 1]`
   - Context keys: positions `[0 .. ctx_len - 1]`
   - Proposal keys: positions `[ctx_len .. ctx_len + block_size - 1]`

7. **Stateless per step**: No persistent KV cache for the draft's own
   computation (the context KV cache in the z-lab reference is an optimization
   to avoid recomputing context projections, not a fundamental requirement).

#### Forward Pass

```
Input:
  noise_embedding = target.embed_tokens([last_token, MASK, MASK, ..., MASK])  # [1, 16, 5120]
  target_hidden = captured features from target's last verify step            # [1, ctx_len, 25600]

1. context = hidden_norm(fc(target_hidden))           # [1, ctx_len, 25600] -> [1, ctx_len, 5120]
2. For each of 5 layers:
   a. q = q_proj(rms_norm(h))                         # [1, 16, 4096]
      k_ctx = k_proj(context)                         # [1, ctx_len, 1024]
      k_noi = k_proj(rms_norm(h))                     # [1, 16, 1024]
      v_ctx = v_proj(context)                         # [1, ctx_len, 1024]
      v_noi = v_proj(rms_norm(h))                     # [1, 16, 1024]
   b. Apply per-head RMSNorm to q and concat(k_ctx, k_noi)
   c. Apply RoPE:
      q gets positions [ctx_len .. ctx_len+15]
      k gets positions [0 .. ctx_len+15]
   d. k = concat(k_ctx, k_noi)                        # [1, ctx_len+16, 1024]
      v = concat(v_ctx, v_noi)                        # [1, ctx_len+16, 1024]
   e. Non-causal attention: softmax(q @ k^T / sqrt(128)) @ v
   f. Output projection + residual
   g. SwiGLU FFN + residual
3. Final RMSNorm
4. logits = target.lm_head(output)                    # [1, 16, 248320]
5. Sample tokens from positions 1..15 (position 0 is the real last token)

Output: 15 candidate draft tokens
```

#### Weight Budget (BF16)

| Component | Shape | Params | Size (BF16) |
|---|---|---:|---:|
| fc | [25600, 5120] | 131.1M | 262 MB |
| hidden_norm | [5120] | 5K | 10 KB |
| Per layer (x5): | | | |
| - q_proj | [5120, 4096] | 21.0M | 42 MB |
| - k_proj | [5120, 1024] | 5.2M | 10.5 MB |
| - v_proj | [5120, 1024] | 5.2M | 10.5 MB |
| - o_proj | [4096, 5120] | 21.0M | 42 MB |
| - q_norm | [32, 128] | 4K | 8 KB |
| - k_norm | [8, 128] | 1K | 2 KB |
| - gate_proj | [5120, 17408] | 89.1M | 178 MB |
| - up_proj | [5120, 17408] | 89.1M | 178 MB |
| - down_proj | [17408, 5120] | 89.1M | 178 MB |
| - input_layernorm | [5120] | 5K | 10 KB |
| - post_attn_layernorm | [5120] | 5K | 10 KB |
| norm (final) | [5120] | 5K | 10 KB |
| **Total** | | **1.73B** | **3.46 GB** |

### 3.3 Qwen3-0.6B PFlash Scorer

Standard Qwen3 transformer, 0.6B parameters:

| Parameter | Value |
|---|---:|
| n_layer | 28 |
| hidden_size | 1024 |
| n_head (Q) | 16 |
| n_head_kv | 8 |
| head_dim | 64 |
| intermediate_size | 3072 |
| vocab_size | 151936 |
| rope_theta | 1,000,000 |
| rms_norm_eps | 1e-6 |

Standard causal transformer — no DeltaNet, no hybrid architecture. Dense
attention throughout. Used only for scoring token importance, not for
generation.

---

## 4. Algorithm Specifications

### 4.1 DFlash Speculative Decode Loop

```
PREFILL:
  target_out = target.forward(prompt_tokens, output_hidden_states=True)
  first_token = sample(target_out.logits[:, -1, :])
  target_hidden = extract_features(target_out.hidden_states, capture_layer_ids)
  committed = prompt_tokens + [first_token]

DECODE LOOP:
  while len(committed) < max_length:
    # 1. Build noise block
    block = [committed[-1], MASK, MASK, ..., MASK]  # 16 tokens
    noise_embed = target.embed_tokens(block)         # [1, 16, 5120]

    # 2. Draft forward (non-causal, block diffusion)
    draft_hidden = draft.forward(
        noise_embedding=noise_embed,
        target_hidden=target_hidden,   # [1, ctx_len, 25600]
        position_ids=[ctx_len .. ctx_len+15],
        is_causal=False)
    draft_logits = target.lm_head(draft_hidden)
    draft_tokens = sample(draft_logits[:, 1:, :])    # 15 candidates

    # 3. Target verification (causal forward)
    verify_input = [committed[-1]] + draft_tokens     # 16 tokens
    target_out = target.forward(verify_input, output_hidden_states=True)
    posterior = sample(target_out.logits)              # 16 posterior tokens

    # 4. Acceptance (greedy prefix match)
    acceptance_length = 0
    for i in range(15):
        if draft_tokens[i] == posterior[i]:
            acceptance_length += 1
        else:
            break

    # 5. Commit: accepted draft tokens + 1 bonus (target's correction)
    committed += draft_tokens[:acceptance_length]
    committed += [posterior[acceptance_length]]         # bonus token

    # 6. Update target features (only accepted positions)
    target_hidden = extract_features(
        target_out.hidden_states, capture_layer_ids
    )[:, :acceptance_length + 1, :]

    # 7. Rollback target state to committed position
    #    - For full-attn layers: truncate KV cache
    #    - For DeltaNet layers: restore SSM state from snapshot
    rollback_target_state(len(committed))

    # 8. Check for EOS
    if committed[-1] == eos_token_id:
        break
```

#### Acceptance Logic Detail

The z-lab reference uses exact-match greedy acceptance:
```
match = (draft_tokens == posterior[:-1])  # element-wise compare
acceptance_length = match.cumprod().sum()  # contiguous prefix length
```

At temperature=0, this is lossless (greedy draft == greedy verify). At
temperature>0, this is an approximation (not rejection sampling).

#### DeltaNet State Rollback

For each DeltaNet layer (48 of them), after verification:
1. Before verify: snapshot `ssm_state[128, 128]` and `conv_state[3, 6144]`
2. During verify: store per-token intermediate SSM states
3. After acceptance: read SSM state at `accepted_position` index,
   write back to `ssm_state`. Reconstruct conv_state from the last
   `d_conv - 1 = 3` accepted tokens' conv inputs.

On CPU, the simplest approach is snapshot-and-restore: copy the SSM/conv state
before verification, restore on partial acceptance. This avoids the complexity
of per-token intermediate storage.

### 4.2 DDTree (Tree-Structured Verification)

Instead of verifying a single chain of 15 draft tokens, DDTree verifies a
**tree** of candidates in a single target forward pass.

#### Tree Construction

From the draft's top-K logits at each of 15 positions:

```
BUILD_DDTREE(draft_logprobs, budget=22):
  1. Seed: insert the full top-1 chain as the initial path
     (chain_seed=True by default)
  2. Priority queue (max-heap by cumulative log-probability)
  3. For each node in the chain, insert siblings (top-2, top-3, ...)
  4. Pop highest-priority unexpanded node, expand its children
  5. Repeat until budget (22 nodes) is reached
  6. Return: token_ids[], depths[], parents[], child_maps[], visibility[]
```

`visibility[i][j]` is an ancestor-only boolean mask: token `i` can only attend
to token `j` if `j` is an ancestor of `i` in the tree (or `j == i`).

#### Tree Verification

Single target forward over `1 + tree.n_nodes` tokens:
- Position 0: the last committed token
- Positions 1..n_nodes: tree nodes in DFS order
- Attention mask: `visibility` matrix (ancestor-only, not full causal)
- For DeltaNet layers: `parent_ids` array drives tree-aware recurrence
  (each node's recurrent update is applied to its parent's state, not the
  previous DFS node's state)

#### Tree Walk

Starting from root, follow target's argmax through `child_maps`:
```
FOLLOW_TREE(target_logits, tree):
  pos = root
  accepted = []
  while pos has children in child_maps:
    target_token = argmax(target_logits[pos])
    if target_token in child_maps[pos]:
      accepted.append(target_token)
      pos = child_maps[pos][target_token]
    else:
      accepted.append(target_token)  # bonus token
      break
  return accepted
```

### 4.3 PFlash Speculative Prefill

For long prompts (>32K tokens), compress the prompt before target prefill.

#### Pipeline

```
PFLASH_COMPRESS(prompt_ids, keep_ratio=0.05):
  S = len(prompt_ids)

  # 1. Drafter forward (Qwen3-0.6B, full sequence)
  #    Uses FlashPrefill block-sparse attention for S > 32K
  hidden_states = qwen3_0p6b.forward(prompt_ids)

  # 2. Tail attention scoring
  #    Q = last n_lookahead=8 query positions
  #    K = all S key positions
  #    score[lookahead, S] = Q[-8:] @ K^T / sqrt(d)  per layer, per head
  #    running_max[lookahead, S] = max over (layers, heads)
  running_max = tail_attention_score(hidden_states, n_lookahead=8)

  # 3. Mean over lookahead -> per-token score
  score[S] = mean(running_max, axis=0)

  # 4. AvgPool smoothing (1D, kernel=13)
  smooth[S] = avg_pool_1d(score, kernel_size=13)

  # 5. Chunk-top-K selection
  n_chunks = ceil(S / 32)
  chunk_scores = [mean(smooth[c*32 : (c+1)*32]) for c in range(n_chunks)]
  n_keep = max(1, round(n_chunks * keep_ratio))
  selected = top_k_indices(chunk_scores, n_keep)
  selected.sort()

  # 6. Span merge (adjacent chunks become contiguous spans)
  compressed_ids = []
  for each contiguous run of selected chunk indices:
    compressed_ids.extend(prompt_ids[run_start*32 : run_end*32])

  return compressed_ids  # ~5% of original length
```

At `keep_ratio=0.05`: 128K tokens -> ~2.6K tokens. NIAH retrieval preserved.

#### FlashPrefill Block-Sparse Attention (for the Drafter Forward)

The Qwen3-0.6B forward at 128K tokens would be O(S^2) with dense attention.
FlashPrefill makes the drafter's own attention sparse:

```
FLASH_PREFILL_FORWARD(Q, K, V, config):
  block_size = 128
  M = ceil(S / block_size)  # number of blocks

  # Step 1: Mean K per block
  mean_K[M, n_kv_heads, head_dim] = mean(K.reshape(M, block_size, ...), axis=1)

  # Step 2: Block scores
  for each q_block m in [0, M):
    for each k_block n in [0, m]:  # causal
      score[m, n, h] = sum_j(exp2(Q[m*128+j] . mean_K[n] * scale - max)) for j in q_block

  # Step 3: Block selection
  for each q_block m, head h:
    max_score = max(score[m, :, h])
    threshold = max_score * alpha  # alpha=0.12 default, 0.85 for long ctx
    selected[m, h] = {n : score[m,n,h] >= threshold}
                   ∪ {0..attention_sink-1}      # always keep first 2 blocks
                   ∪ {m-window+1..m}             # always keep last 4 blocks
                   ∪ (if m >= M-2: all blocks)   # last 2 q-blocks attend everywhere

  # Step 4: Sparse attention forward
  for each q_block m, head h:
    O[m] = FlashAttention(Q[m], K[selected[m,h]], V[selected[m,h]])
    # Online softmax (FlashAttention-2 style) over only the selected blocks
```

**On CPU**: Steps 1-3 are straightforward loop nests. Step 4 is a standard
tiled attention loop restricted to selected K/V blocks. No WMMA, no shared
memory staging, no warp shuffles. The algorithm translates directly to scalar
loops in Crux IR.

---

## 5. Weight Format: INT4 GPTQ in Safetensors

### 5.1 Safetensors Container Format

```
[8 bytes: u64 LE header_size]
[header_size bytes: JSON metadata]
[tensor data: contiguous, mmap-friendly]
```

JSON metadata maps tensor names to `{dtype, shape, data_offsets: [start, end]}`.
Offsets are relative to the end of the header.

### 5.2 GPTQ Weight Layout

For each linear layer, GPTQ stores:

| Tensor | Shape | DType | Description |
|---|---|---|---|
| `qweight` | `[in_features/8, out_features]` | int32 | 8 INT4 values packed per int32 |
| `qzeros` | `[in_features/group_size, out_features/8]` | int32 | Zero points, packed INT4 |
| `scales` | `[in_features/group_size, out_features]` | float16 | Per-group scale factors |
| `g_idx` | `[in_features]` | int32 | Group assignment (optional, often sequential) |

Standard `group_size = 128`.

#### Dequantization (per element)

```
group = row_idx / group_size  # (or g_idx[row_idx] if non-sequential)
packed_word = qweight[row_idx / 8, col_idx]
nibble_idx = row_idx % 8
int4_val = (packed_word >> (nibble_idx * 4)) & 0xF  # unsigned 4-bit

zero_word = qzeros[group, col_idx / 8]
zero_nibble = col_idx % 8
zero = (zero_word >> (zero_nibble * 4)) & 0xF

scale = scales[group, col_idx]  # float16

dequantized = (int4_val - zero) * scale  # float32 result
```

### 5.3 Which Models Get INT4 GPTQ

| Model | Size | Quantization | Notes |
|---|---|---|---|
| Qwen3-0.6B (PFlash scorer) | 0.6B | **BF16** (no quantization) | Too small to benefit; only ~1.2 GB |
| Qwen3.5-DFlash draft | 1.73B | **BF16** (no quantization) | Draft quality matters; keep full precision |
| Qwen3.6-27B (target) | 27B | **INT4 GPTQ** | ~8 GB quantized (vs 54 GB BF16) |

The target model's `embed_tokens` and `lm_head` are typically kept in FP16/BF16
even in GPTQ models (embedding lookup doesn't benefit from quantization).

### 5.4 Note on Qwen3.5 (Hybrid) Quantization

GPTQ applies to all linear layers regardless of layer type. DeltaNet layers
have different projection shapes than full-attention layers:
- `wqkv`: `[n_embd, d_inner * 3]` — fused Q/K/V for DeltaNet
- `wqkv_gate`: `[n_embd, d_inner]` — output gate
- `ssm_beta`, `ssm_alpha`: `[n_embd, dt_rank]` — recurrence params
- `ssm_out`: `[d_inner, n_embd]` — output projection

These are still standard linear projections and quantize normally under GPTQ.

---

## 6. Crux Infrastructure Required

### 6.1 Current Crux State (What Exists)

| Capability | Status |
|---|---|
| IR with 67 opcodes (arith, math, compare, bitwise, control flow, reduce) | Working |
| CPU reference interpreter (sequential, scalar) | Working |
| View/Memory/Stream/Device/Program/Event APIs | Working |
| Shape, Strides, DType (i8-i64, u8-u64, f16, f32, f64, bf16) | Working |
| Arena allocator (bump allocation) | Working |
| Named bindings for dispatch | Working |
| View transformations (slice, transpose, reshape, broadcast) | Working |
| Canonical kernels (matmul, map, reduce, transpose) | Working |
| f16/bf16 software conversion in interpreter | Working |

### 6.2 Infrastructure to Build

Each item below is a reusable Crux component, not a one-off hack.

#### 6.2.1 Safetensors Loader (With module)

```
Module: crux.safetensors (or standalone dpflash.safetensors)

Types:
  SafetensorsFile {
    path: str,
    header: SafetensorsHeader,
    mmap: *mut u8,          // mmap'd file data
    data_offset: usize,     // byte offset where tensor data begins
  }
  SafetensorsHeader {
    tensors: Vec[TensorMeta],
  }
  TensorMeta {
    name: str,
    dtype: DType,           // mapped from safetensors dtype strings
    shape: Shape,
    data_start: usize,      // byte offset within data section
    data_end: usize,
  }

Functions:
  open(path: str) -> Result[SafetensorsFile, Error]
  get_tensor_view(file, name: str, device: *mut Device) -> Result[View, Error]
    // Returns a View backed by mmap'd Memory, zero-copy
  get_tensor_copy(file, name: str, device: *mut Device) -> Result[View, Error]
    // Copies tensor data into a new Memory allocation
  close(file)
```

Safetensors dtype string mapping:
- `"F16"` -> `DType.Float16`
- `"BF16"` -> `DType.BFloat16`
- `"F32"` -> `DType.Float32`
- `"I32"` -> `DType.Int32`
- `"I8"` -> `DType.Int8`

JSON parsing: minimal parser for the safetensors header format (flat object
with known structure, no nested objects beyond `data_offsets`).

#### 6.2.2 INT4 Packed DType

Crux's current DType enum does not include INT4. Two options:

**Option A: Treat as opaque bytes.** Store INT4 weights as `DType.UInt8` (two
values per byte) or `DType.Int32` (eight values per int32, matching GPTQ
packing). Dequantization is explicit in the matmul kernel. No dtype system
changes needed.

**Option B: Add `DType.Int4`.** Requires `dtype_size` to return a sub-byte
value, which breaks assumptions throughout Crux.

**Recommendation: Option A.** GPTQ already packs 8 INT4 values per int32. The
`qweight` tensor is simply `DType.Int32` with shape `[in/8, out]`. The kernel
unpacks inline. This is consistent with how every real INT4 implementation
works — the packing is the kernel's concern, not the type system's.

#### 6.2.3 New IR Kernels

All kernels below are Crux IR programs built with the existing instruction set.
No new opcodes needed.

**Tier 1 — Core transformer ops:**

| Kernel | Inputs | Output | Notes |
|---|---|---|---|
| `rms_norm` | x[S, D], weight[D], eps | out[S, D] | `x * weight / sqrt(mean(x^2) + eps)` |
| `rope` | x[S, H, D], freqs[S, D/2] | out[S, H, D] | Rotary position embedding |
| `rope_multi` | x[S, H, D], freqs, sections[4] | out[S, H, D] | Multi-RoPE (Qwen3.5 full-attn) |
| `silu` | x[..] | out[..] | `x * sigmoid(x)` = `x / (1 + exp(-x))` |
| `swiglu_fused` | gate[S, F], up[S, F] | out[S, F] | `silu(gate) * up` |
| `softmax` | x[S, D] | out[S, D] | `exp(x - max(x)) / sum(exp(x - max(x)))` |
| `sigmoid` | x[..] | out[..] | `1 / (1 + exp(-x))` |
| `softplus` | x[..] | out[..] | `log(1 + exp(x))` |
| `l2_norm` | x[S, D] | out[S, D] | `x / sqrt(sum(x^2) + eps)` per row |
| `causal_attention` | Q[S,H,D], K[S,Hk,D], V[S,Hk,D] | out[S,H,D] | Tiled with online softmax |
| `noncausal_attention` | Q, K, V | out | Same but no causal mask |
| `matmul_f32` | A[M,K], B[K,N] | C[M,N] | Dense F32 matmul |
| `matmul_int4_dequant` | qw[K/8,N]i32, qz, scales, x[M,K]f32 | out[M,N]f32 | Fused INT4 dequant+matmul |
| `embedding_lookup` | ids[S]i32, table[V,D] | out[S,D] | Gather rows by index |

**Tier 2 — DeltaNet ops (for target model):**

| Kernel | Inputs | Output | Notes |
|---|---|---|---|
| `conv1d` | x[S+pad, C], weight[K, C] | out[S, C] | 1D conv with d_conv=4 kernel |
| `gated_delta_net_step` | q, k, v, beta, g, S_in | O, S_out | Single-token recurrent update |
| `ssm_state_snapshot` | state[D,D,L] | snap[D,D,L] | Copy for rollback |
| `ssm_state_restore` | snap[D,D,L] | state[D,D,L] | Restore from snapshot |

**Tier 3 — PFlash scoring ops:**

| Kernel | Inputs | Output | Notes |
|---|---|---|---|
| `block_mean` | K[S,Hk,D], block_size | mean_K[M,Hk,D] | Mean per block |
| `block_score` | Q[S,H,D], mean_K[M,Hk,D] | score[M,M,H] | Q-block vs K-block scores |
| `block_select` | score[M,M,H], alpha, sink, window | idx[M,N,H], cnt[M,H] | Threshold selection |
| `sparse_attention` | Q, K, V, idx, cnt | O | Attention over selected blocks only |
| `avg_pool_1d` | x[S], kernel_size | out[S] | 1D average pooling |
| `chunk_topk` | scores[S], chunk_size, keep_ratio | selected_chunks[] | Top-K chunk selection |

#### 6.2.4 Tokenizer

The models use Qwen's BPE tokenizer. Options:

1. **Bind SentencePiece/tiktoken via C FFI** — fastest path, well-tested
2. **Minimal BPE in With** — more self-contained but substantial work
3. **Pre-tokenize externally** — pass token IDs in, defer tokenizer to later

**Recommendation**: Option 3 for initial bring-up (test with pre-tokenized
inputs), then Option 1 for production.

#### 6.2.5 Multi-Program Dispatch Orchestration

The speculative decode loop requires dispatching multiple Crux programs in
sequence with data dependencies. This is application-layer logic in With, not
IR:

```
// Pseudocode for the decode loop in With
fn decode_step(target, draft, state) -> Vec[i32]:
    // 1. Build noise block
    let block = build_noise_block(state.last_token, draft.mask_token_id)
    dispatch(stream, embed_lookup_prog, ..., bind("ids", block), bind("table", target.embed_tokens), bind("out", noise_embed))

    // 2. Draft forward (5 layers)
    dispatch(stream, draft_fc_prog, ..., bind("x", target_hidden), bind("w", draft.fc), bind("out", context))
    dispatch(stream, rms_norm_prog, ..., bind("x", context), bind("w", draft.hidden_norm), bind("out", context))
    for layer in draft.layers:
        dispatch_draft_layer(stream, layer, context, noise_embed, ...)

    // 3. LM head
    dispatch(stream, matmul_prog, ..., bind("a", draft_output), bind("b", target.lm_head), bind("out", logits))

    // 4. Sample draft tokens
    let draft_tokens = argmax(logits)  // or sample with temperature

    // 5. Target verify (64 layers, hybrid)
    dispatch_target_forward(stream, target, verify_input, ...)

    // 6. Acceptance
    let accepted = greedy_accept(draft_tokens, posterior_tokens)

    // 7. Rollback
    rollback_target_state(state, accepted.len())

    return accepted
```

---

## 7. Memory Budget Analysis (CPU)

### 7.1 Model Weights

| Model | Precision | Size |
|---|---|---:|
| Qwen3.6-27B target (INT4 GPTQ) | INT4 + FP16 scales | ~8 GB |
| Qwen3.5-DFlash draft | BF16 | ~3.46 GB |
| Qwen3-0.6B scorer | BF16 | ~1.2 GB |
| Target embed_tokens (shared) | FP16 | ~476 MB (248320 * 5120 * 2) |
| Target lm_head (shared) | FP16 | ~476 MB |
| **Total weights** | | **~13.6 GB** |

### 7.2 Runtime State (at 128K context)

| Component | Size | Notes |
|---|---|---:|
| Full-attn KV cache (16 layers) | ~3.2 GB | 16 * 2 * 128K * 4 heads * 256 dim * 4 bytes |
| DeltaNet SSM state (48 layers) | ~150 MB | 48 * 128 * 128 * 4 bytes * 2 (state + snapshot) |
| DeltaNet conv state (48 layers) | ~7 MB | 48 * 3 * 6144 * 4 bytes * 2 |
| Target features ring buffer | ~13 MB | 25600 * 4096 cap * 2 bytes (bf16) |
| Draft workspace | ~50 MB | Activations, attention scratch |
| PFlash scorer workspace (128K) | ~2 GB | Full 0.6B forward + FlashPrefill scratch |
| **Total runtime** | **~5.4 GB** |

### 7.3 Total

~19 GB for all three models + 128K context. Fits comfortably in 32 GB system
RAM; 64 GB gives ample headroom.

On CPU, the park/unpark choreography from Lucebox is unnecessary. All models
remain resident.

---

## 8. Implementation Phases

Sequenced to get Qwen3.5-27B inference working first. The target model is the
anchor — the draft borrows its embed_tokens and lm_head, so nothing else runs
without it. PFlash (the 0.6B scorer) is deferred since it's an optimization
for long prompts, not a prerequisite for generation.

### Phase 1: Safetensors Loader + Smoke Test

**Goal**: Load a safetensors file and read tensor data into Crux Views.

**Deliverables**:
1. JSON parser (minimal, for safetensors header format)
2. `safetensors.open()` / `get_tensor_view()` / `get_tensor_copy()` / `close()`
3. mmap support (With FFI to `mmap(2)`)
4. Smoke test: load a small safetensors file, verify tensor shapes and
   spot-check values against Python reference

**Crux infra built**: File I/O, mmap, JSON parsing — all reusable.

**Estimated scope**: ~500-800 lines of With.

### Phase 2: Core Transformer Kernels

**Goal**: Implement and test the fundamental ML kernels as Crux IR programs.
Only the kernels needed for the 27B target forward — no PFlash-specific ops
yet.

**Deliverables**:
1. `rms_norm` kernel
2. `rope` kernel (standard Neox-style rotary)
3. `rope_multi` kernel (Qwen3.5 full-attn layers: sections `[11,11,10,0]`)
4. `silu`, `sigmoid`, `softplus` kernels
5. `swiglu_fused` kernel
6. `softmax` kernel
7. `embedding_lookup` kernel
8. `matmul_f32` kernel (tiled, production-quality)
9. `causal_attention` kernel (tiled, online softmax, with sliding window
   support and GQA head broadcast)
10. `l2_norm` kernel (per-row L2 normalization, needed by DeltaNet)
11. Comprehensive tests against NumPy/PyTorch reference values

**Crux infra built**: Complete ML kernel library. Every transformer model
needs these same ops.

**Test strategy**: For each kernel, generate reference inputs/outputs from
PyTorch, save as binary files, load in Crux tests, compare outputs within
tolerance (1e-5 for f32, 1e-3 for f16/bf16).

**Estimated scope**: ~2000-3000 lines of With (kernel builders + tests).

### Phase 3: INT4 GPTQ Matmul Kernel

**Goal**: Implement fused INT4 dequantization + matmul for the 27B target's
linear layers.

**Deliverables**:
1. `matmul_int4_dequant` Crux IR kernel: loop over groups, unpack int32 ->
   8 x int4, subtract zero point, multiply by scale, accumulate into f32
2. GPTQ weight unpacking logic (handle `qweight`, `qzeros`, `scales` layout)
3. Test against PyTorch `torch.nn.Linear` with GPTQ-quantized weights
4. Benchmark: measure throughput, identify optimization targets

**Kernel algorithm** (scalar, for CPU interpreter):
```
for m in [0, M):         # output rows
  for n in [0, N):       # output columns
    acc = 0.0
    for k_group in [0, K/group_size):
      scale = scales[k_group, n]           # f16 -> f32
      zero_packed = qzeros[k_group, n/8]
      zero = (zero_packed >> ((n%8)*4)) & 0xF
      for k_local in [0, group_size):
        k = k_group * group_size + k_local
        packed = qweight[k/8, n]
        int4_val = (packed >> ((k%8)*4)) & 0xF
        dequant = (int4_val - zero) * scale
        acc += x[m, k] * dequant
    out[m, n] = acc
```

**Crux infra built**: INT4 dequantization pattern — reusable for any GPTQ
model.

**Estimated scope**: ~500-800 lines of With.

### Phase 4: DeltaNet Kernels

**Goal**: Implement the Gated DeltaNet primitives needed for 48 of the
target's 64 layers.

**Deliverables**:
1. `conv1d` kernel: 1D depthwise convolution with d_conv=4 kernel, prepending
   conv_state window
2. `gated_delta_net_step` kernel: single-token recurrent update
   `S_new = g * S_old + beta * (K^T @ V)`, `O = Q @ S_new`
3. `ssm_state_snapshot` / `ssm_state_restore`: bulk copy of recurrent state
   (48 layers * 128 * 128 f32) for speculative rollback
4. Conv state window management (slide and update)
5. Tests against PyTorch reference (manually compute DeltaNet recurrence on
   known inputs)

**Crux infra built**: Recurrent model primitives — reusable for any SSM/linear
attention model.

**Dependencies**: Phase 2 (matmul_f32, l2_norm).

**Estimated scope**: ~800-1200 lines of With.

### Phase 5: Qwen3.5-27B Target Forward (Hybrid Architecture)

**Goal**: Run the full 64-layer hybrid target model on CPU. Single-token
autoregressive generation producing correct output. This is the primary
milestone.

**Deliverables**:
1. GPTQ model loader: parse safetensors, construct weight Views for all 64
   layers (handling both full-attn and DeltaNet layer structures), load
   `embed_tokens` and `lm_head` in FP16/BF16
2. Full-attention layer forward: INT4 matmul for QKV/O projections, QKNorm
   (per-head RMSNorm on Q and K), multi-RoPE with sections, sigmoid-gated
   output, GQA attention with sliding window, SwiGLU FFN
3. DeltaNet layer forward: INT4 matmul for projections (wqkv, wqkv_gate,
   ssm_beta, ssm_alpha, ssm_out), conv1d, gated_delta_net_step, L2-norm Q/K,
   group-query repeat, `rms_norm(O) * silu(z)` output gating
4. Hybrid dispatcher: iterate 64 layers, route to full-attn (layers 0,4,8,...)
   or DeltaNet (all others) based on `full_attention_interval = 4`
5. KV cache for full-attn layers (16 layers, F32)
6. SSM state + conv state for DeltaNet layers (48 layers)
7. Hidden state capture at layers `{1, 16, 31, 46, 61}` into target_feat
   ring buffer (for later use by the draft model)
8. Logit sampling (argmax for greedy)
9. Autoregressive generation loop (single-token decode)
10. Validation: compare per-token logits and greedy generation output against
    HuggingFace Transformers or Lucebox

**Dependencies**: Phase 1 (loader), Phase 2 (core kernels), Phase 3 (INT4
matmul), Phase 4 (DeltaNet kernels).

**Crux infra built**: Full model loading and inference orchestration pattern,
hybrid layer dispatch, KV cache + SSM state management.

**Estimated scope**: ~3000-4000 lines of With. This is the largest phase.

### Phase 6: DFlash Draft Forward

**Goal**: Run the 5-layer block-diffusion draft model, producing candidate
tokens conditioned on captured target features.

**Deliverables**:
1. Draft model loader: parse safetensors for the draft's own weights (fc,
   hidden_norm, 5 transformer layers, final norm); bind target's embed_tokens
   and lm_head
2. Feature fusion: `fc` projection (25600 -> 5120) + `hidden_norm` RMSNorm
3. Cross+self attention: project context K/V from target features, project
   proposal K/V from draft hidden states, concatenate, run non-causal attention
4. `noncausal_attention` kernel: variant of causal_attention with no mask
   (all positions attend to all positions)
5. Asymmetric RoPE position assignment: queries at `[ctx_len..ctx_len+15]`,
   keys at `[0..ctx_len+15]`
6. QKNorm on Q and K
7. SwiGLU FFN (same as target's full-attn FFN)
8. Draft logit sampling (argmax + top-K extraction for later DDTree use)
9. Validation: compare draft logits against z-lab Python reference on
   identical inputs

**Dependencies**: Phase 2 (core kernels), Phase 5 (target forward provides
captured features and shared embed_tokens/lm_head).

**Note**: Draft weights are BF16 (not INT4) — uses `matmul_f32` after
bf16->f32 conversion, not the INT4 kernel.

**Estimated scope**: ~800-1200 lines of With.

### Phase 7: Speculative Decode Loop

**Goal**: Wire up the full draft-verify-accept loop. Generate text using
speculative decoding.

**Deliverables**:
1. Main decode loop in With:
   - Build noise block: `[last_token, MASK*15]`
   - Embed via target's embed_tokens
   - Draft forward (Phase 6) -> 15 candidate tokens
   - Target verify forward (Phase 5) on `[last_token] + candidates`
   - Greedy acceptance (contiguous prefix match + bonus token)
2. DeltaNet state snapshot before verify, restore after partial accept
3. KV cache truncation on rollback (full-attn layers)
4. Conv state rollback (DeltaNet layers)
5. Target feature ring buffer accumulation (only accepted positions kept)
6. EOS detection and generation termination
7. End-to-end validation: generate text and compare output quality against
   Lucebox or z-lab reference (token sequences should match at temperature=0)

**Dependencies**: Phase 5 (target forward), Phase 6 (draft forward).

**Estimated scope**: ~500-800 lines of With.

### Phase 8: DDTree (Tree-Structured Verification)

**Goal**: Replace chain verification with tree-structured verification for
higher acceptance rates per target forward pass.

**Deliverables**:
1. Tree construction: best-first heap over per-position top-K log-probability
   prefixes, chain seeding (top-1 chain inserted first), budget=22 nodes
2. Ancestor-only attention mask generation (`visibility` matrix)
3. Tree-aware target forward: modified causal attention using visibility mask
   instead of standard causal mask; `parent_ids` array for DeltaNet tree-aware
   recurrence (each node's SSM update applied to parent's state, not
   DFS-previous node's state)
4. Tree walk: follow target's argmax through `child_maps`
5. Tree-aware rollback: read SSM state at the correct tree node (requires
   per-node intermediate state storage, not simple snapshot/restore)
6. Validation: compare acceptance lengths and generation output against
   Lucebox's DDTree implementation

**Dependencies**: Phase 7 (basic spec decode working).

**Estimated scope**: ~1000-1500 lines of With.

### Phase 9: Qwen3-0.6B Forward Pass (BF16, for PFlash Scorer)

**Goal**: Run Qwen3-0.6B inference end-to-end. This is a standard dense
transformer (no DeltaNet), much simpler than the 27B target.

**Deliverables**:
1. Model loader: parse safetensors, build weight Views for 28 layers
2. KV cache allocation (simple ring buffer, dense attention)
3. Forward pass orchestrator: dispatch kernels layer-by-layer
4. Validation: compare per-token logits against HuggingFace Transformers

**Dependencies**: Phase 1 (loader), Phase 2 (core kernels). No INT4 needed
(BF16 model).

**Note**: Most of the orchestration code from Phase 5 can be reused — 0.6B
is a strict subset of the 27B architecture (dense attention only, no DeltaNet,
no hybrid dispatch, no INT4).

**Estimated scope**: ~800-1200 lines of With (much reuse from Phase 5).

### Phase 10: PFlash Scoring Pipeline

**Goal**: Given a long prompt (>32K tokens), score token importance via the
0.6B scorer and produce a compressed token stream.

**Deliverables**:
1. FlashPrefill kernels (4 Crux IR programs): `block_mean`, `block_score`,
   `block_select`, `sparse_attention` — for making the 0.6B's own forward
   pass tractable at 128K
2. Tail attention scoring: `Q[-8:] @ K^T / sqrt(d)` per layer/head, max
   over layers/heads -> `running_max[8, S]`
3. Mean-over-lookahead, AvgPool smoothing (kernel=13), chunk-top-K
   (chunk_size=32, keep_ratio=0.05), span merge (application-layer With
   code, not IR kernels)
4. End-to-end validation: compare compressed output against Lucebox's C++
   implementation on identical inputs; verify NIAH retrieval preserved

**Dependencies**: Phase 2 (core kernels), Phase 9 (0.6B forward pass).

**Estimated scope**: ~1500-2000 lines of With.

### Phase 11: PFlash + DFlash Integration

**Goal**: Full end-to-end pipeline — compress long prompts via PFlash, then
speculative-decode via DFlash.

**Deliverables**:
1. PFlash compression as preprocessing step
2. Target prefill on compressed prompt (reuse Phase 5 target forward in
   prefill mode)
3. DFlash decode on the prefilled context
4. End-to-end validation: long-context generation with NIAH retrieval test

**Dependencies**: Phase 7 or 8 (spec decode), Phase 10 (PFlash scoring).

**Estimated scope**: ~300-500 lines of With (mostly integration glue).

---

## 9. Dependency Graph

```
Phase 1 (safetensors loader)
    |
    v
Phase 2 (core kernels) ─────────────────────────────┐
    |               |                                |
    v               v                                |
Phase 3          Phase 4                             |
(INT4 matmul)    (DeltaNet kernels)                  |
    |               |                                |
    └───────┬───────┘                                |
            v                                        |
       Phase 5                                       |
       (27B target forward) ◄── PRIMARY MILESTONE    |
            |                                        |
            v                                        |
       Phase 6                                       |
       (draft forward)                               |
            |                                        |
            v                                        |
       Phase 7                                       |
       (spec decode loop)                            |
            |                                        |
            v                                        v
       Phase 8                                  Phase 9
       (DDTree) [optional]                      (0.6B forward)
            |                                        |
            |                                        v
            |                                   Phase 10
            |                                   (PFlash scoring)
            |                                        |
            └──────────────┬─────────────────────────┘
                           v
                      Phase 11
                      (integration)
```

Phases 3 and 4 can run in parallel. Phases 8 and 9 can run in parallel.
The critical path to first useful output is: 1 -> 2 -> 3+4 -> 5 -> 6 -> 7.

---

## 10. Testing Strategy

### 10.1 Reference Value Generation

Python script using PyTorch + HuggingFace Transformers:
1. Load each model
2. Run forward pass on test inputs
3. Dump intermediate values (per-layer activations, attention outputs, etc.)
   as binary f32 arrays
4. Save to test fixture directory

### 10.2 Per-Kernel Testing

For each Crux IR kernel:
- Small-shape exact test (e.g., 2x3 matmul, hand-verified)
- Medium-shape tolerance test (e.g., 128x256 matmul, compared to reference)
- Edge cases (zero input, single element, max rank, etc.)

Tolerance: 1e-5 for f32, 1e-3 for bf16-involved operations, 1e-2 for INT4
dequant (quantization introduces inherent error).

### 10.3 Per-Model Testing

1. **Logit comparison**: Run N tokens through both Crux and PyTorch, compare
   logit vectors (top-5 tokens should match, full vector within tolerance)
2. **Greedy generation**: Generate 100 tokens greedily, compare exact token
   sequence (should match if logits match)
3. **Acceptance rate**: Run spec decode for 100 steps, compare acceptance
   lengths against reference implementation

### 10.4 End-to-End Testing

1. **NIAH (Needle in a Haystack)**: Insert a fact in a long document, verify
   retrieval after PFlash compression
2. **HumanEval / GSM8K**: Run benchmarks, compare accuracy (not speed)

---

## 11. Key Design Decisions

### 11.1 DeltaNet: Snapshot vs. Per-Token Intermediate

Lucebox stores per-token SSM intermediate states in f16 for precise rollback.
This requires `128 * 128 * 48 * max_verify_tokens * 2` bytes of storage.

On CPU, snapshot-and-restore is simpler and adequate:
- Before verification: `memcpy` all 48 SSM states (150 MB total)
- After acceptance: restore snapshot, then replay accepted tokens sequentially
- Trade-off: replay cost is O(accepted_length * 48 layers), but each DeltaNet
  step is a single matrix update (fast on CPU)

For DDTree (Phase 9), per-node state tracking becomes necessary. Defer that
complexity until DDTree is implemented.

### 11.2 Target Feature Storage: BF16 vs F32

Lucebox uses BF16 for target_feat to save VRAM. On CPU, F32 is simpler (no
bf16 conversion overhead in the CPU interpreter's software path) and memory
is not the constraint. Use F32 for target features.

### 11.3 KV Cache Quantization

Lucebox uses Q4_0 or TQ3_0 KV cache to fit 256K context in 24 GB VRAM. On
CPU with ample RAM, use F32 KV cache for simplicity. Quantized KV cache is
a future optimization if memory becomes a concern at very long contexts.

### 11.4 Sliding Window Attention

Lucebox uses a 2048-token sliding window on full-attention layers. On CPU,
implement sliding window from the start — it reduces attention compute from
O(S^2) to O(S * window) and is straightforward (just limit the K/V range in
the attention kernel).

### 11.5 Draft KV Cache

The z-lab reference maintains a KV cache for the draft's context projections
to avoid recomputing them each step. This is an optimization. For initial
implementation, recompute context K/V each step (stateless draft). Add caching
later if profiling shows it matters.

---

## 12. File Layout

```
dpflash/
  dpflash-spec.md                     # this document
  lib/dpflash/
    dpflash.w                         # package root
    safetensors.w                     # safetensors loader
    json.w                            # minimal JSON parser
    tokenizer.w                       # tokenizer bindings (Phase 3+)
    kernels/
      rms_norm.w                      # RMSNorm kernel builder
      rope.w                          # RoPE kernel builder
      activations.w                   # silu, sigmoid, softplus, swiglu
      softmax.w                       # softmax kernel builder
      attention.w                     # causal + noncausal attention
      matmul.w                        # f32 matmul
      matmul_int4.w                   # INT4 GPTQ fused dequant+matmul
      embedding.w                     # embedding lookup
      conv1d.w                        # 1D convolution
      delta_net.w                     # Gated DeltaNet recurrence
      pflash_ops.w                    # block_mean, block_score, block_select, sparse_attn
      reduce_ops.w                    # avg_pool_1d, chunk_topk
    models/
      qwen3_config.w                  # model config types
      qwen3_loader.w                  # weight loading + layer construction
      qwen3_0p6b.w                    # Qwen3-0.6B forward (PFlash scorer)
      qwen35_target.w                 # Qwen3.5-27B hybrid forward
      qwen35_dflash_draft.w           # DFlash 5-layer draft forward
    inference/
      kv_cache.w                      # KV cache management
      ssm_state.w                     # DeltaNet state management
      target_features.w               # target_feat ring buffer
      spec_decode.w                   # speculative decode loop
      ddtree.w                        # DDTree construction + walk
      pflash.w                        # PFlash scoring pipeline
      pipeline.w                      # end-to-end pipeline orchestrator
  test/
    safetensors_test.w
    kernel_tests/                     # per-kernel tests
    model_tests/                      # per-model forward pass tests
    integration_tests/                # end-to-end tests
  fixtures/
    generate_fixtures.py              # PyTorch script to generate test data
```

---

## 13. Performance Expectations (CPU)

Not optimized. Rough estimates for a modern CPU (Apple M-series or x86-64):

| Operation | Estimate | Notes |
|---|---|---|
| 27B INT4 matmul (single token) | ~50-100 ms | Memory-bound: ~8 GB weights @ 50 GB/s bandwidth |
| 27B full forward (single token) | ~3-5 s | 64 layers, each with projections |
| Draft forward (16 tokens) | ~200-400 ms | 5 layers, 1.73B params |
| PFlash 0.6B forward (128K) | ~30-60 min | O(S^2) attention, no sparse opt |
| PFlash with FlashPrefill sparse | ~5-15 min | Only ~5% of blocks attended |
| Spec decode (per step) | ~4-6 s | Draft forward + target verify |
| Spec decode (per output token) | ~0.5-1.5 s | If acceptance length ~4-8 |

These are pre-optimization estimates. SIMD (NEON/AVX) and threading would
improve throughput 4-16x on the matmul-bound operations.

---

## 14. Glossary

| Term | Definition |
|---|---|
| Block diffusion | Denoising multiple mask tokens simultaneously in a single non-causal forward pass |
| DeltaNet | Gated linear attention with learned recurrence (Schlag et al.) |
| DDTree | Tree-structured speculative verification (Ringel & Romano) |
| DFlash | Block Diffusion for Flash Speculative Decoding (z-lab) |
| FlashPrefill | Block-sparse attention for efficient long-context prefill (Fan et al.) |
| GPTQ | Post-training quantization method for LLMs (Frantar et al.) |
| GQA | Grouped Query Attention (fewer KV heads than Q heads) |
| NIAH | Needle In A Haystack — retrieval test for long-context models |
| PFlash | Speculative prefill — compress long prompts via importance scoring |
| QKNorm | Per-head RMSNorm on Q and K before RoPE (Qwen3 convention) |
| RoPE | Rotary Position Embedding (Su et al.) |
| SSM state | The recurrent state matrix in DeltaNet layers |
| SwiGLU | Gated FFN: `silu(gate(x)) * up(x)` |
| Target features | Hidden states captured from target model layers to condition the draft |

---

## 15. Open Questions

1. **Qwen3.5 vs Qwen3.6**: The Lucebox codebase references both "Qwen3.5" and
   "Qwen3.6" names for the 27B target. Need to verify which exact model
   checkpoint to use and whether DFlash drafts exist for both.

2. **GPTQ availability for hybrid models**: Standard GPTQ tooling (AutoGPTQ,
   GPTQModel) may not handle DeltaNet layers correctly. Need to verify that
   llm-compressor supports Qwen3.5's hybrid architecture, or whether
   DeltaNet projections need special handling.

3. **Draft model quantization**: The spec keeps the draft at BF16 for quality.
   If the draft's 3.46 GB footprint is a concern on smaller-RAM systems,
   INT4 or INT8 quantization of the draft is possible but may degrade
   acceptance rates.

4. **Tokenizer compatibility**: Qwen3-0.6B uses vocab 151936 while Qwen3.5-27B
   uses vocab 248320. PFlash scoring uses the 0.6B's tokenizer/vocab. Need to
   handle the vocab mismatch at the PFlash-to-DFlash handoff (re-tokenize the
   compressed text, or use a shared tokenizer throughout).

5. **Multi-RoPE sections**: Qwen3.5's `rope_sections = [11, 11, 10, 0]` for
   full-attention layers vs standard Neox RoPE for the draft. Need to verify
   the exact RoPE implementation for each model component.
