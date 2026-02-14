#!/usr/bin/env python3
"""
Qwen3-Coder-Next (qwen3next) — TinyGrad implementation
Architecture: Hybrid Gated Delta Net + Full GQA Attention, 512-expert MoE + shared expert
Pattern: [DeltaNet×3, Attention×1] × 12 = 48 layers
Params:  ~80B total, ~3B active per token

Loads directly from GGUF (MXFP4_MOE quantized). Expert weights use sparse
access: only the top-K selected experts (~40 MB) are dequantized per layer per
token, instead of all 512 experts (~2.1 GB). This keeps peak nvmap-pinned
memory under ~8 GB, fitting comfortably in Jetson Orin's 64 GB unified memory.

Supports CUDA=1 (PTX) and NV=1 (native Nvidia) backends on Jetson Orin AGX.

Usage:
  # Interactive chat (inside tinygrad nix dev shell)
  NV=1 python3 qwen3next_llm.py

  # Benchmark
  CUDA=1 python3 qwen3next_llm.py --benchmark 20

  # OpenAI-compatible server
  NV=1 python3 qwen3next_llm.py --serve

Reference: docs/qwen3next-architecture.md for full forward pass pseudocode.
"""
from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, math, pathlib
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, dtypes
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored

# ─── Tokenizer (qwen2 preset from llm.py) ──────────────────────────────────────
class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="qwen2"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")
    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
  @staticmethod
  def from_gguf_kv(kv:dict):
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv.get("tokenizer.ggml.pre", "qwen2"))
  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])
  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str): return self.encode("<|im_start|>" + role + "\n")
  def end_turn(self, eos_id:int): return [eos_id] + self.encode("\n")

# ─── RoPE (partial — only first rope_dim dims of head) ─────────────────────────
@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  """Apply RoPE to first rope_dim dims of x: (B,H,T,D). Passes remaining dims through."""
  half = freqs_cis.shape[-1] // 2   # rope_dim/2 pairs to rotate
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)  # each (1,1,T,half)
  x1, x2 = x[..., :half], x[..., half:2*half]
  x_rest = x[..., 2*half:] if x.shape[-1] > 2*half else None
  rotated = (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)
  return rotated.cat(x_rest, dim=-1) if x_rest is not None else rotated

# ─── Expert Weights (sparse — only dequant selected experts) ───────────────────
class ExpertWeights:
  """MoE expert weights with sparse access to avoid full-tensor realization.

  The packed weight tensor (num_experts, out, in) is ~2.1 GB float32 per matrix
  after MXFP4 dequant (512 experts × 512 × 2048). With 48 layers × 3 matrices,
  that's ~307 GB — impossible on 64 GB Jetson.

  After loading, call split_experts() to create per-expert lazy slices.
  During inference, only the top-K selected experts (~40 MB) are realized."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
    self._experts: list[Tensor]|None = None  # set by split_experts()

  def split_experts(self):
    """Split packed (num_experts, out, in) into per-expert lazy slices.
    Each slice is an independent lazy view into the GGUF-backed dequant graph.
    When realized, TinyGrad fuses the slice with the MXFP4 dequant, so only
    ~4 MB per expert is allocated instead of the full ~2.1 GB."""
    n = self.weight.shape[0]
    self._experts = [self.weight[i] for i in range(n)]
    if DEBUG >= 2: stderr_log(f"    split {n} experts, ~{self.weight.nbytes()/1e6:.0f} MB packed → {n}×{self._experts[0].nbytes()/1e6:.1f} MB slices\n")

  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    if self._experts is not None:
      return self._sparse_forward(sel, x)
    # Dense fallback (only used before split_experts is called)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

  def _sparse_forward(self, sel:Tensor, x:Tensor) -> Tensor:
    """Memory-efficient forward: only realize selected experts.
    sel: (B, T, K) expert indices, x: (B, T, 1, D) input.
    For single-token gen: B=1, T=1, K=10 → ~40 MB per matrix instead of ~2.1 GB."""
    sel_np = sel.numpy()  # (B, T, K) — tiny tensor, fast CPU transfer
    unique_ids = sorted(set(int(e) for e in sel_np.flatten()))

    # Stack only the needed experts: each is a lazy slice → dequant only these
    expert_stack = Tensor.stack(*[self._experts[e] for e in unique_ids])  # (n_unique, out, in)

    if len(unique_ids) == sel_np.shape[-1] and all(unique_ids[i] == int(sel_np.flat[i]) for i in range(len(unique_ids))):
      # Fast path: unique_ids already matches sel order (common for single-token)
      selected = expert_stack.unsqueeze(0).unsqueeze(0)  # (1, 1, K, out, in)
    else:
      # General path: remap sel indices to compressed expert dimension
      remap = {orig: new for new, orig in enumerate(unique_ids)}
      remapped_flat = [remap[int(e)] for e in sel_np.flatten()]
      remapped = Tensor(remapped_flat, dtype=dtypes.int32).reshape(sel_np.shape).to(x.device)
      selected = expert_stack[remapped]  # (B, T, K, out, in)

    # Match original ExpertWeights API: x is (B,T,1,D), result is (B,T,K,out)
    # (B,T,1,1,D) @ (B,T,K,D,out) → (B,T,K,1,out) → squeeze → (B,T,K,out)
    return (x.unsqueeze(-2) @ selected.transpose(-1, -2)).squeeze(-2)

# ─── Delta Net Recurrent Block ─────────────────────────────────────────────────
class DeltaNetBlock:
  """Gated Delta Net — linear attention with learnable recurrent state matrix.
  State: [B, S_v, S_k, H_v] = [B, 128, 128, 32] matrix bank per layer.
  Conv state: [B, conv_k-1, conv_channels] = [B, 3, 8192]."""

  def __init__(self, dim:int, d_inner:int, n_k_heads:int, n_v_heads:int,
               head_k_dim:int, head_v_dim:int, conv_kernel:int, norm_eps:float,
               hidden_dim:int, num_experts:int, num_experts_per_tok:int):
    self.dim, self.d_inner = dim, d_inner           # 2048, 4096
    self.n_k_heads, self.n_v_heads = n_k_heads, n_v_heads  # 16, 32
    self.head_k_dim, self.head_v_dim = head_k_dim, head_v_dim  # 128, 128
    self.conv_kernel = conv_kernel                   # 4
    self.conv_channels = d_inner + 2 * n_k_heads * head_k_dim  # 8192
    self.heads_per_group = n_v_heads // n_k_heads    # 2
    self.ba_dim = 2 * self.heads_per_group * n_k_heads  # 64

    # Pre/Post norms
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)

    # Delta Net projections
    self.attn_qkv = nn.Linear(dim, self.conv_channels, bias=False)  # →8192
    self.attn_gate = nn.Linear(dim, d_inner, bias=False)             # →4096 (Z gate)
    self.ssm_ba = nn.Linear(dim, self.ba_dim, bias=False)            # →64 (beta+alpha)

    # SSM parameters
    self.ssm_a = Tensor.zeros(n_v_heads)                             # [32]
    self.ssm_dt_bias = Tensor.zeros(n_v_heads)                       # [32]
    self.ssm_conv1d_weight = Tensor.zeros(conv_kernel, self.conv_channels)  # [4, 8192]
    self.ssm_norm = nn.RMSNorm(head_v_dim, norm_eps)                 # [128]
    self.ssm_out = nn.Linear(d_inner, dim, bias=False)               # 4096→2048

    # MoE FFN
    self.num_experts_per_tok = num_experts_per_tok
    self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
    self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    # Shared expert
    self.ffn_gate_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down_shexp = nn.Linear(hidden_dim, dim, bias=False)
    self.ffn_gate_inp_shexp = Tensor.zeros(dim)  # scalar gate weight

  def init_state(self, B:int=1) -> tuple[Tensor, Tensor]:
    delta_state = Tensor.zeros(B, self.head_v_dim, self.head_k_dim, self.n_v_heads, dtype=dtypes.float32).contiguous()
    conv_state = Tensor.zeros(B, self.conv_kernel - 1, self.conv_channels, dtype=dtypes.float32).contiguous()
    return delta_state, conv_state

  def _delta_net(self, x:Tensor, delta_state:Tensor, conv_state:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Single-token Gated Delta Net. x: (B,1,D)."""
    B = x.shape[0]
    S_k, S_v, H_k, H_v = self.head_k_dim, self.head_v_dim, self.n_k_heads, self.n_v_heads
    hpg = self.heads_per_group  # 2

    # Projections
    qkv = self.attn_qkv(x).reshape(B, self.conv_channels)       # (B, 8192)
    z = self.attn_gate(x)                                         # (B, 1, 4096)
    ba = self.ssm_ba(x).reshape(B, 2 * hpg, H_k)                # (B, 4, 16)

    # Beta-alpha split: first hpg rows are beta, last hpg rows are alpha
    beta_raw = ba[:, :hpg].reshape(B, H_v, 1)                    # (B, 32, 1)
    alpha = ba[:, hpg:].reshape(B, H_v)                           # (B, 32)

    # Decay gate: ssm_a is negative (log-space), alpha+bias gives dt, softplus makes positive
    gate = self.ssm_a * (alpha + self.ssm_dt_bias).softplus()     # (B, 32), negative

    # Causal 1D convolution: prepend conv_state, take last conv_kernel frames
    conv_in = conv_state.cat(qkv.unsqueeze(1), dim=1)            # (B, 4, 8192)
    new_conv = conv_in[:, 1:, :].contiguous()                     # (B, 3, 8192)
    conv_out = (conv_in * self.ssm_conv1d_weight).sum(axis=1).silu()  # (B, 8192)

    # Split into V, Q, K — V gets first d_inner dims, Q/K interleaved in rest
    v = conv_out[:, :self.d_inner].reshape(B, S_v, H_v)          # (B, 128, 32)
    qk = conv_out[:, self.d_inner:].reshape(B, S_k, H_k, 2)
    q_small = qk[:, :, :, 0]                                      # (B, 128, 16)
    k_small = qk[:, :, :, 1]                                      # (B, 128, 16)

    # Expand Q,K from n_k_heads→n_v_heads via repeat_interleave
    # [h0,h1,...,h15] → [h0,h0,h1,h1,...,h15,h15] for heads_per_group=2
    q = q_small.unsqueeze(-1).expand(B, S_k, H_k, hpg).reshape(B, S_k, H_v)
    k = k_small.unsqueeze(-1).expand(B, S_k, H_k, hpg).reshape(B, S_k, H_v)

    # L2 normalize Q, K (over head_dim axis=1), scale Q
    eps = 1e-6
    q = q / (q.square().sum(1, keepdim=True).sqrt() + eps) * (1.0 / math.sqrt(S_v))
    k = k / (k.square().sum(1, keepdim=True).sqrt() + eps)
    beta = beta_raw.sigmoid()                                     # (B, 32, 1)

    # --- Recurrent update on state [B, S_v, S_k, H_v] ---
    # 1. Decay
    g = gate.reshape(B, 1, 1, H_v).exp()                         # decay ∈ (0,1)
    state = delta_state * g

    # 2. Read at key k: matrix-vector product along S_k dim
    k_e = k.reshape(B, 1, S_k, H_v)                              # (B, 1, 128, 32)
    kv_mem = (state * k_e).sum(2, keepdim=True)                   # (B, S_v, 1, H_v)

    # 3. Delta rule: beta * (v - read)
    v_t = v.reshape(B, S_v, 1, H_v)                              # (B, 128, 1, 32)
    delta = beta.reshape(B, 1, 1, H_v) * (v_t - kv_mem)          # (B, 128, 1, 32)

    # 4. Write: rank-1 outer product update
    state = state + k_e * delta                                    # (B, S_v, S_k, H_v)

    # 5. Query: read at key q
    q_e = q.reshape(B, 1, S_k, H_v)                              # (B, 1, 128, 32)
    output = (state * q_e).sum(2)                                  # (B, S_v, H_v)

    # Gated per-head norm + output projection
    output = self.ssm_norm(output.transpose(1, 2))                # (B, H_v, S_v) → normed [128]
    output = output.reshape(B, 1, self.d_inner) * z.silu()        # gate with silu(Z)
    return self.ssm_out(output), state.contiguous(), new_conv

  def _moe_ffn(self, h:Tensor) -> Tensor:
    """MoE FFN with shared expert and scalar sigmoid gate."""
    h_norm = self.post_attention_norm(h)
    x = h_norm.unsqueeze(2)                                       # (B, T, 1, D)

    # Routed experts (top-k selection)
    probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
    x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))
    moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)

    # Shared expert (SwiGLU + scalar sigmoid gate)
    shexp = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
    sg = (h_norm * self.ffn_gate_inp_shexp).sum(-1, keepdim=True).sigmoid()
    return h + moe_out + shexp * sg

  def __call__(self, x:Tensor, ds:Tensor, cs:Tensor) -> tuple[Tensor, Tensor, Tensor]:
    h = self.attn_norm(x)
    attn_out, new_ds, new_cs = self._delta_net(h, ds, cs)
    return self._moe_ffn(x + attn_out).contiguous(), new_ds, new_cs

# ─── Full Attention Block ──────────────────────────────────────────────────────
class FullAttentionBlock:
  """Standard GQA attention with sigmoid gating + MoE FFN. Uses KV cache."""
  def __init__(self, dim:int, n_heads:int, n_kv_heads:int, head_dim:int,
               rope_theta:float, rope_dim:int, norm_eps:float, max_context:int,
               hidden_dim:int, num_experts:int, num_experts_per_tok:int):
    self.dim, self.n_heads, self.n_kv_heads = dim, n_heads, n_kv_heads
    self.head_dim, self.rope_theta, self.rope_dim = head_dim, rope_theta, rope_dim
    self.max_context = max_context

    # Norms
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)

    # Attention: Q projection includes gate (n_heads * head_dim * 2)
    self.attn_q = nn.Linear(dim, n_heads * head_dim * 2, bias=False)  # →8192
    self.attn_k = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # →512
    self.attn_v = nn.Linear(dim, n_kv_heads * head_dim, bias=False)   # →512
    self.attn_output = nn.Linear(n_heads * head_dim, dim, bias=False)  # 4096→2048
    self.attn_q_norm = nn.RMSNorm(head_dim, norm_eps)
    self.attn_k_norm = nn.RMSNorm(head_dim, norm_eps)

    # MoE FFN (same structure as DeltaNetBlock)
    self.num_experts_per_tok = num_experts_per_tok
    self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
    self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
    self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    self.ffn_gate_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up_shexp = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down_shexp = nn.Linear(hidden_dim, dim, bias=False)
    self.ffn_gate_inp_shexp = Tensor.zeros(dim)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    h = self.attn_norm(x)

    # Joint Q+gate → split per head
    qg = self.attn_q(h).reshape(B, T, self.n_heads, self.head_dim * 2)
    q, gate = qg[..., :self.head_dim], qg[..., self.head_dim:]
    k = self.attn_k(h).reshape(B, T, self.n_kv_heads, self.head_dim)
    v = self.attn_v(h).reshape(B, T, self.n_kv_heads, self.head_dim)

    # Per-head QK norm
    q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q, k, v, gate = (t.transpose(1, 2) for t in (q, k, v, gate))  # →(B,H,T,D)

    # Partial RoPE (first rope_dim dims only)
    fc = precompute_freqs_cis(self.rope_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q, k = apply_rope(q, fc), apply_rope(k, fc)

    # KV cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, B, self.n_kv_heads, self.max_context, self.head_dim,
                                   dtype=k.dtype, device=k.device).contiguous().realize()
    self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v)).realize()
    k_full = self.cache_kv[0, :, :, :start_pos+T, :]
    v_full = self.cache_kv[1, :, :, :start_pos+T, :]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k_full, v_full, attn_mask=mask, enable_gqa=True)

    # Sigmoid gating
    attn = (attn * gate.sigmoid()).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _moe_ffn(self, h:Tensor) -> Tensor:
    h_norm = self.post_attention_norm(h)
    x = h_norm.unsqueeze(2)
    probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
    x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))
    moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
    shexp = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
    sg = (h_norm * self.ffn_gate_inp_shexp).sum(-1, keepdim=True).sigmoid()
    return h + moe_out + shexp * sg

  def __call__(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x = x + self._attention(x, start_pos)
    return self._moe_ffn(x).contiguous()

# ─── Qwen3Next Transformer ────────────────────────────────────────────────────
class Qwen3NextTransformer:
  """Hybrid Delta Net + Full Attention with MoE.
  Layers 0..N-1 with full_attn_interval=4 → [delta, delta, delta, attn] repeating."""

  def __init__(self, *, num_blocks:int, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int,
               norm_eps:float, vocab_size:int, head_dim:int, rope_theta:float, rope_dim:int,
               max_context:int, num_experts:int, num_experts_per_tok:int,
               d_inner:int, n_k_heads:int, n_v_heads:int, head_k_dim:int, head_v_dim:int,
               conv_kernel:int, full_attn_interval:int):
    self.num_blocks = num_blocks
    self.max_context = max_context
    self.full_attn_interval = full_attn_interval

    self.token_embd = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)

    # Build hybrid layer lists
    self.delta_blocks: list[DeltaNetBlock] = []
    self.attn_blocks: list[FullAttentionBlock] = []
    self.layer_type: list[str] = []       # "delta" | "attn" per layer index
    self.delta_idx: list[int] = []        # layer → delta_blocks index (-1 if attn)
    self.attn_idx: list[int] = []         # layer → attn_blocks index (-1 if delta)

    for i in range(num_blocks):
      is_rec = (i % full_attn_interval) != (full_attn_interval - 1)
      if is_rec:
        self.layer_type.append("delta")
        self.delta_idx.append(len(self.delta_blocks))
        self.attn_idx.append(-1)
        self.delta_blocks.append(DeltaNetBlock(dim, d_inner, n_k_heads, n_v_heads,
          head_k_dim, head_v_dim, conv_kernel, norm_eps, hidden_dim, num_experts, num_experts_per_tok))
      else:
        self.layer_type.append("attn")
        self.delta_idx.append(-1)
        self.attn_idx.append(len(self.attn_blocks))
        self.attn_blocks.append(FullAttentionBlock(dim, n_heads, n_kv_heads, head_dim,
          rope_theta, rope_dim, norm_eps, max_context, hidden_dim, num_experts, num_experts_per_tok))

    self._delta_states: list[tuple[Tensor, Tensor]] | None = None
    if DEBUG >= 1:
      print(f"  {len(self.delta_blocks)} delta + {len(self.attn_blocks)} attn = {num_blocks} layers")

  def _init_states(self, B:int=1):
    self._delta_states = [blk.init_state(B) for blk in self.delta_blocks]

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    if self._delta_states is None: self._init_states(tokens.shape[0])
    x = self.token_embd(tokens)
    for i in range(self.num_blocks):
      if self.layer_type[i] == "delta":
        di = self.delta_idx[i]
        x, ds, cs = self.delta_blocks[di](x, self._delta_states[di][0], self._delta_states[di][1])
        self._delta_states[di] = (ds, cs)
      else:
        x = self.attn_blocks[self.attn_idx[i]](x, start_pos)
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return self.forward(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf_path:str|pathlib.Path, max_context:int|None=None) -> tuple[Qwen3NextTransformer, dict]:
    """Load from GGUF with memory-safe two-phase realization.
    Non-expert weights are realized immediately (~2 GB fp16).
    Expert weights stay lazy (backed by mmap'd GGUF), dequantized on-the-fly."""

    stderr_log("Loading GGUF metadata...\n")
    # Keep GGUF on DISK device to avoid copying 40+ GB file into NV memory.
    # Using .to(None) would resolve to Device.DEFAULT (NV under NV=1) and fail.
    gguf_tensor = Tensor(pathlib.Path(gguf_path))
    kv, state_dict = nn.state.gguf_load(gguf_tensor)
    # Move per-tensor views to a concrete compute device lazily (copy ops realized later).
    target_device = "NV" if getenv("NV", 0) else ("CUDA" if getenv("CUDA", 0) else None)
    for k in list(state_dict.keys()):
      state_dict[k] = state_dict[k].to(target_device) if target_device is not None else state_dict[k].to(None)

    arch = kv['general.architecture']
    if arch != 'qwen3next': raise ValueError(f"Expected qwen3next, got '{arch}'")

    mc = kv.get(f'{arch}.context_length', 131072)
    max_context = min(max_context, mc) if max_context is not None else min(mc, 2048)  # cap for memory safety
    n_heads = kv[f'{arch}.attention.head_count']
    n_kv_heads = kv[f'{arch}.attention.head_count_kv']
    head_dim = kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads)
    d_inner = kv.get(f'{arch}.ssm.inner_size', 4096)
    n_k_heads = kv.get(f'{arch}.ssm.group_count', 16)
    n_v_heads = kv.get(f'{arch}.ssm.time_step_rank', 32)
    head_k_dim = kv.get(f'{arch}.ssm.state_size', 128)
    head_v_dim = d_inner // n_v_heads
    conv_kernel = kv.get(f'{arch}.ssm.conv_kernel', 4)
    full_attn_interval = kv.get(f'{arch}.full_attention_interval', 4)
    rope_dim = kv.get(f'{arch}.rope.dimension_count', 64)
    num_experts = kv.get(f'{arch}.expert_count', 0)
    num_experts_per_tok = kv.get(f'{arch}.expert_used_count', 0)
    hidden_dim = kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length'])

    if 'output.weight' not in state_dict:
      state_dict['output.weight'] = state_dict['token_embd.weight']

    stderr_log(f"  arch={arch}, layers={kv[f'{arch}.block_count']}, dim={kv[f'{arch}.embedding_length']}, "
               f"experts={num_experts}×top{num_experts_per_tok}, max_ctx={max_context}\n")

    model = Qwen3NextTransformer(
      num_blocks=kv[f'{arch}.block_count'], dim=kv[f'{arch}.embedding_length'],
      hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
      norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim, rope_theta=kv[f'{arch}.rope.freq_base'], rope_dim=rope_dim,
      max_context=max_context, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
      d_inner=d_inner, n_k_heads=n_k_heads, n_v_heads=n_v_heads,
      head_k_dim=head_k_dim, head_v_dim=head_v_dim,
      conv_kernel=conv_kernel, full_attn_interval=full_attn_interval)

    # Remap GGUF flat names (blk.N.xxx) → model's delta_blocks/attn_blocks paths
    _remap_state_dict(model, state_dict)

    # Identify expert weight tensors (these are the huge MXFP4 ones — 40+ GB)
    expert_keys = set()
    for k in state_dict:
      if any(p in k for p in ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')):
        expert_keys.add(k)

    # Cast non-expert tensors to fp16 (expert tensors stay as float32 from MXFP4 dequant)
    use_half = getenv("HALF", 1)
    for k in list(state_dict.keys()):
      if k not in expert_keys and use_half:
        state_dict[k] = state_dict[k].cast(dtypes.float16)

    # Load into model (realize=False — all tensors stay lazy)
    stderr_log(f"  Loading state dict ({len(state_dict)} tensors, {len(expert_keys)} expert tensors kept lazy)...\n")
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)

    # Realize non-expert params to avoid DISK-backed sources in runtime graphs.
    all_params = nn.state.get_parameters(model)
    expert_param_ids = set()
    for blk_list in (model.delta_blocks, model.attn_blocks):
      for blk in blk_list:
        for attr_name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
          expert_param_ids.add(id(getattr(blk, attr_name).weight))

    non_expert_params = [p for p in all_params if id(p) not in expert_param_ids]
    expert_params = [p for p in all_params if id(p) in expert_param_ids]
    stderr_log(f"  Realizing {len(non_expert_params)} non-expert params...\n")
    for s in non_expert_params:
      s.replace(s.contiguous())
      s.realize()
    stderr_log(f"  {len(expert_params)} expert params left lazy for sparse dequant\n")

    # Phase 4: Split expert weights into per-expert lazy slices for sparse access.
    # This is CRITICAL for memory safety on Jetson Orin 64GB:
    #   Without split: self.weight[sel] realizes full (512, out, in) = ~2.1 GB float32 per matrix
    #                  48 layers × 3 matrices = ~307 GB → guaranteed OOM crash
    #   With split:    only top-K experts realized per layer = ~40 MB per matrix
    #                  48 layers × 3 × 40 MB = ~5.8 GB peak → fits comfortably
    stderr_log(f"  Splitting expert weights for sparse access...\n")
    for blk_list in (model.delta_blocks, model.attn_blocks):
      for blk in blk_list:
        for attr_name in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
          getattr(blk, attr_name).split_experts()
    stderr_log(f"  Expert weights split into per-expert lazy slices (memory-safe)\n")

    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id


def _remap_state_dict(model: Qwen3NextTransformer, sd: dict[str, Tensor]):
  """Remap GGUF flat names (blk.N.xxx) → model's delta_blocks/attn_blocks paths."""
  new: dict[str, Tensor] = {}
  fai = model.full_attn_interval

  for name, tensor in list(sd.items()):
    if not name.startswith("blk."):
      new[name] = tensor
      continue

    parts = name.split(".")
    blk_i = int(parts[1])
    rest = ".".join(parts[2:])
    is_rec = (blk_i % fai) != (fai - 1)
    pfx = f"delta_blocks.{model.delta_idx[blk_i]}" if is_rec else f"attn_blocks.{model.attn_idx[blk_i]}"

    # Map GGUF names that don't match model attribute names directly
    mapping = {
      "ssm_a": "ssm_a",
      "ssm_dt.bias": "ssm_dt_bias",
      "ssm_conv1d.weight": "ssm_conv1d_weight",
      "ffn_gate_inp_shexp.weight": "ffn_gate_inp_shexp",
    }
    if rest == "ssm_conv1d.weight":
      tensor = tensor.transpose(-1, -2)
    mapped = mapping.get(rest)
    new[f"{pfx}.{mapped}" if mapped else f"{pfx}.{rest}"] = tensor

  sd.clear()
  sd.update(new)

# ─── HTTP Server (OpenAI-compatible) ──────────────────────────────────────────
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>qwen3next chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'qwen3next', messages: msgs, stream: true})});
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if self.path == "/v1/chat/completions":
      ids: list[int] = []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
        ids += tok.end_turn(eos_id)
      ids += tok.role("assistant")
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())

# ─── Main ──────────────────────────────────────────────────────────────────────
DEFAULT_GGUF = pathlib.Path.home() / ".cache/tinygrad/downloads/qwen3-coder-next/Qwen3-Coder-Next-MXFP4_MOE.gguf"
GGUF_URL = "https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/resolve/main/Qwen3-Coder-Next-MXFP4_MOE.gguf"

def _preflight_memory_check(gguf_path: pathlib.Path):
  """Check system memory and tune kernel params for large model loading on Jetson."""
  import os
  gguf_size_gb = gguf_path.stat().st_size / (1024**3)

  # Read available memory
  try:
    with open("/proc/meminfo") as f:
      meminfo = {k.strip(): int(v.split()[0]) for line in f if ':' in line for k, v in [line.split(':', 1)]}
    avail_gb = meminfo.get('MemAvailable', 0) / (1024**2)
    total_gb = meminfo.get('MemTotal', 0) / (1024**2)
  except Exception:
    avail_gb, total_gb = 0, 0

  stderr_log(f"  System RAM: {total_gb:.1f} GB total, {avail_gb:.1f} GB available\n")
  stderr_log(f"  GGUF file:  {gguf_size_gb:.1f} GB (mmap'd, pages evictable)\n")

  # Estimate memory needs: ~3 GB non-expert params + ~2 GB states + ~3 GB OS headroom
  non_expert_est_gb = 3.0
  os_headroom_gb = 4.0
  min_needed_gb = non_expert_est_gb + os_headroom_gb

  if avail_gb < min_needed_gb:
    stderr_log(f"  WARNING: Only {avail_gb:.1f} GB available, need ~{min_needed_gb:.0f} GB minimum\n")
    stderr_log(f"  Dropping page caches to free memory...\n")
    try:
      os.system("sync")
      with open("/proc/sys/vm/drop_caches", "w") as f: f.write("3")
    except (PermissionError, OSError):
      stderr_log(f"  (could not drop caches — run as root or set vm.drop_caches sysctl)\n")

  # Tune kernel VM params for large mmap workloads
  try:
    with open("/proc/sys/vm/min_free_kbytes") as f: mfk = int(f.read().strip())
    # Ensure at least 256MB min_free to prevent OOM under memory pressure
    if mfk < 262144:
      try:
        with open("/proc/sys/vm/min_free_kbytes", "w") as f: f.write("262144")
        stderr_log(f"  Raised min_free_kbytes: {mfk} → 262144 (256 MB)\n")
      except (PermissionError, OSError): pass
  except Exception: pass

def _decode_with_stats(model: Qwen3NextTransformer, tok: SimpleTokenizer, eos_id: int,
                       ids: list[int], start_pos: int, max_new_tokens: int,
                       stream_output: bool=True) -> tuple[list[int], float, float]:
  t0 = time.perf_counter()
  first_token_ts: float|None = None
  out: list[int] = []

  for next_id in model.generate(ids, start_pos):
    now = time.perf_counter()
    if first_token_ts is None: first_token_ts = now
    if next_id == eos_id or len(out) >= max_new_tokens: break
    out.append(next_id)
    if stream_output:
      sys.stdout.write(tok.decode([next_id]))
      sys.stdout.flush()

  t1 = time.perf_counter()
  prefill_tokens = max(1, len(ids) - start_pos)
  prefill_s = max(1e-9, (first_token_ts or t1) - t0)
  gen_s = max(1e-9, t1 - (first_token_ts or t1))
  prefill_tps = prefill_tokens / prefill_s
  gen_tps = len(out) / gen_s if len(out) > 0 else 0.0
  return out, prefill_tps, gen_tps

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Qwen3-Coder-Next — TinyGrad")
  parser.add_argument("--gguf", type=str, default=str(DEFAULT_GGUF), help="Path to GGUF file")
  parser.add_argument("--max_context", type=int, default=2048, help="Max context length (keep low for memory)")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI API server")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s")
  parser.add_argument("--prompt", type=str, default=None, help="Run one-shot prompt and exit")
  parser.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for --prompt/chat turns")
  args = parser.parse_args()

  gguf_path = pathlib.Path(args.gguf)
  if not gguf_path.exists():
    print(f"GGUF not found at {gguf_path}\nDownload:\n  curl -L -o '{DEFAULT_GGUF}' '{GGUF_URL}'")
    sys.exit(1)

  print(f"Loading Qwen3-Coder-Next from {gguf_path.name} ...")
  _preflight_memory_check(gguf_path)
  t0 = time.perf_counter()
  model, kv = Qwen3NextTransformer.from_gguf(gguf_path, args.max_context)
  print(f"Loaded in {time.perf_counter()-t0:.1f}s")

  if args.benchmark:
    param_bytes = sum(x.nbytes() for x in nn.state.get_parameters(model))
    gen = model.generate([0], 0)
    for i in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s, param {param_bytes/x:7.2f} GB/s"):
        next(gen)
    sys.exit(0)

  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  if args.serve is not None:
    print(f"Serving on http://localhost:{args.serve}")
    TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  if args.prompt is not None:
    ids: list[int] = [bos_id] if bos_id is not None else []
    ids += tok.role("user") + tok.encode(args.prompt) + tok.end_turn(eos_id) + tok.role("assistant")
    out, prefill_tps, gen_tps = _decode_with_stats(model, tok, eos_id, ids, max(len(ids)-1, 0), args.max_new_tokens, stream_output=True)
    print(f"\n[prompt] prefill: {prefill_tps:.1f} tok/s, gen: {gen_tps:.1f} tok/s, out: {len(out)} tok")
    sys.exit(0)

  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    out, prefill_tps, gen_tps = _decode_with_stats(model, tok, eos_id, ids, start_pos, args.max_new_tokens, stream_output=True)
    print(f"\n[turn] prefill: {prefill_tps:.1f} tok/s, gen: {gen_tps:.1f} tok/s, out: {len(out)} tok\n")
