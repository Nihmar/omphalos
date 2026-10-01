# omphalos — technical report

*Status as of 2026-10-01 (after #84). Companion to [PLAN.md](../PLAN.md): PLAN.md says what
the engine is meant to be and why; this report says what is built, the mathematics
behind it, and what was measured. Numbers come from `bench/results/`; every claim
names the issue where it was measured.*

## Contents

1. [Target and constraints](#1-target-and-constraints)
2. [The model as a computation](#2-the-model-as-a-computation)
3. [The decode roofline](#3-the-decode-roofline)
4. [Weights: block formats and lossless re-layouts](#4-weights-block-formats-and-lossless-re-layouts)
5. [The GEMV kernels](#5-the-gemv-kernels)
6. [Kernel boundaries: the launch-gap model](#6-kernel-boundaries-the-launch-gap-model)
7. [Gated DeltaNet](#7-gated-deltanet)
8. [Gated attention and flash-decoding](#8-gated-attention-and-flash-decoding)
9. [The quantized KV cache](#9-the-quantized-kv-cache)
10. [Validation](#10-validation)
11. [Results](#11-results)
12. [Method: what measurement taught us](#12-method-what-measurement-taught-us)
13. [Open work](#13-open-work)

---

## 1. Target and constraints

One model, one GPU: `Qwen3.8-27B` (GSQ-RCO GGUF, ~3.5 bits per weight on average) on an
AMD Radeon RX 9060 XT 16 GB (RDNA4, `gfx1200`, 32 CUs, wave32), ROCm/HIP.
Priorities, in order: **VRAM**, **decode speed**, **prefill speed**.

The non-negotiable constraint shapes everything below: the weights stay **bit-exact** with
the GGUF. Any re-layout must be a bijection on the stored bits; nothing is re-quantized.

Hardware figures used throughout (measured in M0, `bench/bw_membench.hip`):

| quantity | value |
|---|---|
| streaming-read bandwidth $B$ | $318.3$ GB/s (99.5 % of the 320 GB/s spec) |
| LDS per workgroup | 64 KiB |
| VRAM | 16 GiB |

---

## 2. The model as a computation

Hidden size $d = 5120$, vocabulary $V = 248\,320$, 64 layers in a fixed pattern: every
fourth layer is a **gated full-attention** layer (16 of them), the other 48 are **Gated
DeltaNet** layers (linear attention with a matrix-valued recurrent state). A 65th block is
the MTP (multi-token prediction) head, not yet used.

Every layer $\ell$ maps the residual stream $x \in \mathbb{R}^{d}$ as

$$
\begin{aligned}
u &= \mathrm{RMSNorm}(x;\,w^{\text{attn}}_\ell), &
r &= x + \mathrm{Mixer}_\ell(u), \\
v &= \mathrm{RMSNorm}(r;\,w^{\text{post}}_\ell), &
x' &= r + W^{\text{down}}_\ell\big(\mathrm{silu}(W^{\text{gate}}_\ell v)\odot W^{\text{up}}_\ell v\big),
\end{aligned}
$$

with $\mathrm{RMSNorm}(x; w)_i = x_i\,w_i \big/ \sqrt{\tfrac{1}{n}\sum_j x_j^2 + \varepsilon}$,
$\varepsilon = 10^{-6}$, $\mathrm{silu}(z) = z\,\sigma(z)$, and the FFN width
$d_{\text{ff}} = 17\,408$. The mixer is either the DeltaNet block (§7) or the attention
block (§8). After the last layer, $\mathrm{logits} = W^{\text{out}}\,\mathrm{RMSNorm}(x)$.

At decode time (one token) every projection is a **matrix–vector product** (GEMV), and the
step is dominated by reading the weight matrices once.

---

## 3. The decode roofline

A decode step must read every weight that participates in it. For this file that is

$$
W = 10.574\ \text{GiB} = 11.353\ \text{GB}
$$

(all of layers 0–63 and the output head; the token embedding is a row gather and the MTP
head is unused). With $B = 318.3$ GB/s the lower bound on the step is

$$
t_{\min} = \frac{W}{B} = 35.7\ \text{ms} \quad (\approx 28\ \text{tokens/s}).
$$

The achieved fraction $\eta = t_{\min}/t_{\text{step}}$ is the figure of merit. The step
today is $48.48$ ms, $\eta = 0.736$; llama.cpp on the same card decodes in $48.5$ ms
($\eta \approx 0.735$). §6 explains where the remaining quarter goes.

Everything that is not a weight read is, for decode, overhead: the activations are a few
kilobytes, the DeltaNet states are $48 \times 48 \times 128^2 \times 4$ B $= 151$ MB per step
(read and written once, §7), and the KV cache grows with the context (§9).

---

## 4. Weights: block formats and lossless re-layouts

### 4.1 The formats

The file mixes ten storage types, chosen per tensor by RCO under a bit budget. All quantize
blocks of $256$ weights (super-blocks) split into sub-blocks of 16 or 32:

| type | weight $w$ | bits/weight |
|---|---|---:|
| Q4_K | $w = d\,s_j\,q - d_{\min}\,m_j$, $q \in \{0..15\}$, 6-bit $s_j, m_j$ per 32 | 4.5 |
| Q2_K | $w = d\,s_j\,q - d_{\min}\,m_j$, $q \in \{0..3\}$, 4-bit $s_j, m_j$ per 16 | 2.625 |
| Q6_K | $w = d\,s_j\,(q - 32)$, $q \in \{0..63\}$ | 6.5625 |
| IQ4_XS | $w = d\,(s_j - 32)\,\tau(q)$, $\tau$ a fixed 16-entry non-uniform table | 4.25 |
| IQ3_S | $w = d\,(1 + 2s_j)\,\sigma\,g$, $g$ from a 512-entry 4-value grid | 3.4375 |
| IQ3_XXS | $w = \tfrac{d}{2}\,(\tfrac12 + s_j)\,\sigma\,g$, 256-entry 4-value grid | 3.0625 |
| IQ2_S / XS / XXS | $w = \tfrac{d}{4}\,(\tfrac12 + s_j)\,\sigma\,g$, 1024 / 512 / 256-entry 8-value grids | 2.56 / 2.31 / 2.06 |
| IQ1_M | $w = d_\ell\,(g + \delta)$, $g \in \{-1,0,1\}$, $\delta = \pm\tfrac18$ | 1.75 |

For the "IQ" types $g$ is a vector of small non-negative integers read from a codebook
(IQ3_S: odd values $1..15$; IQ3_XXS: $\{4, 12, \dots, 62\}$; IQ2: $\{8, 25, 43\}$) and
$\sigma \in \{\pm 1\}$ a per-weight sign. In IQ3_S and IQ2_S the signs are stored as
plain bits; in IQ3_XXS, IQ2_XXS and IQ2_XS each group of 8 weights stores 7 sign bits and
the 8th is their parity, $\sigma_7 = \bigoplus_{i<7} \sigma_i$, so that an even number of
weights are negative.

### 4.2 Re-layout as a bijection

A GGUF block is an array of structs (110 bytes for IQ3_S: $d$, 64 bytes of low index bits,
8 bytes of high bits, 32 sign bytes, 4 scale bytes). 110 is not a multiple of 16, so a lane
reading "its" block cannot use aligned 16-byte loads. At load time each tensor is
**repacked** into a struct of arrays — one stream per field, each 128-byte aligned — and
the inverse map (`unrepack_*`) is checked to reproduce the GGUF bytes exactly
(`omph-gemv-bench` does this for every tensor it touches). The total size grows by 0.7 %
(alignment of the Q4_K and IQ4_XS scale streams).

Within a stream the bits may also be **permuted**, as long as the permutation is fixed and
invertible. Two permutations are used, both designed so that the kernel needs fewer
instructions (§5.2):

* **Pair-ordered signs** (IQ3_S, IQ2_S; #63, #73). In the GGUF, bit $w$ of a sub-block's
  32-bit sign word is the sign of weight $w$. The repack moves it to
  $$
  \pi(w) = \begin{cases} 15 - w/2 & w \text{ even} \\ 31 - \lfloor w/2 \rfloor & w \text{ odd}\end{cases}
  $$
  so that the two sign bits of the weight pair $p = (2p', 2p'+1)$ end up, after a single
  left shift by $p'$, at bits 15 and 31 — exactly the sign bits of a packed `half2`.
* **Reversed sign fields** (IQ3_XXS, IQ2_XXS, IQ2_XS; #63, #75). The 7-bit sign index of a
  group is split into its even-weight bits (reversed, 4 bits) and odd-weight bits
  (reversed, 3 bits) placed in separate fields, so that the group's pair-ordered word is
  $$
  T = (E \ll 12)\ \big|\ (O \ll 29)\ \big|\ (\mathrm{popcount}(E\,|\,O \ll 4) \bmod 2) \ll 28.
  $$

A third re-layout — interleaving the four rows a warp reads together, block by block
(#77) — was implemented, measured faster in isolation and slower in the model, and
rejected (§12).

---

## 5. The GEMV kernels

### 5.1 Structure

For a weight matrix $W \in \mathbb{R}^{n \times k}$ stored in blocks, $y = W x$ is computed
with one wave32 per 4 output rows and one lane per 32-weight sub-block:

$$
y_r = \sum_{b}\ \underbrace{\mathrm{scale}_{r,b}}_{\text{f32}} \sum_{i=0}^{31} g_{r,b,i}\,\sigma_{r,b,i}\,x_{32b+i},
$$

lane $\lambda$ handling sub-blocks $b \equiv \lambda \pmod{32}$, followed by a wave
reduction. The activation $x$ (f16) is staged once per workgroup in LDS; the weights are
read exactly once from VRAM. Codebooks live in LDS.

### 5.2 The half2 codebook trick (#63)

The inner product per 8 weights used to cost ~55 instructions (convert each codebook byte
to f32, apply each sign, one FMA per weight). Two observations remove most of them.

**Bytes to f16 in two instructions.** For an integer $0 \le b < 1024$, the IEEE half with
bit pattern $\texttt{0x6400} \mid b$ has exponent field $25$, i.e. unbiased exponent $10$,
and therefore value
$$
2^{10}\left(1 + \frac{b}{1024}\right) = 1024 + b .
$$
So one byte-permute (`v_perm_b32`) that interleaves two codebook bytes with the constant
`0x64` builds the half2 $(1024 + b_0,\ 1024 + b_1)$, and one packed subtraction
(`v_pk_add_f16` of $-1024$) gives $(b_0, b_1)$ exactly.

**Signs by OR.** The codebook values are non-negative, so negating them is setting the
sign bit, and with pair-ordered signs (§4.2) the mask for pair $p'$ is
`(S << p') & 0x80008000`, combined with `v_and_or_b32`.

**Two products per instruction.** `v_dot2_f32_f16` computes $a_0 b_0 + a_1 b_1 + c$ with
f32 accumulation. The products are exact: a codebook value has at most 6 significant bits
and an f16 activation 11, so each product fits the 24-bit f32 significand. Only the
accumulation order differs from the scalar loop (KL $< 10^{-6}$ nats).

The IQ3_S inner loop went from 1179 to roughly 600 instructions per 128 weights. Q2_K, which has no codebook, uses
$$
\sum_i (d\,s\,c_i - d_{\min} m)\,x_i = d\,s \sum_i c_i x_i \;-\; d_{\min} m \sum_i x_i
$$
per 16-group, with the $\sum x_i$ shared by the four rows of the warp, and the 2-bit codes
of four consecutive weights extracted from one word as bytes
($(w \gg 2k)\ \&\ \texttt{0x03030303}$) before the same half2/dot2 path (#75).

### 5.3 LDS bank conflicts (#75)

Model the LDS as 64 four-byte banks, word address $a$ in bank $a \bmod 64$ (the
measurements below agree with it). A lane that
reads the 32 activations of its sub-block with 16-byte loads starts at word
$\lambda \cdot P$, where $P$ is the sub-block pitch in words. With the natural
$P = 16$ the lanes' start banks are $16\lambda \bmod 64 \in \{0, 16, 32, 48\}$: 8 lanes per
bank group, an 8-way conflict. With $P = 20$ the starts $20\lambda \bmod 64$ cycle through
16 distinct values ($\gcd(20, 64) = 4$), leaving the unavoidable 2-way conflict of 32 lanes
× 4 words on 64 banks. Q2_K had the worst case: one lane per 128-weight group,
$P = 64$ words, *all* lanes on one bank; a pitch of 66 words fixed it (1.65 → 1.29 ms
for all Q2_K tensors).

### 5.4 Where the kernels stand

Back-to-back over all tensors of a type, launches alternated over two streams (§6):

| type | GiB | time | of $B$ |
|---|---:|---:|---:|
| IQ3_S | 3.39 | 13.44 ms | 85 % |
| IQ4_XS | 2.86 | 10.43 ms | 92 % |
| IQ3_XXS | 1.93 | 8.03 ms | 81 % |
| Q4_K (blocks) | 0.83 | 3.20 ms | 88 % |
| Q4_K (output, one 701 MB kernel) | 0.68 | 2.34 ms | 98.5 % |
| IQ2_S / IQ2_XS / IQ2_XXS / Q2_K | 0.88 | 4.84 ms | 50–68 % |

IQ3_S and IQ3_XXS run within 6 % of a variant that only *loads* the weights: they are at
the ceiling of their access pattern, and the compute is hidden.

---

## 6. Kernel boundaries: the launch-gap model

A kernel that streams $W_k$ bytes takes, to first order,

$$
t_k = \frac{W_k}{\eta_k B} + \tau_k ,
$$

where $\tau_k$ collects the ramp-up (waves being dispatched before the memory pipeline is
full) and the tail (the last waves draining while most of the GPU idles). For the 700 MB
output projection $\tau$ is negligible and $\eta = 0.985$. A decode step, however, is
**~1300 kernels**, most of them a few tens of MB or less, so the $\tau$ terms add up.

The evidence (#71): the same IQ3_S tensors back to back reach 79 % of $B$ on one stream,
**96 %** when consecutive launches alternate over two streams (their ramps and tails
overlap), 93 % with three. A `rocprofv3` trace of a real step showed the GPU idle for
~6 ms in ~1130 gaps (median 4 µs) between kernels.

Three consequences were implemented:

1. **Overlap independent GEMVs** (#71). The projections that read the same input — DeltaNet
   $W^{qkv} \parallel W^{\text{gate}}, W^{\beta}, W^{\alpha}$; attention
   $W^{q} \parallel W^{k}, W^{v}$; FFN $W^{\text{gate}} \parallel W^{\text{up}}$ — are forked
   onto a second stream and joined before their consumer. The build uses HIP's per-thread
   default stream: on the legacy null stream every cross-stream wait serializes the device.
   54.83 → 51.00 ms.
2. **Fewer boundaries** (#66, #78, #79, #83). Chains of small elementwise kernels between
   GEMVs were fused into one kernel each (§7.3, §8.2): ~600 launches fewer per step
   (per-change timings in §11.1).
3. Things that did *not* pay, measured: splitting a dependent GEMV into two row halves on
   two streams (+1 ms: the extra launch and the event cost more than the overlapped tail),
   a persistent/grid-stride GEMV with exactly the resident workgroups (−1 ms on the bench,
   −0.13 ms in the model).

---

## 7. Gated DeltaNet

### 7.1 The recurrence

Each DeltaNet layer has 16 key heads and 48 value heads of size $s = 128$; value head $h$
uses key head $h \bmod 16$. Per head and token, with $q, k \in \mathbb{R}^{s}$ (L2-normalized),
$v \in \mathbb{R}^{s}$, a write strength $\beta \in (0,1)$ and a decay $\alpha \in (0,1)$, the
state $S \in \mathbb{R}^{s \times s}$ evolves by the gated delta rule

$$
S_t = \alpha_t\,S_{t-1} + k_t\,\delta_t^{\top},
\qquad
\delta_t = \beta_t\big(v_t - (\alpha_t S_{t-1})^{\top} k_t\big),
\qquad
o_t = \frac{1}{\sqrt{s}}\,S_t^{\top} q_t ,
$$

i.e. in the order the kernel executes it:

$$
S \leftarrow \alpha S, \qquad
\delta = \beta\,(v - S^{\top}k), \qquad
S \leftarrow S + k\,\delta^{\top}, \qquad
o = \tfrac{1}{\sqrt{s}}\,S^{\top} q .
$$

The gates come from two tiny projections and per-head parameters:
$\beta = \sigma(a_\beta)$ and $\alpha = \exp\!\big(A \cdot \mathrm{softplus}(a_\alpha + b_{dt})\big)$
with $A < 0$ and $\mathrm{softplus}(z) = \log(1 + e^{z})$ (computed as
$\log 1\mathrm{p}(e^{-|z|}) + \max(z, 0)$ for stability).

Before the recurrence, $q, k, v$ go through a depthwise causal convolution of width 4 over
time with a SiLU, and $q, k$ are L2-normalized per head:
$\hat q = q / \sqrt{\sum_i q_i^2 + \varepsilon}$ (implemented as
$\mathrm{RMSNorm}(q; \varepsilon/s)/\sqrt{s}$, the same quantity). After it, the output is
normalized with a gate: $y = \mathrm{RMSNorm}(o; w)\odot \mathrm{silu}(z)$, $z$ from its own
projection.

### 7.2 The state in registers (#66)

The state is the only large per-token read outside the weights: $128^2 \times 4$ B per head.
The first implementation walked it five times (decay with $S^{\top}k$, update, $S^{\top}q$,
and re-reads). One workgroup of 256 threads now owns a head: thread $(c, \text{half})$ keeps
the 64 elements $S_{64\,\text{half} + r,\ c}$ in registers, so $S$ is read once and written
once; the two column reductions $S^{\top}k$ and $S^{\top}q$ combine the two halves in LDS.
44 → 16 µs per layer.

### 7.3 One kernel per layer (#78, #83)

Between the projections and the output GEMV the decode used to launch nine kernels per layer
(sigmoid, softplus, ×A, conv, two norms, delta, gated norm, cast). They are now:

* `conv_silu_split_l2`: one 128-thread workgroup per $q$ head, $k$ head and 128-channel
  slice of $v$; a $q$/$k$ workgroup is exactly one head, so its L2 norm is a block reduction;
* the delta step, which resolves $\beta$ and $\alpha$ from the raw projections itself, and
  whose epilogue applies the gated RMSNorm to the head it just produced (the norm is per
  value head — exactly one workgroup's output) and writes the f16 input of $W^{\text{out}}$.

---

## 8. Gated attention and flash-decoding

### 8.1 The layer

24 query heads, 4 key/value heads (GQA, 6 queries per KV head), head dimension 256. The
query projection also produces an output gate $g$. Per head:

$$
\hat q = \mathrm{RoPE}\big(\mathrm{RMSNorm}(q; w^{q})\big), \quad
\hat k = \mathrm{RoPE}\big(\mathrm{RMSNorm}(k; w^{k})\big), \quad
\mathrm{out} = \sigma(g) \odot \sum_{j \le t} \mathrm{softmax}_j\!\Big(\tfrac{\hat q \cdot \hat k_j}{\sqrt{256}}\Big)\,v_j .
$$

RoPE is partial NeoX-style on the first $n_{\text{rot}} = 64$ dimensions: for
$i < 32$, with $\theta_i = p \,/\, 10^{7\,\cdot\,2i/64}$ at position $p$,

$$
\begin{pmatrix} x'_i \\ x'_{i+32}\end{pmatrix} =
\begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}
\begin{pmatrix} x_i \\ x_{i+32}\end{pmatrix}.
$$

### 8.2 One prep kernel (#79)

Split, both norms, both RoPEs, the Hadamard rotation (§9.2) of $q, k, v$ and the
quantization of $k, v$ into the cache (§9.1) are one launch: one 256-thread workgroup per
query, key and value head. Each of the 8 waves of a key/value workgroup holds exactly one
32-element quantization block, so the block maximum is a wave reduction.

### 8.3 GQA-grouped flash-decoding with split-K (#59)

A workgroup serves a tile of query tokens times the 6 query heads of one KV head, so every
K/V block is read and dequantized once for six heads. The key range is split into
$S$ chunks so that a single decode token still launches ≥ 128 workgroups. Each split $s$
runs the online softmax over its keys and emits, per query row, the triple

$$
m_s = \max_{j \in s} z_j, \qquad
\ell_s = \sum_{j \in s} e^{z_j - m_s}, \qquad
a_s = \sum_{j \in s} e^{z_j - m_s}\, v_j ,
$$

with $z_j = \hat q \cdot \hat k_j / \sqrt{256}$. A merge kernel combines them in a fixed
order (so the result is deterministic):

$$
M = \max_s m_s, \qquad
\mathrm{out} = \frac{\sum_s e^{m_s - M} a_s}{\sum_s e^{m_s - M} \ell_s},
$$

un-rotates (§9.2), applies the gate and writes f16. Within a split the scores of a 16-key
block are computed by thread (key, 16-dimension chunk) pairs with one 16-byte load each and
a 16-lane shuffle reduction. Decode step at 8k context: 99.0 → 61.7 ms (the old kernel
re-read K/V for each of the six heads with uncoalesced row reads).

A race found in the review (#55): the online softmax scanned the block's scores for the
maximum and overwrote them with weights without a barrier in between; threads could disagree
on $m$. With the barrier the engine became bit-deterministic run to run.

---

## 9. The quantized KV cache

### 9.1 Block quantization

K and V are stored per token and KV head in blocks of 32 along the head dimension, with an
f16 scale per block. For a block $x \in \mathbb{R}^{32}$:

$$
\textbf{K (Q8):}\quad s = \frac{\max_i |x_i|}{127}, \quad q_i = \mathrm{clamp}\big(\mathrm{round}(x_i/s), -127, 127\big), \quad \tilde x_i = s\,q_i ;
$$
$$
\textbf{V (Q4):}\quad s = \frac{\max_i |x_i|}{7}, \quad q_i = \mathrm{clamp}\big(\mathrm{round}(x_i/s), -8, 7\big), \quad \tilde x_i = s\,q_i .
$$

Per head and token: K 256 + 16 = 272 bytes, V 128 + 16 = 144 bytes, against 1024 each in
f32. For the 16 attention layers and 4 KV heads that is 26.6 kB per token: **0.87 GB at
32k** instead of 4.29 GB.

The round-to-nearest error is uniform on $[-s/2, s/2]$, with variance $s^2/12$; for a block
the relative error is governed by the ratio between the block's maximum and its typical
value. Errors in K move the scores (and enter through the softmax exponentially), errors
in V are averaged by the attention weights — hence 8 bits for K and 4 for V (PLAN §13.2).

### 9.2 The Hadamard rotation

Let $H_n$ be the Sylvester–Hadamard matrix, $H_1 = (1)$,
$H_{2n} = \begin{pmatrix} H_n & H_n \\ H_n & -H_n \end{pmatrix}$, normalized as
$\bar H = H_{256}/\sqrt{256}$. It is orthogonal and symmetric, so $\bar H^{-1} = \bar H^{\top} = \bar H$.
Rotating queries and keys leaves every score unchanged:

$$
(\bar H \hat q)\cdot(\bar H \hat k) = \hat q^{\top}\bar H^{\top}\bar H\,\hat k = \hat q\cdot\hat k ,
$$

and rotating the values commutes with the attention average, so the output is recovered by
one more rotation:

$$
\sum_j p_j\,(\bar H v_j) = \bar H \sum_j p_j\,v_j \quad\Longrightarrow\quad \mathrm{out} = \bar H\Big(\sum_j p_j\,\bar H v_j\Big).
$$

Why it helps quantization: each rotated coordinate is a $\pm$ sum of all 256 original ones,
$(\bar H x)_i = \tfrac{1}{16}\sum_j \pm x_j$. An outlier channel — common in keys after
RoPE — is spread evenly over all coordinates, so the block maxima drop toward the block
RMS and the quantization step $s$ with them. On the first measurement the V Q4 block
scales went from $0.088$–$0.253$ (outlier-dominated) to a uniform $0.10$–$0.14$.

The rotation must come **after** QK-norm and RoPE, which do not commute with $\bar H$. The
first implementation rotated before them (and rotated a stale $q$); the review found it
(#46): KL 0.803 → 0.0014 nats.

The transform costs $256 \log_2 256 = 2048$ additions per head (a butterfly in LDS) and is
fused into the prep kernel (§8.2) and the attention merge.

### 9.3 The FP16 window

The last 128 tokens are kept exactly (f16, rotated basis) in a ring, and the attention reads
them instead of their quantized form (PLAN §13.4). Cost: 8.4 MB. A position $p$ is in the
ring iff $p \ge \text{seq} - 128$ — relative to the sequence, not to the query: a prefill
chunk longer than the window overwrites its own early slots (#48).

### 9.4 K at 4 bits

With K in V's Q4 format the 32k cache would drop to 0.55 GB. Measured (#81) against the
exact cache, the KL roughly doubles (table in §11.2) — about llama.cpp's own q4_0/q4_0, but
over the budget. It is an option (`OMPH_KV_K4=1`), not the default.

---

## 10. Validation

The reference for every quality figure is the model run with an exact f32 KV cache. The
comparison metric is the Kullback–Leibler divergence of the next-token distributions,
averaged over positions:

$$
\overline{D}_{\mathrm{KL}} = \frac{1}{N}\sum_{n=1}^{N} \sum_{v=1}^{V} p_n(v)\,\log\frac{p_n(v)}{q_n(v)},
\qquad p_n = \mathrm{softmax}(\ell^{\text{ref}}_n),\ q_n = \mathrm{softmax}(\ell^{\text{test}}_n),
$$

computed in f64 by `tools/compare_logits.py`, together with the top-1 agreement
$\frac1N \sum_n [\arg\max p_n = \arg\max q_n]$.

At 32k the f32 cache (4.29 GB) does not fit next to the weights, so the reference keeps it
in pinned host RAM and stages one layer's rows into VRAM before each attention
(`OMPH_KV_HOST=1`; bit-identical to the device f32 cache where both fit). The budget is
llama.cpp's own KV quantization error against its f16 cache at the same context
(`bench/m5_llama_kv_kl.sh`).

Besides KL, the engine is checked against a CPU golden dump of llama.cpp (per-layer
outputs, attention and DeltaNet internals) and a NumPy reference decode (greedy tokens).

Noise floor: since #55 two runs of the engine are bit-identical; before, two identical f32
runs differed by KL $\approx 10^{-5}$.

---

## 11. Results

### 11.1 Decode speed

Decode step, 512-token wikitext prompt, default KV, GPU time, median of 15 steps; each row
measured A/B against its predecessor in one session (`bench/results/`):

| change | issue | ms/token | $\eta$ |
|---|---|---:|---:|
| after the post-M5 review | #57 | 60.5 | 0.59 |
| GQA flash-decoding | #59 | 58.4 | 0.61 |
| IQ1_M GEMV, fused RMSNorm, delta state in registers | #66 | 54.8 | 0.65 |
| independent GEMVs on two streams | #71 | 51.0 | 0.70 |
| IQ2_S kernel | #73 | 50.85 | 0.70 |
| IQ2_XS / IQ2_XXS / Q2_K kernels, LDS bank conflicts | #75 | 50.17 | 0.71 |
| DeltaNet chain fused | #78 | 49.23 | 0.73 |
| attention prep fused | #79 | 48.63 | 0.73 |
| gated norm in the delta step | #83 | **48.48** | **0.74** |
| llama.cpp (M0 baseline) | | 48.5 | 0.74 |

At 8k context: 99.0 → 58.2 ms (the attention no longer scales with the six GQA heads).

### 11.2 KV quantization quality

KL against the exact f32 cache, last 512 positions of a wikitext prompt, FP16 window 128:

| context | K8/V4 (default) | K4/V4 | llama.cpp q8_0/q4_0 (budget) | llama.cpp q4_0/q4_0 |
|---|---:|---:|---:|---:|
| 8k | 0.00078 | 0.00169 | 0.00162 | 0.00252 |
| 16k | 0.00091 | 0.00349 | 0.00149 | 0.00275 |
| 32k | 0.00072 | 0.00211 | 0.00135 | 0.00256 |

### 11.3 VRAM

Weights 11.36 GiB resident (the whole GGUF, embedding and MTP head included, plus 0.7 %
of alignment), of which 10.57 GiB are read per decode step; KV 26.6 kB per token
(0.87 GB at 32k), no f16 copy of any weight in the decode (the last one, a 178 MB cache for
the single IQ1_M tensor, went with its GEMV in #66).

---

## 12. Method: what measurement taught us

* **Measure the step, not the kernel.** Several changes that were faster in an isolated
  benchmark were neutral or slower in the model: the IQ3_S row interleave (+5 % alone,
  0.4 ms slower in the model, #77), the grid-stride GEMV, the IQ3_S/IQ3_XXS rewrites
  (−21 % alone, −0.7 ms together in the model). The A/B on the real step, alternating two
  binaries in one session, decides.
* **Profilers distort small kernels.** `rocprofv3`'s per-kernel times inflated some GEMVs by
  up to 40 %; the per-type cost in the model is measured by ablation
  (`OMPH_SKIP_GEMV_TYPE`): the step with and without that type's GEMVs.
* **Instruments need checking too.** The GEMV benchmark counted bytes with the Q4_K formula
  for every type (overstating IQ2_S by 1.7×), and the phase timer reused one event pair
  (every total was *calls × last interval*). The M3/M4 per-type efficiency figures relied
  on the first.
* **Removing a kernel is not the same as removing its time.** Skipping the SwiGLU kernel
  made the step slower; fusing a whole chain into one kernel made it faster. Boundaries are
  not additive; only whole chains measured cleanly.
* **Determinism is a test.** Once runs were bit-identical (#55), any difference between two
  binaries was a real difference.

---

## 13. Open work

* **MTP (M6)**: draft with the built-in MTP head and verify $k+1$ tokens per weight read.
  This is the only lever left that changes the roofline itself (§3): at acceptance rate
  $a$ and $k$ drafts the expected tokens per step are $\sum_{i=0}^{k} a^i$.
* **The DeltaNet conv inside the delta step** (#83, next): one kernel between the
  projections and the output GEMV.
* **Per-layer K precision** (PLAN §13.5): K4 only where a layer tolerates it.
* **Prefill** (M8): dequant + WMMA GEMM, chunked DeltaNet; today the prefill reuses the
  decode kernels or the f16 path.
