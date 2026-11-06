# PyTorch Coding Cheat Sheet (for ML live-coding interviews)

For someone who knows the math and needs to write **correct, vectorized PyTorch fast**, with the shape bookkeeping and gotchas interviewers probe. Every Python block below was executed against **torch 2.15 (CPU)**. GPU-only lines are guarded with `torch.cuda.is_available()` and marked `# GPU-only`.

**Contents**
1. [Tensors: creation, dtype/device, views & shapes](#1-tensors-creation-dtypedevice-views--shapes)
2. [Broadcasting, einsum, distances](#2-broadcasting-einsum-distances)
3. [Indexing, gather/scatter, sorting, counting, segments](#3-indexing-gatherscatter-sorting-counting-segments)
4. [Numerics & stability](#4-numerics--stability)
5. [Autograd](#5-autograd)
6. [nn.Module patterns](#6-nnmodule-patterns)
7. [Training loop, AMP, checkpoints, determinism](#7-training-loop-amp-checkpoints-determinism)
8. [Data: Dataset, DataLoader, collate, samplers](#8-data-dataset-dataloader-collate-samplers)
9. [Vision ops](#9-vision-ops)
10. [Attention](#10-attention)
11. [Distributed & torch.compile in one screen](#11-distributed--torchcompile-in-one-screen)
12. [Interview gotchas (38)](#12-interview-gotchas)
13. [Debugging & profiling](#13-debugging--profiling)

**Dimension letters used in `# shape` comments:** `B` batch · `T`/`S` query/key sequence length · `D` model dim · `H` heads (in attention) or height (in vision) · `Dh` head dim · `C` channels or classes · `V` vocab · `N`,`M` set sizes · `K` clusters/segments/top-k.

Every block assumes this preamble (each section was run as one script, top to bottom):

```python
import math, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 1. Tensors: creation, dtype/device, views & shapes

### Creation

```python
x = torch.tensor([[1., 2.], [3., 4.]])   # (2,2) float32. torch.tensor COPIES and infers dtype
i = torch.tensor([1, 2, 3])              # int64 ("Long"): Python ints -> int64, floats -> float32
assert x.dtype == torch.float32 and i.dtype == torch.int64

torch.zeros(2, 3); torch.ones(2, 3); torch.full((2, 3), 7.0)
torch.empty(2, 3)                        # uninitialized memory: garbage values
torch.arange(0, 10, 2)                   # [0,2,4,6,8] int64, end exclusive; torch.arange(5.) is float32
torch.linspace(0, 1, 5)                  # 5 points, both endpoints included
torch.eye(3); torch.rand(2, 3); torch.randn(2, 3)
torch.randint(0, 10, (2, 3))             # high is exclusive; size must be a tuple
torch.zeros_like(x); torch.randn_like(x) # same shape/dtype/device as x
x.new_zeros(4, 5); x.new_full((2,), -1.) # same dtype/device as x, new shape (handy inside forward)

a = np.ones(3)
t = torch.from_numpy(a)                  # SHARES memory with numpy, keeps float64
t2 = torch.as_tensor(a, dtype=torch.float32)  # converts (copies only if needed)
back = t2.detach().cpu().numpy()         # .numpy() needs CPU and no grad; shares memory
```

### dtype / device / `.to`

```python
x = torch.randn(4, 3)
x = x.to(device)                 # returns a NEW tensor: you must assign it (no-op if already there)
x = x.to(torch.bfloat16)         # dtype change
x = x.float()                    # shortcuts: .half() .bfloat16() .double() .long() .int() .bool()
y = torch.zeros(3).to(x)         # match dtype AND device of another tensor
print(x.dtype, x.device, x.shape, x.ndim, x.numel())

model = nn.Linear(3, 2).to(device)   # Modules: .to() works IN PLACE (and returns self)
if torch.cuda.is_available():        # GPU-only: async host->device copy needs pinned memory
    xg = torch.randn(4, 3).pin_memory().to("cuda", non_blocking=True)
```

### `no_grad` vs `inference_mode`

```python
w = torch.randn(3, requires_grad=True)
with torch.no_grad():
    y = w * 2                    # no graph recorded
assert not y.requires_grad

with torch.inference_mode():     # like no_grad, a bit faster (skips view/version tracking)
    z = w * 2                    # z is an "inference tensor"
try:
    (w * z).sum().backward()     # inference tensors can't be saved for backward
except RuntimeError as e:
    print("inference tensor in autograd:", str(e)[:50])
# Rule: inference_mode for pure eval/serving; no_grad if the outputs feed later training (e.g. teacher targets).

@torch.no_grad()                 # decorator form; also @torch.inference_mode()
def predict(model, x):
    return model(x)
```

### Contiguity, `view` vs `reshape`

```python
x = torch.arange(6).reshape(2, 3)        # (2,3), stride (3,1): contiguous
y = x.t()                                # (3,2), stride (1,3): a VIEW (same storage), NOT contiguous
assert not y.is_contiguous() and y.data_ptr() == x.data_ptr()
try:
    y.view(6)                            # view needs compatible strides
except RuntimeError as e:
    print("view failed:", str(e)[:45])
y.reshape(6)                             # works: copies only when it must
y.contiguous().view(6)                   # explicit copy, then view
x.view(3, 2); x.view(-1); x.view(2, -1); x.flatten()  # -1 = infer that dim
```

| op | copies? | works on non-contiguous? | note |
|---|---|---|---|
| `view` | never | only if strides allow | guaranteed alias of input |
| `reshape` | if needed | yes | don't rely on it aliasing (or not) |
| `contiguous()` | if not contiguous | yes | needed before `.view` after `permute`/`transpose` |
| `permute`/`transpose`/`expand`/slicing | never | yes | return views with new strides |

### `permute` / `transpose`

```python
img = torch.randn(8, 32, 32, 3)          # (B,H,W,C) channels-last (numpy/PIL order)
x = img.permute(0, 3, 1, 2)              # (B,C,H,W) channels-first: view, non-contiguous
x = x.contiguous()                       # if you'll .view() it, or for speed
q = torch.randn(2, 4, 10, 16)            # (B,H,T,Dh)
assert q.transpose(1, 2).shape == (2, 10, 4, 16)   # (B,T,H,Dh): swaps exactly two dims
assert q.mT.shape == (2, 4, 16, 10)      # batched matrix transpose (last two dims); x.T is for 2-D only
```

### `expand` vs `repeat` vs `repeat_interleave`

```python
v = torch.tensor([1, 2, 3])              # (3,)
e = v.expand(4, 3)                       # (4,3) NO copy: stride 0 on the new dim
assert e.stride() == (0, 1)
v[None].expand(4, -1)                    # -1 = keep that dim; only size-1 (or new leading) dims can expand
try:
    e.add_(1)                            # in-place write into an expanded tensor
except RuntimeError as err:
    print("expand + in-place:", str(err)[:40])     # -> clone() first

v.repeat(2)                              # [1,2,3,1,2,3]  tile (copies); args = repeat count per dim
assert v.repeat(2, 1).shape == (2, 3)
v.repeat_interleave(2)                   # [1,1,2,2,3,3]  each element repeated (= np.repeat)
torch.repeat_interleave(v, torch.tensor([2, 0, 1]))   # [1,1,3] per-element counts

kv = torch.randn(2, 2, 10, 16)           # (B, H_kv, T, Dh): grouped-query attention, 8 q heads
assert kv.repeat_interleave(4, dim=1).shape == (2, 8, 10, 16)   # q head h uses kv head h // 4
```

### `unsqueeze` / `squeeze` pitfalls

```python
x = torch.randn(3)
assert x.unsqueeze(0).shape == x[None].shape == (1, 3)
assert x[:, None].shape == x[..., None].shape == (3, 1)

b = torch.randn(1, 5, 1)                 # a batch of ONE
assert b.squeeze().shape == (5,)         # squeezes ALL size-1 dims, including the batch dim!
assert b.squeeze(-1).shape == (1, 5)     # always pass dim
assert b.squeeze((0, 2)).shape == (5,)   # tuple of dims (2.0+)
assert x.squeeze(0).shape == (3,)        # squeezing a non-size-1 dim is a SILENT no-op
```

### Combine, split, reduce

```python
a, b = torch.randn(2, 3), torch.randn(2, 3)
assert torch.cat([a, b], dim=0).shape == (4, 3)      # existing dim grows
assert torch.stack([a, b], dim=0).shape == (2, 2, 3) # NEW dim

x = torch.randn(2, 12)
x.chunk(3, dim=-1)                       # 3 tensors of (2,4): arg = NUMBER of chunks
x.split(4, dim=-1); x.split([2, 10], dim=-1)   # arg = chunk SIZE, or explicit sizes
q, k, v = torch.randn(2, 5, 3 * 8).chunk(3, dim=-1)   # each (2,5,8)
assert x.unflatten(-1, (3, 4)).shape == (2, 3, 4)
assert torch.randn(2, 3, 4, 5).flatten(1).shape == (2, 60)   # flatten from start_dim=1

assert x.sum(dim=-1, keepdim=True).shape == (2, 1)  # keepdim -> result broadcasts back against x
x.amax(dim=(0, 1))                       # multi-dim max (x.max(dim) takes one dim and returns (values, indices))
x.mean(-1); x.norm(dim=-1)
x.var(-1)                                # UNBIASED (N-1) by default
x.var(-1, correction=0)                  # biased (N), what LayerNorm/BatchNorm use
```

---

## 2. Broadcasting, einsum, distances

**Rules:** align shapes from the **right**; each dim pair must be equal or one of them `1` (or missing); the result takes the max. Size-1 dims are stretched for free (no copy).

| a | b | result |
|---|---|---|
| `(B,C,H,W)` | `(C,1,1)` | `(B,C,H,W)`: per-channel op |
| `(B,C,H,W)` | `(C,)` | aligns `C` with **W**: error, or silently wrong if `C == W` |
| `(N,1)` | `(N,)` | **`(N,N)`**: classic silent bug in losses |
| `(N,1,D)` | `(1,M,D)` | `(N,M,D)`: all pairs |

### Pairwise distances

```python
N, M, D = 5, 7, 3
X, Y = torch.randn(N, D), torch.randn(M, D)

diff = X[:, None, :] - Y[None, :, :]     # (N,1,D) - (1,M,D) -> (N,M,D)
D2 = (diff ** 2).sum(-1)                 # (N,M) squared distances; O(N*M*D) memory

# GEMM version, O(N*M) memory: ||x||^2 + ||y||^2 - 2 x.y
D2b = (X ** 2).sum(-1, keepdim=True) + (Y ** 2).sum(-1) - 2 * X @ Y.T   # (N,1)+(M,)-(N,M) -> (N,M)
D2b = D2b.clamp_min(0)                   # cancellation can give tiny negatives -> sqrt(neg) = NaN
assert torch.allclose(D2, D2b, atol=1e-5)

dist = torch.cdist(X, Y)                 # (N,M) Euclidean (p=2); batched: (B,N,D),(B,M,D) -> (B,N,M)
assert torch.allclose(dist, D2.sqrt(), atol=1e-5)
nn_idx = dist.argmin(dim=1)              # (N,) nearest Y for each X
knn_d, knn_i = dist.topk(3, dim=1, largest=False)   # (N,3) 3 nearest
cos = F.cosine_similarity(X[:, None], Y[None], dim=-1)   # (N,M) via broadcasting
```

### Outer products & contrastive (CLIP) logits

```python
a, b = torch.randn(4), torch.randn(6)
outer = a[:, None] * b[None, :]          # (4,1)*(1,6) -> (4,6)
assert torch.allclose(outer, torch.outer(a, b))
A, Bm = torch.randn(2, 4), torch.randn(2, 6)
batched_outer = A[:, :, None] * Bm[:, None, :]   # (2,4,6)

img = F.normalize(torch.randn(8, 512), dim=-1)   # (B,D) unit-norm image embeddings
txt = F.normalize(torch.randn(8, 512), dim=-1)   # (B,D) unit-norm text embeddings
logits = img @ txt.T / 0.07                      # (B,B) cosine sim / temperature
labels = torch.arange(8)                         # matching pairs on the diagonal
clip_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
```

### `matmul` broadcasting

```python
assert (torch.randn(8, 5, 3) @ torch.randn(3, 4)).shape == (8, 5, 4)          # batch dims broadcast
assert (torch.randn(8, 1, 5, 3) @ torch.randn(6, 3, 2)).shape == (8, 6, 5, 2)
assert (torch.randn(5, 3) @ torch.randn(3)).shape == (5,)                      # 1-D = vector
torch.bmm(torch.randn(8, 5, 3), torch.randn(8, 3, 4))   # strict 3-D, no broadcasting; torch.mm strict 2-D
```

### `einsum` cookbook

| operation | einsum | equivalent |
|---|---|---|
| batched matmul | `bij,bjk->bik` | `A @ B` |
| attention scores | `bhqd,bhkd->bhqk` | `q @ k.mT` |
| attention output | `bhqk,bhkd->bhqd` | `p @ v` |
| outer product | `i,j->ij` | `torch.outer(a, b)` |
| row-wise dot | `nd,nd->n` | `(x * y).sum(-1)` |
| all-pairs dot | `nd,md->nm` | `x @ y.T` |
| bilinear `xᵀWy` per row | `bi,ij,bj->b` | `((x @ W) * y).sum(-1)` |
| `nn.Bilinear` | `bi,oij,bj->bo` | `F.bilinear(x, y, W)` |
| trace / diagonal | `ii->` / `ii->i` | `A.trace()` / `A.diagonal()` |
| 1×1 conv (channel mix) | `bchw,oc->bohw` | `F.conv2d(x, W[:, :, None, None])` |
| any leading dims | `...d,de->...e` | `x @ W` |

```python
B, H, T, S, Dh = 2, 4, 5, 6, 8
q, k, v = torch.randn(B, H, T, Dh), torch.randn(B, H, S, Dh), torch.randn(B, H, S, Dh)
scores = torch.einsum("bhqd,bhkd->bhqk", q, k)          # (B,H,T,S)
assert torch.allclose(scores, q @ k.mT, atol=1e-5)
p = scores.softmax(-1)
out = torch.einsum("bhqk,bhkd->bhqd", p, v)             # (B,H,T,Dh)
assert torch.allclose(out, p @ v, atol=1e-5)

x, y, W = torch.randn(B, 3), torch.randn(B, 3), torch.randn(3, 3)
bil = torch.einsum("bi,ij,bj->b", x, W, y)              # (B,)  x_b^T W y_b
assert torch.allclose(bil, ((x @ W) * y).sum(-1), atol=1e-5)
Wo = torch.randn(5, 3, 3)                               # (out, in1, in2), like nn.Bilinear.weight
assert torch.allclose(torch.einsum("bi,oij,bj->bo", x, Wo, y), F.bilinear(x, y, Wo), atol=1e-5)

A = torch.randn(4, 4)
assert torch.allclose(torch.einsum("ii->", A), A.trace())
assert torch.allclose(torch.einsum("ii->i", A), A.diagonal())

fm, Wc = torch.randn(B, 3, 8, 8), torch.randn(16, 3)    # (B,C,H,W), (O,C)
assert torch.allclose(torch.einsum("bchw,oc->bohw", fm, Wc), F.conv2d(fm, Wc[:, :, None, None]), atol=1e-5)
assert torch.allclose(torch.einsum("...d,de->...e", q, torch.eye(Dh)), q)
```

Gotchas
- A letter that appears in inputs but not in the output is **summed**. Letters are single characters.
- `einsum` is great for clarity in interviews. For hot paths, `@`/`bmm` is at least as fast.

---

## 3. Indexing, gather/scatter, sorting, counting, segments

### Masks, `masked_fill`, `where`, advanced indexing

```python
x = torch.randn(4, 5)
mask = x > 0                              # (4,5) bool
sel = x[mask]                             # 1-D (num_true,): boolean indexing FLATTENS
y = x.masked_fill(~mask, 0.0)             # (4,5) out-of-place; mask may broadcast, e.g. (4,1)
y2 = torch.where(mask, x, 0.0)            # elementwise select (scalars allowed)
assert torch.equal(y, y2)
rows, cols = torch.where(mask)            # 1-arg form = indices of True (like nonzero(as_tuple=True))
mask.nonzero().shape                      # (num_true, 2)
x[mask.any(dim=1)]                        # keep rows that have any positive

x[[0, 2]]                                 # rows 0 and 2 -> (2,5) (advanced indexing COPIES)
x[:, [1, 3]]                              # (4,2)
x[torch.tensor([0, 2]), torch.tensor([1, 3])]   # elements (0,1) and (2,3) -> (2,)  NOT a submatrix
x[torch.tensor([0, 2])[:, None], torch.tensor([1, 3])]   # (2,2) submatrix via broadcasting indices

labels = torch.tensor([1, 0, 4, 2])
true_logit = x[torch.arange(4), labels]   # (4,) per-row pick: logit of the true class

z = torch.zeros(3); idx = torch.tensor([0, 0, 1])
z[idx] += 1                               # duplicates do NOT accumulate -> [1,1,0]
assert z.tolist() == [1, 1, 0]
z2 = torch.zeros(3).index_put_((idx,), torch.ones(3), accumulate=True)   # [2,1,0]
assert z2.tolist() == [2, 1, 0]
```

### `gather` / `take_along_dim` (read along a dim)

`out[i][j] = x[i][idx[i][j]]` for `dim=1`. `idx` is int64 and has the same **ndim** as `x`.

```python
Bsz, C = 4, 6
logits = torch.randn(Bsz, C); y = torch.randint(0, C, (Bsz,))
picked = logits.gather(1, y[:, None]).squeeze(1)          # (B,1) -> (B,)
assert torch.equal(picked, logits[torch.arange(Bsz), y])

T, V = 5, 11                                              # LM per-token log-prob of targets
logp = torch.randn(Bsz, T, V).log_softmax(-1)             # (B,T,V)
tgt = torch.randint(0, V, (Bsz, T))                       # (B,T)
tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) # (B,T)
assert torch.equal(tok_logp, torch.take_along_dim(logp, tgt[..., None], dim=-1).squeeze(-1))
assert torch.allclose(-tok_logp.mean(), F.cross_entropy(logp.reshape(-1, V), tgt.reshape(-1)))
```

### `scatter_` / `scatter_add_` / `scatter_reduce_` / `index_add_` (write along a dim)

`out[i][idx[i][j]] = src[i][j]` for `dim=1`.

```python
Bsz, C = 4, 6
y = torch.tensor([1, 0, 5, 2])
onehot = torch.zeros(Bsz, C).scatter_(1, y[:, None], 1.0)    # (B,C)
assert torch.equal(onehot, F.one_hot(y, num_classes=C).float())   # F.one_hot returns int64

vals = torch.tensor([1., 2., 3., 4.]); seg = torch.tensor([0, 2, 0, 1])
sums = torch.zeros(3).scatter_add_(0, seg, vals)             # [4,4,2]  duplicates accumulate
maxs = torch.zeros(3).scatter_reduce_(0, seg, vals, reduce="amax", include_self=False)  # [3,4,2]
# reduce in {"sum","prod","mean","amax","amin"}. Plain scatter_ with duplicate indices: an arbitrary one wins.

E = torch.randn(10, 4)                                       # (V,D) embedding table
ids = torch.tensor([3, 3, 7])
rows = E.index_select(0, ids)                                # (3,4) == E[ids]
g = torch.ones(3, 4)
acc = torch.zeros(10, 4).index_add_(0, ids, g)               # row 3 gets 2, row 7 gets 1 (sparse-grad style)
assert acc[3, 0] == 2 and acc[7, 0] == 1
```

### Segment ops (sum / mean / max / softmax) and k-means

```python
N, D, K = 8, 3, 3
vals = torch.randn(N, D); seg = torch.tensor([0, 0, 1, 2, 2, 2, 1, 0])   # segment id per row
seg_sum = torch.zeros(K, D).index_add_(0, seg, vals)                      # (K,D)
counts = torch.bincount(seg, minlength=K)                                 # (K,)
seg_mean = seg_sum / counts.clamp_min(1)[:, None]                         # avoid /0 for empty segments
seg_max = torch.full((K, D), float("-inf")).scatter_reduce(
    0, seg[:, None].expand(-1, D), vals, reduce="amax")                   # (K,D)

s = torch.randn(N)                                                        # softmax within each segment
m = torch.full((K,), float("-inf")).scatter_reduce(0, seg, s, reduce="amax")
ex = (s - m[seg]).exp()                                                   # subtract per-segment max
seg_softmax = ex / torch.zeros(K).index_add_(0, seg, ex)[seg]
assert torch.allclose(torch.zeros(K).index_add_(0, seg, seg_softmax), torch.ones(K))

X = torch.randn(100, 2); C = X[torch.randperm(100)[:K]]                   # k-means
for _ in range(10):
    assign = torch.cdist(X, C).argmin(1)                                  # (N,)
    C = torch.zeros(K, 2).index_add_(0, assign, X) / torch.bincount(assign, minlength=K).clamp_min(1)[:, None]
```

### `topk`, `sort`, `argsort`, ranks

```python
x = torch.tensor([[3., 1., 4., 1., 5.]])
vals, idx = x.topk(2, dim=-1)                 # largest=True, sorted=True -> [[5,4]], [[4,2]]
svals, sidx = x.sort(dim=-1, descending=True)
order = x.argsort(dim=-1)
ranks = x.argsort(dim=-1).argsort(dim=-1)     # rank of each element (0 = smallest)
assert torch.equal(x.gather(-1, sidx), svals)
```

### Top-k / top-p (nucleus) sampling

```python
def sample_next(logits, k=None, p=None, temperature=1.0):
    """logits: (B,V) -> token ids (B,)"""
    logits = logits / temperature
    if k is not None:
        kth = logits.topk(k, dim=-1).values[:, -1:]                  # (B,1) k-th largest
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if p is not None:
        sl, si = logits.sort(dim=-1, descending=True)                # (B,V)
        probs = sl.softmax(-1)
        remove = probs.cumsum(-1) - probs > p                        # mass BEFORE token > p (top-1 always kept)
        sl = sl.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, si, sl)   # un-sort
    return torch.multinomial(logits.softmax(-1), num_samples=1).squeeze(-1)

tok = sample_next(torch.randn(3, 50), k=10, p=0.9, temperature=0.8)   # (3,)
greedy = torch.randn(3, 50).argmax(-1)                                 # (3,)
```

### `bincount`, one-hot, `unique`, confusion matrix

```python
y_true = torch.tensor([0, 2, 1, 2, 2]); y_pred = torch.tensor([0, 1, 1, 2, 0]); C = 3
class_counts = torch.bincount(y_true, minlength=C)              # [1,1,3]; 1-D non-negative ints only
cm = torch.bincount(y_true * C + y_pred, minlength=C * C).view(C, C)   # rows = true, cols = pred
acc = cm.diag().sum() / cm.sum()

t = torch.tensor([10, 30, 10, 20, 30, 30])
uniq, inv, cnt = torch.unique(t, return_inverse=True, return_counts=True)   # sorted unique
# uniq=[10,20,30], inv=[0,2,0,1,2,2] (dense relabel 0..K-1), cnt=[2,1,3]
assert torch.equal(uniq[inv], t)
```

### `cumsum` tricks

```python
lengths = torch.tensor([3, 1, 2]); Bsz, T = len(lengths), int(lengths.max())
mask = torch.arange(T)[None, :] < lengths[:, None]               # (B,T) True = real token
offsets = lengths.cumsum(0) - lengths                            # exclusive prefix sum [0,3,4]
seg_ids = torch.repeat_interleave(torch.arange(Bsz), lengths)    # [0,0,0,1,2,2]
packed = torch.arange(6.)                                        # (sum(lengths),) packed sequences
padded = torch.zeros(Bsz, T); padded[mask] = packed              # packed -> padded (row-major fill)
assert torch.equal(padded[mask], packed)                         # padded -> packed

am = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])                  # left-padded attention mask
pos_ids = (am.cumsum(-1) - 1).clamp_min(0)                       # [[0,0,0,1],[0,1,2,3]]

r = torch.tensor([1., 2., 3.])
assert r.flip(-1).cumsum(-1).flip(-1).tolist() == [6., 5., 3.]   # reverse cumsum (reward-to-go)
betas = torch.linspace(1e-4, 0.02, 1000)
alpha_bar = torch.cumprod(1 - betas, dim=0)                      # (1000,) diffusion schedule
```

---

## 4. Numerics & stability

### log-sum-exp, log-softmax, log-sigmoid, softplus

```python
x = torch.tensor([0., 200.])
print(torch.log(torch.softmax(x, -1)))   # [-inf, 0.]: softmax underflowed to 0, then log -> -inf (NaN grads)
print(F.log_softmax(x, -1))              # [-200., 0.]: computed as x - logsumexp(x)

z = torch.randn(4, 10) * 50
lse = torch.logsumexp(z, dim=-1)                                     # (4,)
m = z.amax(-1, keepdim=True)
lse_manual = (m + (z - m).exp().sum(-1, keepdim=True).log()).squeeze(-1)   # subtract max first
assert torch.allclose(lse, lse_manual)
assert torch.allclose(F.log_softmax(z, -1), z - lse[:, None], atol=1e-4)

logit, tgt = torch.randn(6) * 30, torch.randint(0, 2, (6,)).float()
# log(sigmoid(x)) = logsigmoid(x);  log(1 - sigmoid(x)) = logsigmoid(-x)
bce_manual = -(tgt * F.logsigmoid(logit) + (1 - tgt) * F.logsigmoid(-logit)).mean()
assert torch.allclose(bce_manual, F.binary_cross_entropy_with_logits(logit, tgt))
assert torch.allclose(F.softplus(logit), -F.logsigmoid(-logit))      # softplus(x) = log(1 + e^x)
sigma = F.softplus(torch.randn(3)) + 1e-4     # positivity constraint for a predicted std/variance
torch.log1p(torch.tensor(1e-10)); torch.expm1(torch.tensor(1e-10))   # accurate near 0
```

### eps handling: `clamp_min`, `xlogy`, `kl_div`

```python
v = torch.zeros(2, 3)
unit = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)   # what F.normalize does; no 0/0
assert torch.equal(unit, F.normalize(v, dim=-1))
p = torch.tensor([0.0, 0.5, 0.5])
safe_log = torch.log(p.clamp_min(1e-8))                     # never log(0)
entropy = -torch.xlogy(p, p).sum()                           # xlogy defines 0*log(0) = 0

w = torch.zeros(3, requires_grad=True)
torch.sqrt((w ** 2).sum()).backward(); print(w.grad)         # NaN: d sqrt/dx at 0 is inf, times 0
w.grad = None
w.norm().backward(); print(w.grad)                           # 0s: torch.norm handles the 0 case
# manual: torch.sqrt(x.clamp_min(eps)) or (x + eps).sqrt()

P = torch.softmax(torch.randn(4, 5), -1); logQ = torch.log_softmax(torch.randn(4, 5), -1)
kl = F.kl_div(logQ, P, reduction="batchmean")     # KL(P || Q): input = LOG-probs of Q, target = probs of P
assert torch.allclose(kl, (P * (P.log() - logQ)).sum(-1).mean())   # "batchmean", not "mean" (= /numel)
```

### Fully-masked rows: `-inf` + softmax = NaN

```python
scores = torch.randn(2, 4)
keep = torch.tensor([[True, True, False, False],
                     [False, False, False, False]])          # row 1 has nothing to attend to
p = scores.masked_fill(~keep, float("-inf")).softmax(-1)
print(p[1])                                                  # all NaN: exp(-inf - (-inf)) = exp(nan)

# Fix A: softmax, then zero the NaN rows (or torch.nan_to_num(p, nan=0.0))
p_a = torch.nan_to_num(p, nan=0.0)
# Fix B: fill with a large finite negative, then zero masked probabilities
neg = torch.finfo(scores.dtype).min
p_b = scores.masked_fill(~keep, neg).softmax(-1).masked_fill(~keep, 0.0)
assert torch.allclose(p_a, p_b) and p_b[1].sum() == 0
# Fix C: guarantee >= 1 valid key per query (causal masks always allow the diagonal)
```

Gotchas
- **Additive** masks with `finfo.min` can overflow: `score + finfo.min` becomes `-inf` when the score is negative, which brings back the NaN. `masked_fill` *replaces* the value, so it is safe. `-1e9` overflows to `-inf` in fp16 (max 65504). Use `-1e4` or `finfo(dtype).min`.
- In torch 2.15 (verified on CPU), `F.scaled_dot_product_attention` returns **0** for a fully masked row. A hand-written softmax returns **NaN**.

### float16 vs bfloat16 vs float32

| dtype | exp/mantissa bits | max | smallest normal | eps (rel. precision) | notes |
|---|---|---|---|---|---|
| float32 | 8 / 23 | 3.4e38 | 1.2e-38 | 1.19e-7 | default |
| float16 | 5 / 10 | **65504** | 6.1e-5 | 9.8e-4 | overflows/underflows: needs a GradScaler |
| bfloat16 | 8 / 7 | 3.4e38 | 1.2e-38 | **7.8e-3** | fp32 range, low precision: no scaler needed |

```python
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt)
    print(dt, fi.bits, fi.max, fi.tiny, fi.eps)            # tiny = smallest normal
torch.iinfo(torch.int64).max

print(torch.tensor(70000., dtype=torch.float16))           # inf   (overflow)
print(torch.tensor(1e-8, dtype=torch.float16))             # 0.    (underflow: why fp16 grads need scaling)
print(torch.tensor(256., dtype=torch.bfloat16) + 1)        # 256.  (bf16 can't represent 257)
xb = torch.randn(10000).bfloat16()
xb.sum(dtype=torch.float32)                                # accumulate reductions in fp32
torch.softmax(xb, -1, dtype=torch.float32)                 # softmax in fp32
```

---

## 5. Autograd

### Leaves, `.grad` accumulation, zeroing

```python
w = torch.randn(3, requires_grad=True)       # leaf (created by the user)
x = torch.randn(3)
y = (w * x).sum()                            # non-leaf: has grad_fn
assert w.is_leaf and not y.is_leaf and y.grad_fn is not None
y.backward()
assert torch.allclose(w.grad, x)
(w * x).sum().backward()                     # grads ACCUMULATE (+=)
assert torch.allclose(w.grad, 2 * x)
w.grad = None                                # what optimizer.zero_grad(set_to_none=True) does (default since 2.0)

h = w * 2; h.retain_grad()                   # non-leaf .grad is None unless retain_grad()
out = h ** 2                                 # (3,) non-scalar
out.backward(torch.ones_like(out))           # non-scalar needs a grad_output (or call .sum().backward())
assert h.grad is not None

bad = torch.randn(3, requires_grad=True) * 0.01   # result of an op -> NOT a leaf -> .grad stays None
assert not bad.is_leaf
good = (torch.randn(3) * 0.01).requires_grad_()   # leaf; or nn.Parameter(torch.randn(3) * 0.01)
assert good.is_leaf
```

### `detach` vs `no_grad`, straight-through estimator

```python
w = torch.randn(4, requires_grad=True)
target = (w * 2).detach()          # stop-gradient for ONE tensor (shares storage): EMA/target nets, BYOL
with torch.no_grad():              # no graph for a whole BLOCK: eval, manual parameter updates
    w -= 0.1 * torch.ones(4)       # in-place update of a leaf is allowed here

z = torch.randn(4, requires_grad=True)
z_q = torch.round(z)               # non-differentiable quantization (VQ-VAE codebook lookup, binarization)
z_st = z + (z_q - z).detach()      # forward value = z_q, backward = identity
z_st.sum().backward()
assert torch.equal(z_st, z_q) and torch.equal(z.grad, torch.ones(4))
```

### `retain_graph`, `autograd.grad`, higher order, gradient penalty

```python
w = torch.randn(3, requires_grad=True)
y = w.exp().sum()
y.backward(retain_graph=True)      # keep saved tensors for a second backward
y.backward()                       # OK; without retain_graph above:
try:                               # "Trying to backward through the graph a second time"
    y.backward()
except RuntimeError as e:
    print(str(e)[:55])

x = torch.randn(5, requires_grad=True)
(g,) = torch.autograd.grad((x ** 3).sum(), x, create_graph=True)   # 3x^2, still differentiable
(g2,) = torch.autograd.grad(g.sum(), x)                            # 6x
assert torch.allclose(g2, 6 * x)
# autograd.grad returns grads instead of writing .grad; allow_unused=True if some inputs are unused

def gradient_penalty(D, real, fake):                               # WGAN-GP
    eps = torch.rand(real.size(0), 1, 1, 1)                        # (B,1,1,1) broadcast over C,H,W
    xh = (eps * real + (1 - eps) * fake).requires_grad_(True)
    (gr,) = torch.autograd.grad(D(xh).sum(), xh, create_graph=True)   # create_graph: penalty is trained
    return ((gr.flatten(1).norm(dim=1) - 1) ** 2).mean()

Dnet = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.SiLU(), nn.Flatten(), nn.Linear(8 * 8 * 8, 1))
gp = gradient_penalty(Dnet, torch.randn(4, 3, 8, 8), torch.randn(4, 3, 8, 8))
gp.backward()                                                      # reaches Dnet's params (double backward)
assert Dnet[0].weight.grad is not None
```

### Custom `autograd.Function` + `gradcheck`

```python
class SiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)                 # tensors -> save_for_backward; non-tensors -> ctx.attr
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_out):                 # one grad_out per forward output
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * (s + x * s * (1 - s))  # one return per forward input

class GradReverse(torch.autograd.Function):      # DANN domain adaptation
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None              # None for the non-tensor input

x = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)   # gradcheck: float64!
assert torch.autograd.gradcheck(SiLU.apply, (x,), eps=1e-6, atol=1e-4)
assert torch.allclose(SiLU.apply(x), F.silu(x))
GradReverse.apply(x, 0.5).sum().backward()
assert torch.allclose(x.grad, torch.full_like(x, -0.5))
```

### In-place errors and how to fix them

```python
a = torch.randn(3, requires_grad=True)
b = a.sigmoid()            # sigmoid saves its OUTPUT for backward
b.mul_(2)                  # in-place op bumps b's version counter
try:
    b.sum().backward()
except RuntimeError as e:
    print(str(e)[:95])     # "...modified by an inplace operation..."
b = a.sigmoid() * 2        # fix: out-of-place
b.sum().backward()

p = torch.randn(3, requires_grad=True)
try:
    p[p < 0] = 0           # "a leaf Variable that requires grad is being used in an in-place operation"
except RuntimeError as e:
    print(str(e)[:70])
with torch.no_grad():      # fix for init/manual updates: no_grad; for model code: out-of-place
    p[p < 0] = 0
```

Gotchas
- To find the culprit, `torch.autograd.set_detect_anomaly(True)` prints the forward op whose output was modified. See §13.
- `nn.ReLU(inplace=True)` right after an op that saves its *output* (sigmoid, tanh, softmax, `exp`) breaks backward. After conv/linear it is fine.
- Avoid `.data`: it bypasses version tracking and silently gives wrong grads. Use `.detach()` or `no_grad`.

---

## 6. nn.Module patterns

### Skeleton, `ModuleList` vs Python list

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2, p=0.1):
        super().__init__()                               # must come first
        dims = [d_in] + [d_hidden] * n_layers
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(d_hidden, d_out)
        self.drop = nn.Dropout(p)

    def forward(self, x):                                # x: (B, d_in)
        for layer in self.layers:
            x = self.drop(F.gelu(layer(x)))
        return self.head(x)                              # (B, d_out)

class Buggy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(4, 4) for _ in range(3)]   # plain list: NOT registered
        self.heads = {"a": nn.Linear(4, 2)}                 # plain dict: NOT registered

assert len(list(Buggy().parameters())) == 0      # not optimized, not moved by .to(), not in state_dict
m = MLP(8, 16, 3)
assert m(torch.randn(5, 8)).shape == (5, 3)
heads = nn.ModuleDict({"cls": nn.Linear(16, 10), "box": nn.Linear(16, 4)})
scales = nn.ParameterList([nn.Parameter(torch.ones(4)) for _ in range(2)])
seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
```

### Buffers: `register_buffer` (persistent vs not)

```python
class SinusoidalPE(nn.Module):
    def __init__(self, max_len, d):
        super().__init__()
        pos = torch.arange(max_len)[:, None]                          # (T,1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))   # (d/2,)
        pe = torch.zeros(max_len, d)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)              # moves with .to(), NOT in state_dict
        self.register_buffer("causal", torch.ones(max_len, max_len, dtype=torch.bool).tril(), persistent=False)
        self.register_buffer("num_calls", torch.zeros((), dtype=torch.long))   # persistent: in state_dict

    def forward(self, x):                                             # (B,T,D)
        self.num_calls += 1
        return x + self.pe[: x.size(1)]

pe = SinusoidalPE(16, 8)
assert list(pe.state_dict().keys()) == ["num_calls"]
assert len(list(pe.parameters())) == 0                # buffers aren't parameters (no grad, no optimizer)
assert pe(torch.randn(2, 10, 8)).shape == (2, 10, 8)
```

### Init: `nn.init.*`, zero-init last layer

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)          # GPT/ViT style
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")   # He init for ReLU nets
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

model = MLP(8, 16, 3)
model.apply(init_weights)                    # recursive over all submodules
nn.init.zeros_(model.head.weight)            # zero-init last layer of a residual branch / gate:
nn.init.zeros_(model.head.bias)              # block starts as identity (adaLN-Zero, ControlNet zero-conv)
nn.init.xavier_uniform_(torch.empty(4, 4))   # Glorot: tanh/sigmoid/attention projections
with torch.no_grad():                        # manual tweaks to params need no_grad
    model.layers[0].weight.mul_(0.5)
```

### `train()` / `eval()`

| layer | `model.train()` | `model.eval()` |
|---|---|---|
| `Dropout` | zeroes with prob p, scales kept units by `1/(1-p)` | identity |
| `BatchNorm` | normalizes with batch stats, **updates** running stats | uses running stats |
| everything else | same | same |

`eval()` does **not** disable autograd, so pair it with `torch.no_grad()`/`inference_mode()`. Both are recursive and set `module.training`.

```python
bn_net = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))
xb = torch.randn(8, 4)
bn_net.eval()
with torch.no_grad():
    assert torch.equal(bn_net(xb), bn_net(xb))       # deterministic in eval
bn_net.train()
assert bn_net.training and bn_net[2].training
```

### `state_dict`, saving/loading, freezing, param groups, counting

```python
model = MLP(8, 16, 3)
torch.save(model.state_dict(), "mlp.pt")
sd = torch.load("mlp.pt", map_location="cpu", weights_only=True)   # weights_only=True is the default since 2.6
model2 = MLP(8, 16, 3, n_layers=3)                                 # different architecture
res = model2.load_state_dict(sd, strict=False)                     # returns (missing_keys, unexpected_keys)
print("missing:", res.missing_keys, "unexpected:", res.unexpected_keys)
sd_ddp = {"module." + k: v for k, v in sd.items()}                 # DDP adds "module.", compile adds "_orig_mod."
model.load_state_dict({k.removeprefix("module."): v for k, v in sd_ddp.items()})

# Freeze a backbone (fine-tuning / linear probe)
model.layers.requires_grad_(False)          # == for p in model.layers.parameters(): p.requires_grad = False
model.layers.eval()                          # ALSO freeze BN stats / dropout (requires_grad doesn't!), redo after model.train()
opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)

# No weight decay on biases & norm weights (ndim < 2)
decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim >= 2]
no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.05},
                         {"params": no_decay, "weight_decay": 0.0}], lr=3e-4)

n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert n_total == (8 * 16 + 16) + (16 * 16 + 16) + (16 * 3 + 3) and n_train == 16 * 3 + 3
```

Parameter-count mental math: `Linear(i,o)` = `i·o + o`; `Conv2d(ci,co,k)` = `co·ci·k² + co`; `Embedding(V,D)` = `V·D`; `LayerNorm(D)` = `2D`; `MultiheadAttention(D)` = `4D² + 4D`; transformer block ≈ `12D²`.

### Hooks: capture activations / gradients

```python
model = MLP(8, 16, 3)
acts, grads = {}, {}

def save_act(name):
    def hook(module, inputs, output):            # forward hook: (module, input tuple, output)
        acts[name] = output.detach()
    return hook

handles = [m.register_forward_hook(save_act(n))
           for n, m in model.named_modules() if isinstance(m, nn.Linear)]
handles.append(model.head.register_full_backward_hook(
    lambda mod, grad_in, grad_out: grads.__setitem__("head_out", grad_out[0].detach())))

model(torch.randn(5, 8)).sum().backward()
for h in handles:
    h.remove()                                   # always remove, or they keep firing
assert set(acts) == {"layers.0", "layers.1", "head"} and grads["head_out"].shape == (5, 3)

t = torch.randn(3, requires_grad=True)           # tensor hook: inspect/modify a gradient
t.register_hook(lambda g: g * 0.5)
(t * 2).sum().backward()
assert torch.allclose(t.grad, torch.ones(3))
```

### Weight tying

```python
class TinyLM(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.emb = nn.Embedding(V, D)                  # weight (V,D)
        self.lm_head = nn.Linear(D, V, bias=False)     # weight (V,D): same shape
        self.lm_head.weight = self.emb.weight          # tie: SAME Parameter object (do it before creating the optimizer)

    def forward(self, idx):                            # (B,T) int64
        return self.lm_head(self.emb(idx))             # (B,T,V)

lm = TinyLM(100, 32)
assert lm.lm_head.weight is lm.emb.weight
assert sum(p.numel() for p in lm.parameters()) == 100 * 32   # parameters() de-duplicates
assert lm(torch.randint(0, 100, (2, 5))).shape == (2, 5, 100)
```

---

## 7. Training loop, AMP, checkpoints, determinism

### Canonical loop

```python
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
X = torch.randn(512, 20); y = (X[:, :5].sum(1) > 0).long()          # toy binary task
train_loader = DataLoader(TensorDataset(X[:400], y[:400]), batch_size=32, shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X[400:], y[400:]), batch_size=128)

model = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2))
model = model.to(device)                                             # move BEFORE creating the optimizer
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
epochs, warmup = 5, 20
total_steps = epochs * len(train_loader)

def lr_lambda(step):                                                 # linear warmup + cosine decay
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)            # multiplies the base lr

@torch.no_grad()
def evaluate(model, loader):
    model.eval()                                                     # dropout off, BN running stats
    tot_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                                           # (B,C)
        tot_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(-1) == yb).sum().item()
        n += yb.numel()
    return tot_loss / n, correct / n

step = 0
for epoch in range(epochs):
    model.train()                                                    # re-enable after evaluate()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)                                           # (B,C) raw logits
        loss = F.cross_entropy(logits, yb)                           # yb: (B,) int64
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # returns pre-clip norm
        opt.step()
        sched.step()                                                 # per-step schedule: after opt.step()
        step += 1
        if step % 25 == 0:                                           # .item() syncs GPU: log sparingly
            print(f"step {step} loss {loss.item():.4f} gnorm {gnorm.item():.2f} lr {sched.get_last_lr()[0]:.2e}")
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"epoch {epoch} val_loss {val_loss:.4f} val_acc {val_acc:.3f}")
assert val_acc > 0.8
```

### Checkpoint save / resume (model + optimizer + scheduler + scaler + RNG)

```python
def save_checkpoint(path, model, opt, sched, epoch, step, scaler=None):
    state = {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "step": step,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)               # atomic: a crash mid-save can't corrupt the last checkpoint

def load_checkpoint(path, model, opt, sched, scaler=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)   # numpy RNG state isn't weights_only-safe: trusted files only
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])          # casts/moves optimizer state to the params' device
    sched.load_state_dict(ck["scheduler"])
    if scaler is not None and ck["scaler"] is not None:
        scaler.load_state_dict(ck["scaler"])
    random.setstate(ck["rng"]["python"]); np.random.set_state(ck["rng"]["numpy"])
    torch.set_rng_state(ck["rng"]["torch"])
    if ck["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    return ck["epoch"] + 1, ck["step"]            # resume at the next epoch

save_checkpoint("ckpt.pt", model, opt, sched, epoch, step)
expected = torch.rand(3)                          # what the RNG produces right after saving
model_r = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2)).to(device)
opt_r = torch.optim.AdamW(model_r.parameters(), lr=3e-3, weight_decay=0.01)   # same construction as before
sched_r = torch.optim.lr_scheduler.LambdaLR(opt_r, lr_lambda)
start_epoch, step_r = load_checkpoint("ckpt.pt", model_r, opt_r, sched_r)
assert torch.equal(torch.rand(3), expected) and step_r == step
assert sched_r.get_last_lr() == sched.get_last_lr()
assert all(torch.equal(a, b) for a, b in zip(model.parameters(), model_r.parameters()))
```

### Mixed precision

```python
use_cuda = torch.cuda.is_available()
dev = "cuda" if use_cuda else "cpu"
model = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 2)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
xb, yb = torch.randn(32, 20, device=dev), torch.randint(0, 2, (32,), device=dev)

# bf16: same exponent range as fp32, so no loss scaling (preferred on A100/H100/TPU)
with torch.autocast(device_type=dev, dtype=torch.bfloat16):
    logits = model(xb)                      # matmuls run in bf16 -> logits.dtype == bfloat16
    loss = F.cross_entropy(logits, yb)      # autocast runs losses/softmax/norms in fp32
assert logits.dtype == torch.bfloat16 and loss.dtype == torch.float32
loss.backward()                             # backward OUTSIDE autocast; params & grads stay fp32
opt.step(); opt.zero_grad(set_to_none=True)

# fp16 (V100/T4 era): needs GradScaler. enabled=False makes it a transparent no-op (CPU here)
scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_cuda):   # GPU-only when enabled
    loss = F.cross_entropy(model(xb), yb)
scaler.scale(loss).backward()               # multiply loss by S so small grads don't underflow
scaler.unscale_(opt)                        # divide grads by S BEFORE clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(opt)                            # skips the step if grads contain inf/NaN
scaler.update()                             # adapts S
opt.zero_grad(set_to_none=True)
```

### Gradient accumulation

```python
model = nn.Linear(20, 2); opt = torch.optim.SGD(model.parameters(), lr=0.1)
loader = DataLoader(TensorDataset(torch.randn(64, 20), torch.randint(0, 2, (64,))), batch_size=8)
accum = 4                                    # effective batch = 8 * 4 = 32
opt.zero_grad(set_to_none=True)
for i, (xb, yb) in enumerate(loader):
    loss = F.cross_entropy(model(xb), yb) / accum   # scale so the summed grads equal a mean over 32
    loss.backward()                          # grads accumulate in .grad
    # DDP: wrap non-final micro-steps in `with ddp_model.no_sync():` to skip redundant all-reduces
    if (i + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
```

Gotcha: with variable token counts per micro-batch (LM padding), sum the per-token losses and divide by the **total** tokens across all micro-batches. Averaging the per-micro-batch means gives a different result.

### Seeding & determinism

```python
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)                  # seeds CPU and all CUDA devices

def seed_worker(worker_id):                  # DataLoader workers: reseed numpy/random per worker
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

seed_everything(0)
g = torch.Generator().manual_seed(0)         # controls shuffle order independently of the global RNG
loader = DataLoader(TensorDataset(torch.arange(10.)), batch_size=5, shuffle=True,
                    generator=g, worker_init_fn=seed_worker)

torch.use_deterministic_algorithms(True)     # raises on known-nondeterministic ops (e.g. some CUDA scatter_add/index_add)
torch.backends.cudnn.benchmark = False       # benchmark=True picks fastest (possibly nondeterministic) conv algos
# GPU: also set env CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA init
torch.use_deterministic_algorithms(False)
torch.set_float32_matmul_precision("high")   # allow TF32 matmuls on Ampere+ (speed vs. exact fp32)
```

---

## 8. Data: Dataset, DataLoader, collate, samplers

### Map-style `Dataset` + `DataLoader`

```python
from torch.utils.data import Dataset, DataLoader

class PairDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):                     # return a single sample; the loader batches
        return {"x": self.X[i], "y": self.y[i]}

ds = PairDataset(torch.randn(100, 8), torch.randint(0, 3, (100,)))
loader = DataLoader(
    ds, batch_size=16, shuffle=True,
    num_workers=0,                                # >0 = worker processes. In notebooks on macOS/Windows keep 0
    pin_memory=torch.cuda.is_available(),         # page-locked memory -> faster async .to("cuda", non_blocking=True)
    drop_last=True,                               # avoids a tiny last batch (BatchNorm!)
    # persistent_workers=True, prefetch_factor=2  # only valid with num_workers > 0
)
batch = next(iter(loader))                        # default collate stacks per dict key
assert batch["x"].shape == (16, 8) and batch["y"].shape == (16,) and len(loader) == 6
```

### `collate_fn` for variable-length sequences

```python
from torch.nn.utils.rnn import pad_sequence

PAD = 0
seqs = [torch.randint(1, 50, (n,)) for n in (5, 2, 7, 3)]
data = [(s, int(i % 2)) for i, s in enumerate(seqs)]

def collate(batch):                                       # batch: list of (seq (L_i,), label)
    toks, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in toks])
    padded = pad_sequence(toks, batch_first=True, padding_value=PAD)   # (B, L_max)
    mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]    # (B, L_max) True = real
    return padded, mask, torch.tensor(labels)

loader = DataLoader(data, batch_size=4, collate_fn=collate)   # any indexable list works as a dataset
padded, mask, labels = next(iter(loader))
assert padded.shape == (4, 7) and mask.sum().item() == 17
left = pad_sequence(list(seqs), batch_first=True, padding_side="left")   # (2.6+) for decoder-only generation
mean_pooled = (padded.float() * mask).sum(1) / mask.sum(1)               # masked mean over time
```

### `IterableDataset` with worker sharding

```python
# --- standalone script: save as stream_demo.py, run `python stream_demo.py` (needs the __main__ guard) ---
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class Stream(IterableDataset):
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        info = get_worker_info()                      # None in the main process
        wid, nw = (0, 1) if info is None else (info.id, info.num_workers)
        # Without this, EVERY worker yields the full stream -> duplicated data.
        # Under DDP also shard by rank: start = rank * nw + wid, step = world_size * nw.
        return iter(range(wid, self.n, nw))

if __name__ == "__main__":
    out = [int(v) for b in DataLoader(Stream(20), batch_size=4, num_workers=2) for v in b]
    assert sorted(out) == list(range(20)), out        # each item exactly once
    print("ok", out)
```

### Samplers: class imbalance, distributed

```python
from torch.utils.data import WeightedRandomSampler, DistributedSampler, TensorDataset

labels = torch.tensor([0] * 90 + [1] * 10)                 # 9:1 imbalance
ds = TensorDataset(torch.randn(100, 4), labels)
class_w = 1.0 / torch.bincount(labels).float()              # (C,)
sample_w = class_w[labels]                                  # (N,) per-sample weight
sampler = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
loader = DataLoader(ds, batch_size=50, sampler=sampler)     # sampler and shuffle=True are mutually exclusive
ys = torch.cat([yb for _, yb in loader])
print("fraction of class 1 after resampling:", ys.float().mean().item())   # about 0.5

# DistributedSampler: each rank sees a disjoint 1/world_size shard
s0 = DistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=0)  # normally rank/world come from dist
s1 = DistributedSampler(ds, num_replicas=2, rank=1, shuffle=True, seed=0)
for epoch in range(2):
    s0.set_epoch(epoch); s1.set_epoch(epoch)                # REQUIRED, or every epoch reuses the same order
    i0, i1 = list(s0), list(s1)
    assert len(i0) == 50 and not set(i0) & set(i1)
```

---

## 9. Vision ops

### Conv arithmetic

Output size per spatial dim: **`out = floor((in + 2p − d·(k−1) − 1) / s) + 1`**
Transposed conv: **`out = (in − 1)·s − 2p + d·(k−1) + output_padding + 1`**

| config | effect on `H` |
|---|---|
| `k=3, s=1, p=1` | same size (`padding="same"` works only for stride 1) |
| `k=3, s=2, p=1` | `ceil(H/2)` |
| `k=P, s=P, p=0` | `H/P` (patch embedding) |
| `ConvTranspose k=4, s=2, p=1` | `2H` |

Weight shape: `(C_out, C_in / groups, kH, kW)`. Depthwise conv: `groups=C_in`. Params: `C_out·(C_in/groups)·k² + C_out`.

```python
def conv_out(h, k, s=1, p=0, d=1):
    return (h + 2 * p - d * (k - 1) - 1) // s + 1

x = torch.randn(2, 3, 32, 32)                                          # (B,C,H,W)
assert nn.Conv2d(3, 16, 3, stride=2, padding=1)(x).shape == (2, 16, conv_out(32, 3, 2, 1), 16)
assert nn.Conv2d(3, 16, 5, dilation=2)(x).shape[-1] == conv_out(32, 5, d=2) == 24
assert nn.ConvTranspose2d(3, 8, 4, stride=2, padding=1)(x).shape == (2, 8, 64, 64)
dw = nn.Conv2d(3, 3, 3, padding=1, groups=3)                           # depthwise
assert dw.weight.shape == (3, 1, 3, 3)
w = torch.randn(16, 3, 3, 3)
assert F.conv2d(x, w, padding=1).shape == (2, 16, 32, 32)              # functional form
```

### `F.unfold` / `F.fold` (im2col)

```python
B, C, H, W, O, k = 2, 3, 8, 8, 4, 3
x = torch.randn(B, C, H, W); w = torch.randn(O, C, k, k); b = torch.randn(O)
cols = F.unfold(x, kernel_size=k, padding=1)                  # (B, C*k*k, L), L = H_out*W_out = 64
assert cols.shape == (B, C * k * k, H * W)
out = (w.view(O, -1) @ cols + b[:, None]).view(B, O, H, W)    # (O,Ckk) @ (B,Ckk,L) -> (B,O,L)
assert torch.allclose(out, F.conv2d(x, w, b, padding=1), atol=1e-4)

folded = F.fold(cols, output_size=(H, W), kernel_size=k, padding=1)   # SUMS overlapping patches
overlap = F.fold(F.unfold(torch.ones_like(x), k, padding=1), (H, W), kernel_size=k, padding=1)
assert torch.allclose(folded / overlap, x, atol=1e-5)          # divide by the overlap count to invert
```

### Patchify / unpatchify (ViT, MAE, DiT)

```python
B, C, H, W, P = 2, 3, 32, 32, 8
x = torch.randn(B, C, H, W)
nh, nw = H // P, W // P
patches = (x.reshape(B, C, nh, P, nw, P)              # split H->(nh,P), W->(nw,P)
             .permute(0, 2, 4, 1, 3, 5)               # (B, nh, nw, C, P, P)
             .reshape(B, nh * nw, C * P * P))         # (B, N, C*P*P), patches in row-major order
assert torch.equal(patches, F.unfold(x, kernel_size=P, stride=P).transpose(1, 2))
p2 = x.unfold(2, P, P).unfold(3, P, P)                # Tensor.unfold: (B, C, nh, nw, P, P)
assert torch.equal(p2.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * P * P), patches)

x_rec = (patches.reshape(B, nh, nw, C, P, P)          # unpatchify = exact inverse
                .permute(0, 3, 1, 4, 2, 5)            # (B, C, nh, P, nw, P)
                .reshape(B, C, H, W))
assert torch.equal(x_rec, x)

D = 64
embed = nn.Conv2d(C, D, kernel_size=P, stride=P)      # ViT patch embedding == Linear on flattened patches
tokens = embed(x).flatten(2).transpose(1, 2)          # (B,D,nh,nw) -> (B,D,N) -> (B,N,D)
lin = patches @ embed.weight.view(D, -1).T + embed.bias
assert torch.allclose(tokens, lin, atol=1e-4)
```

### Layout & normalization

```python
img = torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8)   # HWC uint8 from PIL/cv2/numpy
x = img.permute(2, 0, 1).float() / 255.0                         # (3,H,W) float in [0,1]
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)         # ImageNet stats; (3,1,1) to broadcast
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
x = (x - mean) / std                                             # batched: .view(1, 3, 1, 1)
batch = x.unsqueeze(0)                                           # (1,3,H,W)
assert (torch.tensor([100], dtype=torch.uint8) - 200).item() == 156   # uint8 math wraps: convert to float first

imgs = torch.rand(16, 3, 32, 32)
ch_mean, ch_std = imgs.mean(dim=(0, 2, 3)), imgs.std(dim=(0, 2, 3))   # (C,) dataset stats per channel
```

Which dims each norm reduces over, for `x: (B,C,H,W)`:

```python
x = torch.randn(4, 6, 5, 5); eps = 1e-5
def norm(x, dims):
    mu = x.mean(dims, keepdim=True)
    var = x.var(dims, keepdim=True, correction=0)        # BIASED variance
    return (x - mu) / torch.sqrt(var + eps)

bn = nn.BatchNorm2d(6, affine=False).train()
assert torch.allclose(norm(x, (0, 2, 3)), bn(x), atol=1e-4)                     # BN: per channel over (B,H,W)
assert torch.allclose(norm(x, (1, 2, 3)), F.layer_norm(x, x.shape[1:]), atol=1e-4)  # LN: per sample over (C,H,W)
assert torch.allclose(norm(x, (2, 3)), F.instance_norm(x), atol=1e-4)           # IN: per (sample, channel)
G = 3
gn = norm(x.view(4, G, 6 // G, 5, 5), (2, 3, 4)).view_as(x)                       # GN: per (sample, group)
assert torch.allclose(gn, F.group_norm(x, G), atol=1e-4)
```

### `F.interpolate`, `align_corners`

```python
x = torch.arange(4.).view(1, 1, 1, 4)                         # (B,C,H,W) = [0,1,2,3]
F.interpolate(x, scale_factor=2, mode="nearest")
up_f = F.interpolate(x, size=(1, 8), mode="bilinear", align_corners=False)   # default: pixel EDGES aligned
up_t = F.interpolate(x, size=(1, 8), mode="bilinear", align_corners=True)    # corner pixel CENTERS aligned
print(up_f.flatten()); print(up_t.flatten())                  # different values: keep train & inference consistent
assert up_t[..., 0] == 0 and up_t[..., -1] == 3               # True keeps corner values exactly
down = F.interpolate(torch.randn(1, 3, 64, 64), size=(16, 16), mode="bilinear",
                     antialias=True, align_corners=False)     # antialias when downsampling (PIL-like)
masks = torch.randint(0, 5, (1, 1, 8, 8)).float()
F.interpolate(masks, scale_factor=4, mode="nearest")          # labels/masks: nearest, never bilinear
```

### `F.grid_sample` / `F.affine_grid`

```python
B, C, H, W = 2, 3, 16, 16
x = torch.randn(B, C, H, W)
theta = torch.tensor([[1., 0., 0.], [0., 1., 0.]]).expand(B, 2, 3)        # identity affine (B,2,3)
grid = F.affine_grid(theta, size=(B, C, H, W), align_corners=False)      # (B,H,W,2), coords in [-1,1]
out = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
assert torch.allclose(out, x, atol=1e-5)

ys, xs = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
grid = torch.stack([xs, ys], dim=-1).expand(B, H, W, 2)      # last dim is (x, y) = (W, H) order!
assert torch.allclose(F.grid_sample(x, grid, align_corners=True), x, atol=1e-5)
flipped = F.grid_sample(x, torch.stack([-xs, ys], -1).expand(B, H, W, 2), align_corners=True)
assert torch.allclose(flipped, x.flip(-1), atol=1e-5)        # sample at -x = horizontal flip
```

### `F.pixel_shuffle` (sub-pixel upsampling)

```python
r = 2
x = torch.randn(1, 3 * r * r, 8, 8)                          # (B, C*r^2, H, W)
y = F.pixel_shuffle(x, r)                                    # (B, C, H*r, W*r)
assert y.shape == (1, 3, 16, 16)
assert torch.equal(F.pixel_unshuffle(y, r), x)               # inverse (space-to-depth)
```

---

## 10. Attention

### Manual scaled-dot-product attention with causal + padding masks

```python
def attention(q, k, v, causal=False, key_padding_mask=None, dropout_p=0.0, training=False):
    """q: (B,H,T,Dh), k/v: (B,H,S,Dh), key_padding_mask: (B,S) True = real token."""
    T, S = q.size(-2), k.size(-2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))       # (B,H,T,S)
    allowed = torch.ones(T, S, dtype=torch.bool, device=q.device)
    if causal:  # bottom-right aligned: query i sees keys <= i + (S - T) (right for KV caches)
        allowed = allowed.tril(diagonal=S - T)
    if key_padding_mask is not None:
        allowed = allowed & key_padding_mask[:, None, None, :]        # (B,1,1,S) -> (B,1,T,S)
    scores = scores.masked_fill(~allowed, float("-inf"))
    attn = scores.softmax(dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)                            # rows with no allowed key -> 0
    attn = F.dropout(attn, p=dropout_p, training=training)            # F.dropout defaults to training=True!
    return attn @ v                                                   # (B,H,T,Dh)

B, H, T, Dh = 2, 4, 6, 8
q, k, v = (torch.randn(B, H, T, Dh) for _ in range(3))
lengths = torch.tensor([6, 4])
kpm = torch.arange(T)[None, :] < lengths[:, None]                     # (B,T) right-padded

ref = attention(q, k, v, causal=True, key_padding_mask=kpm)
allowed = torch.ones(T, T, dtype=torch.bool).tril() & kpm[:, None, None, :]   # bool: True = ATTEND
sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
torch.testing.assert_close(ref, sdpa, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(attention(q, k, v, causal=True), F.scaled_dot_product_attention(q, k, v, is_causal=True))
# Float attn_mask is ADDED to the scores (0 = keep, -inf = block). is_causal=True together with attn_mask raises an error.
```

### Multi-head attention module (head split/merge)

```python
class MHA(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x, key_padding_mask=None, causal=False):        # x: (B,T,D)
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D // self.h)           # (B,T,3,H,Dh)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)                           # each (B,H,T,Dh)
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :]                  # (B,1,1,T) True = attend
            if causal:
                mask = mask & torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        out = out.transpose(1, 2).reshape(B, T, D)                     # merge heads; reshape (non-contiguous)
        return self.proj(out)

d, h = 32, 4
x = torch.randn(2, 6, d)
pad = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
mine = MHA(d, h)
ref = nn.MultiheadAttention(d, h, batch_first=True)                    # default batch_first=False -> (T,B,D)!
with torch.no_grad():                                                  # same weights -> same outputs
    ref.in_proj_weight.copy_(mine.qkv.weight); ref.in_proj_bias.copy_(mine.qkv.bias)
    ref.out_proj.weight.copy_(mine.proj.weight); ref.out_proj.bias.copy_(mine.proj.bias)
out_ref, _ = ref(x, x, x, key_padding_mask=~pad)                       # nn.MHA: True = IGNORE (opposite of SDPA)
torch.testing.assert_close(mine(x, key_padding_mask=pad), out_ref, atol=1e-5, rtol=1e-5)
```

Cross-attention (text attends to image patches) uses the same code: `q` comes from `(B,T,D)` text, `k,v` from `(B,N,D)` patches, and the scores are `(B,H,T,N)`.

### Causal alignment gotcha (KV-cache decoding)

```python
from torch.nn.attention.bias import causal_lower_right

q1 = torch.randn(1, 1, 1, 8)                     # 1 new query token
k5, v5 = torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)   # 5 cached keys
o_top_left = F.scaled_dot_product_attention(q1, k5, v5, is_causal=True)
assert torch.allclose(o_top_left[0, 0, 0], v5[0, 0, 0])      # is_causal is TOP-LEFT aligned: sees only key 0!
o_ok = F.scaled_dot_product_attention(q1, k5, v5, attn_mask=causal_lower_right(1, 5))
assert torch.allclose(o_ok, F.scaled_dot_product_attention(q1, k5, v5), atol=1e-6)   # sees all 5 keys
```

Fused SDPA backends (flash / mem-efficient) are picked on GPU. A boolean or float `attn_mask` can rule out the flash kernel; `is_causal=True` without a mask keeps it.

---

## 11. Distributed & torch.compile in one screen

```python
# --- standalone script: save as ddp_demo.py, run `python ddp_demo.py` (CPU, gloo backend) ---
import os
import torch, torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def worker(rank, world_size):
    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29512",   # "nccl" for GPUs
                            rank=rank, world_size=world_size)
    t = torch.tensor([float(rank + 1)])
    dist.all_reduce(t, op=dist.ReduceOp.SUM)              # in place on every rank: 1 + 2 = 3
    assert t.item() == 3.0

    parts = [torch.empty(2) for _ in range(world_size)]
    dist.all_gather(parts, torch.full((2,), float(rank)))  # list of each rank's tensor
    flat = torch.empty(2 * world_size)
    dist.all_gather_into_tensor(flat, torch.full((2,), float(rank)))   # concatenated: [0,0,1,1]
    assert flat.tolist() == [0., 0., 1., 1.]

    out = torch.empty(2)                                   # reduce, then rank r keeps chunk r
    dist.reduce_scatter_tensor(out, torch.arange(4.) + rank)   # rank0: [0,1]+[1,2]=[1,3]; rank1: [5,7]
    assert out.tolist() == ([1., 3.] if rank == 0 else [5., 7.])

    torch.manual_seed(0)                                   # same init on all ranks (DDP also broadcasts rank 0's)
    model = DDP(nn.Linear(4, 2))                           # GPU: DDP(model.to(rank), device_ids=[rank])
    torch.manual_seed(rank)                                # different data per rank
    model(torch.randn(8, 4)).pow(2).mean().backward()      # backward all-reduces (AVERAGES) grads
    g = model.module.weight.grad.clone()                   # .module = the wrapped model (save model.module.state_dict())
    gs = [torch.empty_like(g) for _ in range(world_size)]
    dist.all_gather(gs, g)
    assert torch.allclose(gs[0], gs[1])                    # identical grads on every rank
    if rank == 0:
        print("all collectives OK")                        # log/save only on rank 0
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2, join=True)
```

```bash
# Real launches use torchrun, which sets RANK / LOCAL_RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT:
torchrun --nproc_per_node=8 train.py
# in train.py: dist.init_process_group("nccl"); local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
```

DDP checklist: use a `DistributedSampler` and call `set_epoch`. Convert BN with `nn.SyncBatchNorm.convert_sync_batchnorm(model)` if per-GPU batches are small. Set `find_unused_parameters=True` only if some params get no grad. Use `no_sync()` for grad accumulation. Use FSDP2 (`torch.distributed.fsdp.fully_shard`) when params plus optimizer state don't fit on one GPU.

```python
model = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64))
cmodel = torch.compile(model)       # also: mode="reduce-overhead" (CUDA graphs), "max-autotune"; fullgraph=True to forbid graph breaks
xin = torch.randn(8, 64)
torch.testing.assert_close(cmodel(xin), model(xin), atol=1e-5, rtol=1e-5)   # 1st call compiles (slow)
assert list(cmodel.state_dict())[0].startswith("_orig_mod.")   # save model.state_dict() (the original), not cmodel's
```

`torch.compile` pays off when there are many small elementwise/normalization ops to fuse, shapes are stable, and the run is long (training, serving). It doesn't help, or costs time, for one-off scripts, highly dynamic shapes (recompiles, so try `dynamic=True`), or Python control flow on tensor values (graph breaks). Most interview code should stay eager.

---

## 12. Interview gotchas

| # | gotcha | fix |
|---|---|---|
| 1 | `CrossEntropyLoss` takes **raw logits** `(B,C)` and **int64 class indices** `(B,)`. It applies log-softmax itself | don't softmax first. Float targets of shape `(B,C)` are treated as class **probabilities** (label smoothing / distillation) |
| 2 | CE over sequences: logits `(B,T,V)` | `F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))` or `logits.transpose(1, 2)` → `(B,V,T)`; segmentation `(B,C,H,W)` with targets `(B,H,W)` works directly |
| 3 | padding tokens counted in the loss | `ignore_index=-100` (default) on targets, or a masked mean `(loss * m).sum() / m.sum()` |
| 4 | `BCELoss` on sigmoid outputs: unstable, and raises inside CUDA autocast | `BCEWithLogitsLoss` on logits, float targets of the same shape, `pos_weight` for imbalance |
| 5 | `NLLLoss` expects **log**-probs | `F.nll_loss(F.log_softmax(x, -1), y)` == `F.cross_entropy(x, y)` |
| 6 | forgot `model.eval()`: dropout active, BN uses batch stats and **updates running stats** on val data | `model.eval()` for eval, `model.train()` back before training |
| 7 | `eval()` ≠ `no_grad()`: eval still builds graphs → OOM | `with torch.no_grad():` / `inference_mode()` |
| 8 | `total_loss += loss` keeps every graph alive (memory leak) | `total_loss += loss.item()` (or `loss.detach()`) |
| 9 | `x.view(...)` after `permute`/`transpose` | `.reshape(...)` or `.contiguous().view(...)` |
| 10 | `/` on ints is true division (float); `//` floors toward −∞ (`-5 // 2 == -3`); `.mean()` on int tensors errors | `.float()` first; `torch.div(a, b, rounding_mode="trunc")` |
| 11 | dtype promotion: an `int` tensor plus a Python float gives float32; a 0-dim tensor doesn't up-cast a dimensioned one (`fp16 + fp32 scalar → fp16`); uint8 wraps | cast explicitly; images → `.float()` before arithmetic |
| 12 | `torch.max(x, dim)` returns `(values, indices)`; `torch.max(x)` is a scalar; `torch.max(a, b)` is elementwise | `x.max(-1).values`, `x.amax(dims)`, `x.argmax(-1)` |
| 13 | `dim` vs `axis`, `keepdim` vs `keepdims` | torch accepts both in most reductions, but `dim`/`keepdim` are canonical |
| 14 | `nn.Softmax()` / `F.softmax(x)` without `dim` (deprecated implicit dim) | always `dim=-1` |
| 15 | `argmax`/`sum` drop the reduced dim | `keepdim=True` when you'll broadcast back |
| 16 | `squeeze()` without dim kills the batch dim when `B == 1` | `squeeze(-1)` |
| 17 | `(N,1)` prediction vs `(N,)` target broadcasts to `(N,N)` (MSE only warns) | match shapes: `pred.squeeze(-1)` |
| 18 | in-place op on a tensor saved for backward (`relu_`, `+=`, `x[mask] = 0`) | out-of-place; `set_detect_anomaly` to locate |
| 19 | `BatchNorm` with batch size 1 in train mode raises "Expected more than 1 value per channel" | `drop_last=True`, `GroupNorm`/`LayerNorm`, or eval mode |
| 20 | optimizer created, then modules/params replaced (new head, tying, `.half()` with overwrite, FSDP wrap) → optimizer updates orphans | move/modify the model **first**, then build the optimizer |
| 21 | forgot `zero_grad()`: grads accumulate across steps | `opt.zero_grad(set_to_none=True)` each step |
| 22 | `torch.randn(..., requires_grad=True) * 0.01` is a non-leaf → `.grad` is None | `nn.Parameter(torch.randn(...) * 0.01)` |
| 23 | layers in a Python `list`/`dict` aren't registered | `nn.ModuleList` / `nn.ModuleDict` / `nn.ParameterList` |
| 24 | `x.to(device)` is not in place for tensors | `x = x.to(device)` |
| 25 | tensors created in `forward` land on CPU (`torch.zeros(B, D)`) → device mismatch | `torch.zeros(B, D, device=x.device, dtype=x.dtype)`, `x.new_zeros`, `*_like`, or a buffer |
| 26 | `F.dropout(x, p)` defaults to `training=True`: still active in eval | `F.dropout(x, p, training=self.training)` or `nn.Dropout` |
| 27 | `x.var()`/`std()` are unbiased (N−1); LayerNorm/BN use biased | `correction=0` |
| 28 | `F.kl_div(input, target)`: `input` must be **log**-probs of Q, target = P; computes KL(P‖Q); `"mean"` divides by numel | `F.kl_div(logQ, P, reduction="batchmean")` |
| 29 | `nn.MultiheadAttention`/`nn.LSTM`/`nn.GRU` default to `batch_first=False` `(T,B,D)`; MHA `key_padding_mask` True = **ignore**, SDPA bool mask True = **attend** | pass `batch_first=True`; invert masks carefully |
| 30 | `-inf` masking with fully masked rows → NaN; `-1e9` overflows fp16 | see §4: `nan_to_num`, `finfo.min` with `masked_fill` |
| 31 | `x[idx] += 1` with duplicate indices adds once | `index_add_`, `index_put_(..., accumulate=True)`, `scatter_add_` |
| 32 | `torch.load` defaults to `weights_only=True` (2.6+): full checkpoints with numpy/RNG objects fail | `weights_only=False` for **trusted** files, or store only tensors/primitives |
| 33 | AMP fp16: clipping scaled grads, or forgetting `scaler.update()` | `scaler.unscale_(opt)` → clip → `scaler.step(opt)` → `scaler.update()` |
| 34 | `scheduler.step()` per epoch vs per step mixed up; called before `opt.step()` (warning) | decide the unit; call it after `opt.step()` |
| 35 | `torch.tensor(t)` on an existing tensor (copy + warning); `.item()` on a multi-element tensor errors | `t.clone().detach()`; `.tolist()` |
| 36 | `Embedding` indices must be integer; `labels` from numpy are often int32/float | `.long()` |
| 37 | `model.parameters()` is a generator, so it's empty the 2nd time you iterate it | `params = list(model.parameters())` |
| 38 | interpolating label maps with `bilinear` invents classes; train/infer `align_corners` mismatch | `mode="nearest"` for labels; keep `align_corners` consistent |

Spot-checks for the less obvious rows:

```python
logits, y = torch.randn(4, 3), torch.tensor([0, 2, 1, 2])
assert torch.allclose(F.cross_entropy(logits, y), F.nll_loss(F.log_softmax(logits, -1), y))    # 1, 5
soft = F.one_hot(y, 3).float()
assert torch.allclose(F.cross_entropy(logits, soft), F.cross_entropy(logits, y))               # 1: prob targets
ce_smooth = F.cross_entropy(logits, y, label_smoothing=0.1)

lt, yt = torch.randn(2, 5, 7), torch.randint(0, 7, (2, 5))                                      # 2
assert torch.allclose(F.cross_entropy(lt.reshape(-1, 7), yt.reshape(-1)), F.cross_entropy(lt.transpose(1, 2), yt))
yt[0, 3:] = -100                                                                                # 3: ignore_index
assert torch.isfinite(F.cross_entropy(lt.reshape(-1, 7), yt.reshape(-1)))

assert (torch.tensor(5) / 2).item() == 2.5 and (torch.tensor(-5) // 2).item() == -3            # 10
assert (torch.tensor([1, 2], dtype=torch.int32) + 1.5).dtype == torch.float32                   # 11
assert (torch.ones(2, dtype=torch.float16) + torch.tensor(1.0)).dtype == torch.float16

vals, idx = torch.tensor([[1., 5., 3.]]).max(dim=-1)                                            # 12
assert vals.item() == 5 and idx.item() == 1
assert torch.sum(torch.ones(2, 3), axis=0, keepdims=True).shape == (1, 3)                       # 13

try:
    nn.BatchNorm1d(4).train()(torch.randn(1, 4))                                                # 19
except ValueError as e:
    print("BN:", str(e)[:45])

model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))                                         # 20
opt = torch.optim.SGD(model.parameters(), lr=0.1)
model[1] = nn.Linear(4, 3)                                                  # new head AFTER the optimizer
in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
assert id(model[1].weight) not in in_opt                                    # never updated!

assert not F.dropout(torch.ones(1000), 0.5).eq(1).all()                                         # 26
assert torch.equal(F.dropout(torch.ones(10), 0.5, training=False), torch.ones(10))

gen = nn.Linear(2, 2).parameters()                                                              # 37
assert len(list(gen)) == 2 and len(list(gen)) == 0
```

---

## 13. Debugging & profiling

### Shape asserts

```python
def check(x, spec, env):
    """check(q, "B H T Dh", env): binds names on first use, then enforces consistency."""
    names = spec.split()
    assert x.ndim == len(names), f"{spec}: got {tuple(x.shape)}"
    for n, s in zip(names, x.shape):
        if n.isdigit():
            assert int(n) == s, f"{spec}: got {tuple(x.shape)}"
        else:
            assert env.setdefault(n, s) == s, f"{n}: expected {env[n]}, got {s} in {tuple(x.shape)}"

env = {}
check(torch.randn(2, 4, 10, 16), "B H T Dh", env)
check(torch.randn(2, 4, 10, 10), "B H T T", env)
try:
    check(torch.randn(3, 4, 10, 16), "B H T Dh", env)
except AssertionError as e:
    print("shape error ->", e)
```

### Anomaly detection, NaN hunting, grad norms

```python
x = torch.zeros(1, requires_grad=True)
with torch.autograd.detect_anomaly():            # slow: debugging only. Global form: torch.autograd.set_detect_anomaly(True)
    try:
        (x.sqrt() * 0).sum().backward()          # 0 * inf = NaN in SqrtBackward
    except RuntimeError as e:
        print(str(e)[:70])                       # "Function 'SqrtBackward0' returned nan values..."

model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
def nan_hook(name):                              # first module whose output goes non-finite
    def hook(m, inp, out):
        if isinstance(out, torch.Tensor) and not torch.isfinite(out).all():
            raise RuntimeError(f"non-finite output in {name}")
    return hook
hs = [m.register_forward_hook(nan_hook(n)) for n, m in model.named_modules() if n]
model(torch.randn(4, 8)).sum().backward()
for h in hs:
    h.remove()

per_param = {n: p.grad.norm().item() for n, p in model.named_parameters() if p.grad is not None}
total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))   # total norm, no clipping
print({k: round(v, 3) for k, v in per_param.items()}, "total:", round(total.item(), 3))
# Healthy signs: loss finite, grad norm not exploding/vanishing, update/weight ratio ~1e-3 per step
```

### Overfit one batch (the first sanity check)

```python
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 5))
xb, yb = torch.randn(16, 10), torch.randint(0, 5, (16,))      # one FIXED batch, random labels
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for it in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
print("final loss", loss.item())
assert loss.item() < 0.05      # if it can't memorize 16 samples: bug in model/loss/labels/optimizer wiring
```

### Profiler, memory, timing

```python
from torch.profiler import profile, record_function, ProfilerActivity

model = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
x = torch.randn(64, 256)
acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
with profile(activities=acts, record_shapes=True, profile_memory=True) as prof:
    with record_function("forward"):              # custom label in the trace
        model(x)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))  # GPU: sort_by="cuda_time_total"
# prof.export_chrome_trace("trace.json")         # open in chrome://tracing or Perfetto

if torch.cuda.is_available():                     # GPU-only
    torch.cuda.reset_peak_memory_stats()
    y = model.cuda()(x.cuda())
    print(f"alloc {torch.cuda.memory_allocated() / 2**20:.1f} MiB, "
          f"peak {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")   # allocated by tensors (reserved >= this)
    print(torch.cuda.memory_summary(abbreviated=True))
    model = model.cpu()

from torch.utils.benchmark import Timer
a, b = torch.randn(256, 256), torch.randn(256, 256)
m = Timer(stmt="a @ b", globals={"a": a, "b": b},          # syncs CUDA & warms up for you
          num_threads=torch.get_num_threads()               # Timer defaults to 1 CPU thread!
          ).blocked_autorange(min_run_time=0.2)
print(f"matmul median {m.median * 1e6:.1f} us")

def time_it(fn, iters=20, warmup=3):
    """Manual timing. CUDA kernels launch ASYNC: without synchronize you time the launch, not the work."""
    for _ in range(warmup):                       # warmup: allocator, cuDNN autotune, compile
        fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters

print(f"manual {time_it(lambda: a @ b) * 1e6:.1f} us")
# GPU alternative: start = torch.cuda.Event(enable_timing=True); end = ...; start.record(); fn(); end.record();
#                  torch.cuda.synchronize(); start.elapsed_time(end)  # ms
```

Memory back-of-envelope for training with AdamW: about **16 bytes/param** of model state either way — fp32 training is weights 4 + grads 4 + Adam m 4 + v 4, and bf16/fp16 mixed precision is bf16 weights 2 + bf16 grads 2 + fp32 master weights 4 + m 4 + v 4 (the ZeRO paper's 2 + 2 + 12). Implementations that keep fp32 grads, or extra copies, cost more. Activations scale with `B·T·D·layers` and usually dominate. To cut them: activation checkpointing (`torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)`), a smaller micro-batch with grad accumulation, or bf16.
