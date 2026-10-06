# nanoGPT Notes — Part 2 (Attention through the full model)

Continues `nanogpt-notes-pt-1.md`. Part 1 ended with the slow bag-of-words
loop: each position is the average of the token vectors at and before it.
This part is the rest of "Let's build GPT": that average becomes a matrix
multiply, the weights become data-dependent (self-attention), and the pieces
are stacked into the model in `bigram_v6.py`.

The small teaching sizes (`C = 2`, then `n_embd = 32`, `head_size = 16`)
are the notebook. The file at the end uses `n_embd = 384`. Both are the
same wiring.

---

## 1. The average as a matrix multiply

The Python loops in part 1 compute, for every position `t`, the mean of
rows `0..t`. The same result is one multiply.

For 3 tokens and uniform past-only weights:

```text
wei = [[1.00, 0.00, 0.00],
       [0.50, 0.50, 0.00],
       [0.33, 0.33, 0.33]]

xbow = wei @ x
```

Row `t` of `wei` says how much of each token goes into position `t`.
Row 0 is "100% of token 0". Row 1 is "half of token 0, half of token 1".
Multiplying by `x`, whose shape is `(T, C)`, mixes the rows of `x` and
leaves the channels alone. `xbow` has the same shape as `x`.

`torch.tril(torch.ones(T, T))` builds the lower triangle of ones (the
allowed positions). Dividing each row by its sum turns those ones into
the fractions above. A batched `x` of shape `(B, T, C)` uses the same
`(T, T)` matrix; it broadcasts over the batch.

### Softmax gives the same uniform weights  ← (doubt)

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
```

`masked_fill` writes `-inf` everywhere the triangle is 0, the future.
Softmax of a row of zeros is uniform. Softmax of `-inf` is 0. So a row
with two finite zeros and one `-inf` becomes `[0.5, 0.5, 0]`. Same
matrix as the row-normalized triangle. The point of going through softmax
is that the finite numbers no longer have to be zero. They can be scores,
and softmax still turns each row into weights that add up to 1.

---

## 2. One attention head: query, key, value  ← (doubt)

A single `Head` reads `x` of shape `(B, T, n_embd)` and writes
`(B, T, head_size)`.

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

`x` is not overwritten. The three linear layers are three readings of the
same token vector:

- **query** — what this token is looking for
- **key** — the label other tokens match against
- **value** — the vector that gets copied if the match is strong

In the notebook demo, `n_embd` was 32 and `head_size` was 16. Each token's
32 numbers were dotted with a `(16, 32)` weight and became 16 numbers.
`q`, `k`, and `v` are each `(B, T, 16)`.

### Why three readings instead of using `x` directly

Matching and copying are different jobs. A raw dot product of two token
vectors only matches tokens that already point the same way. The query
matrix can rotate "what I am looking for" into a different direction from
"what I contain", and the value matrix can choose which information is
worth copying. At initialization the three tables are random, so the
readings look meaningless. Training moves them.

### The affinity matrix

```python
wei = q @ k.transpose(-2, -1)   # (B, T, 16) @ (B, 16, T) -> (B, T, T)
```

`transpose(-2, -1)` swaps the last two axes of `k`, so each query can be
dotted with every key. `wei[b, i, j]` is query of token `i` dotted with
key of token `j`: how much token `i` wants token `j`.

Then the same mask and softmax as section 1, and the weighted sum uses
values rather than the raw `x`:

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
out = wei @ v    # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
```

A tiny case that makes the last line obvious. Token 1's softmax row is
`[0.46, 0.54, 0]`, and the value rows are `v0`, `v1`, `v2`. Its output is

```text
0.46 * v0 + 0.54 * v1 + 0 * v2
```

Token 0's row is `[1, 0, 0]`, so its output is exactly its own value.
Future tokens contribute nothing.

### The formula

```text
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V
```

`d_k` is the length of each key, which is `head_size`. In the notebook
demo that is 16, so the divisor is 4. A dot product of two 16-long random
vectors is a sum of 16 products, so the scores come out large. Softmax of
a large score becomes a spike (one weight near 1, the rest near 0) and the
gradient through it dies. Dividing by `sqrt(d_k)` brings the scores back
to a modest size before softmax.

In `bigram_v6.py` the line is `* C**-0.5`, and `C` is taken from
`x.shape`, so it is `n_embd` (384), not `head_size`. The formula's `d_k`
is the key length. The video uses `head_size**-0.5` (equivalently
`k.shape[-1]**-0.5`).

### Decoder mask vs encoder  ← (doubt)

The triangle exists because this model predicts the next character. If
position `t` could see position `t+1`, the value of the answer would be
mixed into the token that is supposed to guess it.

An encoder in a translation model reads a source sentence that is already
fully known. Every source token may look at every other source token, so
that model deletes the `masked_fill` line. "Self-attention" means queries,
keys, and values all come from the same sequence. Cross-attention uses
queries from the decoder and keys/values from the encoder.

### Why `tril[:T, :T]` and not the whole buffer  ← (doubt)

```python
self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
```

The buffer is `(block_size, block_size)` — 8×8 in the small model, 256×256
in `bigram_v6.py`. It is created once and is not a parameter (no gradient).

`wei` is `(B, T, T)`, and `T` changes. Training uses `T = block_size`, so
the slice is the whole matrix. `generate` starts from a single token, so
`T = 1` and `wei` is `(1, 1, 1)`. An 8×8 mask cannot be applied to a 1×1
grid. `[:T, :T]` is the top-left corner of the stored triangle, the
past-only mask for a sequence of the length that just arrived.

`register_buffer` also moves `tril` with `.to(device)`. A plain tensor
attribute would stay on CPU while `wei` is on the GPU.

---

## 3. Position embeddings

A self-attention mix has no notion of order. The set of value vectors is
all it sees, unless order is put into the vectors themselves.

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
tok = self.token_embedding_table(idx)                         # (B, T, n_embd)
pos = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
x = tok + pos
```

Row `t` of the position table is a learned  `n_embd`-vector added to every
token that sits in slot `t`. Token id 43 (`e`) at the start of the chunk
and the same id four characters later get different totals, because the
position rows differ.

`T` has to be read from `idx.shape` before this line. Later in `forward`
there is `B, T, C = logits.shape`, which makes `T` a local variable for
the whole function. Using `T` above that assignment raises
`UnboundLocalError`.

`generate` crops to the last `block_size` tokens before calling the model
(`idx[:, -block_size:]`). The position table only has `block_size` rows.
A longer context would index off the end of it.

---

## 4. Multi-head attention  ← (doubt)

One head is one way of looking at the past, and its output width is
whatever `head_size` you pass. The notebook's single head used
`head_size = 16`, so a 32-number token came out as 16 numbers. That 16
was a demo size. It is not a piece of the multi-head module.

```python
self.sa = MultiHeadAttention(num_heads=n_head, head_size=n_embd // n_head)
```

With `n_embd = 32` and 4 heads, `head_size` is `32 // 4 = 8`. Each head
receives the **entire** 32-number token. The 32 is not chopped into four
slices of 8. Each head has its own query, key, and value tables and writes
8 numbers. `torch.cat(..., dim=-1)` sets the four outputs side by side:

```text
8 + 8 + 8 + 8 = 32
```

The joined result is `(B, T, n_embd)` again, which is what the next layer
and the residual add need. Four heads of 16 would be 64, too wide to add
back onto a 32-number token. One head of 32 is the other way to keep the
width; multi-head just does the looking several ways at once.

In `bigram_v6.py`, `n_embd = 384` and the blocks are built with `n_head=4`,
so each head writes `384 // 4 = 96` numbers and four of them join back to
384. The variable `n_head = 6` at the top of the file is not what the
blocks use.

### The name `n_head` vs `num_heads`  ← (doubt)

`Block` takes `n_head`. `MultiHeadAttention.__init__` takes `num_heads`.
Calling `MultiHeadAttention(n_head=n_head, ...)` raises

```text
TypeError: __init__() got an unexpected keyword argument 'n_head'
```

Keyword arguments match the parameter name exactly. The working call is
`MultiHeadAttention(num_heads=n_head, head_size=...)`.

### Projection

`torch.cat` does not mix the heads. It only lays them next to each other.
`self.proj = nn.Linear(n_embd, n_embd)` is a `32 → 32` (or `384 → 384`)
mix applied after the join, so each output number can depend on all four
heads. Width stays `n_embd`.

---

## 5. Feed-forward  ← (doubt)

Attention gathers from other tokens. The feed-forward then thinks about
one token alone. The same weights run at every position.

```python
nn.Linear(n_embd, 4 * n_embd),   # 32 → 128, or 384 → 1536
nn.ReLU(),
nn.Linear(4 * n_embd, n_embd),   # back to 32, or 384
```

Per token:

```text
32  →  Linear  →  128  →  ReLU  →  128  →  Linear  →  32
```

The first linear spreads the token into a wider workspace: 128 rows of
weights, each dotted with the 32 numbers. `ReLU` does not change the
count. It keeps positive numbers and replaces negative numbers with 0:

```text
[ 1.5,  -0.4,  0.2,  -3.0 ]
[ 1.5,   0.0,  0.2,   0.0 ]
```

Two linear layers with nothing between them multiply out to one linear
layer. The wide middle would be fake. `ReLU` is the bend: which of the
128 coordinates stay on can differ from token to token, so the second
linear mixes a different set of numbers for each token.

Zeroing negatives does discard "how negative". The first layer can flip a
sign during training, so a useful signal can be moved onto the positive
side, where it survives. The negative side is the off state: `-0.4` and
`-3.0` both mean "this coordinate is absent."

---

## 6. A block: residual, layer norm, then a stack

```python
def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
```

### Residual  ← (doubt)

`x = self.sa(x)` would replace the token with the attention output.
`x = x + self.sa(x)` keeps the incoming 32 (or 384) numbers and adds the
block's update on top. The feed-forward does the same. After several
blocks the original token embedding is still in the sum; each block only
contributes a change.

The two tensors in the add must share a shape. That is why the multi-head
output is projected back to `n_embd` and the feed-forward ends at
`n_embd`. A 16-wide head output cannot be added to a 32-wide `x`.

There is a gradient reason too. The add is a path that carries the
upstream gradient straight back to `x`, plus the gradient through the
block. Deep stacks of pure replacements are harder to train. Initializing
so the added branch is small makes a block start close to "pass `x`
through unchanged."

### Layer norm

`nn.LayerNorm(n_embd)` runs on the last dimension, separately for every
token. For one token's 384 numbers it subtracts their mean and divides by
their standard deviation, then multiplies by a learned scale and adds a
learned shift (both length 384). The token's channels are pulled to a
standard size before attention and again before the feed-forward, so a
layer that has started producing huge values does not hand those magnitudes
to the next layer.

This is pre-norm: normalize, then the branch, then add back onto the
un-normalized `x`. The paper's original block normalized after the add.
The video uses pre-norm. A final `self.ln_f` normalizes once more before
`lm_head`.

### `self.sa_head` is gone  ← (doubt)

Once attention and the feed-forward live inside `Block`, the model holds

```python
self.blocks = nn.Sequential(*[Block(n_embd, n_head=4) for _ in range(n_layer)])
```

`forward` must call `x = self.blocks(x)`. `self.sa_head` and `self.ffwd`
are not attributes of `BigramLanguageModel` anymore. Looking them up raises
`AttributeError`. `Sequential` runs the blocks in order. Each block does
layer-norm + attention + residual, then layer-norm + feed-forward + residual.

`n_layer = 6` in `bigram_v6.py`, so six of those blocks.

---

## 7. Dropout

```python
dropout = 0.2
```

`nn.Dropout(0.2)` , during training, sets each number to 0 with probability
0.2 and scales the survivors by `1/0.8` so the expected total stays the
same. It is applied to the attention weights inside a head, to the
multi-head projection output, and at the end of the feed-forward.

`model.eval()` turns dropout off. `estimate_loss` calls `model.eval()`
before measuring and `model.train()` after, so the reported loss is the
model with every connection present. `model.train()` is the mode the
optimizer steps run in, where dropout is active.

Dropout is a regularizer. A head cannot rely on one exact attention weight
always being present, so the model is less able to memorize the training
text. At `dropout = 0` the layers are the identity and nothing is dropped.

---

## 8. The forward pass of the finished model

```text
idx                         (B, T) integer ids
token embedding             (B, T, n_embd)
+ position embedding        (B, T, n_embd)
6 × Block
    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))
LayerNorm
lm_head                     (B, T, vocab_size) logits
cross-entropy against y     one scalar, only if targets were passed
```

`lm_head` is still `Linear(n_embd, vocab_size)`. For the small model that
is `32 → 65`. For `bigram_v6.py` it is `384 → 65`. Column `i` of that last
layer is still the score for character id `i`. Nothing about cross-entropy
changed from part 1.

`generate` is the same loop as part 1, with one extra crop: only the last
`block_size` tokens are fed in, because the position table and the mask
are sized for that context. The sampled id is appended to the full `idx`,
not to the crop.

---

## 9. The scaled-up run, and where it actually computes

The teaching model (`n_embd = 32`, `block_size = 8`, a few thousand steps
of tiny batches) finishes quickly on CPU. `bigram_v6.py` is the video's
final config:

| | teaching model | `bigram_v6.py` |
|---|---|---|
| `n_embd` | 32 | 384 |
| `block_size` | 8 | 256 |
| `batch_size` | 32 | 64 |
| layers | 3 | 6 (`n_layer`) |
| heads per block | 4 | 4 (hardcoded; `n_head = 6` at the top is unused) |
| steps | 3000 | 5000 |
| learning rate | `1e-3` | `3e-4` |
| dropout | none | 0.2 |

That is on the order of 10 million parameters. Attention cost grows with
`T²`, and `T` is 256, so a CPU run takes many hours.

```python
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

`cuda` is an NVIDIA GPU. This MacBook Pro is an Apple M3 Max; it has no
NVIDIA chip, so a `cuda`-only check falls through to `cpu`. The GPU is a
section of the M3 Max chip itself. PyTorch's name for it is `mps` (Metal
Performance Shaders). `torch.backends.mps.is_available()` is true on this
machine, and `.to(device)` then puts the weights and the batches there.
64 GB of memory is enough for this config. If a smaller Mac runs out of
memory, lowering `batch_size` to 32 or 16 is the knob. The same file on a
machine with an NVIDIA GPU selects `cuda` only if that check is put back;
as written, a CUDA box would also need the `'cuda'` branch or it would
stay on CPU.

`eval_iters = 200` means each printed loss is 200 train batches plus 200
val batches, and that happens every 500 steps. The first print is not
instant.

---

## 10. What this model still is

It is a decoder-only transformer trained to predict the next character of
Tiny Shakespeare. The loss is the same cross-entropy as the bigram. The
difference is that the vector used to produce the 65 logits is no longer a
single row of a lookup table. It is the token embedding plus a position
embedding, rewritten by six blocks of masked multi-head attention and a
feed-forward, with residuals so each block adds a change rather than
replacing the token.

The paper's formula `softmax(Q K^T / sqrt(d_k)) V` is one head. Multi-head
is several of those, concatenated and projected. The encoder half of the
paper (a second sequence, no causal mask, cross-attention) is not in this
file.
