# Attention Notes — Chapter 3

Notes from Sebastian Raschka's *Build a Large Language Model (From Scratch)*,
chapter 3, worked in `attention_mechanism.ipynb`. Same habit as the data-prep
notes: the numbers are the ones this notebook printed, and the confusing spots
are marked ← (doubt).

The input is six tokens, each a vector of 3 numbers. The words are Your,
Journey, Starts, With, One, Step. Nothing is trained in the first half. The
goal of one attention step is a context vector: the same shape as a token,
but a weighted mix of the other tokens.

```python
inputs = torch.tensor([
    [0.43, 0.15, 0.89],  # Your
    [0.55, 0.87, 0.66],  # Journey
    [0.57, 0.85, 0.64],  # Starts
    [0.22, 0.58, 0.33],  # With
    [0.77, 0.25, 0.10],  # One
    [0.05, 0.80, 0.55],  # Step
])                   # shape [6, 3]
```

This is the same multiply as nanoGPT. There the line is `out = wei @ v` in
`Head.forward`. Here `wei` is the attention weights and `v` starts out as the
inputs themselves.

---

## 1. Scores for one query

The query is token 2, "Starts", `inputs[2] = [0.57, 0.85, 0.64]`. Its score
against every token is a dot product.

```python
query = inputs[2]
for i, x in enumerate(inputs):
    attention_scores_2[i] = torch.dot(x, query)
# [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605]
```

A larger dot product means the two vectors point in a more similar direction,
so this query attends more strongly to that token. The largest score is
`1.4754`, against "Journey".

Dividing by the sum turns the scores into weights that add to 1:

```text
[0.1454, 0.2277, 0.2248, 0.1280, 0.1104, 0.1637]    # sum 1
```

Softmax does the same job and keeps every weight positive. `torch.exp` of a
large score would overflow, so the notebook uses `torch.softmax`, which is
stable. On a 1-dimensional vector `dim=0` and `dim=-1` are the same axis.

```python
attention_weights_2 = torch.softmax(attention_scores_2, dim=0)
# [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565]
```

---

## 2. The context vector

Each input is scaled by its weight and the scaled vectors are added.

```python
context_vector_2 = torch.zeros(query.shape)
for i, x in enumerate(inputs):
    context_vector_2 = context_vector_2 + attention_weights_2[i] * x
# [0.4431, 0.6496, 0.5671]
```

The first addition is `0.1390 * [0.43, 0.15, 0.89] = [0.0598, 0.0209, 0.1237]`.
Six additions later the vector is still length 3. It is an enriched "Starts":
mostly Journey and itself, because those two weights were the large ones.

---

## 3. All tokens at once

The scores for every query against every key are a `[6, 6]` matrix.
`attention_scores[i, j]` is the dot product of token `i` with token `j`.
Row 2 is the six scores from section 1.

```python
attention_scores = inputs @ inputs.T     # [6, 3] @ [3, 6] -> [6, 6]
```

`@` is a matrix multiply. Each output entry is one row dotted with one
column. It is not an element-wise product.

### `dim=1` and `dim=-1`  ← (doubt)

```python
attention_weights = torch.softmax(attention_scores, dim=1)
attention_weights.sum(dim=1)     # six 1s
```

`attention_scores` has two axes, so the last axis is axis 1. `dim=1` and
`dim=-1` both softmax across a row: the six scores of one query become six
weights that sum to 1. `dim=0` would softmax down a column instead.

They stop being the same once a batch axis exists. On a tensor of shape
`[8, 6, 6]`, `dim=1` is the middle axis and `dim=-1` is still the row of
scores. Attention keeps using `-1`.

The context vectors for all six tokens are one multiply:

```python
context_vectors = attention_weights @ inputs     # [6, 6] @ [6, 3] -> [6, 3]
```

Row 2 is `[0.4431, 0.6496, 0.5671]`, the same vector the loop built for
"Starts".

---

## 4. Trainable query, key, and value

The inputs are projected by three learned matrices before the dots. With
`d_in = 3` and `d_out = 2`, each matrix is `[3, 2]`. Seed 123.

```python
x_2 = inputs[1]                  # Journey, [0.55, 0.87, 0.66]
query_2 = x_2 @ Wq               # [-1.1729, -0.0048]
keys = inputs @ Wk               # [6, 2]
values = inputs @ Wv             # [6, 2]
```

`Wq`, `Wk`, and `Wv` are weight parameters. Training moves them. The
attention weights are not stored. They are recomputed every forward pass from
the current queries and keys, so they change when the input changes even if
the parameters stay fixed.

The query still has to see every key. One score is `query_2.dot(keys[1])`,
which is `0.1376`. All six scores are `query_2 @ keys.T`.

The scores are divided by `sqrt(d_k)` before the softmax. Here `d_k = 2`,
so the divisor is `1.414`. Dot products grow with the dimension. Without the
scale, a large dimension pushes the softmax toward 0 and 1 and the gradient
becomes tiny.

```python
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)
# [0.1704, 0.1611, 0.1652, 0.1412, 0.2505, 0.1117]
context_vector_2 = attention_weights_2 @ values
# [0.2854, 0.4081]
```

The context vector is now length `d_out` (2), not length 3. The mix is over
the value vectors, which are the learned projection of the inputs.

`SelfAttention_V1` packs this into a module. Its row 1 is that same
`[0.2854, 0.4081]`. `SelfAttention_V2` replaces `nn.Parameter` with
`nn.Linear(..., bias=False)`. `nn.Linear` stores the matrix transposed
relative to a hand-built parameter, initializes it differently, and is what
`.parameters()` and `.to(device)` already know how to handle. With seed 789
the six context vectors are around `[-0.07, 0.07]`. Copying
`self_attention_v2.Wq.weight.T` into the hand-built multiply reproduces those
vectors, which is why the `.T` is there.

---

## 5. Causal attention

A language model predicts the next token, so position `t` may use tokens
`0..t` and must not use `t+1..end`. The future sits above the diagonal of the
`[6, 6]` weight matrix.

### `*` is not `@`  ← (doubt)

The first mask is a lower triangle of ones. `1` means keep.

```text
1 0 0 0 0 0
1 1 0 0 0 0
1 1 1 0 0 0
...
```

```python
masked_attention_weights = attention_weights * mask
```

`*` multiplies entries that sit in the same position. Row 0 of the weights is
`[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510]`. Row 0 of the mask is
`[1, 0, 0, 0, 0, 0]`. The product is `[0.1921, 0, 0, 0, 0, 0]`.

`@` dots each row with each column. Row 0 of `attention_weights @ mask` is
`[1.0000, 0.8079, 0.6433, 0.4781, 0.3231, 0.1510]`. Nothing was zeroed. For
this mask, the element-wise product is the one that drops the future.

After the zeros, the row no longer sums to 1, so it is divided by its new sum.
Row 0 becomes `[1, 0, 0, 0, 0, 0]`. Row 1 becomes `[0.5517, 0.4483, 0, 0, 0, 0]`.

Softmax was applied before the mask, so the future scores did affect the
denominator. Dividing again after the zeros removes that. The shorter way is
to put `-inf` on the future scores first. `exp(-inf)` is 0, so those
positions never enter the softmax.

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked_attention_scores = attention_scores.masked_fill(mask.bool(), float('-inf'))
```

`diagonal=1` leaves the diagonal at 0. A token can still attend to itself.
Without it, the diagonal is 1 too, position 0's whole row is `-inf`, and the
softmax is `NaN`.

### Why `triu` here, and why the slice  ← (doubt)

`masked_fill_` writes `-inf` where the mask is 1. The 1s have to sit on the
future, which is the upper triangle, so the mask is `triu`. A `tril` of ones
is the keep-mask used with `*`. Filling that with `-inf` would hide the past
and leave the future visible.

The buffer is allocated at the full `context_length`. A forward pass can be
shorter. `self.mask[:number_of_tokens, :number_of_tokens]` is the top-left
corner that matches this sequence. A `6 × 6` mask cannot be written onto a
`3 × 3` score matrix. Generation starts at length 1 and grows, so each step
crops the same stored mask.

The two procedures produce the same weights. Row 0 is `[1, 0, 0, 0, 0, 0]`
either way.

---

## 6. Dropout  ← (doubt)

Dropout does not write zeros into `Wq`, `Wk`, or `Wv`. It returns a new
tensor for this forward pass. The attention weights are recomputed from the
queries and keys on the next pass.

```python
torch.manual_seed(123)
dropout = torch.nn.Dropout(p=0.5)
dropout(torch.ones(4, 4))
```

```text
2  2  0  2
2  0  0  0
0  2  0  2
2  2  2  2
```

With `p=0.5`, each kept value is multiplied by `1 / (1 - 0.5) = 2`, so the
expected size of the tensor stays the same. In the attention block this runs
after the softmax, on `wei`, before `wei @ values`. A weight that is zeroed
does not contribute to that step's context vector. `model.eval()` turns
dropout off, which is what generation and loss measurement use.

`batch = torch.stack([inputs, inputs])` is two copies of the same six tokens,
shape `[2, 6, 3]`. In training mode the two copies get different dropout
masks, so `CausalAttention` returns two different blocks. Position 0 has only
one legal weight, itself. Dropout either zeros it, and the context vector is
`[0, 0]`, or keeps it and scales it by 2. That is why one sequence's first
row can be all zeros while the other sequence's first row is not.
`causal_attention.eval()` makes the two blocks match.

---

## 7. The wrapper stacks full heads

`MultiHeadAttentionWrapper` builds `num_heads` separate `CausalAttention`
modules and concatenates them on the last axis.

```python
out = torch.cat([head(x) for head in self.heads], dim=-1)
```

The call uses `d_out=2` and `num_heads=2`. Each head writes 2 numbers, so the
joined tensor is `[2, 6, 4]`. Columns 0 and 1 are head 0. Columns 2 and 3 are
head 1. The first row of the second sequence, `[0, 0, 0.9544, 0.2127]`, is
head 0 dropping its only weight and head 1 keeping its own.

This class does not use `head_dim = d_out // num_heads`. Each head receives
the full `d_out`. The width 4 is `num_heads * d_out`, not the width of one
head.

`CausalAttention` also registers `triu` without `diagonal=1`, and its
`forward` reads the notebook global `mask` rather than `self.mask`. It works
in this notebook only because an earlier cell set that global to
`triu(..., diagonal=1)`. The efficient class below uses its own buffer.

---

## 8. One module, heads sliced out of `d_out`

`head_dim` is the width of one head: its query, key, value, and the context
vector `wei @ v`.

```python
self.head_dim = d_out // num_heads          # 2 // 2 = 1
```

`Wq`, `Wk`, and `Wv` are each `nn.Linear(d_in, d_out)`, so one projection
produces all heads at once. `.view` splits the last axis.

```python
# [2, 6, 2] -> [2, 6, 2, 1]
queries.view(batch_size, number_of_tokens, num_heads, head_dim)
```

`2 = 2 * 1`. The two numbers for one token become one number for each head.
The count of values does not change. `.view(2, 6, 2, 2)` would ask for 48
numbers and this tensor has 24, so it raises.

`transpose(1, 2)` moves the head axis next to the batch:

```text
[b, t, nh, hd] -> [b, nh, t, hd]
```

Then the same attention math runs inside each head.

```python
attention_scores = queries @ keys.transpose(-2, -1)
# [b, nh, t, hd] @ [b, nh, hd, t] -> [b, nh, t, t]
```

`keys.shape[-1]` is `head_dim`, so the scale is `sqrt(head_dim)`. The mask
slice, the softmax on `dim=-1`, and the dropout follow. The context vector of
every head is

```python
context_vector = attention_weights @ values
# [b, nh, t, t] @ [b, nh, t, hd] -> [b, nh, t, hd]
```

`transpose` back to `[b, t, nh, hd]`, then `.contiguous().view(...)` joins the
heads into `[b, t, d_out]`. `.contiguous()` is required because a transpose
does not leave the numbers in the order `.view` expects.

### `out_proj` is not the join  ← (doubt)

The `.view` only changes the shape. For one token, `[head 0, head 1]` is
still those two numbers sitting side by side. Head 0 occupies the first
`head_dim` slots and head 1 the next.

```python
self.out_proj = nn.Linear(d_out, d_out)
context_vector = self.out_proj(context_vector)
```

`out_proj` is a learned `d_out × d_out` matrix. Each of the `d_out` outputs
can use every head, so the next block receives one mixed vector.

With seed 123, `d_out=2`, and 2 heads, the module returns shape `[2, 6, 2]`.
The two sequences still differ because dropout is on. The GPT-2 sized check
uses 12 heads, `d_in = d_out = 768`, context 1024, and batch 20. `head_dim`
is `768 // 12 = 64`, and the output shape is `[20, 1024, 768]`.
