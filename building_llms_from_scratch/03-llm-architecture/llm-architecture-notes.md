# LLM Architecture Notes — Chapter 4

Notes from Sebastian Raschka's *Build a Large Language Model (From Scratch)*,
chapter 4, worked in `llm_architecture.ipynb`. Same habit as the earlier
notes: the numbers are the ones this notebook printed, and the confusing
spots are marked ← (doubt).

The chapter assembles the pieces from chapter 2 and chapter 3 into one GPT.
The small config is the GPT-2 124M shape.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}
```

`head_dim` is `768 // 12 = 64`. Twelve blocks are stacked. The output width
is the vocabulary, 50,257.

---

## 1. The placeholder forward pass

`DummyGPT` has the real wiring and empty blocks. Token ids go in. Logits
come out.

```text
token embedding + position embedding
dropout
12 transformer blocks
final layer norm
linear head to vocab_size
```

The two sentences are tokenized with the GPT-2 tokenizer and stacked.

```text
"Every effort moves you"  -> [6109, 3626, 6100, 345]
"Every day holds a"       -> [6109, 1110, 6622, 257]
```

`batch` has shape `[2, 4]`. Seed 123, `DummyGPT` returns

```text
logits.shape == [2, 4, 50257]
```

Two sequences, four positions, one score per vocabulary token. The dummy
blocks return their input unchanged, so this shape is only the embedding
plus the final linear layer. The real blocks keep that same shape.

---

## 2. LayerNorm  ← (doubt)

A deep stack of layers makes gradients vanish or explode, so the weights
never settle on a small loss. LayerNorm forces each token's activation
vector to mean 0 and variance 1 before the next layer uses it. The
normalization is over the last axis, the embedding dimension, and each token
is normalized on its own.

Seed 123, a toy batch of shape `[2, 5]`:

```text
[[0.2961, 0.5166, 0.2517, 0.6886, 0.0740],
 [0.8665, 0.1366, 0.1025, 0.1841, 0.7264]]
```

A linear layer `5 → 6` plus ReLU produces only non-negative numbers, because
ReLU replaces every negative with 0. Before normalization the two rows have
means `0.2432` and `0.1928`, and variances `0.0799` and `0.0670`. After
`(x - mean) / sqrt(variance + 1e-5)` the means are about `0` and the
variances print as `0.9999`.

```python
class LayerNorm(nn.Module):
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalized_x + self.shift
```

`eps = 1e-5` keeps the square root off zero. `scale` starts as ones and
`shift` starts as zeros, both of shape `embedding_dim`, both learned. At
initialization the layer really is mean 0 and variance 1. Training can
stretch and slide that if a different scale fits the data better.

On `batch_example` of width 5, the class prints means around `1e-8` and
variances `0.9998` and `0.9999` when variance is measured with
`unbiased=False`. The same layer on a tensor of shape `[2, 4, 5]`
normalizes each of the 8 tokens separately. Every row of 5 numbers gets its
own mean and variance. The batch axis is not mixed in.

### Why the first print was 1.25, not 1

`tensor.var()` defaults to `unbiased=True`, which divides by `N - 1`.
LayerNorm divides by `N`. Here `N = 5`, and `5 / 4 = 1.25`. The first print,
before `unbiased=False` was added, was `1.2497` and `1.2499`. The hair under
`1.25` is `eps` inside the square root. Measuring with `unbiased=False`
matches the layer and comes back at about `1`.

---

## 3. GELU and the feed-forward width

ReLU is `max(0, x)`. A negative input contributes nothing, and the gradient
there is 0.

GELU multiplies `x` by the Gaussian cumulative distribution, so a negative
input is shrunk instead of cut off.

```text
GELU(x) = x Φ(x)

GELU(x) ≈ 0.5 x (1 + tanh(sqrt(2/π) (x + 0.044715 x³)))
```

`GeLU` in the notebook is that tanh approximation. The plot from `-3` to `3`
shows ReLU as a hinge and GELU as a smooth curve that is already near 0
around `x = -0.75` but not exactly zero, so a negative input still has a
gradient.

The feed-forward module is expand, GELU, contract.

```python
nn.Linear(embedding_dim, 4 * embedding_dim)   # 768 → 3072
nn.GELU()
nn.Linear(4 * embedding_dim, embedding_dim)   # 3072 → 768
```

Each token's 768 numbers become 3072 learned features. GELU runs on each of
those 3072 numbers separately. Two linear layers with a nonlinearity between
them do not collapse into one linear layer. The second layer mixes the wide
features back to 768, which is the width the residual add expects.

`torch.randn(2, 5, 768)` through this module stays `[2, 5, 768]`.

The first `FeedForward` takes the whole config dict. The class defined later,
next to `TransformerBlock`, takes `embedding_dim` directly, and that is the
one the block calls: `FeedForward(config["embedding_dim"])`. Re-running the
earlier cell afterward would make the block pass an integer into a function
that indexes `config["embedding_dim"]`.

---

## 4. Shortcut connections

A shortcut adds a layer's input to a later output, `x = x + layer(x)`. The
gradient then has a path that does not pass through every nonlinearity, so
early layers still receive a usable update.

Five layers, widths `[3, 3, 3, 3, 3, 1]`, input `[[1., 0., -1.]]`, seed 123.
The shortcut is used only when the layer output has the same shape as its
input, so the last layer (`3 → 1`) never adds. Mean absolute gradient on the
weight matrices:

| layer | no shortcut | shortcut |
|---|---|---|
| 0 | 0.000202 | 0.222 |
| 1 | 0.000120 | 0.207 |
| 2 | 0.000715 | 0.329 |
| 3 | 0.001399 | 0.267 |
| 4 | 0.005050 | 1.326 |

Without the shortcut the first layer's gradient is about 25 times smaller
than the last layer's. With the shortcut the first layer stays on the same
scale as the later ones.

---

## 5. The transformer block

One block is the diagram in `transformer-block.png`. Pre-norm: layer norm
before each sublayer, dropout after it, then the shortcut around the whole
sublayer. The original GPT was post-norm. Pre-norm is the stabler order.

```python
ln1_out = self.ln1(x)
mha_out = self.mha(ln1_out)
x = x + self.dropout(mha_out)

ln2_out = self.ln2(x)
ffn_out = self.ffn(ln2_out)
x = x + self.dropout(ffn_out)
```

The shortcut is the `x` from before that branch's layer norm, not the
normalized tensor. Both branches therefore add a 768-vector to a 768-vector.
`MultiHeadAttention` is the efficient chapter 3 class: `head_dim = 64`,
causal mask with `triu(..., diagonal=1)`, dropout on the attention weights,
and `out_proj` mixing the 12 heads back to 768. The block's own dropout is a
second one, on the branch output.

A random input of shape `[2, 4, 768]` comes back as `[2, 4, 768]`.

---

## 6. GPTModel

`GPTModel` replaces the dummy pieces with `TransformerBlock` and `LayerNorm`.
The forward pass is the one in `gpt-model.png`.

```text
token ids [2, 4]
token embedding + position embedding     [2, 4, 768]
dropout
12 blocks
final LayerNorm
linear head                              [2, 4, 50257]
```

The same two sentences produce logits of shape `[2, 4, 50257]`. The last row
of each sequence is the score vector for the next token. Nothing has been
trained, so those scores are an initialization, not a prediction.

### Parameter count and weight tying

```text
total parameters                         163,059,793
token embedding                          [50257, 768]
output head                              [50257, 768]
total without the output head            124,412,160
```

The two matrices have the same shape. The original GPT-2 ties them: one
weight matrix is both the token embedding and the output head, so it is
stored once. This model stores both, which is why the raw count is 163M and
the published GPT-2 small count is 124M. Subtracting the output head,
including its bias, is the difference.

Per block, after dividing the 12-block sums by 12:

```text
feed-forward     4,722,432
attention        2,360,064
```

The feed-forward count is the `768 → 3072` layer, its bias, the
`3072 → 768` layer, and its bias. The attention count is `Wq`, `Wk`, `Wv`,
and `out_proj`. `qkv_bias` is false, so the three QKV linears have no bias.
`out_proj` still has a bias of 768.

Float32 storage of the 163,059,793 parameters is 4 bytes each:

```text
163059793 * 4 / (1024 * 1024) = 622.0 MB
```

The larger configs, parameters and float32 size:

| config | embedding | heads | layers | parameters | size |
|---|---|---|---|---|---|
| 124M | 768 | 12 | 12 | 163.1M | 622 MB |
| medium | 1024 | 16 | 24 | 406.3M | 1.51 GB |
| large | 1280 | 20 | 36 | 838.3M | 3.12 GB |
| xl | 1600 | 25 | 48 | 1637.8M | 6.10 GB |

`head_dim` stays 64 in every row, because `embedding_dim // n_heads` is 64
for all four.

---

## 7. Generation from an untrained model

```python
def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        tokens = torch.cat((tokens, next_token), dim=-1)
    return tokens
```

`model.eval()` turns dropout off. Each step reads only the last position's
logits, shape `[1, 50257]`. `argmax` picks the vocabulary id with the largest
score. Softmax would not change that choice, because it keeps the same
ordering, so the code can argmax the logits directly. The chosen id is
appended and the whole sequence is run again.

`"Hello, how are you?"` encodes to `[15496, 11, 703, 389, 345, 30]`. After
100 new tokens the decode is nonsense (`charbf PLAYDevandro ...`). The
weights are still the random initialization. The loop shows the shape of
generation. It does not yet know English.
