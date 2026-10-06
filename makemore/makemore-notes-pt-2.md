# Makemore Notes — Part 2 (MLP Character-Level Model)

Notes from Andrej Karpathy's "makemore part 2" — building a **multi-layer perceptron
(MLP)** language model following the Bengio et al. 2003 paper. We go from a bigram
lookup (part 1) to a network that looks at **3 previous characters** and learns
**embeddings** for them. Organized around the concepts + the specific doubts I hit
(marked ← (doubt)).

---

## 0. What's new vs. part 1

| Part 1 (bigram) | Part 2 (MLP) |
|---|---|
| context = **1** previous char | context = **block_size** chars (here 3) |
| `W` (27×27), one matrix | embedding `C` + hidden layer (`W1,b1`) + output (`W2,b2`) |
| char → one-hot → logits | chars → **embeddings** → tanh hidden layer → logits |
| no hidden layer | one hidden layer **with `tanh`** (the non-linearity) |
| whole dataset each step | **minibatches** + train/val/test split |

The big new ideas: **embeddings**, a **hidden layer with a non-linearity**,
**minibatch** training, a **learning-rate search**, and **train/val/test splits**.

---

## 1. Dataset: the context window  ← (doubt)

Each example = the last `block_size` characters → predict the next one.

```python
block_size = 3                       # how many chars of context
X, Y = [], []
for w in words:
    context = [0] * block_size       # start: '...' (all '.')
    for ch in w + '.':
        ix = c2i[ch]
        X.append(context)            # input  = current 3-char context
        Y.append(ix)                 # label  = the char that follows
        context = context[1:] + [ix] # slide the window forward
X = torch.tensor(X)                  # (num_examples, 3)
Y = torch.tensor(Y)                  # (num_examples,)
```

### `context = context[1:] + [ix]` — the sliding window  ← (doubt)
Shifts the fixed-length window right by one:
- `context[1:]` drops the **oldest** (leftmost) char.
- `+ [ix]` appends the **newest** char on the right.
- Length stays `block_size`.

```
[0,0,0] --ix=5-->  [0,0,5]  ('..e')
[0,0,5] --ix=13--> [0,5,13] ('.em')
[0,5,13]--ix=13--> [5,13,13]('emm')
```

Same idea is reused later to feed the model its own predictions when **sampling**.

Full dataset (all 32,033 words) → `X.shape = (228146, 3)`.

---

## 2. The embedding matrix `C`  ← (many doubts)

Instead of one-hot → big matrix, we **embed** each character into a small vector.

```python
C = torch.randn((27, 10))    # 27 chars -> 10-dim embedding each (video uses 2 then 10)
```

### `C[5]` == `one_hot(5) @ C`  ← (doubt: is C the weight matrix?)
```python
C[5]                                              # -> tensor([-0.4713, 0.7868])
F.one_hot(torch.tensor(5), num_classes=27).float() @ C   # -> SAME
```
Both identical → the embedding lookup **is** a matrix multiply. So:
- **`C` is the weight matrix of the first (linear) layer.** ✔
- Indexing `C[5]` is just the fast way to compute `one_hot(5) @ C` (the "pluck trick"
  from part 1). We write it as indexing because it's faster and cleaner.

### Embedding = a linear layer with NO non-linearity  ← (doubt)
A normal layer is `activation(input @ W + b)`. The embedding is:
- `one_hot(X) @ C` = `input @ W` (with `C` as `W`),
- **no bias**, **no `tanh`** → **purely linear** (no bending).
Because the input is one-hot, the "weighted sum" degenerates into **selecting a row**
of `C`. That's why we can skip the multiply and just do the lookup.

### How many neurons is `C`?  ← (doubt)
For `C` shape `(27, D)` (D = embedding dim):
- **neurons = D = columns** (embedding dimension). (D=2 or D=10 in the video.)
- **weights per neuron = 27 = rows** (one per input / one-hot slot).
- A **single neuron = one column** of `C` (its 27 weights).

Two readings of the same numbers:
- **column** `C[:, j]` = neuron `j`'s weights.
- **row** `C[i]` = character `i`'s **embedding vector**.

### `emb = C[X]` and the `(N, 3, D)` shape  ← (doubt)
```python
emb = C[X]          # X is (N, 3) -> emb is (N, 3, D)
emb.shape           # e.g. (228146, 3, 10)
```
`C[X]` replaces **each index** in `X` with its D-number embedding:
```
emb[i, j, k]
     │  │  └── k: which of the D embedding numbers   (embedding dim)
     │  └───── j: which of the 3 context positions   (block_size)
     └──────── i: which training example             (dataset size)
```
So `(N, 3, D)` = **examples × context chars × embedding dims**. (This shape is NOT a
minibatch thing — see §6.)

---

## 3. Gluing the 3 embeddings into one input row

The hidden layer wants **one flat vector per example**, so concatenate the 3 embeddings:
`(N, 3, D)` → `(N, 3*D)`.

### `emb[:, 0, :]` — slicing a 3-D tensor  ← (doubt)
Three positions, one per axis:
- `:` axis 0 → **all examples**
- `0` axis 1 → **only context position 0** (a number drops that axis)
- `:` axis 2 → **all embedding dims**

→ shape `(N, D)`: the first context char's embedding, for every example.
```
emb[:, 0, :]   # 1st context char
emb[:, 1, :]   # 2nd
emb[:, 2, :]   # 3rd
```
Rule: `:` = "keep this axis"; a **number** = "pick one index and drop this axis".

### Three ways to flatten (all equivalent)
```python
# 1) manual (hardcoded to block_size=3)
torch.cat([emb[:,0,:], emb[:,1,:], emb[:,2,:]], dim=1)     # (N, 3*D)

# 2) unbind (adapts to any block_size)  ← (doubt)
torch.cat(torch.unbind(emb, 1), 1)

# 3) view (best — no copy, just re-reads memory)
emb.view(emb.shape[0], -1)                                 # (N, 3*D)
```

### `torch.unbind(emb, 1)`  ← (doubt)
`unbind(t, dim)` = "unstack": removes a dim and returns a **tuple of all slices** along
it. `torch.unbind(emb, 1)` → `(emb[:,0,:], emb[:,1,:], emb[:,2,:])` — exactly the manual
list, but generated automatically (works for any block_size). `cat` then glues them.

### `emb.view(...)` — the efficient one
`view` just **reinterprets the same memory** with a new shape (no copying/concatenating),
so it's faster. `emb.view(-1, 30)` or `emb.view(emb.shape[0], -1)`:
- `-1` = "infer this dim". `emb.view(emb.shape[0], -1)` is safest — it always computes
  `3*D` for you, so it adapts if block_size or D change (see the bug in §7).

---

## 4. Hidden layer (the actual non-linearity)

```python
W1 = torch.randn((30, 200))   # 30 = block_size*D = 3*10 ; 200 neurons
b1 = torch.randn(200)
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)    # (N, 200)
```
- Input dim **30** must equal `block_size * embedding_dim`.
- **200** = number of hidden neurons (a knob).
- `tanh` = the **non-linearity** that lets the network learn curved functions (the
  embedding layer had none; this layer does).

---

## 5. Output layer + loss  ← (doubt: loss = inf)

```python
W2 = torch.randn((200, 27))
b2 = torch.randn(27)
logits = h @ W2 + b2          # (N, 27) — one score per possible next char
```

Manual softmax + NLL (same as part 1, vectorized):
```python
counts = logits.exp()
probs  = counts / counts.sum(1, keepdim=True)
loss   = -probs[torch.arange(N), Y].log().mean()
```

### `loss = tensor(inf)`  ← (doubt)
Cause: a probability for a **correct** char came out as exactly `0.0`, then `log(0) =
-inf` → loss `inf`. The `0.0` is a **numerical underflow**: randomly-initialized weights
make some logits huge, so within a row one logit is ~60+ below the max → `exp()` rounds
to `0.0` in float32 → that prob is 0 → `log(0)` blows up. One bad entry poisons the mean.

### Fix: `F.cross_entropy`  ← (doubt)
```python
loss = F.cross_entropy(logits, Y)     # same value, numerically STABLE
```
Why it's better (Andrej's points):
- **Numerically stable:** internally subtracts each row's **max logit** before `exp`
  (log-sum-exp trick), so nothing overflows to `inf` or underflows to a hard `0`.
- **Faster / less memory:** no intermediate `counts`/`probs` tensors.
- **Simpler backward pass.**
Always use `F.cross_entropy(logits, Y)` instead of hand-rolled softmax+log.

---

## 6. Full network + minibatch training  ← (doubt)

### The parameters
```python
g = torch.Generator().manual_seed(2147483647)
C  = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g); b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g); b2 = torch.randn(27,  generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters: p.requires_grad = True
sum(p.numel() for p in parameters)     # ~11,897 params
```

### The training loop
```python
for i in range(40000):
    # 1) minibatch
    ix  = torch.randint(0, X_train.shape[0], (32,))   # 32 random example indices
    # 2) forward
    emb = C[X_train[ix]]                              # (32, 3, 10)
    h   = torch.tanh(emb.view(-1, 30) @ W1 + b1)      # (32, 200)
    logits = h @ W2 + b2                              # (32, 27)
    loss = F.cross_entropy(logits, Y_train[ix])
    # 3) backward
    for p in parameters: p.grad = None                # reset grads (accumulate otherwise)
    loss.backward()
    # 4) update
    for p in parameters: p.data += -lr * p.grad       # step downhill
```

### Minibatches  ← (doubt)
- The full dataset (~228k) is huge; forward+backward over **all** of it every step is
  slow (one update per full sweep).
- Instead each step grabs a **random 32** examples (`torch.randint` → `ix`), and updates
  on just those.
- `torch.randint(0, X.shape[0], (32,))` = 32 random row indices into the dataset;
  `X[ix]` / `Y[ix]` = the batch.
- **Trade-off:** the batch gradient is a **noisy approximation** of the true gradient,
  but each step is ~1000s× cheaper → take **many more** steps. Karpathy: *"better to have
  an approximate gradient and take many steps than the exact gradient and take few."*
- **The two 32s are different things:** early on (5 words) 32 was the *whole* dataset;
  now 32 is a **batch size you choose**. Batch loss printed during training looks
  **jumpy** because each step measures a different random subset.

---

## 7. The `view` shape-mismatch bug  ← (doubt)

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (48x20 and 30x200)
```
- `emb.view(-1, 20)` made `(48, 20)`, but `W1` is `(30, 200)` → inner dims `20 ≠ 30`.
- Root cause: with `D=10, block_size=3`, `emb` is `(32, 3, 10)` = 960 numbers. Reshaping
  to 20 cols forces 48 rows (`960/20`) — wrong. Must reshape to **30** (`= block_size*D`):
```python
emb.view(-1, 30)                # (32, 30)  ✔
emb.view(emb.shape[0], -1)      # better: auto-computes 30, adapts to any D/block_size
```
Rule: **`emb.view(-1, N) @ W1` needs `N == W1.shape[0] == block_size * embedding_dim`.**
Debug habit: `print(emb.shape, W1.shape)` right before a failing matmul.

---

## 8. Finding a good learning rate

Instead of guessing, **search** over a range and watch the loss.
```python
lre = torch.linspace(-3, 0, 1000)   # exponents from -3 to 0
lrs = 10**lre                       # candidate LRs: 0.001 .. 1.0 (log-spaced)
```
- Use **log spacing** (`10**...`) because LR matters multiplicatively (0.001→0.01 is a
  big jump; 0.5→0.51 isn't).
- Run training stepping the LR through `lrs`, record `lri` (the exponent/LR) and `lossi`
  (loss), then plot `lri` vs `lossi`.
- Pick the LR near the **bottom** of the curve (before it spikes up):
```python
eff_lr = lri[lossi.index(min(lossi))]   # ~0.1 here
```
- Typical scheme: train at a good LR for a while, then **decay** it (e.g. ÷10) near the
  end for fine-tuning ("learning-rate decay").

---

## 9. Train / val / test split (overfitting)

As the model gets bigger it can **memorize** the training set. Split the data:
```python
def build_dataset(words):
    ...                                  # same context-window builder as §1
    return X, Y

random.shuffle(words)
n1 = int(0.8*len(words)); n2 = int(0.9*len(words))
X_train, Y_train = build_dataset(words[:n1])   # 80% — train params
X_val,   Y_val   = build_dataset(words[n1:n2]) # 10% — tune hyperparams (size, LR, ...)
X_test,  Y_test  = build_dataset(words[n2:])   # 10% — final check ONCE
```
- **Train (80%)**: fit the weights.
- **Val/dev (10%)**: choose hyperparameters (hidden size, embedding dim, LR, steps).
- **Test (10%)**: touch **rarely** (ideally once) — an honest estimate of real performance.
- **Overfitting signal:** train loss ≪ val loss. Here they're close (train ≈ 2.18,
  val ≈ 2.21) → we're actually mildly **under**fitting → room to grow the model.

### Evaluating on a whole split (no minibatch)
```python
emb = C[X_val]
h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Y_val)     # honest full-split loss
```
Always judge progress on the **full** split, not a noisy minibatch.

---

## 10. Visualizing the embeddings (when D=2)

```python
plt.scatter(C[:,0].detach().numpy(), C[:,1].detach().numpy(), s=200)
for i in range(27):
    plt.text(C[i,0].item(), C[i,1].item(), i2c[i], ha='center', va='center')
```
- `.detach()` = drop the autograd tracking so it can go to numpy (we're just plotting,
  not training).
- After training, similar characters cluster: the **vowels** end up near each other,
  special `.` off on its own, etc. The network **learned** that some characters behave
  alike — that's the payoff of embeddings over one-hot.

---

## 11. Sampling from the model

Generate names by feeding predictions back in (autoregressive), reusing the §1 window slide.
```python
g = torch.Generator().manual_seed(2147483647)
for _ in range(20):
    out = []
    context = [0] * block_size                 # start with '...'
    while True:
        emb = C[torch.tensor([context])]       # (1, 3, 10)
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)       # softmax here (not cross_entropy) — we need probs
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]           # slide window with the new char
        out.append(ix)
        if ix == 0:                            # sampled '.' -> end of name
            break
    print(''.join(i2c[i] for i in out))
```
- `F.softmax(logits, dim=1)` turns logits → probabilities (used here because we need the
  actual distribution to sample; `cross_entropy` was for training loss).
- `torch.multinomial(probs, 1)` = weighted dice → a next-char index (part-1 idea).
- Same window-slide `context = context[1:] + [ix]` feeds the prediction back in.
- Samples (`maleah`, `samiyah`, `keya`, ...) look more name-like than the bigram model —
  more context (3 chars) + learned embeddings help.

---

## Quick reference / cheat sheet

- **block_size** = # context chars (3); each example = last 3 chars → next char.
- **`context = context[1:] + [ix]`** = slide the window (drop oldest, add newest).
- **`C` (27×D)** = embedding table AND the first (linear, no-bias, no-tanh) layer's weights.
- **`C[X]`** = embed each index; shape `(N, block_size, D)` = examples × context × emb-dims.
- **`C` neurons** = D (columns); weights/neuron = 27 (rows); `C[i]` = char i's embedding.
- **`emb[:, 0, :]`** = 1st context char's embedding for all examples (`:` keeps axis, number drops it).
- **flatten**: `emb.view(emb.shape[0], -1)` (best) ≡ `cat(unbind(emb,1),1)` ≡ manual `cat`.
- **`unbind(t, dim)`** = unstack into a tuple of slices; **`view`** = re-shape same memory (cheap).
- **hidden layer**: `h = torch.tanh(emb.view(-1, 30) @ W1 + b1)`; `tanh` = the non-linearity.
- **view rule**: `N` in `emb.view(-1, N)` must equal `W1.shape[0] = block_size * D` (30).
- **`logits = h @ W2 + b2`**; **loss = `F.cross_entropy(logits, Y)`** (stable; avoids `inf`).
- **`loss = inf`** = a correct-char prob underflowed to 0 → `log(0)`; fixed by cross_entropy.
- **minibatch**: `ix = torch.randint(0, X.shape[0], (32,))`; train on `X[ix], Y[ix]` — noisy but fast.
- **LR search**: `lrs = 10**torch.linspace(-3,0,1000)`; plot loss vs LR; pick near the min; decay later.
- **splits**: 80/10/10 train/val/test; train≪val ⇒ overfit; here train≈val ⇒ underfit (grow model).
- **eval** on the FULL split, not a minibatch.
- **`.detach()`** before numpy (drop autograd) for plotting embeddings.
- **sampling**: `F.softmax` → `torch.multinomial` → slide window → stop at `.` (index 0).
