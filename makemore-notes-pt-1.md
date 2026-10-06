# Makemore Notes — Part 1 (Bigram Language Model)

Notes from Andrej Karpathy's "The spelled-out intro to language modeling: makemore"
(part 1), organized around the concepts and the specific doubts I worked through.
Builds a character-level bigram model two ways: (1) by counting, and (2) as a
one-layer neural net trained with gradient descent — and shows they're equivalent.

---

## 0. The dataset

```python
words = open('names.txt', 'r').read().splitlines()
len(words)                          # 32033 names
min(len(w) for w in words)          # 2
max(len(w) for w in words)          # 15
```

`words` is a list of strings (names). We'll break each into character **bigrams**
(adjacent character pairs) and model "what character comes next."

---

## 1. Bigrams and the special token

A **bigram** = a pair of adjacent characters. We wrap each word with a special
token so the model can learn where names **start** and **end**.

- Video first uses two tokens `<S>` / `<E>`, then refactors to a **single** `.` token.
- The `.` marks both start and end. `.` = index 0, letters `a..z` = 1..26.

```python
for w in words:
    chs = ['.'] + list(w) + ['.']            # e.g. '.emma.'
    for c1, c2 in zip(chs, chs[1:]):         # slide a 2-char window
        bigram = (c1, c2)
```

`zip(chs, chs[1:])` pairs each char with the next (same trick as micrograd):
`.emma.` → `(.,e), (e,m), (m,m), (m,a), (a,.)`.

---

## 2. Counting bigrams into a matrix `N`  ← (doubts)

Count how often each pair occurs in a 27×27 tensor.

```python
chars = sorted(list(set(''.join(words))))    # ['a'..'z']
stoi = {c: i+1 for i, c in enumerate(chars)} # a=1 ... z=26
stoi['.'] = 0                                 # '.' gets its OWN index 0
itos = {i: c for c, i in stoi.items()}

N = torch.zeros((27, 27), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        N[stoi[c1], stoi[c2]] += 1
```

Row `i` = "counts of what character follows character `i`".
`N[i, j]` = how many times `j` followed `i`.

### Doubt A: two tokens vs one — indexing convention
- Two tokens (`<S>`=26, `<E>`=27, 28×28) is the video's FIRST version. It leaves an
  all-zero column (`<S>`) and all-zero row (`<E>`) → wasteful → motivates the switch.
- Single `.`=0 (27×27, letters 1..26) is the final version. Use this so the rest of
  the video (sampling, NN) lines up. Counts are the same numbers, just relocated.

### Doubt B: the `.`/`a` collision bug
```python
ctoi = {c:i for i,c in enumerate(chars)}   # a=0 ...
ctoi['.'] = 0                              # BUG: '.' and 'a' BOTH map to 0!
```
Must shift letters to `i+1` so `.` gets its own slot:
```python
ctoi = {c:i+1 for i,c in enumerate(chars)}; ctoi['.'] = 0
```

### Doubt C: counts doubled — forgot to reset `N`
`N[ix1, ix2] += 1` **accumulates**. Running the fill loop twice without re-running
`N = torch.zeros(...)` doubles every count. Always reset the accumulator first.
(Same lesson as zero-grad in micrograd: any `+=` buffer needs a clean start.)

---

## 3. Counts → probabilities  ← (doubts on sum / keepdim)

Normalize each row so it sums to 1 (a probability distribution over the next char).

```python
P = N.float()
P = P / P.sum(1, keepdim=True)     # each ROW divided by its own total
```

### `dim` in `.sum(dim)`
- `dim=0` = rows axis; `dim=1` = columns axis. **The axis you pass gets collapsed.**
- `P.sum(1)` → per-**row** totals (sum across columns) → what we want.
- `P.sum(0)` → per-**column** totals (sum down rows).

### `keepdim`
- `keepdim=True` keeps the reduced axis as size 1: `P.sum(1, keepdim=True).shape == (27,1)`.
- `keepdim=False` (default) removes it: shape `(27,)`.

### Why `keepdim=True` matters (broadcasting trap)
```
P (27,27) / (27,1)  -> divide each ROW by its total     ✓ correct
P (27,27) / (27,)   -> aligns to last axis -> (1,27)     -> divides COLUMNS  ✗ wrong (silent!)
```
So `sum(1, keepdim=True)` gives a (27,1) column that broadcasts per-row. Verify:
`(P / P.sum(1, keepdim=True))[0].sum()  # -> 1.0`.

### Fun fact: row-sums == col-sums here
Every name starts AND ends with `.`, so each token's "left-neighbor" and
"right-neighbor" counts balance → `P.sum(1)` and `P.sum(0)` hold the **same values**
(different shapes: (27,1) vs (1,27)). But they are NOT interchangeable for
normalizing — the shape decides whether you divide rows or columns.

---

## 4. Sampling names with `multinomial`  ← (big doubt)

```python
g = torch.Generator().manual_seed(2147483647)
for _ in range(5):
    out = []
    ix = 0                                  # start at '.'
    while True:
        p = P[ix]                           # distribution over next char
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:                         # sampled '.' = end
            break
    print(''.join(out))
```

### `torch.multinomial` = a weighted random picker
- Input: a probability list. Output: an **index**, chosen in proportion to weights.
- Picture: split the 0–1 line into chunks sized by each probability; throw a dart;
  return whichever chunk it lands in. Bigger prob = bigger chunk = picked more often.
- Returns an **index**, not a value → look it up with `itos[ix]`.
- `replacement=True`: same index can be drawn again (needed for >1 sample).

### `torch.Generator` = controllable random source
- `manual_seed(n)` fixes the random stream → reproducible results.
- Same seed + same probabilities ⇒ identical draws.

### Sampled ≠ most likely  ← (doubt)
`multinomial` returns a **random draw**, not the argmax. Getting index 3 (`c`) once
does NOT mean `c` is most likely — it's just what the weighted dice rolled. Most
likely = `p.argmax()`. We sample (not argmax) so generated names are varied.

### Why output didn't match the video  ← (doubt)
- `torch.multinomial`'s RNG changed across PyTorch versions. On torch 2.8 (mine) the
  same seed gives different samples than the video's torch 1.x. Not a bug — counts
  match, model is correct; the random names just differ. (Downgrade to torch==1.13 to
  match exactly, or just accept it.)

Bigram samples look rough (`mor`, `axx`, ...) because the model only sees ONE previous
character — that's the motivation for better models later.

---

## 5. Evaluating the model: negative log-likelihood (NLL)  ← (doubt)

How good is the model? Score it by the probability it assigns to the real data.

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for c1, c2 in zip(chs, chs[1:]):
        prob = P[stoi[c1], stoi[c2]]
        log_likelihood += torch.log(prob)
        n += 1
nll = -log_likelihood
print(f'{nll/n=}')          # average NLL ≈ 2.45 for the bigram model
```

Reasoning chain (each piece added for a reason):
1. **Likelihood** = product of probabilities the model gives to all real bigrams.
   Higher = better model. But the product of ~228k tiny numbers **underflows** to 0.
2. **log** turns the product into a **sum** (`log(a*b)=log a+log b`): numerically safe,
   and monotonic so it doesn't change which model is better.
3. Each `log(prob)` is ≤ 0 (prob ≤ 1). Sum is negative; **closer to 0 = better**.
4. **Negative** flips it into a **loss** you minimize (0 = perfect). Low NLL = good.
5. **Mean** (÷ n) makes it a per-bigram average, comparable across dataset sizes.

NLL is the natural loss for probabilistic predictions: rewards being confident+right,
punishes being confident+wrong (assigning ~0 prob to something that happens → huge loss).

### Vectorized NLL (the one-liner)  ← (doubt)
The neural-net version computes the SAME loss without a Python loop — over the whole
dataset at once:
```python
loss = -torch.log(probs[torch.arange(num), ys]).mean()
```
- `probs` = `(num, 27)`: each row a prob distribution over the next char.
- `ys` = `(num,)`: the correct next-char index per example.
- **`probs[torch.arange(num), ys]`** = the **pluck trick** (advanced indexing). Two
  parallel index lists paired element-wise: row `torch.arange(num)` = `[0,1,2,...]`,
  column `ys`. Grabs **one number per row** → the prob the model gave to the *correct*
  char, shape `(num,)`. The other 26 numbers per row are ignored.
- **`torch.log(...)`** → log of each (≤ 0; closer to 0 = better).
- **`.mean()`** → average log-likelihood (one scalar).
- **`-`** → flip to a positive loss to minimize (0 = perfect).

Same NLL as the loop above, just: (a) vectorized over all examples, and (b) fed by
`probs` from `xenc @ W` → softmax instead of the count table `P`.

### Model smoothing (counting version)
Add fake counts to avoid zero probabilities (which give infinite loss):
```python
P = (N + 1).float()      # add-one smoothing
P = P / P.sum(1, keepdim=True)
```
More added → distribution more uniform. This is regularization (see §8).

---

## 6. The neural-net version: one-hot + a weight matrix  ← (many doubts)

Same job (given a char, predict next char), but with **learnable weights** instead
of fixed counts.

### One-hot encoding: a character → 27 numbers  ← (doubt)
A single character is fed in as a length-27 vector: all zeros except a 1 at its index.
```python
import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=27).float()   # shape (N, 27)
```
- `N` (rows) = number of input examples (characters fed in).
- `27` (cols) = the **feature dimension** — how many numbers describe ONE input.
- The "27 inputs" = these 27 one-hot slots, NOT the number of examples.

### `W`: the weights  ← (doubts)
```python
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)
```
Read `W` (shape 27×27) two ways — both true because a neuron has one weight per input:
- **rows (27)** = the 27 inputs (one per one-hot slot).
- **columns (27)** = 27 **neurons**; each column = one neuron's 27 weights.
- `W[i, j]` = "the weight neuron `j` puts on input `i`".

`W` = **weights** (learnable). Inputs = `xenc` (the data). Different roles: the one-hot
input just **selects which row of `W`** to read; `W` holds the learned content.

### `xenc @ W`: firing all neurons  ← (doubts)
- **Dot product** = multiply matching elements & add = one neuron's **firing**
  (its output number; bigger = "fires" more strongly = higher score).
- `xenc @ W` does every input-row · neuron-column dot product at once → shape `(N, 27)`.
- **One-hot pluck trick:** because the input is one-hot, `xenc @ W` just selects the
  input character's **row of `W`**. "logits are the 5th row of W" = for input index 5,
  the outputs are literally `W[5]`.

### What the 27 output numbers are  ← (doubt, plain version)
For one input, the 27 outputs = 27 **scores**, one per possible next character.
Bigger score = model prefers that next letter. (Like a judge rating 27 contestants.)

### logits → probabilities (softmax)  ← (doubt)
Raw scores can be negative and don't sum to 1, so convert them:
```python
logits = xenc @ W          # "log-counts" (any real number)
counts = logits.exp()      # exp -> always POSITIVE (like N's counts)
probs  = counts / counts.sum(1, keepdim=True)   # sum to 1 -> probabilities
```
(exp + normalize = **softmax**.)

### What are "log-counts"?  ← (doubt)
Counts must be positive, but the network outputs any real number. Run outputs through
`exp` → positive count-like numbers. Since `exp` undoes `log`, the raw outputs are "the
log of the counts" → **log-counts**. It's just the bridge from arbitrary numbers to
valid positive counts.

---

## 7. Training loop (gradient descent)  ← (doubts)

Exactly the micrograd loop, with `W` as the parameter and NLL as the loss.

```python
for k in range(100):
    # forward
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()  # NLL + reg

    # backward
    W.grad = None            # reset grad FIRST (see below)
    loss.backward()

    # update
    W.data += -50 * W.grad   # step downhill (lr positive, minus = descent)
```

### Why reset `W.grad` before backward  ← (doubt, repeated)
`backward()` **accumulates** into `.grad` (`+=`). Without resetting, gradients from
previous steps pile up → wrong, exploding updates. `W.grad = None` (or `.zero_()`)
gives a clean slate each step. PyTorch accumulates by default so you *can* sum grads
across passes on purpose; for a normal step, resetting is your job.

### Why random init, not zeros  ← (doubt)
- For THIS single linear layer, zeros would actually train fine (each row/col gets a
  distinct gradient from the data).
- But in general, **all-equal weights → symmetry problem**: neurons in a layer would
  compute the same thing, get the same gradient, update identically, never specialize.
  Random init breaks symmetry. Also prefer **small** random values (too large = over-
  confident start). `torch.randn` gives small, centered-at-0 values → good habit.

---

## 8. Regularization & "W = 0"  ← (doubts)

### `W = 0` → uniform predictions
All logits 0 → `exp(0)=1` everywhere → all probs equal `1/27`. So `W=0` = a
"know-nothing" uniform model. Larger weights → more peaked/confident predictions.

### Regularization = smoothing
Add a penalty for large weights to the loss:
```python
loss = nll + 0.01 * (W**2).mean()
```
- `(W**2).mean()` is small when weights are near 0 → acts like a spring pulling `W`→0.
- Pulling `W`→0 pulls probabilities toward **uniform** → **smooths** them → fights
  over-confidence and overfitting.
- Strength (0.01) is a tug-of-war: bigger = smoother (underfit risk); smaller = sharper
  (overfit risk).

| Counting model | Neural model |
|---|---|
| add-one to `N` | add `α·(W**2).mean()` to loss |
| more → more uniform | bigger `α` → smaller `W` → more uniform |

(Karpathy says "smoothing" loosely; technically this is **L2 / weight decay**.)

### Overfitting (asked up front)
Gradient descent minimizes loss on the **training** data only, so with enough capacity
and steps it can memorize noise → low train loss, high unseen loss. Detect via a
train/val/test split; curb with regularization, early stopping, more data, smaller
model. The bigram model is tiny (729 numbers) so it barely overfits — matters more later.

---

## 9. Counting vs neural net — the punchline  ← (doubt)

Both approaches solve the same problem and **converge to the same probabilities**:
- Counting: `P` = exact bigram frequencies (one pass).
- Neural: gradient descent nudges `W` until its softmax probs ≈ those frequencies.

A single-layer softmax trained on NLL provably approaches the count frequencies (with
enough training + little regularization). Final loss should match (~2.45–2.47).

### Why my two samplings differed  ← (doubt)
1. **Main cause:** the neural sampling cell didn't re-seed `g` (no `manual_seed` line),
   so it used a leftover RNG state → different draws than the count cell. Fix: seed `g`
   identically at the top of BOTH loops.
2. **Secondary:** NN probs equal count probs only approximately (undertraining or
   regularization causes small diffs). Sampling is sensitive — one different draw makes
   the whole name diverge.

To match: seed both identically + train to low loss + minimal regularization.

---

## Quick reference / cheat sheet

- **bigram** = adjacent char pair; wrap words with `.` (index 0), letters 1..26.
- **`N[i,j]`** = count of `j` following `i`; reset `N` before re-counting (`+=` accumulates).
- **`P = N/N.sum(1, keepdim=True)`**: normalize ROWS; `keepdim=True` → (27,1) broadcasts per row.
- **`sum(dim)`** collapses that dim; `dim=1` = across columns (per-row totals).
- **`multinomial(p)`** = weighted dice → returns an INDEX (not most likely; that's argmax).
- **`Generator().manual_seed(n)`** = reproducible randomness; same seed+probs → same draws.
- **NLL loss** = `-log(prob of real data).mean()`; low = good; 0 = perfect.
- **vectorized NLL** = `-torch.log(probs[torch.arange(num), ys]).mean()`; pluck trick grabs correct-char prob per row.
- **one-hot**: a char → 27 numbers (the "27 inputs"); `N` examples = rows of `xenc`.
- **`W` (27×27)** = 27 neurons (cols) × 27 weights (rows, one per input); `W[i,j]` = neuron j's weight on input i.
- **`xenc @ W`** = fire all neurons; one-hot input just plucks a row of `W`.
- **logits → exp → normalize** = softmax → probabilities ("log-counts" → counts → probs).
- **training**: forward → `W.grad=None` → backward → `W.data += -lr*W.grad`; reset grad each step.
- **random init** (small) breaks neuron symmetry; **`W=0`** → uniform predictions.
- **regularization** `α·(W**2).mean()` = smoothing; pulls `W`→0 → probs → uniform; fights overfitting.
- **punchline**: counting and gradient descent reach the SAME bigram probabilities.
