# nanoGPT Notes — Part 1 (Bigram Language Model)

Notes from Andrej Karpathy's "Let's build GPT" video, through the bigram model
in `bigram.py` and the start of the attention math (the bag-of-words average).
Same idea as makemore part 1: a character only predicts the next character.
Here the counts live in a `65 × 65` embedding table trained with cross-entropy,
instead of a hand-counted matrix.

---

## 1. The data

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
len(text)          # 1115394
```

Tiny Shakespeare is one long string, 1,115,394 characters.

```python
chars = sorted(list(set(text)))
vocab_size = len(chars)          # 65
ctoi = {ch: i for i, ch in enumerate(chars)}
itoc = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [ctoi[c] for c in s]
decode = lambda l: ''.join([itoc[i] for i in l])
```

`encode("hello")` → `[46, 43, 50, 50, 53]`. One integer per character.
`data = torch.tensor(encode(text))` has shape `[1115394]` because the tensor
has one id for every character. There is no batch dimension yet. The whole
play is one long sequence of token ids.

90 / 10 split:

```python
n = int(0.9 * len(data))         # 1003854
train_data = data[:n]            # [1003854]
val_data = data[n:]              # [111540]
```

Training updates weights using `train_data` only. `val_data` is the held-out
last 10%, used later only to measure loss.

---

## 2. A batch: `ix`, `x`, `y`  ← (doubt)

```python
batch_size = 32    # how many independent snippets in parallel
block_size = 8     # how many characters in each snippet

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

### What `ix` is

`torch.randint(high, size)` draws integers from `0` up to but not including
`high`, and returns a tensor of shape `size`.

```text
high = len(train_data) - block_size = 1003854 - 8 = 1003846
size = (batch_size,) = (32,)
```

So `ix` is shape `(32,)`, dtype `int64`. With seed `1337` the first call is:

```text
[76049, 234249, 934904, 560986, ...]
```

Each number is a starting index into `data`. The trailing comma in `(32,)`
makes a one-element tuple. `(32)` would just be the integer 32, and
`randint` wants a shape.

### Why subtract `block_size`

A start `i` reads:

```text
x = data[i   : i+8]     # 8 inputs
y = data[i+1 : i+9]     # 8 targets, one step to the right
```

The last index touched is `i+8`. That index must exist, so the biggest legal
`i` is `len(data) - 8`. `randint`'s first argument is already an exclusive
upper bound, so the call is `len(data) - block_size`.

A start of `1003845` reads through index `1003853`, the last training token.
A start of `1003850` would ask for index `1003858`, past the end, and the
slice would be shorter than 8. `torch.stack` would then fail because the
rows would have different lengths.

### `x` and `y`

For `i = 76049` the 9 tokens are the characters `Let's hea`:

```text
index   76049  76050  76051  76052  76053  76054  76055  76056  76057
token      24     43     58      5     57      1     46     43     39
char        L      e      t      '      s             h      e      a
```

```text
x = data[76049:76057] = [24, 43, 58, 5, 57, 1, 46, 43]   "Let's he"
y = data[76050:76058] = [43, 58, 5, 57, 1, 46, 43, 39]   "et's hea"
```

Position `t` of `y` is the character that should follow the first `t+1`
characters of `x`. Input `[24]` (`L`) predicts `43` (`e`). The full 8-token
input predicts `39` (`a`).

`torch.stack` lines the 32 slices into tensors of shape `(32, 8)`.
Both `x` and `y` have that shape.

### `.to(device)`

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

`train_data` stays in normal memory. Only the small batch is moved, so the
token ids and the embedding weights sit on the same device. On this Mac
`device` is `'cpu'` and the tensors are already there. On a GPU machine the
same lines copy the batch before the forward pass.

---

## 3. `B`, `T`, `C`

A tensor of scores has three axes:

| Name | Stands for | In a training batch | Meaning |
|---|---|---|---|
| `B` | batch | 32 | how many independent snippets |
| `T` | time | 8 | characters per snippet (`block_size`) |
| `C` | channels | 65 | one score per vocabulary character |

`idx` arrives as `(B, T)`. The embedding replaces every id with its row of
65 scores, so logits are `(B, T, C) = (32, 8, 65)`.

Cross-entropy wants one row of scores per prediction, so training flattens:

```python
logits.view(B*T, C)     # (256, 65)
targets.view(B*T)       # (256,)
```

`32 * 8 = 256` separate "current character → next character" predictions.

---

## 4. What `m` is, and why `m(x)` runs `forward`  ← (doubt)

```python
m = BigramLanguageModel(vocab_size)   # vocab_size = 65
m.to(device)
```

`m` is one object. Printed, it is:

```text
BigramLanguageModel(
  (token_embedding_table): Embedding(65, 65)
)
```

The only thing inside it is a `65 × 65` table of numbers. That table is the
weights: `m.token_embedding_table.weight`, shape `(65, 65)`.
`m.parameters()` yields that one tensor, because it was registered when it
was assigned.

### `self` is that same object

```python
class Dog:
    def __init__(self, name):
        self.name = name
    def speak(self):
        print(self.name)

rex = Dog("rex")
rex.speak()     # inside speak, self is rex
```

`m(xb, yb)` and `m.generate(...)` both receive `m` as `self`.
`self.token_embedding_table` is `m.token_embedding_table`.

Inside `__init__` the name `m` does not exist yet. Python creates a blank
object, calls `__init__` with that object as `self`, fills in the table, and
only then binds the name `m` to that same object. `id(self)` during
`__init__` equals `id(m)` afterward.

### `super().__init__()`

`BigramLanguageModel` inherits from `nn.Module`. `super().__init__()` runs
`nn.Module.__init__`, which creates the empty containers `_modules`,
`_parameters`, and the hook dicts, and sets `training = True`.

The next line is:

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

`nn.Embedding` is itself a module, so that assignment does more than attach
an attribute. `nn.Module.__setattr__` sees a module and stores it:

```text
m._modules = {'token_embedding_table': Embedding(65, 65)}
```

The optimizer and `loss.backward()` find the `65 × 65` weight by walking
`_modules`. Skipping `super().__init__()` leaves `_modules` uncreated, and
the assignment raises:

```text
AttributeError: cannot assign module before Module.__init__() call
```

### `m(xb, yb)` calls `forward`, not `generate`

Python's rule: `m(...)` runs `__call__` if the object has one.
`nn.Module` sets `__call__ = _wrapped_call_impl`, which calls `_call_impl`.
The top of `_call_impl` is:

```python
forward_call = self.forward          # unless torch.jit is tracing
return forward_call(*args, **kwargs)
```

So these are the same call:

```python
m(xb, yb)
m.forward(xb, yb)
```

`generate` runs only when the code says `m.generate(...)`. Having both
methods on the class means the object knows two instructions. The
parentheses are wired to the method named `forward`.

`*args` is the positional inputs packed as a tuple. `**kwargs` is the
`name=value` inputs packed as a dict.

```python
m(xb, yb)            # args = (xb, yb),  kwargs = {}
m(xb, targets=yb)    # args = (xb,),      kwargs = {"targets": yb}
```

`forward_call(*args, **kwargs)` unpacks them back into a normal call.

Hooks are extra functions you can attach so PyTorch runs them around
`forward` without editing `forward`. A pre-hook runs before `forward` and
can replace the inputs. A hook runs after and can replace the output.
Backward hooks run while gradients flow. This model registers none, so
`_call_impl` returns `forward(...)` immediately.

`_slow_forward` is used only while `torch.jit.trace` is recording. It still
calls `self.forward`. Normal training uses `self.forward` directly.

### `targets=None`

```python
def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)   # (B, T, 65)
    if targets is None:
        loss = None
    else:
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    return logits, loss
```

`m(xb, yb)` fills `targets` with `yb` and returns a loss.
`m(idx)` inside `generate` leaves `targets` as `None`: there is no correct
next character to score, so `loss` is `None`.

The logits returned to `generate` must stay `(B, T, 65)`. Flattening them
before the `return` breaks `logits[:, -1, :]`, which needs three dimensions.
Flatten only the copy that goes into `cross_entropy`.

---

## 5. Why a row of 65 numbers is "scores for the next character"  ← (doubt)

The embedding lookup does not know what a "next character" is. At
initialization, row `43` (the row for `e`) is 65 random numbers.

The meaning is in how the loss uses them. The current character selects the
row. The true next character's id selects the column.

If the current character is `e` (id `43`) and the true next character is `h`
(id `46`), `cross_entropy` says: of these 65 numbers, the correct one is
whatever sits at index `46`. Make that number large, and make the others
small. The gradient writes that into `weight[43, 46]`.

Column `46` means "next character is `h`" because `h` was given id `46` in
`ctoi`, and `y` is `x` shifted one character to the right. The column index
and the character id are the same numbering.

Before any training, the numbers are random scores that the loss *treats*
as next-character scores. Minimizing the loss is what moves probability onto
characters that actually came next. Softmax can be applied at either time.
Before training it just renormalizes the random scores. After training the
probabilities pile up on the likely next characters.

A uniform guess over 65 characters gives each one probability `1/65`.
The loss of knowing nothing is:

```text
-ln(1/65) = ln(65) ≈ 4.17
```

The first batch loss is near that (a run here printed `4.8786`, because a
random row is not exactly uniform). Training should push the loss below
`4.17`.

---

## 6. Cross-entropy, and the softmax inside it  ← (doubt)

`F.cross_entropy` takes raw scores and an integer target. You do not
softmax first. Softmax happens inside the loss.

Toy scores `[2.0, 1.0, 0.1]`, true character in column `0`.

**Step 1. Softmax** = `exp(score) / sum(exp(scores))`:

```text
exp(2.0) = 7.389
exp(1.0) = 2.718
exp(0.1) = 1.105
sum      = 11.213

probabilities = [0.659, 0.242, 0.099]
```

**Step 2.** Read the probability at the target column. Column `0` holds
`0.659`. Every other probability is ignored for the loss number.

**Step 3.** Loss = `-ln` of that one probability:

```text
target 0:  -ln(0.659) = 0.417
target 2:  -ln(0.099) = 2.317
```

A perfect row puts probability `1` on the true column, and `-ln(1) = 0`.
Being confident and wrong makes the loss huge, because `-ln` of a tiny
probability blows up. This is the same negative log-likelihood as makemore:
`-log(prob of the real next character)`, averaged over the batch.

For the real table: current character `e` selects row `43`. If the next
character is `h` (id `46`), the loss for that position is
`-ln(softmax(row 43)[46])`. The batch loss is the average over all
`B * T` positions.

---

## 7. The gradient: `p − 1` on the true column  ← (doubt)

`loss.backward()` fills `.grad`. It does not change `.data`.

The weights have both:

```text
weight.data    the 65 × 65 scores
weight.grad    a second (65, 65) tensor, filled by backward
```

The input ids do not. `xb` is integers (`43` for `e`). An id is an index,
not a weight, so there is no `.grad` on `xb` to update. Backward writes into
the rows of `weight.grad` that those ids selected.

For one row, the gradient is:

```text
true column:      probability − 1     (negative, because p < 1)
every other col:  probability         (positive)
```

Same toy, target at column `0`:

```text
probabilities = [0.659, 0.242, 0.099]
gradients     = [0.659 − 1,  0.242,  0.099]
              = [-0.341,     0.242,  0.099]
```

We minimize the **loss**, not the gradient. The gradient is the direction
that makes the loss go up. The update steps the opposite way, which is the
micrograd rule `p = p - lr * grad`.

```text
2.0 − (−0.341) = 2.341     true column goes up
1.0 − 0.242    = 0.758     other columns go down
0.1 − 0.099    = 0.001
```

New probabilities `[0.768, 0.158, 0.074]`. Loss falls from `0.417` to
`0.264`. Column `0` was short of probability `1` by `0.341`, so its gradient
was `-0.341`. Subtracting a negative number raises that score.

That is why column `46` of row `43` grows whenever `e` is followed by `h`,
and the other 64 columns of that row shrink.

---

## 8. The training loop, and how it maps to micrograd

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):          # 3000
    xb, yb = get_batch('train')        # always the training split
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

Micrograd did the same four steps by hand:

```python
for p in m.parameters():
    p.grad = 0.0
loss.backward()
for p in m.parameters():
    p.data += -0.05 * p.grad
```

| This file | Micrograd |
|---|---|
| `logits, loss = m(xb, yb)` | forward, then squared-error loss |
| `optimizer.zero_grad()` | `p.grad = 0.0` |
| `loss.backward()` | `loss.backward()` |
| `optimizer.step()` | `p.data += -lr * p.grad` |

`zero_grad` matters because `backward` adds into the existing gradient
(`+=` in micrograd). Leaving the old gradient in place piles steps on top
of each other. `set_to_none=True` clears them by setting each `.grad` to
`None`.

`AdamW` still steps against the gradient. `lr=1e-3` is the step size
`0.001`. It also keeps a moving average of recent gradients so a weight
that has been pushed the same way for several batches takes a smoother
step. The `W` is a small pull of the weights toward zero, the same idea as
the `0.01 * (W**2).mean()` penalty in makemore.

One call of `m(xb, yb)` does not train the table. Each of the 3000 steps
draws a **new** batch of 32 snippets and nudges only the rows for characters
that appeared in that batch. After many steps the rows stop being random.

The numbers live in memory on the module:
`m.token_embedding_table.weight`. Nothing writes them to a file unless the
model is saved.

---

## 9. `estimate_loss`: measuring both splits without updating  ← (doubt)

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)          # 200
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

Called every `eval_interval` steps (`300`):

```python
if iter % eval_interval == 0:
    losses = estimate_loss()
    # {'train': ..., 'val': ...}
```

The training loop's `get_batch('train')` plus `optimizer.step()` is the only
place weights change. `estimate_loss` also runs a forward pass, but on both
splits, and it never calls `backward` or `step`.

`@torch.no_grad()` skips storing the graph. `loss.item()` copies the scalar
out as a Python float. 200 batches are averaged because one batch of 32
snippets jumps around. `200 * 32 = 6400` snippets per split.

`model.eval()` / `model.train()` flip evaluation mode. This bigram model has
no dropout, so the scores do not change between the two modes. The switch is
there so a later model that does have dropout is measured with dropout off.

`get_batch(split)` is how both splits get a forward pass:

```python
data = train_data if split == 'train' else val_data
```

Train loss: how well the table fits the text it learns from.
Val loss: how well it fits the held-out last 10%. If train keeps falling and
val does not, the model is memorizing the training text.

---

## 10. `generate`  ← (doubt)

```python
def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits, loss = self(idx)          # (B, T, 65); loss is None
        logits = logits[:, -1, :]         # (B, 65)
        probs = F.softmax(logits, dim=-1) # (B, 65), each row sums to 1
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        idx = torch.cat((idx, idx_next), dim=-1)            # T grows by 1
    return idx
```

`max_new_tokens` is an argument. Nothing inside the model computes it.
`m.generate(idx, max_new_tokens=3)` runs the loop 3 times and appends 3
characters to every row.

`self(idx)` is `m(idx)`, which runs `forward`. It is one forward pass to
read scores. It does not compute a training loss and it does not update
the table. The thing that must be repeated to improve the scores is the
training loop in section 8. `generate` is what you run after that, to sample
text from the updated table.

### Shapes, using a `(4, 8)` batch and `max_new_tokens = 3`

Length after pass `k` is `8 + k`. When the loop ends, `k = 3`, so the
returned tensor has shape `(4, 11)`.

| Pass | `idx` in | `self(idx)` | `[:, -1, :]` | draw | `idx` out |
|---|---|---|---|---|---|
| 1 | `(4, 8)` | `(4, 8, 65)` | `(4, 65)` | `(4, 1)` | `(4, 9)` |
| 2 | `(4, 9)` | `(4, 9, 65)` | `(4, 65)` | `(4, 1)` | `(4, 10)` |
| 3 | `(4, 10)` | `(4, 10, 65)` | `(4, 65)` | `(4, 1)` | `(4, 11)` |

The usual call at the bottom of `bigram.py` starts from a single newline
(token `0`), one sequence:

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
decode(m.generate(context, max_new_tokens=500)[0].tolist())
```

First pass: `idx` is `(1, 1)`, logits are `(1, 1, 65)`, the last-step slice
is `(1, 65)`. After 500 new tokens the length is `501`.

### `[:, -1, :]`

For logits of shape `(4, 8, 65)`, this keeps only the last time position of
each snippet. Shape becomes `(4, 65)`: the 65 scores for what comes after
the current last character.

The integer `-1` drops the time axis. A list index `[:, [-1], :]` keeps a
size-1 axis and yields `(4, 1, 65)`. `multinomial` cannot take a 3D tensor,
so the integer index is the one this line needs.

Only the last position is used because the new character is being appended
at the end. The predictions at the earlier positions are for characters
already sitting in `idx`. For a bigram model the earlier characters do not
affect the scores anyway (the row depends only on the current id). The loop
still feeds the whole sequence so the same `generate` works later, when the
model actually uses the full context.

### `dim=-1`

`dim=-1` means the last axis. On a `(4, 65)` tensor that axis is the 65
scores. Softmax along it makes each row add up to 1, and does not mix the
4 snippets together.

```text
[[1, 2, 3],          dim=-1 (across each row)
 [1, 1, 1]]

[[0.090, 0.245, 0.665],     each row sums to 1
 [0.333, 0.333, 0.333]]
```

`dim=0` would walk down the columns instead. `torch.cat(..., dim=-1)` joins
on that same last axis, which for `idx` of shape `(B, T)` is the length.

### Why softmax here, after training already softmaxed

The softmax inside `cross_entropy` is used to compute the training loss and
is not returned. `forward` returns raw scores. Some are negative, and they
do not add up to 1. `generate` needs probabilities, so it softmaxes the last
position's scores itself.

### Multinomial is a weighted roll

One draw is random. Across many draws, a character with probability `0.70`
shows up about 7 times as often as a character with probability `0.10`.

```text
h  0.70
e  0.20
q  0.10
```

1000 draws from that row:

```text
h  706
e  184
q  110
```

Picture one row as a stick of length 1, cut into slices whose widths are the
probabilities. Pick a random point on the stick. The slice you land in is
the character id. `num_samples=1` draws one id per row, so `idx_next` has
shape `(B, 1)`.

Before training every probability is near `1/65`, so the text looks like
noise. After training the large slices sit on characters that really
followed, and the roll mostly lands there. Always taking the single biggest
score (`argmax`) would repeat the same favorite character every time. Drawing
keeps the less common, still-plausible characters.

---

## 11. Bag of words: the running mean  ← (start of the attention section)

The bigram table throws away every character except the current one. The
next construction gives each position a summary of the characters up to and
including itself, by averaging their vectors.

```python
# x has shape (B, T, C) = (4, 8, 2)
# we want xbow[b, t] = mean of x[b, i] for all i <= t
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]          # (t+1, C)
        xbow[b, t] = torch.mean(xprev, dim=0)
```

### Why `x[b, :t+1]` is `(t+1, C)`

`x` is `(4, 8, 2)`: 4 sequences, 8 tokens, 2 channels each. `x[0]` is one
sequence, shape `(8, 2)`. `:t+1` keeps rows `0` through `t` and both
channels.

Seed `1337`, batch item 0:

```text
x[0] =
              channel 0     channel 1
t=0            0.1808        -0.0700
t=1           -0.3596        -0.9152
t=2            0.6258         0.0255
t=3            0.9545         0.0643
t=4            0.3612         1.1679
t=5           -1.3499        -0.5102
t=6            0.2360        -0.2398
t=7           -0.9211         1.5433
```

```text
t=0   x[0, :1]   shape (1, 2)   row 0 only
t=1   x[0, :2]   shape (2, 2)   rows 0 and 1
t=2   x[0, :3]   shape (3, 2)   rows 0, 1, 2
```

`torch.mean(..., dim=0)` averages down the rows. Channel 0 is averaged only
with earlier channel-0 values. The channels are not averaged with each
other. `dim=1` would collapse each vector to a single number. That is a
different operation.

For `t=1`:

```text
channel 0:  (0.1808 + -0.3596) / 2 = -0.0894
channel 1:  (-0.0700 + -0.9152) / 2 = -0.4926
```

`xbow[0]` ends up:

```text
t=0   [ 0.1808, -0.0700]    mean of 1 row
t=1   [-0.0894, -0.4926]    mean of 2 rows
t=2   [ 0.1490, -0.3199]    mean of 3 rows
t=3   [ 0.3504, -0.2238]    mean of 4 rows
t=4   [ 0.3525,  0.0545]
t=5   [ 0.0688, -0.0396]
t=6   [ 0.0927, -0.0682]
t=7   [-0.0341,  0.1332]    mean of all 8 rows
```

`xbow` has the same shape as `x`, `(4, 8, 2)`. Position `t` can see tokens
`0..t` and cannot see tokens `t+1..7`. That is the same "don't look into the
future" rule the later attention mask enforces. The Python loops are the
slow version of a matrix multiply that produces the same averages.
