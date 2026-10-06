# Micrograd Notes — Part 2

Continuation of Part 1. Covers automating backprop, extending the `Value` class,
comparing to PyTorch, and building `Neuron` → `Layer` → `MLP` up through gradient
descent. Organized around the concepts and the specific doubts I worked through.

---

## 1. Method syntax vs function syntax: `n.tanh()` == `tanh(n)`  ← (doubt)

`n.tanh()` and `tanh(n)` are the *same* operation, written two ways:

```python
tanh(n)      # function style:  pass n INTO tanh
n.tanh()     # method style:    ask n to tanh ITSELF
```

We write `n.tanh()` because `tanh` is defined as a **method on the `Value` class**.
When you call `n.tanh()`, Python passes `n` in as `self`.

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    return Value(t, (self, ), 'tanh')
```

Mental translation whenever you see method syntax — flip it to function syntax:

| Written | Means |
|---|---|
| `n.tanh()` | `tanh(n)` |
| `a * b` | `a.__mul__(b)` |
| `a + b` | `a.__add__(b)` |

The thing **left of the dot** is the input that becomes `self`. Operations are
methods on `Value` so the graph (`_prev`, `_op`, `_backward`) gets recorded
automatically as a side effect of computing.

---

## 2. How `_backward` fills a parent's grad — closures  ← (doubt)

When you build a node, a `_backward` function is created and stored **on that node**,
and it "remembers" its inputs via a **closure**.

```python
def tanh(self):
    x = self.data
    t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
    out = Value(t, (self, ), 'tanh')
    def _backward():
        self.grad += out.grad * (1 - t**2)   # self, out, t are "captured"
    out._backward = _backward
    return out
```

When you called `o = n.tanh()`:
- `self` was bound to `n`, `out` to `o`, `t` to `o.data`.
- The inner `_backward` **captured** those variables (a closure), so calling
  `o._backward()` later still knows `self` means `n`.

So `o._backward()` runs `n.grad += o.grad * (1 - o.data**2)`. That's how calling a
method *on `o`* writes into *`n`*.

Local backward rules stored per operation:

| Node built by | Its output's `_backward` does |
|---|---|
| `+` | copies `out.grad` into both inputs (`+=`) |
| `*` | each input `+= out.grad × (other input's data)` |
| `tanh` | input `+= out.grad × (1 − out.data²)` |

`x._backward()` means: "I already know my `.grad`; push it **one edge** back to the
node(s) that made me."

---

## 3. Topological sort: ordering the backward pass  ← (doubt)

A node's grad can't be computed until every node **downstream** (closer to the
output) already has its grad. So we sort nodes in dependency order.

```python
topo = []
visited = set()
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:   # recurse into inputs FIRST
            build_topo(child)
        topo.append(v)          # add self only AFTER all children done
build_topo(o)
```

- **post-order DFS**: a node is appended only after everything it depends on.
- result: inputs first, output last → `[x1, w1, x1*w1, ..., n, o]`.
- `visited` prevents processing a node twice (nodes reachable via multiple paths).

For backprop you walk it in **reverse** (output → inputs):

```python
o.grad = 1.0
for node in reversed(topo):
    node._backward()
```

Reversing guarantees each node's `.grad` is already filled before its `_backward`
runs. This is the automation of doing `o._backward()`, `n._backward()`, ... by hand.

---

## 4. Where does `_prev` come from?  ← (doubt)

Two kinds of nodes:

**Leaf nodes** you create directly — only `data`/`label` passed, so `_prev` is empty:

```python
x1 = Value(2.0, label='x1')
# __init__: self._prev = set(())  -> empty set
```

**Computed nodes** made by operations — the operator fills `_children` for you:

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')  # (self, other) -> _children
    ...
```

So `x1w1 = x1*w1` automatically gets `x1w1._prev = {x1, w1}`. The graph structure is
built as a **side effect of the forward pass** — you only ever pass `data`/`label`,
the links assemble themselves.

| Line | Operator | `_prev` |
|---|---|---|
| `x1w1 = x1*w1` | `__mul__` | `{x1, w1}` |
| `n = ... + b` | `__add__` | `{..., b}` |
| `o = n.tanh()` | `tanh` | `{n}` |

That's why `build_topo` can walk `v._prev`: leaves stop the recursion, computed
nodes point back to their inputs.

---

## 5. The `backward()` method + two classic bugs  ← (doubts)

Full auto-backward on the `Value` class:

```python
def backward(self):
    self.grad = 1.0
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)               # <-- must be self, NOT a hardcoded node
    for node in reversed(topo):
        node._backward()
```

### Bug A: hardcoded root (`build_topo(o)`)
I had `build_topo(o)` — it always traversed the leftover global `o` instead of the
node I called `.backward()` on. Fix: `build_topo(self)`. Symptom was `a.grad` staying
`0` for `b = a+a; b.backward()` because the wrong graph was walked.

### Bug B: gradient accumulation (`+=` not `=`)
For `b = a + a`, `a` feeds the `+` node twice. The `_backward` uses **both** `self`
and `other`, which are the *same object* `a`:

```python
def _backward():
    self.grad  += out.grad   # a.grad: 0 -> 1
    other.grad += out.grad   # a.grad: 1 -> 2
```

- With `+=` (accumulate): `a.grad = 2` ✓  (correct: `b = 2a`, `db/da = 2`)
- With `=` (overwrite): would wrongly give `1`.

**Rule:** when a value is used in multiple places, its gradient is the **sum** of
gradients from each use (multivariable chain rule). Always `+=`, and remember to
**zero grads** before each new `backward()` (see §12).

---

## 6. Extending `Value`: sub, neg, pow, div, radd/rmul  ← (doubts)

### `__sub__` (correct pattern)
`out = self - other`; local derivs are `+1` and `-1`:

```python
def __sub__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data - other.data, (self, other), '-')
    def _backward():
        self.grad  += out.grad
        other.grad += -out.grad
    out._backward = _backward
    return out
```

Karpathy's shortcut reuses existing ops instead of new backward logic:

```python
def __neg__(self):        return self * -1
def __sub__(self, other): return self + (-other)
```

### `__pow__` (the bug I hit)
Exponent is a **constant**, not a graph node. WRONG version stored it as a child and
used `other.data`:

```python
# BUG:
out = Value(self.data**other, (self, other), '**')   # int in _prev  -> build_topo crash
self.grad += out.grad * other.data * self.data**(other.data-1)  # int has no .data
```

FIX — don't add the exponent as a child; use `other` directly:

```python
def __pow__(self, other):
    assert isinstance(other, (int, float)), "only int/float powers"
    out = Value(self.data**other, (self,), f'**{other}')
    def _backward():
        self.grad += out.grad * other * self.data**(other-1)   # k * x^(k-1)
    out._backward = _backward
    return out
```

Symptom of the bug: `AttributeError: 'int' object has no attribute '_prev'` during
`backward()`, triggered via `__truediv__` (which does `self * other**-1`).

### Reflected operators — needed so numbers can be on the left
Python calls these when the `Value` is on the **right** of the operator:

```python
def __radd__(self, other): return self + other   # other + self
def __rmul__(self, other): return self * other   # other * self
def __rsub__(self, other): return self*-1 + other
```

**Why `__radd__` is mandatory:** `sum()` starts from the int `0` and does
`0 + Value` first. Without `__radd__`, that raises
`TypeError: unsupported operand type(s) for +: 'int' and 'Value'`. Karpathy hits this
same error on the loss `sum(...)` and adds `__radd__` — it's in his class too.

| Method | Handles |
|---|---|
| `__add__` / `__radd__` | `Value + x` / `x + Value` |
| `__mul__` / `__rmul__` | `Value * x` / `x * Value` |
| `__truediv__` | `self / other` = `self * other**-1` |

---

## 7. PyTorch: leaf vs non-leaf tensors  ← (doubt)

In micrograd every node keeps `.grad`. PyTorch, by default, only keeps `.grad` on
**leaf** tensors and discards intermediates (to save memory).

- **Leaf**: created directly, e.g. `torch.tensor(2.0, requires_grad=True)` (your
  weights/inputs). `.grad` is populated after `backward()`.
- **Non-leaf**: produced by an op, e.g. `n = x1*w1 + b`, `o = tanh(n)`. `.grad` comes
  back `None` with a warning.

The intermediate grads *are* computed during backward (needed by the chain rule) —
they're just **freed** afterward because you normally only update the leaves
(parameters).

To keep one: call `.retain_grad()` before `backward()`:

```python
n.retain_grad(); o.retain_grad()
o.backward()
print(n.grad)   # now populated
```

| | micrograd | PyTorch default |
|---|---|---|
| leaf `.grad` | kept | kept |
| intermediate `.grad` | kept | computed then discarded |
| keep intermediate | automatic | `x.retain_grad()` |

---

## 8. The `Neuron` class: inputs vs weights  ← (doubt)

```python
class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]  # one weight per input
        self.b = Value(random.uniform(-1,1))                        # one bias total
    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        return act.tanh()
```

- **`nin`** = number of inputs the neuron takes.
- **`self.w`** = a **list** of `nin` weights (`self.w[0]`=w1, `self.w[1]`=w2, ...).
  Started **random** because the net hasn't learned yet — training tunes them.
- **`self.b`** = a **single** bias (no list). Baseline firing tendency.
- Weights are `plural` (a list, one per input); bias is `singular`.

`Neuron(2)` → `self.w = [Value, Value]`, `self.b = Value`. Same computation as the
hand-written `tanh(x1*w1 + x2*w2 + b)`, generalized to any `nin`.

`__call__` lets you call the object like a function: `n(x)` runs `n.__call__(x)`,
where `x` is the list of input values.

---

## 9. `zip` structure  ← (doubt)

`zip` pairs items from lists by position (like a zipper):

```python
list(zip(['w1','w2'], [2.0, 3.0]))   # [('w1', 2.0), ('w2', 3.0)]
```

In the neuron, `for wi, xi in zip(self.w, x)` unpacks one pair per iteration:
- iter 1: `wi=w1, xi=2.0` → `w1*2.0`
- iter 2: `wi=w2, xi=3.0` → `w2*3.0`

This is what guarantees each weight multiplies its matching input. Notes:
- it's a lazy generator (wrap in `list(...)` to view),
- it stops at the **shorter** list (so `len(x)` should equal `nin`).

---

## 10. Don't touch `.data` in the forward pass  ← (doubt/bug)

Two versions of `__call__` I wrote were numerically right but **broke the graph**:

```python
# BUG 1: pulling out .data does plain float math -> no graph built
sum += w.data * x
out  = Value(sum).tanh()   # fresh leaf, no _prev -> backward can't reach weights

# BUG 2: re-wrapping an already-Value in Value(...)
out = Value(out).tanh()    # out is already a Value -> nests Value in .data, severs graph
```

**Rule of thumb:** in the forward pass, **never touch `.data`** and never re-wrap a
`Value`. Keep everything as `Value` objects so `_prev`/`_backward` get recorded.
`.data` is only for reading the final number (printing), never for computing.

Correct:

```python
def __call__(self, x):
    out = self.b
    for wi, xi in zip(self.w, x):
        out += wi * xi     # Value ops build the graph
    out = out.tanh()       # call tanh on the Value directly
    return out
```

Also: don't reuse loop var names (`for w, x in zip(self.w, x)` clobbers input `x`);
use `wi, xi`.

---

## 11. `Layer` and `MLP`  ← (doubt / visualization)

Nesting: `MLP` = list of `Layer`s; `Layer` = list of `Neuron`s; `Neuron` = weights+bias.

### Layer = a column of independent neurons, all fed the same inputs
```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]   # nout neurons
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs          # unwrap single output
```
- `nin` = inputs per neuron; `nout` = number of neurons = number of outputs.
- fully connected: every input feeds every neuron; each connection has its own weight.
- `outs[0] if len==1 else outs`: return a bare `Value` for a 1-neuron layer, else a list.

### MLP = stack of layers, output of one feeds the next
```python
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts                                  # e.g. [3,4,4,1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)                                    # reassign: data flows fwd
        return x
```
For `MLP(3, [4,4,1])`:
- `sz = [3,4,4,1]` → `Layer(3,4)`, `Layer(4,4)`, `Layer(4,1)`.
- each layer's `nout` = next layer's `nin` (guaranteed by consecutive `sz` pairs).
- forward pass: `[3 vals] -> [4] -> [4] -> single Value`.
- params: 4×(3+1) + 4×(4+1) + 1×(4+1) = 16 + 20 + 5 = **41** learnable Values.

---

## 12. Training: loss, gradient descent, the loop  ← (doubts)

### Loss (mean/sum of squared errors)
```python
loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
```
- single `Value`; `0` only when every prediction matches its target.
- squaring makes errors positive and punishes big misses more.
- needs `__radd__` (see §6) because `sum` starts at int `0`.

### Predictions differ from Andrej's — that's fine
Weights are `random.uniform(-1,1)` with no seed, so untrained predictions differ
every run and won't match the video. What matters is the loss going **down** after
training. Optional: `random.seed(1337)` for reproducible *own* runs.

### Gradient descent scheme
After `loss.backward()`, each param has `p.grad = ∂loss/∂p`. Update rule:

\[ p \leftarrow p - \eta \cdot p.\text{grad} \]

```python
for p in m.parameters():
    p.data += -learning_rate * p.grad
```

- gradient points **uphill** (toward higher loss) → step the **opposite** way (the
  minus sign) to go downhill. This is "descent".
- **learning rate `η` is positive** — it's the step size. It is NOT a "negative
  learning rate"; the descent comes from the minus sign. A literally negative `η`
  would be gradient *ascent* (loss goes up).
- too small `η` → slow; too big → overshoot/diverge.

Per-parameter this self-corrects:
- `grad>0` → `-η·grad<0` → `p` decreases; `grad<0` → `p` increases. Either way loss ↓.

### The full loop (+ the zero-grad gotcha)
```python
for step in range(100):
    # forward
    ypred = [m(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    # zero grads FIRST (because _backward uses +=)
    for p in m.parameters():
        p.grad = 0.0
    # backward
    loss.backward()
    # update
    for p in m.parameters():
        p.data += -0.05 * p.grad
    print(step, loss.data)
```

**Critical gotcha:** zero the grads before each `backward()`. Since `_backward`
accumulates (`+=`), skipping the reset makes gradients from previous steps pile up →
wrong, exploding updates. Classic micrograd bug.

---

## 13. Inspecting weights vs inputs  ← (doubt)

- **Weights** are stored `Value`s → readable: `m.layers[0].neurons[0].w[0].grad`.
- **Inputs** in `xs` are plain floats → when used in `w*x`, `__mul__` wraps the float
  in a **throwaway `Value`** you have no handle to, so their `.grad` isn't accessible.

To see input grads, wrap inputs as `Value` yourself:

```python
xs = [[Value(v) for v in row] for row in raw_xs]
# ... loss.backward()
print(xs[0][0].data, xs[0][0].grad)
```

Why micrograd leaves inputs as floats: during training you only update
**parameters**; the gradient w.r.t. inputs isn't needed (it matters for things like
adversarial examples, not normal training).

---

## Quick reference / cheat sheet

- **method == function**: `n.tanh()` is `tanh(n)`; left-of-dot is `self`.
- **closures**: each `_backward` captures its `self`/`out`, so it can write parents' grads.
- **topo sort**: post-order DFS lists inputs→output; run `reversed(topo)` for backprop.
- **`_prev`**: empty for leaves; auto-filled by operators for computed nodes.
- **always `+=`** in `_backward` (multi-use = sum of grads); **`build_topo(self)`** not a fixed node.
- **`__pow__`**: exponent is a constant — not a child, use `other` (not `other.data`).
- **`__radd__`/`__rmul__`**: needed for number-on-the-left; `sum()` requires `__radd__`.
- **PyTorch**: only leaves keep `.grad`; use `.retain_grad()` for intermediates.
- **Neuron** = weights(list)+bias(single) → `tanh(Σ wᵢxᵢ+b)`; **Layer** = list of neurons; **MLP** = list of layers.
- **forward pass**: never touch `.data`, never re-wrap a `Value`.
- **gradient descent**: `p.data += -lr * p.grad`; lr positive, minus sign = downhill.
- **training loop**: forward → zero grads → backward → update; DON'T forget to zero grads.
