# Micrograd Notes — Part 1

Notes from working through Andrej Karpathy's micrograd video, organized around
the concepts and the specific doubts I worked through.

---

## 1. What a derivative actually means

A derivative answers: **"if I nudge the input by a tiny amount `h`, how much does the output change, per unit of `h`?"**

For `f(x) = 2x² - 3x + 4`, we estimate the slope numerically:

```python
h = 0.0000001
x = 3.0
(f(x + h) - f(x)) / h   # ≈ 9.0   -> slope at x = 3
```

- At `x = 3.0` the slope is `≈ 9` (positive → increasing `x` increases `f`).
- At `x = -3.0` the slope is `≈ -15` (negative → increasing `x` decreases `f`).
- At `x = 2/3` the slope is `≈ 0` (flat spot / minimum of the parabola).

**Key idea:** the sign tells you the *direction* of change, the magnitude tells you *how sensitive* the output is to that input. This "nudge and measure" idea is the foundation of everything else.

### Same idea with multiple inputs

For `d = a*b + c` we can nudge one variable at a time:

```python
h = 0.0001
a = 2.0; b = -3.0; c = 10.0
d1 = a*b + c
a += h
d2 = a*b + c
(d2 - d1) / h    # ≈ -3.0  -> slope of d w.r.t. a  (which equals b)
```

- Nudging `a` → slope `≈ b = -3.0`
- Nudging `b` → slope `≈ a = 2.0`
- Nudging `c` → slope `≈ 1.0`

---

## 2. The `Value` object and expression graphs

Instead of plain numbers, we wrap values in a `Value` object that remembers **how it was made**. This lets us build a graph of operations.

```python
class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)   # the inputs that produced this value
        self._op = _op                # the operation that produced it ('+', '*', ...)
        self.label = label
        self.grad = 0.0               # dL/d(this value); starts at 0

    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), '*')
```

Building an expression:

```python
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label = 'e'      # e = -6.0
d = e + c; d.label = 'd'    # d =  4.0
f = Value(-2.0, label='f')
L = d*f; L.label = 'L'      # L = -8.0
```

Each `Value` stores `_prev` (its inputs) and `_op` (how it was combined), so the
whole computation forms a **DAG (directed acyclic graph)** we can walk and visualize.

### Visualizing the graph (`draw_dot`)

We use `graphviz` to render the graph. Each node shows its label, data, and grad.

```python
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # left -> right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid,
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                 shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot

draw_dot(L)
```

**Gotcha I hit:** `graphviz` (the Python package) needs the *system* Graphviz
binaries too. On macOS: `brew install graphviz`. If the kernel was already
running, either restart it or add:

```python
import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
```

---

## 3. What `.grad` actually means  ← (big doubt I cleared up)

**The one rule:** for *any* node `x` in the graph,

```
x.grad = ∂L / ∂x
```

where `L` is the single **final output** of the graph.

Two things to keep straight for every node:
- **`.data`** = the node's actual value.
- **`.grad`** = how much the *final output `L`* changes if you nudge *this node*.

The confusing part I had to internalize:
- `.grad` is **NOT** "the derivative of this node with respect to its inputs."
- `.grad` is **ALWAYS** "the derivative of the final node `L` with respect to this node."
- `L` is the fixed reference point for *everything*.

### Worked examples on `L = d*f` (d=4.0, f=-2.0, L=-8.0)

**`L.grad = ∂L/∂L = 1.0`**
"If I wiggle `L` by `h`, how much does `L` change?" By `h`. Ratio = 1.
→ This is always where backprop *starts*.

**`d.grad = ∂L/∂d = f = -2.0`**
`L = d*f`, so nudging `d` by `h` changes `L` by `f*h = -2h`.

**`f.grad = ∂L/∂f = d = 4.0`**
Nudging `f` by `h` changes `L` by `d*h = 4h`.

> For a multiply node, each input's local derivative is the *other* input's value.

Sensitivity table (same target `L`, different knob wiggled):

| Node | Meaning of its `.grad` | Value |
|------|------------------------|-------|
| `L.grad` | wiggle `L` → effect on `L` | `1.0` |
| `d.grad` | wiggle `d` → effect on `L` | `-2.0` |
| `f.grad` | wiggle `f` → effect on `L` | `4.0` |
| `a.grad` | wiggle `a` → effect on `L` | `6.0` |

`d.grad` isn't `1.0` because `d` is *not* the output — it sits upstream, and its
influence on `L` gets scaled by the multiply with `f`.

---

## 4. The chain rule: `c.grad` and `e.grad`  ← (doubt I cleared up)

`c` and `e` don't appear in `L`'s formula directly — they only affect `L`
*through* `d` (because `d = e + c`). So we chain two hops.

**`c.grad = ∂L/∂c`:**

1. Local: how does `c` affect `d`? `d = e + c` → `∂d/∂c = 1` (addition passes through).
2. Downstream: how does `d` affect `L`? `d.grad = -2.0`.
3. **Multiply them:** `∂L/∂c = ∂L/∂d · ∂d/∂c = -2.0 · 1 = -2.0`.

**`e.grad = ∂L/∂e`:** same logic → `-2.0 · 1 = -2.0`.

**Key takeaways:**
- **Addition distributes the gradient.** Local derivative of `+` is `1` for every
  input, so an addition node copies its output's grad unchanged to all inputs.
  That's why `c.grad == e.grad == d.grad == -2.0`.
- **General backprop pattern:**
  ```
  this_node.grad = (local derivative w.r.t. this input) × (downstream node's .grad)
  ```
- This is why backprop goes **backwards**: you need `d.grad` *before* you can get
  `c.grad` and `e.grad`.

---

## 5. Chaining through a multiply: the `a` example  ← (doubt I confirmed)

Question I checked: *"nudging `a` by `h` increases `e` by `-3h` and `L` by `6h` — right?"* **Yes.**

- `e = a*b`, `b = -3.0` → `∂e/∂a = b = -3.0`, so `e` changes by `-3h`. ✓
- Chain to `L`: `∂L/∂a = ∂L/∂e · ∂e/∂a = (-2.0)·(-3.0) = 6.0`, so `L` changes by `6h`. ✓

Intuition in words:
- Wiggle `a` by `h` → `e` moves by `-3h` (rate `-3`, from `b`).
- That change in `e` → `L` moves by `-2 ×` (change in `e`) = `6h` (rate `-2`, from `e.grad`).
- The two rates **multiply**: `-3 × -2 = 6`. The negatives cancel — that's why
  increasing `a` *increases* `L` even though `b` is negative.

Numeric check:

```python
h = 0.0001
L1 = (2.0 * -3.0 + 10.0) * -2.0        # a = 2.0
L2 = ((2.0 + h) * -3.0 + 10.0) * -2.0  # a = 2.0 + h
(L2 - L1) / h    # ≈ 6.0
```

### Full backward pass for `L = (a*b + c) * f`

```
L.grad = 1.0
  d.grad = f.data * L.grad = -2.0        (through the * node)
    c.grad = 1 * d.grad = -2.0           (through the + node)
    e.grad = 1 * d.grad = -2.0           (through the + node)
      a.grad = b.data * e.grad =  6.0    (through the * node)
      b.grad = a.data * e.grad = -4.0    (through the * node)
```

---

## 6. Modeling a neuron: inputs vs weights, bias, and tanh  ← (doubt I cleared up)

A single neuron:  `output = f( Σ wᵢxᵢ + b )`.

| Symbol | What it is | Role |
|---|---|---|
| `xᵢ` | **inputs** | the data (fixed, given to us) |
| `wᵢ` | **weights** | how much the neuron *cares* about each input (learned) |
| `wᵢ·xᵢ` | scaled input | strength of that connection |
| `Σ wᵢ·xᵢ + b` | weighted sum + **bias** | everything combined into one number |
| `f` (tanh) | **activation function** | squashes the sum + adds non-linearity |

### Inputs vs weights (the intuition)

- **Inputs (`x`) are the data. Weights (`w`) are the knobs.**
- Inputs are given — you don't control them (pixels, previous layer's outputs).
- Weights are what the network **learns**; they control how strongly each input matters:
  - large positive `w` → input matters a lot, pushes neuron *on*
  - large negative `w` → input matters a lot, pushes neuron *off*
  - `w ≈ 0` → ignore this input
- **Bias `b`** = the neuron's baseline eagerness to fire, regardless of inputs.

Analogy: deciding whether to go outside. Inputs = facts (temperature, raining,
weekday). Weights = how much *you* care about each fact. Someone who hates rain
has a big negative weight on "is it raining." Facts are the same for everyone;
weights make *this* neuron behave the way it does.

### Why `tanh` (the activation function)

The raw sum `Σ wᵢxᵢ + b` can be any number. `tanh` squashes it into `(-1, 1)`:

```
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

- big positive sum → output near `+1` (fully fires)
- big negative sum → output near `-1` (stays off)
- sum near `0` → output near `0`, curve is steepest (most sensitive)

Two reasons it's needed:
1. Bounds the output (like a real neuron that can't fire infinitely hard).
2. Adds **non-linearity** — without it, stacking neurons collapses into a single
   linear function and depth buys you nothing. The "kink" lets the network learn
   curvy decision boundaries.

### My neuron in the notebook

```python
# inputs
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights
w1 = Value(3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias
b = Value(6.7, label='b')

# forward pass
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

draw_dot(o)
```

This is the same shape as the `a*b + c` graph, just bigger: `*` nodes for the
`xᵢwᵢ`, `+` nodes to sum them with `b`, and one final `tanh` node → output `o`.

---

## 7. Output vs Loss  ← (doubt I cleared up)

**The output of a neuron is NOT the loss.** They're different things.

- **`o` — the neuron's output (the prediction).** `o = tanh(n)` is what the
  neuron *says*. Just the answer for this input.
- **`L` — the loss (how *wrong* the prediction is).** Computed *afterwards* by
  comparing `o` to the correct target/label, e.g.:

  ```
  L = (o - y_target)²
  ```

  If it should have output `1.0` but produced `0.7`, loss = `(0.7 - 1.0)² = 0.09`.

Pipeline:

```
inputs + weights ─► n ─► o = tanh(n) ─► compare with target ─► L (loss)
                        (prediction)                          (how wrong)
```

**Why differentiate the loss, not the output?** Training = make the network less
wrong = make the *loss* small. So backprop starts from `L`:

- `w1.grad = ∂L/∂w1` = "how does my *mistake* change if I nudge this weight?"
- Then nudge each weight in the direction that *reduces* `L`.

If you backpropped from `o`, you'd learn how to change the prediction but have no
notion of "correct" to aim at. The loss is what injects the target into the math.

> Note: in the earlier toy graph the final node was labeled `l`/`L` and we
> differentiated it directly — that was just a demo of backprop, with no
> prediction-vs-target story. In a real neuron, the final node is the **loss**.

**Where I am now:** my current cell stops at `o` — I've built the **forward pass
of one neuron**. Loss enters later, when we:
1. run the neuron on inputs to get predictions,
2. compare to targets → single loss `L`,
3. call `L.backward()` to get every weight's `.grad`,
4. nudge weights to shrink `L`.

---

## Quick reference / cheat sheet

- **derivative** = sensitivity of output to a tiny nudge of an input.
- **`.data`** = the value; **`.grad`** = `∂L/∂(this node)`, always w.r.t. the final node `L`.
- **backprop rule:** `input.grad = (local derivative) × (downstream .grad)`.
- **`+` node:** passes gradient through unchanged to every input (local deriv = 1).
- **`*` node:** each input's local derivative = the *other* input's value.
- **backprop runs backwards** because each node needs its downstream node's grad first.
- **inputs = data (fixed); weights = learnable knobs; bias = baseline firing tendency.**
- **tanh = activation:** squashes to `(-1, 1)` and adds the non-linearity.
- **output `o` ≠ loss `L`:** loss compares the prediction to the target; backprop starts from the loss.
