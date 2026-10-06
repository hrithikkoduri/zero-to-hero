# Zero to Hero

Notes and code from working through [Andrej Karpathy's](https://karpathy.ai/) neural-net series: micrograd, makemore parts 1 and 2, and Let's build GPT.

Each folder is one project. The notebook or scripts are the implementation. The markdown files are the write-up of the mechanics, with the shapes and numbers from these runs, and the places that were confusing marked as doubts.

The thread through all three is the same. A small computation has a derivative. That derivative trains a network. The network's job is to predict the next character.

## Contents

- [Micrograd](#micrograd) — derivatives, a `Value` graph, backprop, a tiny MLP
- [Makemore](#makemore) — character-level names, first by counting, then with an MLP
- [nanoGPT](#nanogpt) — the same next-character idea, grown into a transformer
- [How to run](#how-to-run)

## Micrograd

A neural net is a graph of small operations. Micrograd builds that graph by hand. Each number is a `Value` with `.data` and `.grad`. `+`, `*`, and `tanh` record how they were produced. Calling `backward()` walks the graph in reverse and fills every `.grad` with the chain rule. Part 2 turns that into a neuron, a layer, and an MLP, then trains it with gradient descent: subtract learning-rate times gradient.

| | |
|---|---|
| Code | [`micrograd/micrograd.ipynb`](micrograd/micrograd.ipynb) |
| Part 1 | [`micrograd/micrograd-notes-pt-1.md`](micrograd/micrograd-notes-pt-1.md) |
| Part 2 | [`micrograd/micrograd-notes-pt-2.md`](micrograd/micrograd-notes-pt-2.md) |

Part 1 — what a derivative is, and how one loss sends a gradient back through a single neuron.

1. What a derivative actually means
2. The `Value` object and expression graphs
3. What `.grad` actually means
4. The chain rule
5. Chaining through a multiply
6. Modeling a neuron: inputs, weights, bias, tanh
7. Output vs loss

Part 2 — the same idea, automated, then stacked into a network and trained.

1. `n.tanh()` and `tanh(n)`
2. How `_backward` fills a parent's grad
3. Topological sort
4. Where `_prev` comes from
5. The `backward()` method and two bugs
6. Extending `Value` (sub, neg, pow, div)
7. PyTorch leaf vs non-leaf tensors
8. The `Neuron` class
9. `zip`
10. Don't touch `.data` in the forward pass
11. `Layer` and `MLP`
12. Training: loss and the gradient-descent loop
13. Inspecting weights vs inputs

## Makemore

Makemore generates names, one character at a time. The dataset is [`makemore/names.txt`](makemore/names.txt): 32,033 names, each wrapped with a `.` so the model can learn where a name starts and ends.

Part 1 is a bigram. Count how often each character follows each other character, turn the counts into probabilities, and sample. The same table can be learned instead of counted: one-hot the current character, multiply by a `27 × 27` weight matrix, and train with negative log-likelihood. At the optimum the network matches the counts.

Part 2 looks at the last three characters, not one. Each character is an embedding (a learned vector). The three vectors are concatenated, passed through a tanh hidden layer, and scored over the 27 possible next characters. Training uses minibatches, a learning-rate search, and a train / val / test split.

| | |
|---|---|
| Part 1 code | [`makemore/makemore.ipynb`](makemore/makemore.ipynb) |
| Part 1 notes | [`makemore/makemore-notes-pt-1.md`](makemore/makemore-notes-pt-1.md) |
| Part 2 code | [`makemore/makemore-2.ipynb`](makemore/makemore-2.ipynb) |
| Part 2 notes | [`makemore/makemore-notes-pt-2.md`](makemore/makemore-notes-pt-2.md) |
| Data | [`makemore/names.txt`](makemore/names.txt) |

Part 1

1. Bigrams and the special token
2. Counting into a matrix
3. Counts to probabilities
4. Sampling with `multinomial`
5. Negative log-likelihood
6. The neural-net version: one-hot plus a weight matrix
7. The training loop
8. Regularization
9. Counting vs the neural net

Part 2

1. The context window
2. The embedding matrix
3. Gluing three embeddings into one row
4. The hidden layer
5. Output layer and loss
6. Minibatch training
7. The `view` shape bug
8. Finding a learning rate
9. Train / val / test
10. Visualizing embeddings
11. Sampling

## nanoGPT

Same task as makemore, on a different dataset: Tiny Shakespeare, [`nanogpt/input.txt`](nanogpt/input.txt), 1,115,394 characters, 65 unique. A character id indexes a row of learned numbers. Those numbers become scores for the next character. Cross-entropy trains them. `generate` takes the last position's scores, softmaxes them, and draws the next character with `multinomial`.

The notebook and `bigram.py` start with a pure bigram: the embedding itself is a `65 × 65` table of next-character scores. Each later script adds one piece of the transformer. `bigram_v6.py` is the model from the end of the video: token and position embeddings, six blocks, multi-head self-attention, a feed-forward layer, residual connections, layer norm, and dropout. Width is 384. It is about 10 million parameters.

| File | What it adds |
|---|---|
| [`bigram.py`](nanogpt/bigram.py) | Bigram table. One row of 65 scores per character. |
| [`bigram_v2.py`](nanogpt/bigram_v2.py) | A 32-d token embedding, a position embedding, and a linear head back to 65 logits. |
| [`bigram_v3.py`](nanogpt/bigram_v3.py) | One self-attention head. Future positions are masked. |
| [`bigram_v4.py`](nanogpt/bigram_v4.py) | Four heads, each of size 8, concatenated back to 32. |
| [`bigram_v5.py`](nanogpt/bigram_v5.py) | A feed-forward (`32 → 128 → ReLU → 32`) and a stack of blocks. |
| [`bigram_v6.py`](nanogpt/bigram_v6.py) | Residuals, layer norm, dropout, and the scaled-up sizes. |

| | |
|---|---|
| Notebook | [`nanogpt/nanogpt.ipynb`](nanogpt/nanogpt.ipynb) |
| Part 1 notes | [`nanogpt/nanogpt-notes-pt-1.md`](nanogpt/nanogpt-notes-pt-1.md) |
| Part 2 notes | [`nanogpt/nanogpt-notes-pt-2.md`](nanogpt/nanogpt-notes-pt-2.md) |

Part 1 — the bigram, and the bag-of-words average that attention grows out of.

1. The data, the 90/10 split, encode and decode
2. A batch: `ix`, `x`, `y`
3. `B`, `T`, `C`
4. What `m` is, and why `m(x)` runs `forward`
5. Why a row of 65 numbers is scores for the next character
6. Cross-entropy and the softmax inside it
7. The gradient: `p − 1` on the true column
8. The training loop, and how it maps to micrograd
9. `estimate_loss`
10. `generate`
11. Bag of words: the running mean

Part 2 — from that average to the full model.

1. The average as a matrix multiply
2. One attention head: query, key, value
3. Position embeddings
4. Multi-head attention
5. The feed-forward
6. A block: residual, layer norm, then a stack
7. Dropout
8. The forward pass of the finished model
9. The scaled-up run, and where it computes
10. What this model still is

## How to run

From this folder, with PyTorch installed:

```bash
python nanogpt/bigram.py
python nanogpt/bigram_v6.py
```

The scripts find `input.txt` next to themselves, so either of those commands works from the repo root. `bigram.py` through `bigram_v5.py` are the small teaching models (`n_embd = 32` after v1) and finish on CPU. `bigram_v6.py` uses the Apple GPU when Metal is available (`device = 'mps'`), otherwise CPU.

The notebooks read their data file from the working directory, and also from `makemore/names.txt` or `nanogpt/input.txt` if that is where the file is. Open `makemore.ipynb` and `makemore-2.ipynb` for parts 1 and 2. Open `nanogpt.ipynb` for the lecture walkthrough; the `.py` files are the same model taken out of the notebook one version at a time.
