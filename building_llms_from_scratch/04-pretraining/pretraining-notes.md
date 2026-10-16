# Pretraining Notes — Chapter 5

Notes from Sebastian Raschka's *Build a Large Language Model (From Scratch)*,
chapter 5, worked in `pretraining.ipynb` and
`pretraining_load_our_model.ipynb`. The text is still *The Verdict*
(`the-verdict.txt`, 20,479 characters, 5,145 GPT-2 tokens). Same habit as
the earlier notes: the numbers are the ones this notebook printed, and the
confusing spots are marked ← (doubt).

The chapter trains the chapter 4 GPT, then loads the official GPT-2 124M
weights into that same class. Training on this short story memorizes the
story. The official weights are what make the model write normal English.

---

## 1. One loss for every position

The two example sentences are a batch of shape `[2, 3]`.

```text
input   "every effort moves"     [16833, 3626, 6100]
        "I really like"          [40, 1107, 588]
target  " effort moves you"      [3626, 6100, 345]
        " really like chocolate" [1107, 588, 11311]
```

`model(input)` returns logits `[2, 3, 50257]`. Softmax on the last axis
makes `probs` the same shape: one distribution per position. ← (doubt) An
argmax of that tensor is not one next token for the batch. `dim=-1` only
collapses the vocabulary, so

```python
torch.argmax(probs, dim=-1, keepdim=True)   # [2, 3, 1]
```

is one chosen id at each of the 2 × 3 positions. The single next token used
in generation is the last position only: `probs[:, -1, :]`.

The probability of the correct target at each position is one cell of that
distribution:

```python
probs[0, [0, 1, 2], target_token_ids[0]]
# probs[0, 0, 3626], probs[0, 1, 6100], probs[0, 2, 345]
```

The first index is the sentence. Both lines have to use their own sentence.
`probs[0, ..., target_token_ids[1]]` looks up the second sentence's answers
inside the first sentence's distribution.

Cross-entropy is the negative log of those correct-token probabilities,
averaged over the batch and the positions. At random initialization the
printed loss is about 10.98. A uniform guess over 50,257 tokens is
`ln(50257) ≈ 10.82`, so 10.98 means the model is still guessing.
Perplexity is `exp(loss)`. A loss near 10.8 is a perplexity near 50,000,
which is the vocabulary size.

---

## 2. Train and validation split

← (doubt) `train_ratio = 0.9` produced a training slice of 4,630 and a
validation slice of 15,849. The cut was `int(5145 * 0.9) = 4630`, which is
90% of the tokens, but it was used to slice `text_data`, the 20,479-character
string. The first 4,630 characters are about 23% of the story.

Cut the string with its own length:

```python
split_index = int(len(text_data) * 0.9)   # 18431
train_data = text_data[:split_index]      # 18431 characters
val_data = text_data[split_index:]        # 2048 characters
```

---

## 3. The dataset windows

```python
for i in range(0, len(token_ids) - context_length, stride):
    input_ids = token_ids[i:i + context_length]
    target_ids = token_ids[i + 1:i + context_length + 1]
```

`range` takes start, stop, step. `i` is the start of each window, and
`stride` is how far the next start jumps. The stop is
`len(token_ids) - context_length` and is not included, because the target
needs one token past the end of the input.

With context 4 and stride 2 on 10 tokens, `i` is `0, 2, 4`. Each window
shares `4 - 2 = 2` tokens with the one before it. The target is always the
input shifted by one. Stride does not change that shift.

The training loader uses context 256, stride 256, batch size 2. A batch is
`[2, 256]`. Stride equal to the context places the windows end to end.

---

## 4. The training loop

The diagram in `training-loop.png` is one epoch, then every batch inside it.

```python
optimizer.zero_grad()
logits = model(input_batch)
loss = nn.functional.cross_entropy(
    logits.view(-1, logits.shape[-1]),
    target_batch.view(-1),
)
loss.backward()
optimizer.step()
```

`loss.backward()` fills each weight's `.grad`. It does not change the
weight. `optimizer.step()` is the AdamW update that writes the new weight.
`zero_grad()` runs first so this batch's gradients replace the previous
batch's.

← (doubt) The first version of `train_simple_model` returned inside the
batch loop, so training stopped after batch 1. The `return` belongs after
both loops.

← (doubt) `tokens_seen += input_batch.shape[1]` adds 256. The batch is
`[2, 256]`, so the step saw `2 * 256 = 512` tokens. The count is
`input_batch.numel()`.

`model.train()` turns dropout on. `model.eval()` turns it off for the loss
check and for the sample. After the sample, `model.train()` runs again so
the next epoch still drops activations. The sample itself is once per epoch,
after that epoch's batches. Inside the sample, only the last
`context_size` tokens are passed in, because the position table has that
many rows.

---

## 5. Why the first runs did not learn

A run with `learning_rate = 1e-5` printed a training loss between 10.97 and
11.00 for all 10 epochs, and the validation loss was 10.9855 on every
print. The generated paragraph was identical every epoch. The update ran.
The step was too small to move a 12-layer, 768-wide model off a random
guess. The validation number stayed still because that loader is one fixed
batch. The training number jittered because `eval_iter=5` averages five
shuffled batches.

`1e-4` is `0.0001`. `4e-4` is `0.0004`, four times larger. `1e-5` is
`0.00001`. The rate that moves this chapter's loss is `4e-4` or `5e-4`,
with AdamW and `weight_decay=0.1`. The model has to be created, moved to
the device, and only then attached to the optimizer. Building the optimizer
before `model.to(device)` makes `step()` update a copy the forward pass
does not use.

With `learning_rate = 0.0005` the training loss fell from about 10 to well
under 1 within 10 epochs, and the sample from `"Every effort moves"`
started reciting *The Verdict*. The validation loss stayed near 6 to 7.
That is overfitting: the same short story was shown for many epochs, so the
training loss can fall to about 0.005 while the validation loss does not.

At a training loss of 0.005 the model gives the memorized next token about
99.5% of the probability. More epochs on the same text do not change the
paragraph. ← (doubt) Reloading the checkpoint and training another 10
epochs left the sample unchanged. The loss at the start of that second run
was already 0.0073, and it ended at 0.0052. Validation stayed near 7.18.

---

## 6. Decoding

Argmax is greedy decoding. It returns the single largest probability, so
1,000 calls on the same `probs` are the same token. In the notebook that
token was `"up"`, 1,000 times.

`torch.multinomial` draws in proportion to the probabilities. `"up"` only
had about 2% of the mass, so in one sample of 1,000 draws the counts were

```text
31  ,"
29  'd
29  Herm
22  up
```

← (doubt) Running it again printed the same counts. `print_sampled_tokens`
starts with `torch.manual_seed(123)`, which restarts the random generator.
Remove that line and the counts move. Argmax does not use the generator, so
it stays `1000 x up`.

Temperature divides the logits before the softmax. Below 1 the largest
logit pulls further ahead. Above 1 the distribution flattens. Top-k keeps
only the largest logits:

```python
masked_logits = torch.where(
    logits < top_logits[0][-1],
    torch.tensor(float("-inf"), device=logits.device),
    logits,
)
```

`top_logits[0][-1]` is the smallest score that still made the cut.
Everything below it becomes `-inf`, and softmax turns `-inf` into
probability 0. A score equal to the cutoff stays, because the test is `<`.

← (doubt) `generate_text` defaulted `device` to `"cpu"` while the model was
on MPS. The embedding lookup then died with `Placeholder storage has not
been allocated on MPS device`. The prompt has to be moved to the same
device as the weights, and the `-inf` tensor has to be created on that
device too.

---

## 7. Saving and loading our own weights

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, "model_and_optimizer.pth")
```

`model_state_dict` is the model: 174 tensors. Twelve of them are the causal
masks, shape `[256, 256]`, which are buffers, not trained weights. The
other 162 tensors are the parameters. Their elements add up to
**162,469,969**.

`Wq`, `Wk`, and `Wv` are in the file, one set per block, blocks 0 through
11. `qkv_bias` was `False` for the trained model, so those three have a
weight and no bias. `out_proj` has a weight and a bias. Each block also
stores two LayerNorm scale/shift pairs and the feed-forward expand
(`[3072, 768]`) and contract (`[768, 3072]`). Outside the blocks: the token
embedding `[50257, 768]`, the position embedding `[256, 768]`, the final
LayerNorm, and `out_head` weight plus bias.

GELU and dropout have no tensors in this file. GELU is a fixed function.
Dropout only stores the rate 0.1.

`optimizer_state_dict` is AdamW's memory, not part of the network. For
every parameter Adam stores a momentum tensor and a squared-gradient
tensor, 324,940,100 extra numbers, plus `lr` and `weight_decay`. A forward
pass never reads them. Generation loads only the model state:

```python
model = GPTModel(GPT_CONFIG).to(device)
model.load_state_dict(check_point["model_state_dict"])
model.eval()
```

`GPTModel(GPT_CONFIG)` by itself is a new random model. `load_state_dict`
is what replaces those random numbers. The class still has to exist.
A state dict is names and tensors. It has no `forward`.

← (doubt) `model.to("mps")` inside the training function moves the weights
and not Adam's momentum tensors. `optimizer.step()` then fails with
`Expected all tensors to be on the same device, mps:0 and cpu`.
`optimizer.to` does not exist on this PyTorch. Move the model to the device
before `optimizer.load_state_dict`, and move any already-loaded state
tensors by hand:

```python
for state in optimizer.state.values():
    for key, value in state.items():
        if torch.is_tensor(value):
            state[key] = value.to(device)
```

The checkpoint file is about 1.8 GB and is not in the repo.

---

## 8. The official GPT-2 weights

`download_and_load_gpt2(model_size="124M", models_dir="gpt2")` reads
OpenAI's TensorFlow checkpoint. The parameter dict has five keys.

| key | what it is | shape | our name |
|---|---|---|---|
| `wte` | token embedding | `[50257, 768]` | `token_embedding.weight` |
| `wpe` | position embedding | `[1024, 768]` | `positional_embedding.weight` |
| `blocks` | the 12 transformer blocks | list of 12 dicts | `transformer_blocks` |
| `g` | final LayerNorm scale | `[768]` | `final_layer_norm.scale` |
| `b` | final LayerNorm shift | `[768]` | `final_layer_norm.shift` |

`wpe` has 1024 rows. The model that receives these weights has to be built
with `context_length` 1024, not 256. ← (doubt) There is no `lm_head` key.
GPT-2 reuses `wte` as the output projection, and `out_head.weight` has the
same shape `[50257, 768]`, so the assignment is `params["wte"]` with no
transpose. There is no output bias in the checkpoint. `params["b"]` is the
final LayerNorm shift.

Inside one block:

```text
attn.c_attn.w    (768, 2304)    fused Wq, Wk, Wv
attn.c_attn.b    (2304,)        bias for that fused layer
attn.c_proj.w    (768, 768)     attention out_proj
attn.c_proj.b    (768,)
ln_1.g, ln_1.b   (768,)         first LayerNorm scale and shift
mlp.c_fc.w       (768, 3072)    feed-forward expand
mlp.c_fc.b       (3072,)
mlp.c_proj.w     (3072, 768)    feed-forward contract
mlp.c_proj.b     (768,)
ln_2.g, ln_2.b   (768,)         second LayerNorm
```

`c_attn["w"]` is the matrix. `c_attn["b"]` is the bias added after it.
Splitting the last axis into 3 gives three `(768, 768)` matrices. Block 1's
`c_attn` is a different matrix from block 0's. The checkpoint stores a
linear weight as `(in, out)`. `nn.Linear` stores it as `(out, in)`. Both
can be `(768, 768)`, so the shape check in `assign` accepts an
untransposed matrix. The copy uses `.T`.

`assign` does not write into `left`. `left` is only there for the shape
check. `right` is a NumPy array from TensorFlow, not a torch tensor.
`torch.tensor(right)` copies it, and the caller has to store the returned
parameter:

```python
gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params["wte"])
```

← (doubt) A dtype check of `left.dtype != right.dtype` raises even when
both are 32-bit floats, because one is `torch.float32` and the other is
`numpy.float32`.

The feed-forward weights are `mlp`, not `attn`. GELU and dropout still have
nothing to load. ← (doubt) The first loader assigned the feed-forward
weights and both block LayerNorms onto `mha.out_proj`, and it looked up
`gpt.output_head` and `params["lm_head"]`. The layer name is `out_head`.
The feed-forward expand is `ffn.layers[0]`, the contract is `ffn.layers[2]`,
and the two block norms are `ln1` and `ln2`.

← (doubt) `import tensorflow` inside `gpt_download.py` aborted this
environment's TensorFlow 2.20 with `mutex lock failed` and the Jupyter
kernel died. The script's argument is `models_dir`, not `model_dir`.
TensorFlow 2.16.2 imports here. The downloaded checkpoint is about 475 MB
and is not in the repo. `gpt_download.py` is.
