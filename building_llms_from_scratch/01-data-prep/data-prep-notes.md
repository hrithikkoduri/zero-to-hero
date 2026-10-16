# Data Prep Notes — Chapter 2

Notes from Sebastian Raschka's *Build a Large Language Model (From Scratch)*,
chapter 2, worked in `data_prep.ipynb`. The text is Edith Wharton's short story
*The Verdict* (`the-verdict.txt`). Same habit as the nanoGPT notes: the numbers
below are the ones this notebook actually printed, and the confusing spots are
marked ← (doubt).

The chapter turns a string into the tensor a transformer can train on.
Characters become token ids, ids become rows of an embedding table, and a
second table adds the position. Nothing here is attention yet. The output of
the chapter is one batch of input embeddings with shape `[8, 4, 256]`.

---

## 1. The text

```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
len(raw_text)          # 20479
raw_text[:100]
# "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no g"
```

20,479 characters, one string. No batch, no token ids yet.

---

## 2. Splitting into tokens

```python
def preprocess_text(text):
    result = re.findall(r"\w+|[^\w\s]", text)
    result = [item.strip() for item in result if item.strip()]
    return result
```

`\w+` takes a run of letters, digits, or `_`. `[^\w\s]` takes one character
that is neither a word character nor whitespace, so punctuation stays as its
own token instead of being deleted.

```python
preprocess_text("Hello, world! This is a test.")
# ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
```

On the story:

```python
preprocessed_text = preprocess_text(raw_text)
len(preprocessed_text)     # 4827
```

20,479 characters become 4,827 tokens. The opening is

```text
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', ...]
```

`--` in the story is two `-` tokens, because `[^\w\s]` matches one character
at a time. `It's` becomes three tokens: `It`, `'`, `s`. That split is what
makes the decode look wrong later.

---

## 3. The vocabulary

```python
all_words = sorted(set(preprocessed_text))
vocab_size = len(all_words)          # 1148
vocab = {token: integer for integer, token in enumerate(all_words)}
```

`sorted` puts punctuation first, then capital words, then lowercase. The id is
the position in that sorted list, so the mapping is fixed once the vocab is
built.

| id | token |
|---|---|
| 0 | `!` |
| 2 | `'` |
| 5 | `,` |
| 7 | `.` |
| 38 | `Gisburn` |
| 44 | `HAD` |

1,148 is the number of distinct tokens in this one story. A word that never
appears, such as `Hello` or `world`, has no row. Looking it up raises
`KeyError`. That is the hole version 2 fills.

---

## 4. SimpleTokenizerV1  ← (doubt)

`encode` turns a string into ids. `decode` turns ids back into a string.

The first version of `encode` ignored its argument and tokenized `self.text`,
which was the entire story. `encode("Hello, world!")` therefore returned
4,827 ids, the whole book. The first ones were the opening line:

| id | token |
|---|---|
| 53 | I |
| 44 | HAD |
| 163 | always |
| 1021 | thought |

`tokenize` has to receive the string you want to encode:

```python
def encode(self, text):
    tokenized_text = self.tokenize(text)
    return [self.string_to_ids[token] for token in tokenized_text]
```

After that fix, a sentence from the story encodes:

```python
tokenizer.encode("It's the last he painted")
# [56, 2, 867, 1006, 615, 546, 760]
```

| id | token |
|---|---|
| 56 | It |
| 2 | `'` |
| 867 | s |
| 1006 | the |
| 615 | last |
| 546 | he |
| 760 | painted |

### Why decode prints `It' s`  ← (doubt)

`decode` joins the tokens with spaces, so the string is first

```text
It ' s the last he painted
```

Then this substitution runs:

```python
text = re.sub(r"\s+([^\w\s])", r"\1", text)
```

`\s+([^\w\s])` is "spaces sitting immediately before a punctuation mark."
The space before `'` matches, so `It '` becomes `It'`. The space before `s`
does not match, because `s` is a word. The result is `It' s the last he painted`.

A second substitution pulls the letter back onto the apostrophe:

```python
text = re.sub(r"'\s+", "'", text)    # It' s  ->  It's
```

The notebook still has only the first substitution, so `It' s` is the current
output.

`Hello` and `world` are still absent from the 1,148-word vocab, so
`encode("Hello, world!")` raises `KeyError` even after `encode` uses its
argument.

---

## 5. SimpleTokenizerV2: unknown words and `<|endoftext|>`

Two extra rows are appended after the sorted vocab, so they get the next ids:

| token | id |
|---|---|
| `<\|unk\|>` | 1148 |
| `<\|endoftext\|>` | 1149 |

Any token that is not in the vocab is replaced with `<|unk|>` before the
lookup.

### Why `<|endoftext|>` disappeared  ← (doubt)

The first regex was still `\w+|[^\w\s]`. `<`, `|`, and `>` are punctuation, so

```text
<|endoftext|>
```

was cut into five pieces: `<`, `|`, `endoftext`, `|`, `>`. None of those five
strings is the vocab entry `<|endoftext|>`, so each became `<|unk|>` (id
1148). That was the run of five `1148`s in the encoded list.

The special token has to be matched as one alternative, and it has to come
first so the single-character pattern never sees it:

```python
re.findall(r"<\|endoftext\|>|<\|unk\|>|\w+|[^\w\s]", text)
```

After that change, the joined sample

```text
Hello, This is a test. <|endoftext|> We are joinging texts.
```

encodes as

```text
[1148, 5, 97, 598, 129, 1148, 7, 1149, 107, 182, 1148, 1148, 7]
```

`1149` is the one `<|endoftext|>`. `Hello`, `test`, `joinging`, and `texts`
are unknown, so they are `1148`. `This`, `is`, `a`, `We`, `are`, the comma,
and the periods are words or punctuation that actually occur in the story.

`decode` still glues some of these together. Joining with spaces and then
deleting the space before `<` (because `<` matches `[^\w\s]`) turns
`<|unk|> <|unk|>` into `<|unk|><|unk|>`. The printed decode is

```text
<|unk|>, This is a<|unk|>.<|endoftext|> We are<|unk|><|unk|>.
```

`<|endoftext|>` is present. The spaces around it were eaten by that same
punctuation rule.

---

## 6. Byte pair encoding

The hand-built vocab is this one story. GPT-2's tokenizer is a fixed
vocabulary of 50,257 ids (`0` through `50256`), trained with byte pair
encoding. A rare word is broken into subwords that are already in the
vocabulary, so there is no `KeyError` and no `<|unk|>` for ordinary text.

```python
tokenizer = tiktoken.get_encoding("gpt2")    # tiktoken 0.14.0
```

```python
text = "Hello, would you like tea? <|endoftext|> In the sunlit terraces of someunknownplace"
tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# [15496, 11, 561, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 5372]
```

`50256` is `<|endoftext|>`. `allowed_special` is required because tiktoken
refuses to encode that marker unless you name it. `decode` of those ids
returns the original string, including the marker.

`someunknownplace` does not need an unknown-token. BPE splits it into pieces
it already knows (`some`, `unknown`, `place` is the idea; the ids above are
the actual pieces).

A made-up word shows the pieces directly:

```python
tokenizer.encode("Akwirw ier")
# [33901, 86, 343, 86, 220, 959]
# Ak      w   ir  w  ' '  ier
```

The story under this tokenizer:

```python
encoded_text = tokenizer.encode(raw_text)
len(encoded_text)          # 5145
encoded_text[:8]
# [40, 367, 2885, 1464, 1807, 3619, 402, 271]
```

| id | piece |
|---|---|
| 40 | I |
| 367 | H |
| 2885 | AD |
| 1464 | always |
| 1807 | thought |
| 3619 | Jack |

`HAD` is two BPE tokens, `H` and `AD`. That is why the story is 5,145 BPE
tokens and only 4,827 word tokens. BPE is a smaller vocabulary of reusable
pieces, and a long word can cost more than one id.

From here on, the notebook uses the GPT-2 tokenizer.

---

## 7. Input and target are the same window, shifted by 1

```python
encoded_sample = encoded_text[50:]
context_size = 4
x = encoded_sample[:context_size]          # [290, 4920, 2241, 287]
y = encoded_sample[1:context_size+1]       # [4920, 2241, 287, 257]
```

Decoded, one position at a time:

```text
and                         ---> established
and established             ---> himself
and established himself     ---> in
and established himself in  ---> a
```

`x` is the context. `y` is that context shifted one token to the right. At
every position the model is asked for the next token. The last target, `a`,
is the token just past the input window. The shift is always 1. It does not
depend on the stride in the next section.

---

## 8. The dataset and the stride  ← (doubt)

```python
for i in range(0, len(token_ids) - max_length, stride):
    input_ids = token_ids[i:i+max_length]
    target_ids = token_ids[i+1:i+max_length+1]
```

`max_length` is the context size. `stride` is how far the next **input**
window moves. The target is always `input` shifted by one, whatever the
stride is.

Opening tokens again: `[40, 367, 2885, 1464, 1807, ...]` = `I H AD always thought ...`

`batch_size = 1`, `context_size = 4`, `stride = 1`:

| call | input | target |
|---|---|---|
| first `next()` | [40, 367, 2885, 1464] | [367, 2885, 1464, 1807] |
| second `next()` | [367, 2885, 1464, 1807] | [2885, 1464, 1807, 3619] |

Stride 1 means the next input starts one token later. The windows overlap.

`context_size = 2`, `stride = 2`:

| i | input | target |
|---|---|---|
| 0 | [40, 367] | [367, 2885] |
| 2 | [2885, 1464] | [1464, 1807] |

The two **inputs** share no tokens. That is what "stride equal to the context
size prevents overlap" means. `[367, 2885]` is not the next input. It is the
target of the first input. The last target token of one pair (`2885`) is the
first input token of the next pair, because the target looks one past the
input. That one-token overlap is the prediction shift, and it stays for any
stride.

`context_size = 8`, `stride = 2` overlaps the inputs on purpose. The second
input starts at token index 2, so its first six tokens are the last six tokens
of the first input.

A batch is several of these pairs stacked. With `batch_size = 8`,
`context_size = 4`, `stride = 4`, one `next()` returns

```text
inputs:  [8, 4]
targets: [8, 4]
```

Eight sequences, each four tokens. The dataloader still holds more batches.
`next()` only pulled the first group of eight. `shuffle=False` keeps them in
story order, so row 0 is the start of the file and row 1 is the next window
of stride 4.

---

## 9. Token embeddings

An embedding layer is a lookup table. Row `i` is the vector for id `i`.

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(6, 3)     # 6 tokens, each a vector of 3
embedding_layer(torch.tensor([1, 2, 3, 4]))
```

Id `1` returns row 1, which under this seed is `[0.9178, 1.5810, 1.3010]`.
Same id, same row, wherever that id sits in the sequence. The table is
`requires_grad=True`, so training can move the rows. It is the same job as
one-hot times a weight matrix, without building the one-hot.

The GPT-2 sized table in the notebook:

```python
token_embedding_layer = torch.nn.Embedding(50257, 256)    # weight shape [50257, 256]
first_batch_token_embeddings = token_embedding_layer(first_batch_inputs)
# first_batch_inputs is [8, 4]
# embeddings are        [8, 4, 256]
```

`50257` because an id can be any GPT-2 token. `256` is the width chosen here
(`embedding_output_dim`), the `C` from the nanoGPT notes. Each of the 8
sequences has 4 tokens, and each token is 256 numbers.

---

## 10. Absolute positional embeddings  ← (doubt)

Self-attention, as built so far, sees a set of vectors. The same token id is
the same row no matter which slot it occupies. Position has to be added in.

Absolute positional encoding: a learned vector for slot `0`, another for slot
`1`, and so on, added to the token vector. Relative positional encoding does
not add a position vector. It changes the attention score using the distance
between two positions.

```python
positional_embedding_layer = torch.nn.Embedding(context_size, embedding_output_dim)
# context_size = 4, embedding_output_dim = 256
# weight shape [4, 256]

positional_embeddings = positional_embedding_layer(torch.arange(context_size))
# arange -> tensor([0, 1, 2, 3])
# positional_embeddings shape [4, 256]
```

The first argument of `nn.Embedding` is how many rows the lookup might need.
The token table uses `50257` because the index is a token id. The position
table uses `context_size` because the index is `0, 1, 2, 3` and a position
index never reaches 50,256. `torch.arange(context_size)` does not read the
batch. Every sequence uses row 0 for its first token, row 1 for its second,
and so on.

### Why the position is not row `1` of the token table  ← (doubt)

Adding the token-table row for the position index collides. Token 35 in slot
1 and token 1 in slot 35 become the same vector, because addition ignores
order:

```text
E[35] + E[1]    token 35 at position 1
E[1]  + E[35]   token 1  at position 35
```

A second table makes `token[35] + position[1]` a different vector from
`token[1] + position[35]`.

### The add covers the whole batch, and only the batch you fetched  ← (doubt)

```python
first_batch_input_embeddings = first_batch_token_embeddings + positional_embeddings
# [8, 4, 256] + [4, 256]  ->  [8, 4, 256]
```

`[4, 256]` broadcasts across the 8 sequences. All 8 rows of this batch get
positions `0, 1, 2, 3`. `first_batch_token_embeddings[0]` is one sequence,
printed only to compare shapes. The add uses the full `[8, 4, 256]` tensor.

`next()` was called once, so only that one batch of 8 was embedded. Later
batches are still sitting in the dataloader. The same `[4, 256]` position
matrix is reused for each of them:

```python
positional_embeddings = positional_embedding_layer(torch.arange(context_size))

for batch_inputs, batch_targets in dataloader:
    token_embeddings = token_embedding_layer(batch_inputs)       # [8, 4, 256]
    input_embeddings = token_embeddings + positional_embeddings   # [8, 4, 256]
```

In the model this addition lives in `forward`. Each training step does it for
whichever batch the dataloader just yielded. There is one position table, not
one table per batch.
