# Model Architecture — Conceptual Notes

> Scope: this document explains **two architectural concepts** you asked about —
> (1) the maths of mean pooling across tokens, and (2) why the head uses a
> per-logit sigmoid instead of a softmax.
>
> ⚠️ **No puzzle spoilers.** This file deliberately does **not** identify the
> non-linear feature F or speculate about its geometric structure. It only
> builds the conceptual scaffolding you need to reason about the model. The
> final section defines what "linearly represented" *means* (you need that to
> even state the puzzle) but stops short of any analysis or answer.

---

## 0. The pipeline at a glance

```
text  ──►  MiniLM encoder (6-layer transformer)  ──►  per-token embeddings  (L × 384)
                                                              │
                                                       mean pooling          ◄── Part 1
                                                              │
                                                        sentence vector  (384)
                                                              │
                              5 Linear layers (384→64→64→64→64→8) with ReLU between
                                                              │
                                                         8 logits  (z₁…z₈)
                                                              │
                                                  per-logit sigmoid           ◄── Part 2
                                                              │
                                            8 independent probabilities (p₁…p₈)
```

The diagram in `model_architecture.png` shows exactly this. Two things are
worth internalising before anything else:

- The encoder is **frozen / pretrained** (`all-MiniLM-L6-v2`). It was *not*
  trained on this 8-feature task. Only the MLP head was trained.
- The "interesting" layer L the puzzle asks about lives **inside the MLP head**
  (post-ReLU of hidden layer 2), *after* pooling. So pooling is upstream
  context, not where the puzzle's structure lives — but understanding it tells
  you what the head receives as input.

---

## Part 1 — Mean pooling across tokens: the maths

### 1.1 What the encoder produces

The text is first **tokenized** (WordPiece). A sentence becomes a sequence of
sub-word token IDs, wrapped with special tokens:

```
"What is the population of Canada in 2023?"
   ─► [CLS] what is the population of canada in 2023 ? [SEP]
```

The transformer runs self-attention over this sequence and outputs one
**contextualised embedding per token**:

$$
H = [\,h_1, h_2, \dots, h_L\,], \qquad h_i \in \mathbb{R}^{384}
$$

`h_i` is *contextual*: the vector for "canada" already reflects the whole
sentence around it, because every layer of self-attention mixes information
across positions. So token order **does** matter — it's baked into each `h_i`
via positional encodings and attention. (Pooling itself, below, is
order-agnostic; the order-sensitivity is already inside the `h_i`.)

### 1.2 Padding and the attention mask

To process sentences in a batch, shorter sequences are **padded** with a
`[PAD]` token up to the batch's max length. An **attention mask**
`m ∈ {0,1}^L` marks which positions are real:

```
tokens:  [CLS] what is the ... ?  [SEP] [PAD] [PAD]
mask m:    1    1   1   1  ...  1    1     0     0
```

We must **not** let padding pollute the sentence vector. The mask is how we
exclude it.

### 1.3 The formula

The reference implementation from the `all-MiniLM-L6-v2` model card:

```python
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]                       # (B, L, 384)
    input_mask_expanded = (attention_mask
                           .unsqueeze(-1)
                           .expand(token_embeddings.size())
                           .float())                          # (B, L, 384)
    return (torch.sum(token_embeddings * input_mask_expanded, 1)
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9))
```

In maths, for a single sentence with token embeddings $h_1,\dots,h_L$ and mask
$m_1,\dots,m_L$:

$$
\boxed{\;e \;=\; \dfrac{\displaystyle\sum_{i=1}^{L} m_i \, h_i}{\displaystyle\sum_{i=1}^{L} m_i}\;}
$$

Read it term by term:

| Code step | Maths | Why |
|---|---|---|
| `token_embeddings * input_mask_expanded` | $m_i \, h_i$ | zero out padding tokens (their contribution becomes the zero vector) |
| `torch.sum(..., 1)` | $\sum_i m_i h_i$ | add up the surviving token vectors |
| `input_mask_expanded.sum(1)` | $\sum_i m_i = n$ | count of **real** tokens (not the padded length L) |
| division | $/\,n$ | turns the sum into an **average** |
| `torch.clamp(..., min=1e-9)` | guard | avoids `0/0` for a degenerate empty input; never bites in practice |

So the sentence embedding is simply the **arithmetic mean of the real token
embeddings**. Nothing fancier than that.

### 1.4 Properties worth having in your head

These are the properties that matter when you later reason about *what
information survives* into the head's input.

1. **It is a linear operator (for a fixed mask).**
   $e = \frac{1}{n}\sum_i h_i$ is a fixed linear map $\mathbb{R}^{L\times 384}\!\to\!\mathbb{R}^{384}$.
   A linear (in fact, convex / averaging) combination of the token vectors —
   no nonlinearity is introduced *by the pooling step itself*.

2. **It is a centroid.** $e$ is the **center of mass** of the token-embedding
   cloud (real tokens only). Geometrically, every token "votes" with equal
   weight $1/n$ and the embedding lands at the average position.

3. **Permutation-invariant over the pooled set.** Reordering the *summation*
   doesn't change $e$. (The sentence's meaning still changes with word order —
   but only because the *values* of the $h_i$ change upstream, not the pooling.)

4. **Length dilution.** A single salient token contributes with weight $1/n$.
   In a long sentence ($n$ large) any one token's pull on the centroid is
   weaker; in a short sentence it dominates. This is a real, often-overlooked
   bias of mean pooling: signal from one decisive word is *averaged down* by
   surrounding filler.

5. **Mixing.** If token "japan" carries a strong component along some
   "country-ness" direction, the mean inherits a *fraction* ($1/n$) of that
   component, plus fractions of every other token's directions. The sentence
   vector is a blend; features become **superposed** in the 384-dim space.

### 1.5 Why mean pooling (not the `[CLS]` token)?

`all-MiniLM-L6-v2` was explicitly fine-tuned with **mean pooling** as its
sentence-representation objective (its `modules.json` is
`Transformer → Pooling(mean) → Normalize`). Raw BERT-style `[CLS]` vectors are
poor sentence embeddings unless a model is specifically trained to make them
so. Averaging spreads the "summarise the sentence" burden across all tokens and
empirically yields more robust similarity-space embeddings — which is what this
encoder was optimised for.

### 1.6 One extra step the starter code hides: L2 normalization

The puzzle's starter code calls `enc.encode(...)`, which runs the model's
**full** module pipeline — including a final **`Normalize`** module. That means
the 384-dim vector the head actually receives is (very likely) **L2-normalized
to unit length**:

$$
\hat{e} = \frac{e}{\lVert e \rVert_2}, \qquad \lVert \hat{e}\rVert_2 = 1
$$

Why this is worth flagging *now* (not a spoiler, just geometry hygiene):

- Unit-norm inputs live on a **hypersphere**, not all of $\mathbb{R}^{384}$.
  Magnitude information is discarded; only **direction** survives into the head.
- Any later talk of "a direction encodes a feature" is happening downstream of
  an input that was itself already direction-only.

You can confirm this yourself with a one-liner (`np.linalg.norm` of an encoded
batch) whenever you want — I'm leaving that for you to drive.

---

## Part 2 — Why a per-logit sigmoid, not a softmax?

Your instinct ("shouldn't softmax work, like in LLMs?") is a great question.
The short answer: **this is a different *kind* of classification problem than
next-token prediction.**

### 2.1 Multi-class vs. multi-label

| | **Multi-class** (softmax) | **Multi-label** (sigmoid) |
|---|---|---|
| Question answered | "**Which one** of K?" | "**Which subset** of K?" |
| Labels | mutually exclusive | independent, can co-occur |
| Targets | one-hot (exactly one 1) | bit-vector (any number of 1s) |
| Output constraint | $\sum_k p_k = 1$ | each $p_k \in (0,1)$, no sum constraint |
| Decision | `argmax` | per-feature threshold (`p_k > 0.5`) |
| Example | next token from a vocab | "tags on a photo" |

The 8 features here — `number, question, color, food, sentiment, country,
person, body_part` — are **not mutually exclusive**. The sentence

> "Alice ate 3 red apples in Japan."

is simultaneously `number=1, color=1, food=1, country=1, person=1`. A softmax
would force these to **compete for a fixed probability budget of 1**: asserting
"this contains a number" would have to *suck probability away from* "this
contains a color". That is semantically nonsensical here. Hence the README's
line: *"the eight probabilities don't need to sum to 1 because the features
aren't mutually exclusive."*

### 2.2 The probabilistic models are fundamentally different

**Softmax** models a single **Categorical** random variable (one winner out of
K):

$$
p_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

Every $p_k$ depends on **all** logits — they are coupled through the
denominator.

**Per-logit sigmoid** models **K independent Bernoulli** random variables (K
separate yes/no coins):

$$
p_k = \sigma(z_k) = \frac{1}{1+e^{-z_k}}
$$

Each $p_k$ depends on **only its own logit**. The joint model is a *product* of
8 Bernoullis, versus softmax's *single* Categorical. This network is, in
effect, **8 logistic-regression heads sharing one feature backbone.**

### 2.3 The loss makes the independence concrete

Training used **per-feature Binary Cross-Entropy**, summed over the 8 outputs:

$$
\mathcal{L} = \sum_{k=1}^{8} \Big[ -\,y_k \log \sigma(z_k) - (1-y_k)\log\big(1-\sigma(z_k)\big) \Big]
$$

The gradient of this loss w.r.t. a logit is beautifully clean:

$$
\frac{\partial \mathcal{L}}{\partial z_k} = \sigma(z_k) - y_k
$$

— it depends **only on feature $k$'s own logit and own label**. Feature 3's
error never flows into feature 5's logit. Under **softmax + categorical
cross-entropy** the gradient is $\text{softmax}(z)_k - \text{onehot}_k$, which
**couples every logit to every label**. Independent targets ⇒ independent
losses ⇒ sigmoid.

### 2.4 Sigmoid *is* a 2-class softmax (the unifying view)

These aren't unrelated functions. Sigmoid is the binary special case of
softmax:

$$
\sigma(z) = \frac{e^{z}}{e^{z} + e^{0}} = \text{softmax}\big([\,z,\;0\,]\big)_1
$$

So "8 independent sigmoids" = "8 independent **2-way** softmaxes"
(present-vs-absent for each feature), as opposed to "one **8-way** softmax"
(pick one feature). Same family, different factorisation of the problem.

One subtle consequence: **softmax is shift-invariant**
($\text{softmax}(z) = \text{softmax}(z + c\mathbf{1})$) — only logit
*differences* matter, the absolute zero point is meaningless. **Sigmoid is
not**: $z_k = 0 \Rightarrow p_k = 0.5$ is a hard, absolute decision boundary.
For independent binary detectors you *want* an absolute "is this feature
present?" threshold per feature — exactly what sigmoid gives and softmax
structurally cannot.

### 2.5 So why do LLMs use softmax then?

Because **next-token prediction is genuinely single-label multi-class**: the
next token is *exactly one* token out of the vocabulary V. The choices really
are mutually exclusive and really do form a probability distribution that
should sum to 1. That's the textbook softmax setting.

The distinction is not "LLM vs. classifier" — it's:

- **"Exactly one of K is true"** → softmax (next token, ImageNet class, …)
- **"Any subset of K can be true"** → per-logit sigmoid (tagging, attribute
  detection, *this puzzle*)

Same modelling toolkit; the data's label structure picks the output layer.

---

## Part 3 — How this connects to the puzzle (framing only, no answer)

You need one definition to even *state* the puzzle, so here it is —
deliberately stopping at the definition:

- At layer L (post-ReLU of hidden 2), each input is a **64-dim activation
  vector** $a \in \mathbb{R}^{64}$.
- A feature is **"linearly represented"** if there exists a single direction
  $w \in \mathbb{R}^{64}$ (and bias $b$) such that a *linear* readout
  $w^\top a + b$ separates that feature's positives from negatives — i.e. a
  linear probe / logistic regression on $a$ recovers the feature with high
  accuracy. Geometrically: positives and negatives sit on opposite sides of a
  single hyperplane.
- The puzzle's claim is that **7 of 8** features satisfy this, and **one, F,
  does not** — its positives/negatives are arranged so that *no single
  hyperplane* cleanly splits them, implying some richer geometric structure.

That is the *question*. **Finding F, and characterising its geometry, is your
investigation to drive — this document intentionally goes no further.**

### Neutral brainstorming prompts (spoiler-free)

Questions to think about that don't presuppose any answer:

- What are *all* the ways a binary feature could be encoded **without** a single
  linear direction? (e.g. requiring two coordinates jointly, lying in a
  ring/shell, being a union of separated clusters, XOR-like interactions,
  magnitude-vs-direction splits, …) — building this menu *before* looking at
  data keeps you honest.
- What measurements would *distinguish* those structures from each other? (a
  linear probe's ceiling accuracy vs. a small MLP probe; clustering;
  2-component vs full-rank PCA of the activations conditioned on the label;
  nearest-neighbour label purity; …)
- Mean pooling **superposes** features (Part 1.4) and ReLUs can **gate** /
  fold space — which of your candidate structures could plausibly *emerge*
  from "average of token embeddings → linear → ReLU → linear → ReLU"?
- What's a fair **control**? How will you show the other 7 *are* linear, so
  "F is different" is a contrast and not an artifact of your method?

---

## Sources

- [sentence-transformers/all-MiniLM-L6-v2 — model card (mean-pooling reference code)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [all-MiniLM-L6-v2 README on Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/README.md)
- [Interpreting logits: Sigmoid vs Softmax — Nandita Bhaskhar (Stanford)](https://web.stanford.edu/~nanbhas/blog/sigmoid-softmax/)
- [Understanding Cross-Entropy / BCE / Softmax losses — Raúl Gómez](https://gombru.github.io/2018/05/23/cross_entropy_loss/)
- [Introduction to Multi-Label Classification — Datature](https://datature.io/blog/introduction-to-multi-label-classification)
- [Taming the Sigmoid Bottleneck (sparse multi-label) — arXiv:2310.10443](https://arxiv.org/pdf/2310.10443)