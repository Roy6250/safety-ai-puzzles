# Research Synthesis — Can Natural Language Autoencoders Help Solve This Puzzle?

> **Question put to me:** read Anthropic's *Natural Language Autoencoders*
> announcement and "retrospect deeply" on whether it can help solve the
> BlueDot TAIS puzzle, with references and proof.
>
> **Format:** honest verdict up front, then a precise summary of NLA, a
> point-by-point case for why it's a poor fit *here*, then the body of related
> interpretability research that **is** directly applicable — most importantly
> Engels et al. (ICLR 2025) — followed by a research-grounded recipe and a
> reference list. Still spoiler-free: I'll name the *kinds* of geometric
> structures documented in the literature, but won't claim which (if any)
> applies to your F.

---

## TL;DR

**Short answer: NLA is the wrong tool for this puzzle.** The technique is
designed for *behavioural* interpretation of *billion-parameter language model*
activations, requires reinforcement-learning two full LMs, and explicitly does
not analyse geometric structure. The puzzle's target is a **64-dimensional
post-ReLU activation in a 5-layer MLP head** — a setting where sklearn-grade
geometric tools (linear probes, kNN, PCA, clustering) are vastly more
appropriate, cheaper, and more informative.

**But sibling Anthropic-adjacent research absolutely is relevant.** The single
most directly applicable paper to your puzzle is:

> **Engels, Liao, Michaud, Gurnee, Tegmark (2024). *Not All Language Model
> Features Are (One-Dimensionally) Linear.* arXiv:2405.14860, ICLR 2025.**

That paper asks *exactly the same question* — "are some features encoded
multi-dimensionally rather than as a single direction?" — and ships the
methodology (SAE decomposition + geometric inspection + intervention) you
would adapt for this puzzle. The puzzle reads as a teaching exercise in
the same conceptual family.

---

## 1. What an NLA actually is (with the equations)

Source: [Anthropic — *Natural Language Autoencoders: Turning Claude's thoughts
into text*](https://www.anthropic.com/research/natural-language-autoencoders),
backed by the full paper at
[transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html).

### 1.1 Three components

- **Target model M**: a *frozen* LLM whose activations $h_\ell$ at layer
  $\ell$ we want to interpret. (Claude Haiku 3.5 / 4.5 / Opus 4.6 in the
  paper.)
- **Activation Verbalizer (AV)**: a copy of $M$'s architecture, fine-tuned to
  take $h_\ell$ as a special token and emit a natural-language explanation
  $z$ via autoregressive sampling at temperature 1.
- **Activation Reconstructor (AR)**: $M$ truncated to its first $\ell$
  layers, followed by a learned affine map. It consumes the text $z$ (wrapped
  in a fixed prompt) and predicts the original activation
  $\hat h_\ell = \mathrm{AR}(z)$.

### 1.2 Joint loss

The system is trained to minimise round-trip MSE:

$$
\mathcal{L} \;=\; \mathbb{E}_{h_\ell \sim \mathcal H}\,\mathbb{E}_{z \sim \mathrm{AV}(\cdot\,|\,h_\ell)}\,\big\lVert h_\ell - \mathrm{AR}(z)\big\rVert_2^2
$$

The AR is updated by supervised MSE regression. The AV — which has to
*sample text* and is therefore non-differentiable through $z$ — is updated by
**GRPO reinforcement learning** with reward
$r(h_\ell, z) = -\log \lVert h_\ell - \mathrm{AR}_\theta(z)\rVert_2^2$ and a
KL penalty to the warm-started init.

### 1.3 Reconstruction quality metric (FVE)

$$
\mathrm{FVE} \;=\; 1 - \dfrac{\mathcal{L}}{\mathbb{E}\,\lVert h_\ell - \bar h_\ell\rVert_2^2}
$$

The reported numbers in the paper are **0.6–0.8 FVE** — solidly above
mean-baseline but far from perfect.

### 1.4 What it produced (the headline findings)

- **Evaluation awareness.** On blackmail-style safety evals, NLA explanations
  showed signs of "the model thinks this is a test" without that ever
  surfacing in the model's visible output (~16% on a destructive-code eval,
  ~26% on SWE-bench, vs. <1% on real claude.ai usage).
- **Hidden motivation auditing.** In an auditing game with intentionally
  misaligned models, an NLA-equipped auditor uncovered the hidden motivation
  12–15% of the time vs. <3% without NLAs.
- **Behavioural planning.** Catching the model planning a future rhyme while
  writing a couplet.

### 1.5 Limitations the authors themselves flag

- **Confabulation.** "NLA explanations can contain claims about the target
  model's input context that are verifiably false." Multi-token consistency
  is *some* signal but not a proof.
- **Blackbox by construction.** "We cannot determine which aspects of an
  activation drove a given component of an explanation."
- **Excess expressivity.** The AV is a full LM — it can infer beyond what
  the activation actually encodes.
- **Cost.** Joint RL on two full LMs; hundreds of tokens per activation at
  inference.
- **Degenerate solutions.** The AV could in principle achieve good
  reconstruction by reproducing context verbatim or by emitting
  uninterpretable text the AR has learnt to decode. The authors don't see
  this in practice but flag it as a risk.

---

## 2. Why NLA is the wrong tool for *this* puzzle — five concrete reasons

### Reason 1 — Architectural mismatch

NLA assumes the target is **a language model whose architecture the AV/AR
can copy**. The puzzle's target is a tiny `nn.Sequential(Linear, ReLU,
Linear, ReLU, ...)` MLP head. There is no autoregressive structure to
inherit, no attention to truncate, and no text tokens to emit from a 64-dim
post-ReLU vector. The AV concept is meaningful only when the activation
*came from text-generating computation*. The puzzle's layer L sits inside a
**classifier**, not a generator.

### Reason 2 — The bottleneck is the wrong modality

NLA's bottleneck is **natural-language prose**. The whole point is that the
intermediate representation is human-readable. But the puzzle is asking a
**geometric** question: "is there a single direction, or is the structure
richer?" English prose can't distinguish "encoded along axis $w$" from
"encoded as a 2-D ring in $\{w_1, w_2\}$" — both reduce to "this activation
represents *country-ness*" in text. The Anthropic NLA paper *explicitly does
not* analyse geometric structure — the WebFetched summary confirms there is
**no analysis of linear vs nonlinear activation geometry, clustering,
manifold dimensionality, or symmetries** in the paper.

### Reason 3 — Cost is wildly disproportionate

NLA needs RL on two full Claude-scale LMs and produces hundreds of tokens
per activation. Your activation is a 64-dim vector with 7000 training
examples. A logistic regression on 64 features fits in under a second; an
8-hidden-unit MLP in seconds; a kNN in tens of seconds. Reaching for NLA
here is like using a particle accelerator to weigh flour.

### Reason 4 — NLA is *behavioural*, the puzzle is *geometric*

Task 1 of the puzzle is "find F"; task 2 is "explain the **geometric
structure** the model uses." That structure question demands tools that
*see* directions, manifolds, clusters — PCA, SAEs, kNN, intervention. NLA
outputs *text descriptions* of what an activation "means." Even at its best
it would say something like *"this activation seems to represent the
presence of a country name"* — a true statement that's worthless for
distinguishing a linear country direction from a clustered country
representation.

### Reason 5 — Confabulation risk is amplified by small dimensionality

The AV is a powerful LM that can infer plausible explanations from *very
little signal*. For a 64-dim vector that's already L2-normalisable, the AV
would have enormous freedom to make up coherent-sounding stories. The
paper's own confabulation caveat ("verifiably false claims") becomes acute
when there is so little information in the bottleneck for the AR to anchor
on.

### Verdict

The NLA technique is genuinely impressive, but using it for this puzzle
would be a *category error*: behavioural method for a geometric question,
billion-parameter scaffolding for a 64-dim problem.

There is *one* idea from NLA worth borrowing in spirit — **round-trip
reconstruction as a fidelity test** — but you're already doing the
classifier-scale equivalent every time you compute "linear-probe accuracy"
vs "MLP-probe accuracy" on layer L. That gap *is* a measure of how much of
the label-information survives a linear bottleneck of the activation.

---

## 3. What *would* help: Engels et al. — the directly relevant paper

> Joshua Engels, Eric J. Michaud, Isaac Liao, Wes Gurnee, Max Tegmark.
> *Not All Language Model Features Are (One-Dimensionally) Linear.*
> arXiv:[2405.14860](https://arxiv.org/abs/2405.14860). Accepted to **ICLR
> 2025**. Code: [github.com/JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures).

This is the paper your puzzle is most likely a teaching shadow of.

### 3.1 Definition they actually use

A feature is **irreducible multi-dimensional** if it *cannot be decomposed
into independent or non-co-occurring lower-dimensional features*. That is:
no rotation of the activation space turns it into a collection of separate
one-dimensional concepts.

### 3.2 The detection pipeline

1. **Decompose** activations with an SAE to get a basis of candidate
   features.
2. **Cluster** SAE features that co-activate on the same inputs.
3. **Project** the activations restricted to that cluster onto its top
   principal components (often just 2–3).
4. **Inspect the geometry**: is it a line (1-d linear), a circle (cyclic),
   a more complex manifold?
5. **Intervene**: ablate / rotate / shift the discovered structure and
   re-measure performance on tasks that should require that feature. If
   accuracy on those tasks drops *while accuracy elsewhere stays put*, the
   geometry is mechanistically causal, not decoration.

### 3.3 What they found (the famous examples)

- **Days of the week** form a **circle** in activation space of GPT-2 and
  Mistral 7B.
- **Months of the year** form another circle.
- These exact circles are used to solve modular-arithmetic tasks ("three
  days after Friday is …") — verified by intervention.

### 3.4 Why this is the relevant paper, not NLA

| | NLA | Engels et al. |
|---|---|---|
| Question answered | What does this activation *mean*? | How is this concept *geometrically encoded*? |
| Target scale | Billion-param Claude | Smaller (GPT-2 124M to Mistral 7B) — methodology scales down further |
| Output | Natural-language explanation | A geometric object (line, circle, manifold) + intervention proof |
| Cost | RL on two LMs | SAE training + PCA + intervention scripts |
| **Fit to the puzzle** | **Mismatched** | **Direct match** |

### 3.5 What you'd translate to the puzzle

You don't even need the SAE step — your activation is already 64-dim, not
12,288-dim. The SAE existed in Engels et al. only to *carve up* the huge
residual stream into manageable pieces. For a 64-d activation you can
skip directly to:

- **Condition activations on the suspect feature's label.**
- **PCA inside the positive class** (and inside the negative class).
- **Inspect the top 2–3 components** visually — line vs ring vs disjoint
  clusters vs more.
- **Stratify** by another covariate (length, lexical content, another
  feature's label) and see whether the geometry sorts by it.
- **Intervene** in the head: project an activation onto the suspect
  structure's axes / subspace, perturb it, push it back through the rest
  of the head, and measure the change in *that feature's* logit vs the
  other seven. (This is exactly the puzzle's task 2 — "show the analysis
  you used to convince yourself.")

That is the **Engels methodology translated to a 5-layer MLP**. The
toolkit doc you already have (`probing_toolkit.md`) implements the
*detection* half of this; the *intervention* half is the natural follow-on
for task 2.

---

## 4. The supporting research landscape

The puzzle sits in a long-running debate in mech interp about how features
are encoded. The relevant priors are:

### 4.1 The Linear Representation Hypothesis (LRH)

The default belief — high-level concepts live on **one-dimensional
directions** ("a `country` axis") — has deep roots:

- **Mikolov et al. (2013).** Word2Vec famously showed
  *king − man + woman ≈ queen*, suggesting linear semantic structure in
  word embeddings.
- **Park, Choe, Veitch (2023). *The Linear Representation Hypothesis and
  the Geometry of Large Language Models.* arXiv:[2311.03658](https://arxiv.org/abs/2311.03658).** Formalises LRH and
  proposes a non-Euclidean inner product to make linearity precise.
- Reviewed in
  [Representation Engineering survey, arXiv:2502.17601](https://arxiv.org/html/2502.17601v1):
  "high-level concepts and functions are encoded ... as linear or
  near-linear features, identified as directions or subspaces."

Many features genuinely are linear (the puzzle says 7/8 are). The
interesting question is the exceptions.

### 4.2 Sparse Autoencoders & Monosemanticity

- **Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language
  Models With Dictionary Learning.* [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features).** Demonstrates that
  SAEs can extract monosemantic features from polysemantic neurons.
- **Anthropic (2024). *Scaling Monosemanticity: Extracting Interpretable
  Features from Claude 3 Sonnet.* [transformer-circuits.pub](https://transformer-circuits.pub/2024/scaling-monosemanticity/).** Scaled SAEs to
  production models; ~70% of features cleanly map to single concepts.
- **Cunningham et al. (2023). *Sparse Autoencoders Find Highly
  Interpretable Features in Language Models.* arXiv:[2309.08600](https://arxiv.org/abs/2309.08600).** Independent reproduction.
- For this puzzle, SAEs are mostly *conceptual scaffolding*: 64-d isn't
  big enough to need dictionary learning, but the *vocabulary* of
  "feature directions in superposition" is exactly what frames the
  question.

### 4.3 Probing classifiers — the methodological cautionary tale

- **Belinkov (2022). *Probing Classifiers: Promises, Shortcomings, and
  Advances.* [Computational Linguistics 48(1)](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and).**
  The canonical methodology paper. Establishes the central tension:
  > A non-linear probe might solve a task by *computation* rather than by
  > *reading out* information already present in the representation. With a
  > linear probe, success implies the information is *there* in a
  > readable form.
- **Hewitt & Liang (2019). *Designing and Interpreting Probes with Control
  Tasks.* [aclanthology.org](https://aclanthology.org/D19-1275/).**
  Introduces control tasks (random labels) to measure probe expressivity vs.
  representation quality.

**Concrete implication for the puzzle.** When you use a small MLP probe in
§1 of `probing_toolkit.md`, keep the hidden width *small* (8–16). If an
MLP probe beats the linear probe by a lot, you must rule out "the MLP
computed it from scratch." Two ways to rule that out:
1. Show the same MLP **fails or barely beats linear** on the *other*
   features at the same width.
2. Show the linear probe **succeeds** on the same features on adjacent
   layers (h=1 or h=3), confirming the deficit is specific to layer L.

### 4.4 Other recent geometric-representation work worth knowing

- **Decomposing MLP Activations into Interpretable Features via
  Semi-Nonnegative Matrix Factorization** (arXiv:[2506.10920](https://arxiv.org/pdf/2506.10920)) — alternative to SAEs that
  doesn't require sparsity priors; could be useful if you find SAE bases
  hard to interpret.
- **Identifying Linear Relational Concepts in Large Language Models**
  (arXiv:[2311.08968](https://arxiv.org/html/2311.08968v2)) — concept of
  *linear relational* features, useful contrast class for "this concept
  has internal relational structure."

---

## 5. A research-grounded recipe for *this* puzzle (still no spoilers)

Combining the Engels methodology with the probing-classifier
discipline and the toolkit you already have:

1. **Establish the linear baseline rigorously.** For each feature `k`, fit
   a *well-regularised* logistic regression on `A_train[2]`. Sweep `C` and
   take the best test AUROC. Call this the **linear ceiling** $L_k$.

2. **Find the linearity gap.** Fit a small MLP probe (8–16 hidden units,
   ≥3 seeds). Call its mean test AUROC $M_k$. Compute $\Delta_k = M_k - L_k$.
   Apply the Belinkov check: confirm the small MLP can *not* solve random
   control labels at the same width.

3. **Compare against adjacent-layer baselines.** Repeat at `h=1`, `h=3`,
   and the raw 384-dim encoder output. The puzzle's claim implies the gap
   for F should peak at `h=2`. If it peaks elsewhere you have a wrong-layer
   confound, not the puzzle's structure.

4. **Engels-style geometric inspection on the candidate.** Take
   `A_test[2]` restricted to the suspect feature's positives. Run PCA;
   plot the top 2–3 components, then 3D; colour by the feature label and
   then by every *other* feature label (those colourings often reveal what
   the geometry is keying on). The Engels prior is to look for: ring /
   circle (cyclic concept), two-or-more disjoint clusters
   (taxonomic / multi-mode), interleaving along a manifold, magnitude
   bands.

5. **Hypothesise → predict → test.** This is the part most undergrads
   skip. Whatever geometric structure you propose, write down a
   *quantitative* prediction it implies — e.g. "if positives form two
   clusters, then a *2-Gaussian* mixture should beat *1-Gaussian* in BIC,
   and within-cluster labels of *another* feature should differ" — and
   check it. A single visualisation isn't proof; a successful pre-registered
   prediction is.

6. **Intervene (the strongest evidence).** Project a positive example onto
   the structure's discovered axes/subspace and either (a) ablate that
   subspace, (b) swap it with a negative example's projection, or (c)
   rotate around an axis. Push the modified activation through the
   remaining head layers (`head.layers[6:]`) and observe the change in the
   target feature's logit vs the others. If feature F's logit flips while
   the other 7 stay roughly unchanged, your geometry is causally
   responsible. This is exactly the Engels intervention pattern from §3.

7. **(Optional, for task 3.) Train a new head where you *deliberately*
   bake in a non-linear structure** for some feature — using a custom loss
   that penalises linear separability at layer L, or a synthetic dataset
   where the feature is naturally polar/cyclic/clustered. The Engels
   replication suggestion ("Train an MLP to predict day-of-week from
   sinusoidal position encodings; verify the hidden representation forms a
   circle") is a one-evening project that would directly demonstrate
   research-grade mechanistic interpretability on a tiny model.

---

## 6. Why I think this puzzle is intentionally an Engels-style exercise

Three converging hints (not spoilers — pattern-matching to the *kind* of
puzzle, not its answer):

1. **The exact framing.** "Seven features are represented linearly … one
   is represented in a different way." This is the LRH-vs-Engels debate
   rephrased as a puzzle.
2. **The layer choice.** The puzzle nails a *single* hidden layer post-ReLU
   — exactly where geometric structure has space to live and where SAEs
   typically target.
3. **Task 3.** "Train a model with an even weirder representation" is
   literally Engels' replication advice repurposed as a homework
   extension. The course is *Technical AI Safety*; the literature it
   teaches from absolutely includes Engels et al., Bricken et al.,
   Scaling Monosemanticity, and the LRH papers.

This is good — it means you can (and should) read those papers in parallel
with your investigation. The mental model they install is exactly the one
you need to write a strong submission.

---

## 7. References

### NLA (the thing you asked about)

1. **Anthropic (2026).** *Natural Language Autoencoders: Turning Claude's
   thoughts into text.*
   [anthropic.com/research/natural-language-autoencoders](https://www.anthropic.com/research/natural-language-autoencoders)
2. **Tschopp K. et al. (2026).** *Natural Language Autoencoders.* Full
   paper.
   [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)
3. Code: [github.com/kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders)
4. Interactive demo: [neuronpedia.org/nla](https://neuronpedia.org/nla)
5. Third-party walkthrough: [MarkTechPost coverage (May 2026)](https://www.marktechpost.com/2026/05/08/anthropic-introduces-natural-language-autoencoders-that-convert-claudes-internal-activations-directly-into-human-readable-text-explanations/)

### The directly-relevant paper

6. **Engels, Liao, Michaud, Gurnee, Tegmark (2024, ICLR 2025).** *Not All
   Language Model Features Are (One-Dimensionally) Linear.*
   arXiv:[2405.14860](https://arxiv.org/abs/2405.14860).
   Code: [github.com/JoshEngels/MultiDimensionalFeatures](https://github.com/JoshEngels/MultiDimensionalFeatures).
   Paper page: [HuggingFace](https://huggingface.co/papers/2405.14860).
   Conference PDF: [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/d3221cdb27e49d9c1cd35ad254feccfe-Paper-Conference.pdf).

### Linear Representation Hypothesis & geometry

7. **Park, Choe, Veitch (2023).** *The Linear Representation Hypothesis
   and the Geometry of Large Language Models.* arXiv:[2311.03658](https://arxiv.org/abs/2311.03658).
8. **Mikolov, Yih, Zweig (2013).** *Linguistic Regularities in Continuous
   Space Word Representations.* The original "king − man + woman ≈ queen"
   observation.
9. **Representation Engineering for LLMs — Survey** (arXiv:[2502.17601](https://arxiv.org/html/2502.17601v1)).

### Sparse autoencoders & monosemanticity

10. **Bricken et al. (2023).** *Towards Monosemanticity: Decomposing
    Language Models With Dictionary Learning.*
    [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)
11. **Templeton et al. (Anthropic, 2024).** *Scaling Monosemanticity:
    Extracting Interpretable Features from Claude 3 Sonnet.*
    [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
12. **Cunningham et al. (2023).** *Sparse Autoencoders Find Highly
    Interpretable Features in Language Models.* arXiv:[2309.08600](https://arxiv.org/abs/2309.08600).
13. **Survey on SAEs for LLM interpretability** (arXiv:[2503.05613](https://arxiv.org/html/2503.05613v3)).

### Probing classifiers (methodology you must respect)

14. **Belinkov (2022).** *Probing Classifiers: Promises, Shortcomings, and
    Advances.* [Computational Linguistics 48(1)](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and).
15. **Hewitt & Liang (2019).** *Designing and Interpreting Probes with
    Control Tasks.* [ACL Anthology](https://aclanthology.org/D19-1275/).
16. **Practical guide:** [Brenndoerfer — *Probing Classifiers*](https://mbrenndoerfer.com/writing/probing-classifiers).

### Adjacent recent work

17. **Decomposing MLP Activations into Interpretable Features via
    Semi-Nonnegative Matrix Factorization** (arXiv:[2506.10920](https://arxiv.org/pdf/2506.10920)).
18. **Identifying Linear Relational Concepts in Large Language Models**
    (arXiv:[2311.08968](https://arxiv.org/html/2311.08968v2)).
19. **Recurrent Neural Networks Learn to Store and Generate Sequences using
    Non-Linear Representations** (arXiv:[2408.10920](https://arxiv.org/html/2408.10920v1)) — examples of non-linear
    representations outside the LM setting.

---

## 8. One-line takeaway

> *NLA is a beautiful technique for the wrong question. Read **Engels et al.
> (arXiv:2405.14860)** instead — it's the same question as your puzzle,
> answered with tools that will actually fit on your laptop.*
