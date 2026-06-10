# Probing Toolkit — Distinguishing Linear vs Non-Linear Feature Representations

> Scope: a hands-on guide to the **measurements** that can tell apart "this
> feature is encoded as a single direction" from "this feature is encoded as
> something more interesting." Each section gives the *idea*, a *runnable
> skeleton*, *how to read the result*, and *what can fool you*.
>
> ⚠️ **No puzzle spoilers.** Every example talks about a generic feature
> `k ∈ {0,…,7}` or "your suspect feature." I'm intentionally not naming F or
> sketching its structure — your investigation, your call.

---

## 0. Shared setup — extract layer-L activations once, probe many times

You'll re-use the same arrays for every measurement, so it pays to extract
them once and cache. Skeleton (run from the repo root with `.venv/bin/python`):

```python
# probe_setup.py
import json, torch, torch.nn as nn, numpy as np
from sentence_transformers import SentenceTransformer

class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(384, 64), nn.ReLU(),   # 0,1  hidden 0
            nn.Linear(64, 64),  nn.ReLU(),   # 2,3  hidden 1
            nn.Linear(64, 64),  nn.ReLU(),   # 4,5  hidden 2  ← layer L lives here (post-ReLU)
            nn.Linear(64, 64),  nn.ReLU(),   # 6,7  hidden 3
            nn.Linear(64, 8),                # 8    logits
        )
    def forward(self, x):
        return self.layers(x)

def load_jsonl(path):
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(r["labels"])
    return texts, np.array(labels, dtype=np.int64)        # (N, 8)

def acts_at(layer_idx_inclusive, embeddings, head):
    """Forward through head.layers[:layer_idx_inclusive] and return activations."""
    with torch.no_grad():
        return head.layers[:layer_idx_inclusive](embeddings).cpu().numpy()

# --- load everything once ---
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
head = Head(); head.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=False)); head.eval()

train_texts, y_train = load_jsonl("data/train.jsonl")
test_texts,  y_test  = load_jsonl("data/test.jsonl")

with torch.no_grad():
    X_train_emb = torch.from_numpy(enc.encode(train_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True))
    X_test_emb  = torch.from_numpy(enc.encode(test_texts,  convert_to_numpy=True, batch_size=64, show_progress_bar=True))

# Cache activations at several layers — you'll want them for controls.
# Indices into nn.Sequential: post-ReLU of hidden h is index 2*(h+1).
A_train = {h: acts_at(2*(h+1), X_train_emb, head) for h in range(4)}   # h = 0,1,2,3
A_test  = {h: acts_at(2*(h+1), X_test_emb,  head) for h in range(4)}
RAW_train, RAW_test = X_train_emb.numpy(), X_test_emb.numpy()           # 384-d encoder output (control)

# np.savez_compressed("acts_cache.npz", **{f"A_train_{h}": v for h,v in A_train.items()}, ...)  # optional
FEATURE_NAMES = json.load(open("feature_names.json"))
print("shapes:", {h: a.shape for h,a in A_train.items()})  # expect (~7000, 64)
```

**Tip:** also probe at `h=1` and `h=3` and the raw 384-d encoder output. The
puzzle says the interesting layer is `h=2`, but adjacent layers and the raw
input are essential *controls* — see §7.

---

## 1. The headline test — linear probe vs. small MLP probe ("linearity gap")

**Idea.** Train two probes on the *same* activations to predict feature `k`:

- a **linear** probe (logistic regression): can only carve a hyperplane.
- a **small non-linear** probe (1-hidden-layer MLP, e.g. 16 hidden units):
  can carve curved / disjoint boundaries.

The metric you actually care about is the **gap**:

$$
\Delta_k \;=\; \text{MLP-probe acc}(k) \;-\; \text{linear-probe acc}(k)
$$

If a feature is genuinely linear in the activation space, both probes do about
equally well (gap ≈ 0). If a feature is encoded non-linearly, the linear
probe hits a ceiling well below what the MLP can reach (gap is materially
positive).

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def probe_one(X_tr, y_tr, X_te, y_te, kind, seed=0):
    if kind == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500,
                            random_state=seed, early_stopping=True)
    clf.fit(X_tr, y_tr)
    p = clf.predict_proba(X_te)[:, 1]
    yhat = (p > 0.5).astype(int)
    return {
        "acc": (yhat == y_te).mean(),
        "bal_acc": balanced_accuracy_score(y_te, yhat),
        "auroc": roc_auc_score(y_te, p),
    }

def gap_table(A_tr, A_te, y_tr, y_te, names):
    sc = StandardScaler().fit(A_tr); Xtr, Xte = sc.transform(A_tr), sc.transform(A_te)
    rows = []
    for k, name in enumerate(names):
        lin = probe_one(Xtr, y_tr[:,k], Xte, y_te[:,k], "linear")
        mlp = probe_one(Xtr, y_tr[:,k], Xte, y_te[:,k], "mlp")
        rows.append((name, lin["bal_acc"], mlp["bal_acc"], mlp["bal_acc"]-lin["bal_acc"], lin["auroc"], mlp["auroc"]))
    return rows

# Run it at layer L (h=2)
rows = gap_table(A_train[2], A_test[2], y_train, y_test, FEATURE_NAMES)
# Print sorted by gap; the bigger the gap, the more non-linear the encoding.
```

**How to read it.**
- Look at *balanced accuracy* (and AUROC) — many of these features have heavy
  class imbalance and raw accuracy lies.
- Seven features should land roughly together (linear ≈ MLP). One should
  stand out with a clearly larger MLP-vs-linear gap. That's your candidate F.
- Don't decide on a single random seed. Run 3–5 seeds, report
  mean ± stddev. The MLP in particular is seed-sensitive at small widths.

**What can fool you.**
- **Class imbalance.** A linear probe can get 90% raw accuracy by always
  predicting "negative" if positives are 10%. Always use balanced acc / AUROC.
- **Linear-probe regularization.** Sweep `C ∈ {0.01, 0.1, 1, 10, 100}` and
  take the *best* linear test score as the linear ceiling — otherwise you risk
  calling a poorly-regularized probe "the linear limit."
- **Overfitting the MLP.** A 64-wide MLP can fit anything in 64-d. Keep the
  hidden width small (8–16). The question is "is there a *modest*
  non-linearity that suffices?", not "can I memorise?"
- **Probe ≠ model.** A probe failing to find a direction doesn't mean the
  model can't use the feature downstream — the model has *two more*
  Linear+ReLU layers after L to disentangle whatever structure is at L. The
  classifier still hits >95% on every feature; the puzzle is about the
  *geometry at L*, not about the model's overall ability.

---

## 2. Visual diagnosis — conditional PCA

**Idea.** Pick feature `k`. Project layer-L activations to 2D with PCA and
colour points by their label `y[:,k]`. The *shape* of the coloured cloud often
tells you the structure at a glance:

| What you see | Likely structure |
|---|---|
| Two cleanly separated blobs | linear (single hyperplane works) |
| Two overlapping blobs | linear-ish, weak signal |
| Positives forming a *ring* / shell around negatives | radial / magnitude-based |
| Positives forming **two+ disjoint clusters** with negatives between | multi-modal / cluster-based |
| Salt-and-pepper interleaving | XOR-like or higher-order |

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_pca_by_label(A, y_col, name, n_components=2):
    Z = PCA(n_components=n_components, random_state=0).fit_transform(A)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(Z[y_col==0,0], Z[y_col==0,1], s=4, alpha=0.4, label="0")
    ax.scatter(Z[y_col==1,0], Z[y_col==1,1], s=4, alpha=0.4, label="1")
    ax.set_title(f"PCA of layer-L acts, coloured by {name}")
    ax.legend(); plt.show()

for k, name in enumerate(FEATURE_NAMES):
    plot_pca_by_label(A_test[2], y_test[:,k], name)
```

**Refinements worth trying.**
- **Class-conditional PCA:** fit PCA on positives only, then on negatives
  only. If the positive-class principal axes look qualitatively different
  from the negatives' (e.g. higher rank, different orientation), that's a
  signal.
- **Mean-difference projection.** Take $\mu_+ - \mu_-$ (the class-mean
  difference vector in $\mathbb{R}^{64}$), project all points on it. Histogram
  of positives vs negatives along this 1-d axis is *the* linear test — if
  they're nicely separated, it's linear; if they overlap heavily, look for
  non-linear structure elsewhere.
- **t-SNE / UMAP.** Use *only* as a visual sanity check, never as evidence —
  they distort distances. PCA is honest.
- **Full-rank PCA scree.** Plot explained-variance per component. Most of
  the signal in a 64-d activation usually lives in the first 5–10 components
  — if a feature's PCA looks featureless in 2D, try a 3-D scatter and the
  next few components.

**Reading caveat.** A feature can be linearly separable *in 64-d* even when it
looks tangled in 2-D PCA — high dimensions hide hyperplanes. Use PCA for
intuition, but confirm with §1's numbers.

---

## 3. Clustering inside each label class

**Idea.** A *single-direction* representation puts all positives on one side
of a hyperplane — they form one rough cloud. A non-linear representation
might put positives into **two or more disjoint clusters** (e.g. *number-as-digit*
vs *number-as-word* could each be their own cluster, conceptually). Same for
negatives.

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def cluster_within(A, y_col, label_value, k_max=5):
    sub = A[y_col == label_value]
    out = {}
    for k in range(2, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(sub)
        out[k] = silhouette_score(sub, km.labels_)   # higher = more cluster-like
    return out

for k, name in enumerate(FEATURE_NAMES):
    print(name, "positives:",  cluster_within(A_test[2], y_test[:,k], 1))
    print(name, "negatives:",  cluster_within(A_test[2], y_test[:,k], 0))
```

**How to read it.**
- A clear silhouette peak at `k ≥ 2` for one class hints at *modes* within
  that class — incompatible with a "single linear direction" story unless
  the modes happen to be collinear.
- Compare across features. The feature whose positive (or negative) class
  shows much higher multi-cluster silhouette than the others is suspicious.

**Caveats.** Silhouette has its own biases (favours convex round clusters);
try DBSCAN or a Gaussian mixture (`sklearn.mixture.GaussianMixture` with BIC
selection) as cross-checks. Don't over-interpret silhouette differences of
±0.02 — look for clear, qualitative gaps.

---

## 4. Nearest-neighbour label purity

**Idea.** kNN doesn't care whether the decision boundary is linear; it only
asks "do nearby points share my label?". So:

- **kNN ≈ Linear-probe accuracy** → the structure that exists is mostly
  linear (a hyperplane and locality agree).
- **kNN ≫ Linear-probe accuracy** → there's *real signal* (neighbours agree
  with you), but no global hyperplane catches it. Strong evidence the
  representation is non-linear.
- **Both low** → the feature genuinely isn't in this layer at all.

```python
from sklearn.neighbors import KNeighborsClassifier

def knn_acc(X_tr, y_tr, X_te, y_te, k=15):
    clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="cosine")
    clf.fit(X_tr, y_tr)
    yhat = clf.predict(X_te)
    return balanced_accuracy_score(y_te, yhat)

for k, name in enumerate(FEATURE_NAMES):
    acc_knn = knn_acc(A_train[2], y_train[:,k], A_test[2], y_test[:,k])
    print(f"{name:10s}  kNN bal-acc = {acc_knn:.3f}")
```

**Tips.**
- Try `metric="cosine"` *and* `metric="euclidean"`. If the sentence
  embeddings going in were unit-norm (see Part 1.6 of the architecture
  notes), cosine and Euclidean are monotone-related but differ subtly inside
  the head's nonlinear layers.
- Sweep `k ∈ {5, 15, 50}`. A consistent kNN-vs-linear gap across `k` is more
  convincing than a single number.
- Compute the gap explicitly: `knn_acc − linear_acc` per feature, alongside
  §1's `mlp_acc − linear_acc`. The two gaps usually agree on the suspect.

---

## 5. Useful bonus probes

These are quick, often illuminating, and rule things in or out:

- **LDA / mean-direction probe (closed-form linear).** Compute
  $w = \Sigma^{-1}(\mu_+ - \mu_-)$ (or use `sklearn.discriminant_analysis
  .LinearDiscriminantAnalysis`). This is the *optimal* linear probe under
  Gaussian-equal-covariance assumptions. If even LDA can't separate a
  feature, the linear story is genuinely poor — not a regularization
  artifact.
- **Kernel SVM.** `SVC(kernel='rbf', C=1, gamma='scale')` on a subsample.
  If RBF crushes linear by a wide margin for one feature only, it's almost
  certainly non-linear at L.
- **Depth-1 decision-tree baseline.** A single axis-aligned threshold is the
  weakest possible "linear-ish" probe; a wide gap to a depth-4 tree is
  another non-linearity tell.
- **Per-feature explained variance.** For each feature `k`, regress each
  activation dimension on the label and sum the per-dim $R^2$. A feature
  that's linearly encoded usually has one or two dimensions doing most of
  the work; a non-linearly encoded feature smears its signal across many
  dimensions weakly.
- **Concept-direction agreement across seeds.** Train the linear probe on
  random halves of the training set. Compute cosine similarity between the
  learned weight vectors. A stable feature direction is highly self-similar
  (>0.9); a feature with no genuine single direction yields noisy,
  inconsistent vectors (<0.5).

---

## 6. Putting it together — one scoring grid

Don't decide F from a single probe. Build one table; it makes the suspect
obvious.

```python
import pandas as pd

def scoreboard(A_tr, A_te, y_tr, y_te, names):
    sc = StandardScaler().fit(A_tr); Xtr, Xte = sc.transform(A_tr), sc.transform(A_te)
    rows = []
    for k, name in enumerate(names):
        lin = probe_one(Xtr, y_tr[:,k], Xte, y_te[:,k], "linear")["bal_acc"]
        mlp = probe_one(Xtr, y_tr[:,k], Xte, y_te[:,k], "mlp")["bal_acc"]
        knn = knn_acc(A_tr,  y_tr[:,k], A_te,  y_te[:,k])
        rows.append(dict(feature=name,
                         linear=lin, mlp=mlp, knn=knn,
                         mlp_minus_lin=mlp - lin,
                         knn_minus_lin=knn - lin))
    return pd.DataFrame(rows).sort_values("mlp_minus_lin", ascending=False)

print(scoreboard(A_train[2], A_test[2], y_train, y_test, FEATURE_NAMES))
```

**Decision pattern you're looking for.** Across `mlp_minus_lin`, `knn_minus_lin`,
clustering silhouettes, and the concept-direction stability check —
**one feature should consistently look anomalous**. If only *one* method
flags a candidate, distrust it. If 3–4 methods independently point at the
same feature, that's your F.

---

## 7. Controls you must run (otherwise you'll fool yourself)

- **Adjacent-layer baseline.** Re-run §1 at `h=1` and `h=3`. If a feature
  looks "non-linear" at every layer, you may just be measuring an
  intrinsically hard feature, not the puzzle's structure. The puzzle's
  claim is specifically about **h=2**, so the anomaly should *peak* there.
- **Raw-encoder baseline.** Run §1 on the 384-d encoder output (before any
  MLP layer). This tells you which features are "easy on a generic
  sentence embedding." If a feature is hard everywhere — including the raw
  encoder — its non-linearity at L might just be inherited, not learned by
  the head. The interesting case is a feature that's *learnable* (high MLP
  probe) but *not linearly readable* at L specifically.
- **The model's own readout is downstream of L.** Tempting to inspect the
  final `Linear(64, 8)`'s row `k` as "the model's direction for feature k"
  — but that's a readout from `h=3`, not from L. Two Linear+ReLU layers sit
  between L and the logits, and that's exactly enough capacity to
  *un-superpose* whatever structure F has at L. So row `k` of the readout
  is not a probe of L.
- **Hold-out hygiene.** Always fit `StandardScaler`, PCA, probes on
  `train`; evaluate on `test`. Don't fit anything on `test`.
- **Multiple seeds.** Report mean ± std over ≥3 seeds for any probe with a
  random component (MLP, KMeans, SVM with subsampling).

---

## 8. What this whole exercise will tell you (and what it won't)

**Will tell you:**
- Whether each feature has a clean linear direction at layer L (the linearity
  gap).
- Whether non-linearly-encoded features have *cluster*, *radial*, or
  *interleaved* structure (PCA + clustering + kNN together).
- Whether the structure is specific to layer L or inherited from upstream
  (the controls in §7).

**Won't tell you (without more work):**
- *Why* the head learned that structure — that requires understanding the
  data distribution + training dynamics.
- The *causal* role of any direction or cluster in the final prediction —
  for that you'd need ablations / patching (zero out a direction, swap a
  cluster centroid, etc.) and re-measure the model's accuracy on the
  affected feature.
- How to *replicate* it in a new model — task 3 of the puzzle, which builds
  on understanding gathered here.

Once you have the scoreboard from §6 and the PCA panels from §2, you'll
likely have a strong hypothesis. From there, the right next step is usually
to **stratify** — re-run the same probes on subsets of the data (split by
length, by token vocabulary, by another feature's label) — to see what
property of the input the geometry is actually keying on. That's where the
interesting story lives, and it's *exactly* the part I'm leaving for you.

---

## Quick reference — recommended order of operations

1. Build `probe_setup.py`; cache activations at `h ∈ {1,2,3}` and the raw
   384-d encoder output.
2. Run §1's `gap_table` at `h=2`. Note the feature with the biggest gap.
3. Repeat §1 at `h=1`, `h=3`, and raw encoder (the §7 controls). Confirm the
   gap is *largest at h=2*.
4. Run §2 (conditional PCA) for the suspect feature only — look at the shape.
5. Run §3 (clustering) and §4 (kNN purity) for all features — fill the
   scoreboard in §6.
6. Pick the most consistent suspect; sanity-check with §5 bonus probes.
7. Hypothesise the *structure* (cluster, ring, interleaved, …); design a
   stratified slicing of the data that would confirm or refute it.

Good hunting.