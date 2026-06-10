import torch.nn as nn
import json, numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Resolve file paths relative to this script, not the current working directory,
# so `python probe_setup.py` works no matter where you run it from.
HERE = Path(__file__).resolve().parent
class Head(nn.Module):

    def __init__(self):
        
        super(Head, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(384,64), nn.ReLU(), # 0,1

            nn.Linear(64,64), nn.ReLU(), # 2,3
            nn.Linear(64,64), nn.ReLU(), # 4,5
            nn.Linear(64,64), nn.ReLU(), # 6,7

            nn.Linear(64,8)

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

    with torch.no_grad():
        return head.layers[:layer_idx_inclusive](embeddings).cpu().numpy()
    


# --- load everything once ---
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
head = Head(); head.load_state_dict(torch.load(HERE / "model.pt", map_location="cpu", weights_only=False)); head.eval()

train_texts, y_train = load_jsonl(HERE / "data" / "train.jsonl")
test_texts,  y_test  = load_jsonl(HERE / "data" / "test.jsonl")

with torch.no_grad():
    X_train_emb = torch.from_numpy(enc.encode(train_texts, convert_to_numpy=True, batch_size=64, show_progress_bar=True))
    X_test_emb  = torch.from_numpy(enc.encode(test_texts,  convert_to_numpy=True, batch_size=64, show_progress_bar=True))


A_train = {h: acts_at(2*(h+1), X_train_emb, head) for h in range(4)}   # h = 0,1,2,3
A_test  = {h: acts_at(2*(h+1), X_test_emb,  head) for h in range(4)}
RAW_train, RAW_test = X_train_emb.numpy(), X_test_emb.numpy()           # 384-d encoder output (control)

# np.savez_compressed("acts_cache.npz", **{f"A_train_{h}": v for h,v in A_train.items()}, ...)  # optional
FEATURE_NAMES = json.load(open(HERE / "feature_names.json"))
print("shapes:", {h: a.shape for h,a in A_train.items()})  # expect (~7000, 64)


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
print(rows)