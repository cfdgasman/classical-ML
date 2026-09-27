"""Draw the 12-panel course banner shown at the top of the root README (assets/banner.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer, make_blobs, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

from mlaz import PALETTE, set_style

set_style()
rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 6, figsize=(15, 5.2))
axes = axes.ravel()
titles = ["01 Framing & splits", "02 Cleaning", "03 EDA", "04 Features", "05 Regression", "06 Classification",
          "07 Trees & ensembles", "08 Evaluation", "09 Tuning", "10 Unsupervised", "11 Interpretability",
          "12 Capstone"]

# 01 train / validation / test bar
ax = axes[0]
for i, (w, c, lab) in enumerate([(0.6, PALETTE[0], "train"), (0.2, PALETTE[1], "val"), (0.2, PALETTE[3], "test")]):
    left = [0, 0.6, 0.8][i]
    ax.barh(0, w, left=left, color=c)
    ax.text(left + w / 2, 0, lab, ha="center", va="center", color="white", fontweight="bold")
ax.set_ylim(-1, 1)

# 02 missing-value matrix
ax = axes[1]
m = rng.random((30, 8)) < np.linspace(0.02, 0.5, 8)
ax.imshow(~m, aspect="auto", cmap="Greys_r", vmin=-0.3)

# 03 histogram with KDE-ish curve
ax = axes[2]
x = np.concatenate([rng.normal(0, 1, 400), rng.normal(3, 0.7, 200)])
ax.hist(x, bins=30, color=PALETTE[5], density=True)

# 04 polynomial feature basis
ax = axes[3]
t = np.linspace(-1, 1, 100)
for d in range(1, 5):
    ax.plot(t, t ** d, lw=2)

# 05 regression line
ax = axes[4]
xr = rng.uniform(0, 10, 60)
yr = 1.5 * xr + rng.normal(0, 2, 60)
ax.scatter(xr, yr, s=12)
ax.plot([0, 10], [0, 15], color=PALETTE[3], lw=2.5)

# 06 SVM boundary on moons
ax = axes[5]
Xm, ym = make_moons(200, noise=0.2, random_state=0)
clf = SVC(gamma=2).fit(Xm, ym)
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
zz = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, zz > 0, alpha=0.25, cmap="coolwarm")
ax.scatter(*Xm.T, c=ym, cmap="coolwarm", s=8)

# 07 tree step function
ax = axes[6]
xs = np.sort(rng.uniform(0, 6, 120))
ys = np.sin(xs) + rng.normal(0, 0.2, 120)
tree = DecisionTreeRegressor(max_depth=3).fit(xs[:, None], ys)
ax.scatter(xs, ys, s=8, color=PALETTE[5])
ax.plot(xs, tree.predict(xs[:, None]), color=PALETTE[2], lw=2.5)

# 08 ROC curve
ax = axes[7]
Xb, yb = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(Xb[:, :3], yb, random_state=0)
p = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
fpr, tpr, _ = roc_curve(yte, p)
ax.plot(fpr, tpr, lw=2.5, color=PALETTE[0])
ax.plot([0, 1], [0, 1], "--", color="grey")

# 09 grid-search heatmap
ax = axes[8]
g = np.exp(-((np.arange(6)[:, None] - 3) ** 2 + (np.arange(6)[None, :] - 2) ** 2) / 4)
ax.imshow(g, cmap="viridis")

# 10 k-means clusters
ax = axes[9]
Xc, _ = make_blobs(300, centers=4, random_state=3)
km = KMeans(4, n_init=10, random_state=0).fit(Xc)
ax.scatter(*Xc.T, c=[PALETTE[i] for i in km.labels_], s=8)
ax.scatter(*km.cluster_centers_.T, marker="X", s=120, color="black")

# 11 feature importance bars
ax = axes[10]
imp = np.sort(rng.random(6))
ax.barh(range(6), imp, color=PALETTE[4])

# 12 predicted vs actual
ax = axes[11]
yt = rng.uniform(5, 80, 120)
ax.scatter(yt, yt + rng.normal(0, 5, 120), s=8, color=PALETTE[1])
ax.plot([5, 80], [5, 80], color="black", lw=1.5)

for ax, t in zip(axes, titles):
    ax.set_title(t, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
fig.suptitle("ML A-to-Z · Classical Machine Learning with scikit-learn & pandas", fontsize=16, fontweight="bold")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "assets" / "banner.png"
fig.savefig(out, dpi=110)
print(out)
