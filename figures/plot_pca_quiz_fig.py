import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Correlated 2D Gaussian cloud whose main axis points roughly along (2, 1).
angle = np.arctan2(1, 2)
rotation = np.array(
    [[np.cos(angle), -np.sin(angle)],
     [np.sin(angle), np.cos(angle)]]
)
n_samples = 300
cloud = rng.normal(size=(n_samples, 2)) * np.array([2.0, 0.9])
cloud = cloud @ rotation.T

# Candidate vectors shown to the student (letter, x, y, color).
arrows = [
    ("A", -1, 2, "tab:blue"),
    ("B", 0, 2, "tab:orange"),
    ("C", 2, 1, "tab:red"),
    ("D", 2, 0, "tab:green"),
]

plt.figure(figsize=(7, 5.5))
plt.scatter(cloud[:, 0], cloud[:, 1], alpha=0.4, edgecolors="none")

for label, x, y, color in arrows:
    plt.annotate(
        "",
        xy=(x, y),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=color, lw=2.5),
    )
    plt.annotate(
        label,
        xy=(x, y),
        color=color,
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", fc="white", ec=color),
    )

plt.title("Principal Component Analysis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(-5.5, 6)
plt.ylim(-4, 4)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pca_cloud_quiz.png", dpi=100)
plt.show()


from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

feature_names = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium"]
X = load_wine(as_frame=True).data[feature_names]

n_components = 3
pc_labels = [f"PC{i + 1}" for i in range(n_components)]

pca_raw = PCA(n_components=n_components).fit(X)
pca_scaled = make_pipeline(StandardScaler(), PCA(n_components=n_components)).fit(X)

loadings = {
    "Pipeline A": np.abs(pca_raw.components_),
    "Pipeline B": np.abs(pca_scaled[-1].components_),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (title, matrix) in zip(axes, loadings.items()):
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=30, ha="right")
    ax.set_yticks(range(n_components))
    ax.set_yticklabels(pc_labels)
    for i in range(n_components):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Loading weight")

fig.tight_layout()
fig.savefig("pca_heatmap.png", dpi=100)
plt.show()
