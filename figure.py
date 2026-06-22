import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from itertools import permutations

# ---- font: Palatino with sensible fallbacks ------------------------------
# 'Palatino'          -- macOS native
# 'Palatino Linotype' -- Windows native
# 'TeX Gyre Pagella'  -- free Palatino-clone (Linux / TeX distributions)
# 'URW Palladio L'    -- ghostscript Palatino-clone

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [
    'Palatino', 'Palatino Linotype', 'Palatino LT Std',
    'URW Palladio L', 'TeX Gyre Pagella', 'P052', 'serif',
]
plt.rcParams['mathtext.fontset'] = 'stix'   # matching serif math

OUT = 'outputs'
UNIQUE_LABELS = ['Hunt', 'Wolf', 'Novitsky']

# MATLAB modern default line colors (R2014b+)
MATLAB_BLUE   = '#0072BD'
MATLAB_ORANGE = '#D95319'
MATLAB_YELLOW = '#EDB120'

COLOURS = {
    'Hunt':     MATLAB_BLUE,
    'Wolf':     MATLAB_ORANGE,
    'Novitsky': MATLAB_YELLOW,
}
MARKERS = {'Hunt': 'o', 'Wolf': 's', 'Novitsky': '^'}

# Confusion-matrix colormap: white -> MATLAB blue
CONF_CMAP = LinearSegmentedColormap.from_list(
    'matlab_blue_seq', ['#FFFFFF', MATLAB_BLUE]
)


# -------------------------------------------------------------------------
# Load
# -------------------------------------------------------------------------
D_u = np.load(f'{OUT}/distance_matrix_unscaled.npy')
D_n = np.load(f'{OUT}/distance_matrix_normalized.npy')

labels, rep_ids, km_u, km_n = [], [], [], []
with open(f'{OUT}/labels.txt') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        labels.append(parts[0]); rep_ids.append(int(parts[1]))
        km_u.append(int(parts[2])); km_n.append(int(parts[3]))

labels = np.array(labels)
km_u, km_n = np.array(km_u), np.array(km_n)
true_labels = np.array([UNIQUE_LABELS.index(l) for l in labels])
n = len(labels)


# -------------------------------------------------------------------------
# Best label permutation -> confusion matrix
# -------------------------------------------------------------------------
def best_permutation(predicted, truth, k):
    best_acc, best_perm = -1, None
    for perm in permutations(range(k)):
        remap = np.array([perm[p] for p in predicted])
        acc = (remap == truth).mean()
        if acc > best_acc:
            best_acc, best_perm = acc, perm
    return best_perm, best_acc


def confusion_matrix(predicted, truth, k, perm):
    remap = np.array([perm[p] for p in predicted])
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(truth, remap):
        cm[t, p] += 1
    return cm


perm_u, acc_u = best_permutation(km_u, true_labels, 3)
perm_n, acc_n = best_permutation(km_n, true_labels, 3)
cm_u = confusion_matrix(km_u, true_labels, 3, perm_u)
cm_n = confusion_matrix(km_n, true_labels, 3, perm_n)

# misclassified mask
remap_u = np.array([perm_u[p] for p in km_u])
remap_n = np.array([perm_n[p] for p in km_n])
mis_u = remap_u != true_labels
mis_n = remap_n != true_labels


# -------------------------------------------------------------------------
# MDS
# -------------------------------------------------------------------------
print("Fitting MDS...")
mds_u = MDS(n_components=2, dissimilarity='precomputed',
            random_state=0, n_init=4, max_iter=300,
            normalized_stress='auto', init='random').fit_transform(D_u)
mds_n = MDS(n_components=2, dissimilarity='precomputed',
            random_state=0, n_init=4, max_iter=300,
            normalized_stress='auto', init='random').fit_transform(D_n)


# -------------------------------------------------------------------------
# Plot helpers
# -------------------------------------------------------------------------
def plot_confusion(ax, cm, title, accuracy):
    """3x3 confusion matrix drawn as vector Rectangles (no imshow)."""
    from matplotlib.patches import Rectangle
    k = cm.shape[0]
    norm_cm = cm / cm.sum(axis=1, keepdims=True)
    for i in range(k):
        for j in range(k):
            val = cm[i, j]
            face = CONF_CMAP(norm_cm[i, j])
            ax.add_patch(Rectangle(
                (j - 0.5, i - 0.5), 1.0, 1.0,
                facecolor=face, edgecolor='white', linewidth=0.6,
            ))
            colour = 'white' if norm_cm[i, j] > 0.5 else '#333333'
            ax.text(j, i, f'{val}', ha='center', va='center',
                    color=colour, fontsize=13, fontweight='bold')
    # set ranges so rectangles fill the axes; origin at top-left, square cells
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(UNIQUE_LABELS, fontsize=12)
    ax.set_yticklabels(UNIQUE_LABELS, fontsize=12)
    ax.set_xlabel('predicted (k-medoids cluster)', fontsize=12)
    ax.set_ylabel('true source tree', fontsize=12)
    ax.set_title(f'{title}\naccuracy = {accuracy*100:.1f}%',
                 fontsize=13)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_mds(ax, emb, mis, title):
    """MDS scatter coloured by source tree; misclassified points ringed."""
    # correctly classified points first
    for ul in UNIQUE_LABELS:
        mask = (labels == ul) & ~mis
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=COLOURS[ul], marker=MARKERS[ul],
                   s=36, edgecolor='#444444', linewidth=0.5,
                   alpha=0.85, label=f'{ul} (n={(labels==ul).sum()})')
    # misclassified points on top, thick black ring
    if mis.any():
        for ul in UNIQUE_LABELS:
            mask = (labels == ul) & mis
            if mask.any():
                ax.scatter(emb[mask, 0], emb[mask, 1],
                           c=COLOURS[ul], marker=MARKERS[ul],
                           s=110, edgecolor='black', linewidth=2.0,
                           zorder=10)
        # add one legend entry for the highlight style
        ax.scatter([], [], facecolor='lightgray', marker='o',
                   s=110, edgecolor='black', linewidth=2.0,
                   label=f'misclassified (n={int(mis.sum())})')

    leg = ax.legend(loc='lower left', fontsize=11,
                    facecolor='white', framealpha=0.7,
                    edgecolor='#888888')
    leg.get_frame().set_linewidth(0.6)

    ax.set_xlabel('MDS dim 1', fontsize=12)
    ax.set_ylabel('MDS dim 2', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.25)

    # tick label size
    ax.tick_params(axis='both', labelsize=11)

    # scientific notation on both axes -- the raw labels (e.g. -200000)
    # are wide and were pushing the y-axis label into the confusion matrix.
    ax.ticklabel_format(style='sci', axis='both',
                        scilimits=(0, 0), useMathText=True)
    # the "x10^N" offset text
    ax.xaxis.get_offset_text().set_size(10)
    ax.yaxis.get_offset_text().set_size(10)


# -------------------------------------------------------------------------
# Figure
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11.5, 10),
                         gridspec_kw={'width_ratios': [0.6, 1],
                                      'wspace': 0.22, 'hspace': 0.34})

settings = [
    ('unscaled (original branch lengths)',
     mds_u, mis_u, cm_u, acc_u),
    ('normalized (mean branch length = 1)',
     mds_n, mis_n, cm_n, acc_n),
]

# panel labels in reading order: top-left, top-right, bottom-left, bottom-right
panel_labels = [['A', 'B'], ['C', 'D']]

mds_titles = [
    'MDS — unscaled subsampled trees; markers indicate source tree',
    'MDS — normalized subsampled trees; markers indicate source tree',
]

for row, (ttl, mds_emb, mis, cm, acc) in enumerate(settings):
    plot_confusion(axes[row, 0], cm, ttl, acc)
    plot_mds(axes[row, 1], mds_emb, mis, title=mds_titles[row])

# place A / B / C / D near the upper-left corner of each panel
# (axes-relative coordinates), at a font size only slightly larger
# than the axis titles inside the panel.
for r in range(2):
    for c in range(2):
        x = -0.18 if c == 0 else -0.14
        axes[r, c].text(
            x, 1.03,
            panel_labels[r][c],
            transform=axes[r, c].transAxes,
            fontsize=13, fontweight='bold',
            ha='left', va='bottom',
        )

fig.suptitle('HIV trees: polynomial Canberra clustering '
             '(100 subsamples per source tree, k-medoids only)',
             y=0.995, fontsize=15, fontweight='bold')

for ext, kwargs in [('pdf', dict()), ('svg', dict()), ('png', dict(dpi=600))]:
    path = f'{OUT}/comparison_figure.{ext}'
    plt.savefig(path, bbox_inches='tight', **kwargs)
    size_kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}  ({size_kb:.0f} KB)")

plt.close()

print(f"\nunscaled   accuracy {acc_u*100:.1f}%   misclassified {int(mis_u.sum())}/300")
print(f"normalized accuracy {acc_n*100:.1f}%   misclassified {int(mis_n.sum())}/300")
