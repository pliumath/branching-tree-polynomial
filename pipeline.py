import os
import random
import time
import numpy as np
import dendropy
from scipy.signal import fftconvolve
from scipy.spatial.distance import cdist
from itertools import permutations

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
TARGET_TIPS = 500
DOWNSAMPLES_PER_TREE = 100

INPUT_FILES = [
    ('Hunt',     'Hunt.nwk',     'newick'),
    ('Wolf',     'Wolf.nwk',     'newick'),
    ('Novitsky', 'Novitsky.nex', 'nexus'),
]
UNIQUE_LABELS = ['Hunt', 'Wolf', 'Novitsky']
OUT = 'outputs'
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# Tree handling
# =============================================================================
def dendropy_to_dict_tree(dpy_tree):
    """Convert a dendropy.Tree to the dict-of-nodes representation."""
    nodes = list(dpy_tree.preorder_node_iter())
    idx_of = {id(nd): i for i, nd in enumerate(nodes)}
    out = []
    for i, nd in enumerate(nodes):
        parent = nd.parent_node
        out.append({
            'index': i,
            'parent': None if parent is None else idx_of[id(parent)],
            'tip': nd.is_leaf(),
            'branch length': float(nd.edge.length) if nd.edge.length else 0.0,
        })
    return out


def random_downsample(dpy_tree, k, rng):
    """Return a NEW dendropy.Tree restricted to k random leaves."""
    leaves = dpy_tree.leaf_nodes()
    if len(leaves) <= k:
        return dpy_tree.clone(depth=2)
    keep_labels = [nd.taxon.label for nd in rng.sample(leaves, k)]
    sub = dpy_tree.clone(depth=2)
    sub.retain_taxa_with_labels(set(keep_labels))
    sub.suppress_unifurcations()
    return sub


def normalize_branch_lengths(tree_dict):
    """Scale all branch lengths so their mean equals 1."""
    bls = [nd['branch length'] for nd in tree_dict if nd['parent'] is not None]
    mean_bl = float(np.mean(bls))
    out = []
    for nd in tree_dict:
        new = dict(nd)
        if nd['parent'] is not None:
            new['branch length'] = nd['branch length'] / mean_bl
        out.append(new)
    return out, mean_bl

# =============================================================================
# Polynomial computation (Liu 2021) via FFT
# =============================================================================
def tree_levels(tree):
    """Height (in nodes) of every node above the deepest leaf below it."""
    children = {}
    for nd in tree:
        children.setdefault(nd['parent'], []).append(nd['index'])
    levels = [None] * len(tree)
    order, stack = [], list(children.get(None, []))
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children.get(i, []))
    for i in reversed(order):
        kids = children.get(i, [])
        levels[i] = 0 if not kids else 1 + max(levels[k] for k in kids)
    return levels


def poly_prod_2_fft(p1, p2):
    """Multiply two bivariate polynomials via FFT, strip trailing zeros."""
    a = np.asarray(p1, dtype=np.float64)
    b = np.asarray(p2, dtype=np.float64)
    r = fftconvolve(a, b, mode='full')
    r[np.abs(r) < 1e-12] = 0.0
    while r.shape[0] > 1 and not np.any(r[-1]):
        r = r[:-1]
    while r.shape[1] > 1 and not np.any(r[:, -1]):
        r = r[:, :-1]
    return r


def poly_p(tree):
    """Bivariate tree-distinguishing polynomial of Liu (2021)."""
    polys = []
    for node in tree:
        if node['tip']:
            polys.append(np.array([[0.0, 0.0], [1.0, 0.0]]))
        else:
            polys.append(np.array([[1.0, 0.0]]))
    levels = tree_levels(tree)
    if max(levels) == 0:
        return polys
    children_of = {}
    for nd in tree:
        children_of.setdefault(nd['parent'], []).append(nd['index'])
    for level in range(max(levels)):
        for i in range(len(tree)):
            if levels[i] == level + 1:
                prod = polys[i]
                for child in children_of.get(i, []):
                    pc = polys[child].copy()
                    pc[0, 1] = pc[0, 1] + tree[child]['branch length']
                    prod = poly_prod_2_fft(prod, pc)
                polys[i] = prod
    return polys

# =============================================================================
# Clustering
# =============================================================================
def best_match_accuracy(predicted, truth, k):
    """Best label-permutation accuracy (predicted in 1..k)."""
    best = 0
    for perm in permutations(range(k)):
        remap = np.array([perm[p - 1] for p in predicted])
        best = max(best, (remap == truth).mean())
    return best


def kmedoids(D, k, max_iter=300, n_init=30, seed=42):
    """k-medoids on a precomputed distance matrix (PAM-style)."""
    rng = np.random.default_rng(seed)
    n = len(D)
    best_cost = np.inf
    best_labels = best_medoids = None
    for _ in range(n_init):
        medoids = rng.choice(n, size=k, replace=False)
        for _ in range(max_iter):
            assign = np.argmin(D[:, medoids], axis=1)
            new_medoids = []
            for c in range(k):
                members = np.where(assign == c)[0]
                if len(members) == 0:
                    new_medoids.append(medoids[c])
                else:
                    sub = D[np.ix_(members, members)]
                    new_medoids.append(members[np.argmin(sub.sum(axis=1))])
            new_medoids = np.array(new_medoids)
            if np.array_equal(np.sort(new_medoids), np.sort(medoids)):
                break
            medoids = new_medoids
        assign = np.argmin(D[:, medoids], axis=1)
        cost = sum(D[i, medoids[assign[i]]] for i in range(n))
        if cost < best_cost:
            best_cost = cost
            best_labels = assign
            best_medoids = medoids
    return best_labels, best_medoids, best_cost


# =============================================================================
# Helpers
# =============================================================================
def pad_and_flatten(polys):
    max_r = max(P.shape[0] for P in polys)
    max_c = max(P.shape[1] for P in polys)
    n = len(polys)
    flat = np.zeros((n, max_r * max_c), dtype=np.float64)
    for i, P in enumerate(polys):
        pad = np.zeros((max_r, max_c), dtype=np.float64)
        pad[:P.shape[0], :P.shape[1]] = P
        flat[i] = pad.ravel()
    return flat, max_r, max_c


# =============================================================================
# Main pipeline
# =============================================================================
def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print("Computing polynomials in two settings (this takes a few minutes)")
    print("=" * 72)

    polys_unscaled = []
    polys_normalized = []
    labels = []
    rep_ids = []
    mean_bls = []

    # cache parsed trees
    parsed_trees = []
    for name, path, fmt in INPUT_FILES:
        t = dendropy.Tree.get(path=path, schema=fmt)
        parsed_trees.append((name, t))
        print(f"  loaded {name}: {len(t.leaf_nodes())} tips")

    t_start = time.time()
    for name, t in parsed_trees:
        for r in range(DOWNSAMPLES_PER_TREE):
            # one subsample, used in BOTH settings
            sub = random_downsample(t, TARGET_TIPS, rng)
            td = dendropy_to_dict_tree(sub)
            td_norm, mean_bl = normalize_branch_lengths(td)

            P_u = np.asarray(poly_p(td)[0], dtype=np.float64)
            P_n = np.asarray(poly_p(td_norm)[0], dtype=np.float64)

            polys_unscaled.append(P_u)
            polys_normalized.append(P_n)
            labels.append(name)
            rep_ids.append(r)
            mean_bls.append(mean_bl)

            if (r + 1) % 20 == 0:
                elapsed = time.time() - t_start
                done = len(polys_unscaled)
                print(f"  {name:9s} rep {r+1:3d}/{DOWNSAMPLES_PER_TREE}  "
                      f"({done}/300 done, {elapsed:.0f}s elapsed)")

    print(f"\n  total: {time.time() - t_start:.1f} s, "
          f"{len(polys_unscaled)} polynomials per setting")

    n = len(polys_unscaled)
    true_labels = np.array([UNIQUE_LABELS.index(l) for l in labels])

    # -------------------------------------------------------------------------
    # Pad, flatten, compute distance matrices
    # -------------------------------------------------------------------------
    print("\nPadding + flattening polynomials...")
    flat_u, ru, cu = pad_and_flatten(polys_unscaled)
    flat_n, rn, cn = pad_and_flatten(polys_normalized)
    print(f"  unscaled    padded shape ({ru}, {cu}) -> flat {flat_u.nbytes/1e6:.0f} MB")
    print(f"  normalized  padded shape ({rn}, {cn}) -> flat {flat_n.nbytes/1e6:.0f} MB")

    print("\nComputing Canberra distance matrices...")
    t0 = time.time()
    D_u = cdist(flat_u, flat_u, metric='canberra')
    print(f"  unscaled    {D_u.shape}  {time.time()-t0:.1f}s")
    t0 = time.time()
    D_n = cdist(flat_n, flat_n, metric='canberra')
    print(f"  normalized  {D_n.shape}  {time.time()-t0:.1f}s")

    # release flat arrays
    del flat_u, flat_n

    # -------------------------------------------------------------------------
    # k-medoids
    # -------------------------------------------------------------------------
    print("\nk-medoids (k=3, PAM, 30 restarts)...")
    km_u, med_u, cost_u = kmedoids(D_u, k=3, n_init=30)
    km_n, med_n, cost_n = kmedoids(D_n, k=3, n_init=30)

    acc_u = best_match_accuracy(km_u + 1, true_labels, 3)
    acc_n = best_match_accuracy(km_n + 1, true_labels, 3)

    print(f"  unscaled    medoids = {sorted(med_u.tolist())}  "
          f"accuracy = {acc_u*100:.1f}%")
    print(f"  normalized  medoids = {sorted(med_n.tolist())}  "
          f"accuracy = {acc_n*100:.1f}%")

    # confusion matrices
    def confusion(predicted, truth, k):
        cm = np.zeros((k, k), dtype=int)
        # find best permutation
        best, best_perm = 0, None
        for perm in permutations(range(k)):
            remap = np.array([perm[p - 1] for p in predicted])
            acc = (remap == truth).mean()
            if acc > best:
                best, best_perm = acc, perm
        remap = np.array([best_perm[p - 1] for p in predicted])
        for t, p in zip(truth, remap):
            cm[t, p] += 1
        return cm

    cm_u = confusion(km_u + 1, true_labels, 3)
    cm_n = confusion(km_n + 1, true_labels, 3)
    print(f"\n  confusion (rows=truth, cols=predicted):")
    print(f"  unscaled:\n{cm_u}")
    print(f"  normalized:\n{cm_n}")

    # -------------------------------------------------------------------------
    # Within / between distance summary
    # -------------------------------------------------------------------------
    def within_between(D):
        within, between = {}, {}
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    within.setdefault(labels[i], []).append(D[i, j])
                else:
                    key = tuple(sorted([labels[i], labels[j]]))
                    between.setdefault(key, []).append(D[i, j])
        return within, between

    within_u, between_u = within_between(D_u)
    within_n, between_n = within_between(D_n)

    # -------------------------------------------------------------------------
    # Save artifacts
    # -------------------------------------------------------------------------
    print(f"\nWriting outputs to {OUT}/ ...")
    np.save(f'{OUT}/distance_matrix_unscaled.npy', D_u)
    np.save(f'{OUT}/distance_matrix_normalized.npy', D_n)
    np.savez_compressed(
        f'{OUT}/polynomials_unscaled.npz',
        **{f'{l}_{r}': P for l, r, P in zip(labels, rep_ids, polys_unscaled)}
    )
    # The normalized polynomial file is ~600 MB as a single npz. Some
    # filesystems / artifact stores reject files that large, so we split
    # by source tree (~200 MB each).
    for src in UNIQUE_LABELS:
        idx = [i for i, l in enumerate(labels) if l == src]
        np.savez_compressed(
            f'{OUT}/polynomials_normalized_{src}.npz',
            **{f'rep_{rep_ids[i]}': polys_normalized[i] for i in idx}
        )

    with open(f'{OUT}/labels.txt', 'w') as f:
        f.write('source\trep\tkm_unscaled\tkm_normalized\tmean_bl_orig\n')
        for i in range(n):
            f.write(f'{labels[i]}\t{rep_ids[i]}\t{km_u[i]}\t{km_n[i]}\t'
                    f'{mean_bls[i]:.6f}\n')

    with open(f'{OUT}/results_summary.txt', 'w') as f:
        f.write("HIV polynomial-clustering pipeline -- summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Subsamples per tree: {DOWNSAMPLES_PER_TREE}\n")
        f.write(f"Target tips         : {TARGET_TIPS}\n")
        f.write(f"Random seed         : {SEED}\n")
        f.write(f"Total polynomials   : {n} per setting\n\n")

        f.write("Polynomial sizes (after FFT trailing-zero strip):\n")
        for ul in UNIQUE_LABELS:
            idx = [i for i, l in enumerate(labels) if l == ul]
            yu = [polys_unscaled[i].shape[1] - 1 for i in idx]
            yn = [polys_normalized[i].shape[1] - 1 for i in idx]
            f.write(f"  {ul:9s}  unscaled y-deg = {np.mean(yu):.1f} "
                    f"+/- {np.std(yu):.1f}    "
                    f"normalized y-deg = {np.mean(yn):.1f} "
                    f"+/- {np.std(yn):.1f}\n")
        f.write("\n")

        f.write("Within / between Canberra distances (mean +/- sd):\n")
        for setting_name, w, b in [('UNSCALED', within_u, between_u),
                                    ('NORMALIZED', within_n, between_n)]:
            f.write(f"  -- {setting_name} --\n")
            for ul in UNIQUE_LABELS:
                v = w[ul]
                f.write(f"    within  {ul:9s}: {np.mean(v):10.1f} "
                        f"+/- {np.std(v):.1f}  (n={len(v)})\n")
            for k in [('Hunt','Wolf'),('Hunt','Novitsky'),('Wolf','Novitsky')]:
                v = b[tuple(sorted(k))]
                f.write(f"    between {k[0]:9s}-{k[1]:9s}: "
                        f"{np.mean(v):10.1f} +/- {np.std(v):.1f}  "
                        f"(n={len(v)})\n")
        f.write("\n")

        f.write("Clustering accuracy (k-medoids, k=3, PAM 30 restarts):\n")
        f.write(f"  unscaled   : {acc_u*100:.1f}%\n")
        f.write(f"  normalized : {acc_n*100:.1f}%\n\n")

        f.write("Confusion matrices (rows=truth Hunt/Wolf/Novitsky, "
                "cols=predicted after best label permutation):\n")
        f.write(f"  unscaled:\n")
        for r in cm_u:
            f.write("    " + "  ".join(f"{x:4d}" for x in r) + "\n")
        f.write(f"  normalized:\n")
        for r in cm_n:
            f.write("    " + "  ".join(f"{x:4d}" for x in r) + "\n")

    print(f"\nDone. All outputs in {OUT}/")
    print(f"To produce the comparison figure, run:  python make_figure.py")
    return D_u, D_n, km_u, km_n, acc_u, acc_n


if __name__ == '__main__':
    main()
