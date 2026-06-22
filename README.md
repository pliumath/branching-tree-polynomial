# Polynomial encoding of rooted trees with branch lengths

Code and data accompanying the paper *"Polynomial encoding of rooted trees with
branch lengths"*. The pipeline computes a bivariate tree-distinguishing
polynomial for each of three HIV-1 phylogenies, clusters the resulting
coefficient matrices using a Canberra distance, and reports the recovered
source-tree labels.

## Contents

| File              | Description                                                   |
|-------------------|---------------------------------------------------------------|
| `pipeline.py`     | Subsamples each input tree to 500 leaves (100 replicates per source tree), computes the polynomial under two settings (original and mean-branch-length-normalized branch lengths), builds the 300 × 300 Canberra distance matrices, and clusters with k-medoids (PAM, *k* = 3). |
| `figure.py`       | Reads the outputs of `pipeline.py` and produces the comparison figure (confusion matrices + MDS plots) in PDF, SVG, and PNG. |
| `Hunt.nwk`        | HIV-1 subtype C phylogeny, South Africa (Hunt et al., 2012). |
| `Wolf.nwk`        | HIV-1 subtype B phylogeny, Seattle (Wolf et al., 2017).      |
| `Novitsky.nex`    | HIV-1 subtype C phylogeny, Mochudi, Botswana (Novitsky et al., 2013). |

All three input trees are midpoint-rooted and have branch lengths in units of
expected nucleotide substitutions per site, as inferred by RAxML under a
GTR + CAT model.

## Requirements

Python ≥ 3.8 with:

- `numpy`
- `scipy`
- `dendropy`
- `scikit-learn`
- `matplotlib`

Install everything with:

```bash
pip install numpy scipy dendropy scikit-learn matplotlib
```

Or, recommended, into a virtual environment:

```bash
python -m venv venv
source venv/bin/activate              # on Windows: venv\Scripts\activate
pip install numpy scipy dendropy scikit-learn matplotlib
```

## Usage

Run from the repository root, with the three tree files in the same directory:

```bash
python pipeline.py    # ~3 min on a typical laptop; writes outputs/
python figure.py      # ~10 s; reads outputs/, writes outputs/comparison_figure.{pdf,svg,png}
```

## Outputs

All files are written to `outputs/`.

| File                                       | Description                                                |
|--------------------------------------------|------------------------------------------------------------|
| `distance_matrix_unscaled.npy`             | 300 × 300 Canberra distance matrix, original branch lengths. |
| `distance_matrix_normalized.npy`           | 300 × 300 Canberra distance matrix, mean branch length = 1. |
| `polynomials_unscaled.npz`                 | 300 root coefficient matrices (key format: `{source}_{rep}`). |
| `polynomials_normalized_{Hunt,Wolf,Novitsky}.npz` | 100 normalized coefficient matrices per source tree (key format: `rep_{i}`). |
| `labels.txt`                               | Source tree, replicate id, k-medoids cluster labels (both settings), and original mean branch length per subsample. |
| `results_summary.txt`                      | Within/between-tree distance summaries, confusion matrices, clustering accuracy. |
| `comparison_figure.{pdf,svg,png}`          | Confusion matrices + MDS visualizations.                    |

## Reproducibility

The pipeline uses a fixed seed (`SEED = 42` in `pipeline.py`). Subsampling and
k-medoids initialization are bit-deterministic across machines.

The polynomial multiplication uses FFT (`scipy.signal.fftconvolve`), whose
summation order depends on the BLAS backend (OpenBLAS / MKL / Apple
Accelerate) and the active thread count. This causes the coefficient
matrices to differ at the ~10⁻¹⁵ relative level across machines, which can
occasionally flip the assignment of a handful of border-case trees between
neighbouring clusters in the normalized setting. The qualitative result
(clean separation of all three sources) is unaffected. For bit-identical
results, set `OPENBLAS_NUM_THREADS=1` before running.

## Citation

If you use this code, please cite:

> [paper citation TBD]

The three HIV-1 datasets should additionally be cited:

- Wolf E. *et al.* AIDS Res. Hum. Retroviruses, 33(4):318–322, 2017.
- Hunt G.M. *et al.* Clin. Infect. Dis., 54(Suppl 4):S334–S338, 2012.
- Novitsky V. *et al.* PLoS One, 8(12):e80589, 2013.

## License

[license TBD — MIT or BSD-3-Clause recommended for code; the HIV trees were
released by their original authors and remain under their respective terms.]
