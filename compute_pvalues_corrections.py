"""
compute_pvalues_corrections.py

Computes uncorrected p-values from state-conditioned decoding permutation results,
then applies two multiple comparison correction methods:
    1. Cluster-based correction (non-parametric, controls FWER over time)
    2. Benjamini-Yekutieli (BY) correction (controls FDR under arbitrary dependence)

The key statistic at each timepoint is:
    delta = within-state accuracy - cross-state accuracy

The p-value tests whether delta_HMM > delta_random (one-tailed).

Convention: column 0 of acc arrays = real HMM run, columns 1: = permutations.

Usage:
    python compute_pvalues_corrections.py <mouse_name>
"""

import numpy as np
import pickle
import os
import sys
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

mouse_name_list = ['Buchanan', 'Stella', 'Mitt', 'Barat']

BASE_RESULTS_FOLDER = '/home/administrator/hippocampus_Cooper_Fortin/results'

# Cluster-based correction parameters
CLUSTER_FORMING_ALPHA = 0.05   # uncorrected p-value threshold to form clusters
CLUSTER_ALPHA         = 0.05   # significance threshold for cluster mass test

# Time axis for plotting
FS = 250                        # Hz (after downsampling)
TIME_WINDOW_MS = (-1000, 2000)    # ms around response

# ─────────────────────────────────────────────────────────────────────────────
# Core statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta(acc_within, acc_cross):
    """delta[t, perm] = within[t, perm] - cross[t, perm]"""
    return acc_within - acc_cross


def compute_uncorrected_pvalues(delta):
    """
    One-tailed p-value at each timepoint:
        p[t] = proportion of permutations where delta_perm[t] >= delta_HMM[t]

    Parameters
    ----------
    delta : (n_timepoints, n_permutations)
            column 0 = real HMM, columns 1: = shuffled permutations

    Returns
    -------
    p_uncorrected : (n_timepoints,)
    delta_hmm     : (n_timepoints,)
    delta_perm    : (n_timepoints, n_perm)  — permutation columns only
    """
    delta_hmm  = delta[:, 0]
    delta_perm = delta[:, 1:]

    # fraction of permutations at least as extreme as the real observation
    p_uncorrected = np.mean(delta_perm >= delta_hmm[:, np.newaxis], axis=1)

    return p_uncorrected, delta_hmm, delta_perm


# ─────────────────────────────────────────────────────────────────────────────
# Cluster-based correction
# ─────────────────────────────────────────────────────────────────────────────

def cluster_based_correction(p_uncorrected, delta_hmm, delta_perm,
                              forming_alpha=CLUSTER_FORMING_ALPHA,
                              cluster_alpha=CLUSTER_ALPHA):
    """
    Non-parametric cluster-based correction controlling FWER over time.

    The procedure follows Maris & Oostenveld (2007):

    Step 1 — Form clusters in the REAL data
        Threshold the uncorrected p-values at `forming_alpha`.
        Find contiguous runs of significant timepoints → these are clusters.
        Compute each cluster's mass = sum of delta_hmm within the cluster.

    Step 2 — Build null distribution of maximum cluster mass
        For each permutation p:
            - Treat delta_perm[:, p] as the "observed" statistic under the null.
            - At each timepoint, compute a p-value for this permutation's delta
              against ALL OTHER permutations (leave-one-out).
            - Threshold those p-values at `forming_alpha` to find null clusters.
            - Record the maximum cluster mass across all null clusters.
        This gives a null distribution of max cluster masses.

    Step 3 — Threshold real clusters
        A real cluster is significant if its mass > 95th percentile of the null.
        Equivalently, cluster p-value = fraction of null masses >= cluster mass.


    Parameters
    ----------
    p_uncorrected : (n_timepoints,)  — uncorrected p-values from real data
    delta_hmm     : (n_timepoints,)  — real within-cross delta
    delta_perm    : (n_timepoints, n_perm)  — permutation deltas
    forming_alpha : float  — threshold to form clusters (typically 0.05)
    cluster_alpha : float  — significance level for cluster mass test

    Returns
    -------
    significant   : (n_timepoints,) bool — timepoints in significant clusters
    cluster_pvals : dict  cluster_id -> p-value
    null_dist     : (n_perm,)  null distribution of max cluster masses
    """
    n_timepoints, n_perm = delta_perm.shape
    nan_mask = np.isnan(p_uncorrected)

    # ── Step 1: clusters in real data ─────────────────────────────────────────
    above_thresh = (~nan_mask) & (p_uncorrected < forming_alpha)
    real_clusters = _find_clusters(above_thresh)

    real_cluster_masses = {
        cid: np.sum(delta_hmm[tidx])
        for cid, tidx in real_clusters.items()
    }

    # ── Step 2: null distribution ─────────────────────────────────────────────
    #
    # For each permutation p, we need a p-value at each timepoint so we can
    # apply the same forming_alpha threshold and find clusters.
    #
    # The p-value for permutation p at timepoint t is:
    #   "how extreme is delta_perm[t, p] relative to the other permutations?"
    # We use leave-one-out (exclude permutation p itself) to avoid circularity.
    #
    null_max_masses = np.zeros(n_perm)

    for p in range(n_perm):
        perm_delta = delta_perm[:, p]                        # (n_timepoints,)

        # p-value of this permutation's delta against all OTHER permutations
        other = np.delete(delta_perm.copy(), p, axis=1)             # (n_timepoints, n_perm-1)
        perm_p = np.mean(other >= perm_delta[:, np.newaxis], axis=1)
        perm_p[nan_mask] = np.nan

        # find clusters in this permutation using same forming threshold
        perm_above    = (~nan_mask) & (perm_p < forming_alpha)
        perm_clusters = _find_clusters(perm_above)

        if len(perm_clusters) == 0:
            null_max_masses[p] = 0.0
        else:
            masses = [np.sum(perm_delta[tidx]) for tidx in perm_clusters.values()]
            null_max_masses[p] = np.max(masses)

    # ── Step 3: assess real clusters against null ─────────────────────────────
    significant   = np.zeros(n_timepoints, dtype=bool)
    cluster_pvals = {}

    for cid, tidx in real_clusters.items():
        mass  = real_cluster_masses[cid]
        p_val = np.mean(null_max_masses >= mass)   # fraction of null >= observed
        cluster_pvals[cid] = p_val
        if p_val < cluster_alpha:
            significant[tidx] = True

    return significant, cluster_pvals, null_max_masses


def _find_clusters(binary_mask):
    """
    Find contiguous runs of True in a 1D boolean array.

    Returns
    -------
    clusters : dict  cluster_id (int) -> np.array of timepoint indices
    """
    clusters   = {}
    cid        = 0
    in_cluster = False

    for t, val in enumerate(binary_mask):
        if val and not in_cluster:
            in_cluster = True
            start = t
        elif not val and in_cluster:
            in_cluster = False
            clusters[cid] = np.arange(start, t)
            cid += 1

    if in_cluster:                                 # cluster reaching the end
        clusters[cid] = np.arange(start, len(binary_mask))

    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# Benjamini-Yekutieli correction
# ─────────────────────────────────────────────────────────────────────────────

def benjamini_yekutieli_correction(p_values, alpha=0.05):
    """
    BY correction — valid under arbitrary dependence between tests.
    Appropriate here because adjacent timepoints are temporally autocorrelated,
    which violates the independence assumption of standard BH.

    Parameters
    ----------
    p_values : (n_timepoints,)
    alpha    : FDR level

    Returns
    -------
    rejected    : (n_timepoints,) bool
    p_corrected : (n_timepoints,) BY-adjusted p-values
    """
    nan_mask = np.isnan(p_values)
    p_fill   = p_values.copy()
    p_fill[nan_mask] = 1.0

    rejected, p_corrected, _, _ = multipletests(p_fill, alpha=alpha, method='fdr_by')

    rejected[nan_mask]    = False
    p_corrected[nan_mask] = np.nan

    return rejected, p_corrected


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(acc_within_hmm, acc_cross_hmm,
                 p_uncorrected, sig_uncorrected,
                 sig_by, sig_cluster,
                 mouse_name, save_path=None):
    """
    Three-panel figure, one panel per correction method.
    Each panel shows:
        - within-state decoding accuracy (solid)
        - cross-state decoding accuracy (dashed)
        - shaded regions where within > cross is significant

    Parameters
    ----------
    acc_within_hmm  : (n_timepoints,)  real HMM within-state accuracy
    acc_cross_hmm   : (n_timepoints,)  real HMM cross-state accuracy
    p_uncorrected   : (n_timepoints,)  uncorrected p-values
    sig_uncorrected : (n_timepoints,)  bool, uncorrected significant timepoints
    sig_by          : (n_timepoints,)  bool, BY-significant timepoints
    sig_cluster     : (n_timepoints,)  bool, cluster-significant timepoints
    """
    n_tp   = len(acc_within_hmm)
    time_s = np.linspace(TIME_WINDOW_MS[0], TIME_WINDOW_MS[1], n_tp)

    COLOR_WITHIN  = '#2166ac'   # blue
    COLOR_CROSS   = '#d6604d'   # red-orange
    COLOR_SHADE   = '#fee090'   # yellow shading for significance
    SHADE_ALPHA   = 0.45

    panel_configs = [
        ('Uncorrected (p < 0.05)',            sig_uncorrected),
        ('Benjamini-Yekutieli corrected',      sig_by),
        ('Cluster-based corrected',            sig_cluster),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    fig.suptitle(f'State-conditioned decoding accuracy — {mouse_name}',
                 fontsize=13, fontweight='bold', y=0.98)

    for ax, (title, sig_mask) in zip(axes, panel_configs):

        # significance shading first (so curves sit on top)
        _shade_significant(ax, time_s, sig_mask,
                           color=COLOR_SHADE, alpha=SHADE_ALPHA,
                           label='p < 0.05 (significant)')

        # accuracy curves
        ax.plot(time_s, acc_within_hmm, color=COLOR_WITHIN, lw=2,
                label='Within-state accuracy')
        ax.plot(time_s, acc_cross_hmm,  color=COLOR_CROSS,  lw=2,
                ls='--', label='Cross-state accuracy')

        # reference lines
        ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.7)   # chance
        ax.axvline(0,   color='black', lw=1.0, ls=':', alpha=0.5)   # response

        ax.set_ylabel('Accuracy', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.set_ylim([0.35, 1.05])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # legend only on top panel to avoid repetition
        if ax is axes[0]:
            ax.legend(fontsize=9, loc='upper left', framealpha=0.9)

    axes[-1].set_xlabel('Time relative to response (ms)', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved to {save_path}')
    else:
        plt.show()
    plt.close()


def _shade_significant(ax, time_s, sig_mask, color, alpha, label):
    """Shade contiguous runs of significant timepoints."""
    clusters = _find_clusters(sig_mask)
    for i, (_, tidx) in enumerate(clusters.items()):
        ax.axvspan(time_s[tidx[0]], time_s[tidx[-1]],
                   color=color, alpha=alpha,
                   label=label if i == 0 else None)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

for mouse_name in mouse_name_list

    results_folder = os.path.join(BASE_RESULTS_FOLDER, mouse_name, 'review_analyses')

    result_files = [f for f in os.listdir(results_folder)
                    if f.startswith('NO_LOCO_stand_rand_deco_pos') and f.endswith('.pkl')]

    if len(result_files) == 0:
        raise FileNotFoundError(f'No results file found in {results_folder}')
    if len(result_files) > 1:
        print(f'Warning: multiple results files found, using {result_files[0]}')

    fpath = os.path.join(results_folder, result_files[0])
    print(f'Loading {fpath}')

    with open(fpath, 'rb') as fp:
        results = pickle.load(fp)

    acc_within = results['acc_per_state_general']     # (n_tp, n_perm)
    acc_cross  = results['acc_cross_state_general']   # (n_tp, n_perm)

    n_perm = acc_within.shape[1] - 1
    print(f'Shape: {acc_within.shape} — {n_perm} permutations + 1 real HMM run')

    # Real HMM accuracy curves (column 0)
    acc_within_hmm = acc_within[:, 0]
    acc_cross_hmm  = acc_cross[:, 0]

    # ── Compute delta and uncorrected p-values ────────────────────────────────
    delta                            = compute_delta(acc_within, acc_cross)
    p_uncorr, delta_hmm, delta_perm  = compute_uncorrected_pvalues(delta)

    sig_uncorr = (~np.isnan(p_uncorr)) & (p_uncorr < 0.05)
    print(f'\nUncorrected:   {np.sum(sig_uncorr)} / {len(p_uncorr)} timepoints p < 0.05')

    # ── Benjamini-Yekutieli ───────────────────────────────────────────────────
    sig_by, p_by = benjamini_yekutieli_correction(p_uncorr, alpha=0.05)
    print(f'BY corrected:  {np.sum(sig_by)} / {len(sig_by)} timepoints significant')

    # ── Cluster-based ─────────────────────────────────────────────────────────
    print('Running cluster-based correction (may take a moment with 10k perms)...')
    sig_cluster, cluster_pvals, null_dist = cluster_based_correction(
        p_uncorr, delta_hmm, delta_perm
    )
    print(f'Cluster corr:  {np.sum(sig_cluster)} / {len(sig_cluster)} timepoints significant')
    for cid, pval in cluster_pvals.items():
        print(f'  Cluster {cid}: mass p = {pval:.4f}')

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'acc_within_hmm': acc_within_hmm,
        'acc_cross_hmm' : acc_cross_hmm,
        'delta_hmm'     : delta_hmm,
        'delta_perm'    : delta_perm,
        'p_uncorrected' : p_uncorr,
        'sig_uncorrected': sig_uncorr,
        'p_by'          : p_by,
        'sig_by'        : sig_by,
        'sig_cluster'   : sig_cluster,
        'cluster_pvals' : cluster_pvals,
        'null_dist'     : null_dist,
    }
    out_path = os.path.join(results_folder, f'{mouse_name}_pvalues_corrections_new.pkl')
    with open(out_path, 'wb') as fp:
        pickle.dump(output, fp, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'\nResults saved to {out_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig_path = os.path.join(results_folder, f'{mouse_name}_pvalues_corrections_new.png')
    plot_results(acc_within_hmm, acc_cross_hmm,
                 p_uncorr, sig_uncorr,
                 sig_by, sig_cluster,
                 mouse_name, save_path=fig_path)

"""
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python compute_pvalues_corrections.py <mouse_name>')
        print(f'Available: {mouse_name_list}')
        sys.exit(1)

    mouse = sys.argv[1]
    if mouse not in mouse_name_list:
        print(f'Unknown mouse "{mouse}". Available: {mouse_name_list}')
        sys.exit(1)

    run(mouse)
    
"""
