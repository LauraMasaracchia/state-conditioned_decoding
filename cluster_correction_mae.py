"""
Cluster-based permutation correction for MAE-based decoding data.

Input shapes:
    obs_mae  : (n_neurons, n_timepoints)           — observed mean absolute error
    null_mae : (n_neurons, n_timepoints, n_perms)  — MAE under label shuffling

Logic: lower MAE = better prediction. A point is "significant" when the
observed MAE is surprisingly LOW compared to the null distribution.
p-value = fraction of permutations with MAE <= observed MAE (left tail).

Cluster mass = sum of (null_mean_mae - obs_mae) within the cluster.
This measures how much each point beats its local null, making the
statistic sensitive to effect strength rather than just cluster size.
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────────────────────
# 1. P-VALUES  (left-tailed for MAE)
# ─────────────────────────────────────────────────────────────

def compute_pvalues_mae(obs_mae, null_mae):
    """
    One-tailed (left) p-values: how often is null MAE <= observed MAE?

    A small p-value means the observed MAE is lower than almost all
    permutations — i.e. the prediction is genuinely better than chance.

    Parameters
    ----------
    obs_mae  : (n_neurons, n_timepoints)
    null_mae : (n_neurons, n_timepoints, n_perms)

    Returns
    -------
    pvalues : (n_neurons, n_timepoints)  in [0, 1]
    """
    n_perms = null_mae.shape[2]
    pvalues = (null_mae <= obs_mae[:, :, np.newaxis]).sum(axis=2) / n_perms
    return pvalues


# ─────────────────────────────────────────────────────────────
# 2. CLUSTER FINDING
# ─────────────────────────────────────────────────────────────

def find_clusters(score_map, above_thresh_mask, adjacency='time_only'):
    """
    Find spatiotemporal clusters.

    Parameters
    ----------
    score_map         : (n_neurons, n_timepoints)  values used for cluster mass
    above_thresh_mask : (n_neurons, n_timepoints)  bool, True where p < threshold
    adjacency         : 'time_only' — clusters span adjacent timepoints per neuron
                        'full'      — clusters also span adjacent neurons

    Returns
    -------
    cluster_labels : (n_neurons, n_timepoints) int, 0 = not in any cluster
    cluster_ids    : list of ints
    """
    if adjacency == 'time_only':
        cluster_labels = np.zeros_like(score_map, dtype=int)
        current_id = 1
        for n in range(score_map.shape[0]):
            labeled, num = ndimage.label(above_thresh_mask[n])
            for c in range(1, num + 1):
                cluster_labels[n, labeled == c] = current_id
                current_id += 1
    elif adjacency == 'full':
        structure = np.ones((3, 3), dtype=int)
        cluster_labels, _ = ndimage.label(above_thresh_mask, structure=structure)
    else:
        raise ValueError("adjacency must be 'time_only' or 'full'")

    cluster_ids = [c for c in np.unique(cluster_labels) if c != 0]
    return cluster_labels, cluster_ids


def cluster_mass(score_map, cluster_labels, cluster_id):
    """Sum of score values within a cluster."""
    return score_map[cluster_labels == cluster_id].sum()


# ─────────────────────────────────────────────────────────────
# 3. MAIN CORRECTION PIPELINE
# ─────────────────────────────────────────────────────────────

def cluster_based_correction_mae(obs_mae, null_mae,
                                  alpha=0.05,
                                  cluster_forming_p=0.05,
                                  adjacency='time_only',
                                  verbose=True):
    """
    Cluster-based permutation correction for MAE decoding data.

    The cluster mass statistic is the sum of (null_mean - obs_mae) within
    each cluster. This is positive where the observed MAE beats the null,
    and scales with effect strength rather than just cluster size.

    Parameters
    ----------
    obs_mae           : (n_neurons, n_timepoints)
    null_mae          : (n_neurons, n_timepoints, n_perms)
    alpha             : cluster-level significance threshold
    cluster_forming_p : uncorrected p threshold to form clusters
    adjacency         : 'time_only' or 'full'
    verbose           : print progress

    Returns
    -------
    dict with keys:
        pvalues_uncorrected   : (n_neurons, n_timepoints)
        cluster_labels        : (n_neurons, n_timepoints)
        cluster_pvalues       : {cluster_id: p_value}
        significant_mask      : (n_neurons, n_timepoints) bool
        null_max_cluster_mass : (n_perms,)
    """
    n_neurons, n_timepoints, n_perms = null_mae.shape

    # Score per point = how much obs beats the local null mean (positive = good)
    null_mean  = null_mae.mean(axis=2)          # (n_neurons, n_timepoints)
    obs_score  = null_mean - obs_mae            # observed score
    # For each permutation: score = null_mean - perm_mae
    null_score = null_mean[:, :, np.newaxis] - null_mae  # (n, t, perms)

    # ── Step 1: uncorrected p-values ──
    pvalues = compute_pvalues_mae(obs_mae, null_mae)

    # ── Step 2: threshold mask ──
    above_thresh_obs = pvalues < cluster_forming_p

    if verbose:
        n_sig = above_thresh_obs.sum()
        total = n_neurons * n_timepoints
        print(f"Points below cluster-forming threshold "
              f"(p<{cluster_forming_p}): {n_sig}/{total} "
              f"({100*n_sig/total:.1f}%)")

    # ── Step 3: find clusters in observed data ──
    cluster_labels_obs, cluster_ids_obs = find_clusters(
        obs_score, above_thresh_obs, adjacency=adjacency
    )

    if verbose:
        print(f"Clusters found in observed data: {len(cluster_ids_obs)}")

    # ── Step 4: null distribution of max cluster mass ──
    null_max_cluster_mass = np.zeros(n_perms)

    for perm_idx in range(n_perms):
        perm_mae   = null_mae[:, :, perm_idx]
        perm_score = null_score[:, :, perm_idx]

        perm_ranks = (null_mae <= perm_mae[:, :, np.newaxis]).sum(axis=2)
        perm_pvals = perm_ranks / n_perms
        perm_above = perm_pvals < cluster_forming_p

        perm_labels, perm_ids = find_clusters(perm_score, perm_above, adjacency)

        if perm_ids:
            masses = [cluster_mass(perm_score, perm_labels, cid)
                      for cid in perm_ids]
            null_max_cluster_mass[perm_idx] = max(masses)

        if verbose and (perm_idx + 1) % 1000 == 0:
            print(f"  Permutation {perm_idx+1}/{n_perms}...")

    # ── Step 5: assign corrected p-values ──
    cluster_pvalues  = {}
    significant_mask = np.zeros((n_neurons, n_timepoints), dtype=bool)

    for cid in cluster_ids_obs:
        obs_mass = cluster_mass(obs_score, cluster_labels_obs, cid)
        c_pval   = (null_max_cluster_mass >= obs_mass).sum() / n_perms
        cluster_pvalues[cid] = c_pval
        if c_pval < alpha:
            significant_mask[cluster_labels_obs == cid] = True

    if verbose:
        n_sig_c = sum(1 for p in cluster_pvalues.values() if p < alpha)
        print(f"\nSignificant clusters (p<{alpha}): "
              f"{n_sig_c}/{len(cluster_ids_obs)}")
        for cid, pval in cluster_pvalues.items():
            size = (cluster_labels_obs == cid).sum()
            mark = " ✓" if pval < alpha else ""
            print(f"  Cluster {cid}: {size} points, p={pval:.4f}{mark}")

    return {
        'pvalues_uncorrected':   pvalues,
        'cluster_labels':        cluster_labels_obs,
        'cluster_pvalues':       cluster_pvalues,
        'significant_mask':      significant_mask,
        'null_max_cluster_mass': null_max_cluster_mass,
    }


# ─────────────────────────────────────────────────────────────
# 4. VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_pvalue_comparison(results, time_axis=None, neuron_ids=None,
                           alpha=0.05, cluster_forming_p=0.05,
                           figsize=(14, 5)):
    """
    Two-panel figure comparing uncorrected vs cluster-corrected p-values.

    Both panels share a linear p-value colour scale from 0 to 0.05
    (white = p>=0.05, dark = p~0). This makes it immediately obvious
    which points are nominally significant and which survived correction.

    Left  — all uncorrected p-values, with a contour at the
            cluster-forming threshold.
    Right — same, but only cluster-corrected significant regions
            are coloured; everything else is greyed out, and
            each surviving cluster is labelled with its p-value.
    """
    pvals    = results['pvalues_uncorrected']
    sig_mask = results['significant_mask']
    n_neurons, n_timepoints = pvals.shape

    if time_axis is None:
        time_axis  = np.arange(n_timepoints)
    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)

    # Clamp p-values to [0, 0.05] for display: anything >= 0.05 maps to white
    pvals_display = np.clip(pvals, 0, 0.05)

    # inferno_r: p=0 (most significant) = bright yellow/white, p=0.05 = black
    cmap = plt.colormaps.get_cmap('inferno_r')

    extent = [time_axis[0], time_axis[-1],
              neuron_ids[0] - 0.5, neuron_ids[-1] + 0.5]
    t_grid, n_grid = np.meshgrid(time_axis, neuron_ids)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(1, 3,
                            width_ratios=[1, 1, 0.045],
                            wspace=0.10,
                            left=0.07, right=0.91,
                            top=0.83,  bottom=0.12)
    ax_l  = fig.add_subplot(gs[0])
    ax_r  = fig.add_subplot(gs[1])
    ax_cb = fig.add_subplot(gs[2])

    # ── LEFT: all uncorrected p-values ──
    im = ax_l.imshow(pvals_display, aspect='auto', origin='lower',
                     extent=extent, cmap=cmap, vmin=0, vmax=0.05,
                     interpolation='nearest')

    # Contour at cluster-forming threshold
    ax_l.contour(t_grid, n_grid, pvals,
                 levels=[cluster_forming_p],
                 colors='royalblue', linewidths=1.4, linestyles='--')

    ax_l.set_title(f'Uncorrected p-values\n'
                   f'Blue dashed = cluster-forming threshold '
                   f'(p < {cluster_forming_p})',
                   fontsize=10.5)
    ax_l.set_xlabel('Time', fontsize=10)
    ax_l.set_ylabel('Neuron', fontsize=10)

    # ── RIGHT: cluster-corrected ──
    # Grey background everywhere
    ax_r.imshow(np.full_like(pvals_display, 0.05), aspect='auto', origin='lower',
                extent=extent, cmap='Greys', vmin=0, vmax=0.1, alpha=0.3,
                interpolation='nearest')

    # Show only significant regions with same colour scale
    sig_pvals = np.where(sig_mask, pvals_display, np.nan)
    ax_r.imshow(sig_pvals, aspect='auto', origin='lower',
                extent=extent, cmap=cmap, vmin=0, vmax=0.05,
                interpolation='nearest')

    # Outline surviving clusters
    ax_r.contour(t_grid, n_grid, sig_mask.astype(float),
                 levels=[0.5], colors='black',
                 linewidths=1.6, linestyles='-')

    # Label each significant cluster with its corrected p-value
    cluster_labels = results['cluster_labels']
    for cid, c_pval in results['cluster_pvalues'].items():
        if c_pval < alpha:
            ys, xs = np.where(cluster_labels == cid)
            cx = time_axis[int(xs.mean())]
            cy = neuron_ids[int(ys.mean())]
            ax_r.text(cx, cy, f'p={c_pval:.3f}',
                      ha='center', va='center', fontsize=8,
                      fontweight='bold', color='black',
                      bbox=dict(boxstyle='round,pad=0.25',
                                fc='white', alpha=0.7, ec='none'))

    n_sig = sum(1 for p in results['cluster_pvalues'].values() if p < alpha)
    ax_r.set_title(f'Cluster-corrected (p < {alpha})\n'
                   f'{n_sig} significant cluster(s) — rest greyed out',
                   fontsize=10.5)
    ax_r.set_xlabel('Time', fontsize=10)
    ax_r.set_yticklabels([])

    # ── Shared colourbar ──
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label('p-value', fontsize=10)
    cb.set_ticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    cb.ax.invert_yaxis()   # 0 at top (most significant), 0.05 at bottom

    # Mark cluster-forming threshold on colourbar
    cb.ax.axhline(cluster_forming_p, color='royalblue',
                  linewidth=1.5, linestyle='--')

    fig.suptitle(
        'Effect of cluster-based correction on MAE decoding significance\n'
        'Colour scale: p = 0 (dark) → p = 0.05 (white). '
        'Values ≥ 0.05 map to white.',
        fontsize=10, y=0.98, color='#333333'
    )

    return fig


# ─────────────────────────────────────────────────────────────
# 5. EXAMPLE
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    rng = np.random.default_rng(42)

    n_neurons    = 20
    n_timepoints = 100
    n_perms      = 500    # use 10_000 in real analysis

    null_mae = np.clip(rng.normal(1.0, 0.1, (n_neurons, n_timepoints, n_perms)), 0, None)
    obs_mae  = np.clip(rng.normal(1.0, 0.1, (n_neurons, n_timepoints)), 0, None)

    # True effect: neurons 5-10, timepoints 40-70
    obs_mae[5:10, 40:70] -= 0.40
    # Two isolated spurious points — should be cleaned up by correction
    obs_mae[2, 20]  -= 0.38
    obs_mae[15, 85] -= 0.36

    time_axis = np.linspace(-0.5, 1.0, n_timepoints)

    print("Running cluster-based permutation correction (MAE)...\n")
    results = cluster_based_correction_mae(
        obs_mae=obs_mae,
        null_mae=null_mae,
        alpha=0.05,
        cluster_forming_p=0.05,
        adjacency='full',
        verbose=True,
    )

    fig = plot_pvalue_comparison(
        results,
        time_axis=time_axis,
        alpha=0.05,
        cluster_forming_p=0.05,
    )
    fig.savefig('/mnt/user-data/outputs/pvalue_comparison.png',
                dpi=150, bbox_inches='tight')
    print("\nPlot saved.")
