"""Role-subset selection for plan 2026-07-22-role-count-sweep.

The sweep's independent variable is `n` = the number of role centroids the
manifold is built from. Subsets are chosen so that every `n` is a maximally
*spread* summary of the same 276-role cloud (the plan's selection rule):

    KMeans(k=n) on the 276 role means  ->  keep each cluster's MEDOID role.

Medoid = the member role whose mean is Euclidean-closest to its cluster centroid,
so a kept role is always a real role, never a synthetic average. This is what
makes the small-n manifold a coarse sketch of the *same* surface rather than a
different object. Note the bias it introduces (plan confound 2): medoids are more
mutually orthogonal than a random draw would be, which works *against* finding a
manifold at small n.

`default` (the Assistant persona) is force-included so its position is readable at
every n; it replaces the medoid of its own cluster, which preserves both `n` and
the one-role-per-cluster property.

Everything happens in the SHARED D=50 PCA space of `pipeline.load_cloud` — the
ambient basis is fit once on all 6,900 raw points and never refit per subset, or
nothing would be comparable across `n`.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .pipeline import Cloud

DEFAULT_ROLE = "default"


def kmeans_medoid_roles(cloud: Cloud, n: int, seed: int = 0) -> dict:
    """Select `n` roles by k-means medoid. Returns a dict describing the choice.

    Keys: roles (sorted list), default_forced (bool), n_filled (int, degenerate
    guard firings), labels (cluster id per role, for diagnostics).
    """
    role_names = list(cloud.role_names)
    R = len(role_names)
    if n >= R:
        return {"roles": sorted(role_names), "default_forced": False,
                "n_filled": 0, "labels": np.zeros(R, dtype=int)}

    km = KMeans(n_clusters=n, random_state=seed, n_init=10).fit(cloud.role_means)
    labels = km.labels_

    picked = []
    for c in range(n):
        members = np.where(labels == c)[0]
        if len(members) == 0:                       # degenerate guard (see plan)
            continue
        d = np.linalg.norm(cloud.role_means[members] - km.cluster_centers_[c], axis=1)
        picked.append(int(members[int(np.argmin(d))]))

    n_filled = 0
    if len(picked) < n:
        # Fill the shortfall with the unselected role farthest from what we have.
        remaining = [i for i in range(R) if i not in set(picked)]
        while len(picked) < n and remaining:
            D = np.linalg.norm(
                cloud.role_means[remaining][:, None, :]
                - cloud.role_means[picked][None, :, :], axis=2).min(axis=1)
            take = remaining.pop(int(np.argmax(D)))
            picked.append(take)
            n_filled += 1

    # default force-include: swap it in for the medoid of its OWN cluster.
    default_forced = False
    if DEFAULT_ROLE in role_names:
        di = role_names.index(DEFAULT_ROLE)
        if di not in picked:
            c = labels[di]
            same = [p for p in picked if labels[p] == c]
            picked.remove(same[0] if same else picked[-1])
            picked.append(di)
            default_forced = True

    return {"roles": sorted(role_names[i] for i in picked),
            "default_forced": default_forced, "n_filled": n_filled,
            "labels": labels}


def subset_cloud(cloud: Cloud, roles) -> Cloud:
    """Restrict a Cloud to `roles`, keeping the SHARED PCA basis.

    `global_mean` becomes the mean of the subset's own raw points: it is the TSS
    anchor, and it must be identical for the real fit and every permutation of
    that same subset (only SSR is allowed to move), which it is — the permutation
    reshuffles labels, never points.
    """
    keep = sorted(set(roles))
    mask = np.isin(cloud.roles, keep)
    raw = cloud.raw[mask]
    sub_roles = cloud.roles[mask]
    idx = {r: i for i, r in enumerate(cloud.role_names)}
    means = np.vstack([cloud.role_means[idx[r]] for r in keep])
    return Cloud(raw=raw, roles=sub_roles, role_names=keep, role_means=means,
                 global_mean=raw.mean(0), pca=cloud.pca)


def cloud_scale(cloud: Cloud) -> tuple:
    """(within-role per-dim RMS noise, role-mean spread) — calibrates the
    per-n positive control to that subset's own signal-to-noise."""
    idx = {r: i for i, r in enumerate(cloud.role_names)}
    pt_means = cloud.role_means[[idx[r] for r in cloud.roles]]
    noise = float(np.sqrt(np.mean((cloud.raw - pt_means) ** 2)))
    spread = float(np.sqrt(np.mean(np.sum(
        (cloud.role_means - cloud.global_mean) ** 2, axis=1))))
    return noise, spread
