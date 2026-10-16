"""Discrete curvature of a role's nearest-neighbour graph.

WHY THIS EXISTS BESIDE `curvature_gain`
---------------------------------------
The panel already has a curvature measure, but only one KIND: `spline_r2` and
`curvature_gain` fit a smooth thin-plate surface through the whole cloud and ask
how much better it does than a flat plane. That is a global, model-based answer,
and it has a known weakness — the spline gets 40 anchors and the plane gets 3
dimensions, so the gap mixes real bending with the spline's extra fitting
freedom.

Ollivier-Ricci and Forman-Ricci answer the same question with no surface fitted
at all. They are defined edge by edge on the k-nearest-neighbour graph, so they
are local, non-parametric, and cannot inherit a fitting artefact. Where they
agree with `curvature_gain` the curvature reading is robust to how it was
measured; where they disagree, the spline is the first suspect.

WHAT THE SIGN MEANS
-------------------
Both follow the Riemannian convention.

  positive curvature — neighbourhoods OVERLAP more than flat space would allow.
      Mass moves cheaply between neighbours; the graph is locally sphere-like,
      clustered, redundant.
  zero              — locally flat, Euclidean-like.
  negative          — neighbourhoods diverge. The graph is locally tree-like or
      saddle-shaped; getting from one neighbourhood to the next is expensive.

TWO DISTANCES, NOT ONE
----------------------
Ollivier-Ricci needs both and md/SPACES.md records both:

  1. EUCLIDEAN, to decide who is a neighbour and to weight the edges.
  2. The GRAPH SHORTEST-PATH metric, as the cost of transporting mass between
     two neighbourhoods.

Confusing them is the usual way this measure is got wrong: the transport cost is
a distance along the graph, not a straight line through the ambient space.

SCALE
-----
The graph is built on the cloud rescaled to unit RMS radius (`density.unit_rms`),
so both curvatures are scale-free by construction. Ollivier-Ricci would be
anyway — it is a ratio of two distances — but Forman-Ricci with weighted edges
is not, so they are put on the same footing rather than left inconsistent.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from density import unit_rms

K_GRAPH = 10        # neighbours per node; matches density.K_NEIGHBOURS
ALPHA = 0.5         # Ollivier idleness: mass kept at the node itself


def knn_graph(X: np.ndarray, k: int = K_GRAPH):
    """(adjacency lists, edge weights, edge list) of a symmetrised kNN graph.

    Symmetrised by UNION, not intersection: an intersection graph on 200 points
    in 50 dimensions fragments badly, and a disconnected graph makes the
    shortest-path metric infinite for most pairs.
    """
    import networkx as nx
    k = min(k, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(X)):
        for j, d in zip(idx[i, 1:], dist[i, 1:]):      # skip self
            G.add_edge(int(i), int(j), weight=float(d))
    return G


def _neighbourhood_measure(G, node, alpha: float = ALPHA):
    """Probability mass at `node`: alpha on itself, the rest split over its
    neighbours inversely to edge weight — closer neighbours carry more."""
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return [node], np.array([1.0])
    w = np.array([1.0 / max(G[node][n]["weight"], 1e-12) for n in nbrs])
    w = w / w.sum() * (1.0 - alpha)
    return [node] + nbrs, np.concatenate([[alpha], w])


def ollivier_ricci(G, alpha: float = ALPHA) -> np.ndarray:
    """Ollivier-Ricci curvature of every edge.

        kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y)

    where m_x is the neighbourhood measure at x, W1 is the 1-Wasserstein
    distance under the graph shortest-path metric, and d(x, y) is the edge
    weight. Exact discrete optimal transport via POT — no Sinkhorn
    approximation, because the transport problems here are 11x11 and there is
    nothing to gain by approximating them.
    """
    import networkx as nx
    import ot
    # All-pairs shortest paths once, not per edge: the graph is 200 nodes, so
    # the full matrix is 200x200 and every edge's transport reads from it.
    sp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    n = G.number_of_nodes()
    D = np.full((n, n), np.inf)
    for i, row in sp.items():
        for j, d in row.items():
            D[i, j] = d
    # A disconnected graph would leave inf entries and poison the transport.
    # Cap them at the finite diameter: unreachable is "as far as possible",
    # which is the right reading and keeps the LP solvable.
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = finite.max() if finite.size else 1.0

    out = []
    for u, v, data in G.edges(data=True):
        su, mu = _neighbourhood_measure(G, u, alpha)
        sv, mv = _neighbourhood_measure(G, v, alpha)
        cost = D[np.ix_(su, sv)]
        w1 = float(ot.emd2(mu, mv, np.ascontiguousarray(cost)))
        d_uv = max(data["weight"], 1e-12)
        out.append(1.0 - w1 / d_uv)
    return np.asarray(out, float)


def forman_ricci(G) -> np.ndarray:
    """Forman-Ricci curvature of every edge (weighted, Sreejith et al. 2016).

        F(u,v) = w_uv * [ w_u/w_uv + w_v/w_uv
                          - sum_{e ~ u, e != uv} w_u / sqrt(w_uv * w_e)
                          - sum_{e ~ v, e != uv} w_v / sqrt(w_uv * w_e) ]

    with unit node weights. Purely combinatorial once the weights are fixed — no
    optimal transport, microseconds per edge. It is the cruder of the two and is
    known to agree with Ollivier only loosely, which is exactly why both are
    computed: agreement between two different definitions is worth more than
    either alone.
    """
    out = []
    for u, v, data in G.edges(data=True):
        w_uv = max(data["weight"], 1e-12)
        term = 2.0 / w_uv                       # unit node weights w_u = w_v = 1
        for node, other in ((u, v), (v, u)):
            for nbr in G.neighbors(node):
                if nbr == other:
                    continue
                w_e = max(G[node][nbr]["weight"], 1e-12)
                term -= 1.0 / np.sqrt(w_uv * w_e)
        out.append(w_uv * term)
    return np.asarray(out, float)


def curvature_metrics(X: np.ndarray, k: int = K_GRAPH) -> tuple:
    """(scalar summaries, per-edge values) for one cloud.

    Returns NaNs rather than raising if the graph degenerates: a failure here
    must not lose a role's other 30 metrics.
    """
    nan = {c: np.nan for c in ("orc_mean", "orc_sd", "orc_p10", "orc_p90",
                               "frc_mean", "frc_sd")}
    try:
        G = knn_graph(unit_rms(X), k)
        if G.number_of_edges() == 0:
            return nan, {"orc": np.array([]), "frc": np.array([])}
        orc, frc = ollivier_ricci(G), forman_ricci(G)
    except Exception as e:  # noqa: BLE001 — OT/graph solvers are brittle
        print(f"    [curvature] failed: {e}")
        return nan, {"orc": np.array([]), "frc": np.array([])}
    return ({"orc_mean": float(np.mean(orc)), "orc_sd": float(np.std(orc)),
             "orc_p10": float(np.percentile(orc, 10)),
             "orc_p90": float(np.percentile(orc, 90)),
             "frc_mean": float(np.mean(frc)), "frc_sd": float(np.std(frc))},
            {"orc": orc, "frc": frc})
