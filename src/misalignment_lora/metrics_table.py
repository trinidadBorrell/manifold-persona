"""partExtra/metrics_table.png: the metrics of the top-6 / top-10 figures, full name (abbreviation), family, what it
measures, sign of the relation to position at layer 25.   python -m misalignment_lora.metrics_table"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROWS = [
 ("Mean Forman–Ricci curvature (frc_mean)", "curvature", "Mean Forman–Ricci curvature over the edges of the 10-nearest-neighbour graph of the cloud: 4 minus the degrees of the two end points, so low where the graph is densely connected.", "−"),
 ("Spread of Forman–Ricci curvature (frc_sd)", "curvature", "Standard deviation of the Forman–Ricci curvature over the edges: how uneven the connectivity is across the cloud.", "+"),
 ("Mean Ollivier–Ricci curvature (orc_mean)", "curvature", "Mean Ollivier–Ricci curvature of the same edges: Wasserstein distance between the neighbour distributions of the two end points, relative to their distance. Positive = overlapping neighbourhoods.", "weak +"),
 ("Median graph path / straight-line ratio (path_ratio_p50)", "curvature", "Median over point pairs of the shortest path along the 10-NN graph divided by the straight-line distance. 1 = flat and well sampled; larger = the graph detours.", "−"),
 ("Minimum spanning tree length (H0_total_persistence)", "topology", "Total length of the minimum spanning tree of the cloud = sum of the H0 bar lengths of a Vietoris–Rips filtration (PCA-50, 400-point subsamples). Small = tight, well-connected cloud.", "−"),
 ("Mean 10-nearest-neighbour distance (knn_dist_mean)", "density", "Mean distance from each point to its 10 nearest neighbours, after scaling the cloud to unit RMS radius.", "−"),
 ("Unevenness of neighbour distances (knn_dist_cv)", "density", "Coefficient of variation of the 10-NN distances: how unevenly dense the cloud is.", "+"),
 ("Spread of log kernel density (kde_logdens_sd)", "density", "Standard deviation over points of the log Gaussian kernel-density estimate (bandwidth = median neighbour distance): contrast between dense and sparse regions.", "+"),
 ("Levina–Bickel intrinsic dimension (MLE)", "intrinsic dimension", "Maximum-likelihood local intrinsic dimension from the 10 nearest neighbours of each point, averaged over points.", "+"),
 ("Correlation dimension (corr_dim)", "intrinsic dimension", "Slope of log(number of point pairs closer than r) against log r, between the 5th and 50th percentile of pairwise distances.", "−"),
 ("PCA participation ratio (PCA_participation_ratio)", "intrinsic dimension", "(Σλ)² / Σλ² over the covariance eigenvalues λ: the number of directions that would carry the variance if it were spread evenly.", "−"),
 ("Effective rank (effective_rank)", "spectral shape", "exp of the Shannon entropy of the normalised eigenvalue spectrum: another count of the directions that carry variance.", "−"),
 ("Variance share of the first component (top1_share)", "spectral shape", "Share of the cloud's total variance in its first principal component.", "+"),
 ("Prompt × question interaction share (interaction_frac)", "design decomposition", "The cloud lies on a prompt × question grid. Fit each point as prompt effect + question effect; the interaction share is the variance left over, the part that depends on the combination.", "+"),
 ("Question share of variance (quest_frac)", "design decomposition", "Share of the cloud's variance explained by which question was asked.", "−"),
 ("Variance share along the assistant axis (var_frac_along_axis)", "anisotropy to the assistant axis", "Share of the cloud's variance that lies along the assistant-axis direction.", "weak; sign flips\nafter fine-tune"),
 ("Cosine of PC1 with the assistant axis (cos_pc1_axis)", "anisotropy to the assistant axis", "Absolute cosine between the cloud's first principal component and the assistant axis.", "weak"),
 ("Top-10 subspace overlap with the default role (subspace_overlap_k10)", "anisotropy to the assistant axis", "Mean squared cosine of the principal angles between the persona's top-10 PCA subspace and the default role's: do the two clouds spread in the same directions.", "−"),
 ("Linear CKA with the default role (cka_linear)", "question-geometry similarity to the default role", "Linear centred kernel alignment between the persona's question-mean vectors and the default role's: do the two arrange the questions in the same relative geometry.", "+"),
 ("Mutual nearest-neighbour overlap with the default role (mknn_align)", "question-geometry similarity to the default role", "Share of each question's 10 nearest neighbours (among the other questions) that are the same for the persona and for the default role.", "+"),
 ("Representational similarity with the default role (rsa_spearman)", "question-geometry similarity to the default role", "Spearman correlation between the persona's and the default role's matrices of pairwise distances between question means.", "+"),
 ("Procrustes residual to the default role (procrustes_resid)", "question-geometry similarity to the default role", "Residual after the best rotation and scaling of the persona's question means onto the default role's; 0 = identical arrangement.", "−"),
 ("Centroid agreement between prompt halves (split_half_cos_prompts)", "consistency", "Cosine between the centroid from system prompts 1–3 and the centroid from prompts 4–5: dependence of the persona's position on the prompt wording.", "−"),
 ("Centroid agreement between question halves (split_half_cos_questions)", "consistency", "Cosine between the centroid from the even-numbered questions and from the odd-numbered ones.", "−"),
]
FAMCOL = {"curvature": "#e8eef5", "topology": "#eaf3ec", "density": "#fdf1e8", "intrinsic dimension": "#f3ecf7", "spectral shape": "#f7f3e6",
          "design decomposition": "#e9f4f4", "anisotropy to the assistant axis": "#f5eaea", "question-geometry similarity to the default role": "#eef0f8", "consistency": "#f0f0f0"}


def main():
    out = Path("output/29-oct/partExtra"); out.mkdir(parents=True, exist_ok=True)
    first = {"curvature", "topology", "density", "intrinsic dimension", "spectral shape"}
    render(out / "metrics_table_1_shape.png", [r for r in ROWS if r[1] in first], "Table 1: shape of the cloud (curvature, topology, density, intrinsic dimension, spectral shape)")
    render(out / "metrics_table_2_structure.png", [r for r in ROWS if r[1] not in first], "Table 2: structure of the cloud relative to the design and to the default role (design decomposition, anisotropy, similarity to the default role, consistency)")
    render(out / "metrics_table.png", ROWS, "All metrics")


def render(path, ROWS, subtitle):
    cells = [[textwrap.fill(r[0], 30), textwrap.fill(r[1], 20), textwrap.fill(r[2], 70)] for r in ROWS]
    heights = [max(len(c.split("\n")) for c in row) for row in cells]
    line_h = 0.42
    fig_h = line_h * sum(heights) + 1.6
    fig, ax = plt.subplots(figsize=(16, fig_h), dpi=170); ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=["Metric (abbreviation in the figures)", "Family", "What it measures"],
                   colWidths=[0.27, 0.16, 0.57], loc="upper center", cellLoc="left")
    tab.auto_set_font_size(False); tab.set_fontsize(11)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#c9d1d9"); cell.set_linewidth(0.6); cell.PAD = 0.03
        if r == 0:
            cell.set_text_props(weight="bold", color="#15232e", fontsize=11.5); cell.set_facecolor("#dfe6ee"); cell.set_height(line_h * 1.6 / fig_h)
        else:
            cell.set_facecolor(FAMCOL[ROWS[r - 1][1]]); cell.set_height(line_h * heights[r - 1] / fig_h)
    ax.set_title(subtitle + "\nEach metric is computed on one persona's cloud of answer activations (layer 25, response-token mean, question-centred).",
                 fontsize=11, color="#4e5c68", loc="left", pad=8)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
