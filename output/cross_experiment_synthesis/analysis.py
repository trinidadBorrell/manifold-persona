"""Cross-experiment synthesis for Brendon/Codex runs from 2026-07-30/31.

This is exploratory. It joins stored result-level records and recomputes a
small role-level geometry summary on the exact 40-question E8 activation set.
No source data or experiment outputs are changed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED = 20260731
N_BOOT = 5000
N_PERM = 20000


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def participation_ratio(x: np.ndarray) -> float:
    xc = x - x.mean(axis=0)
    singular = np.linalg.svd(xc, compute_uv=False)
    eigenvalues = singular * singular
    total = eigenvalues.sum()
    return float(total * total / np.sum(eigenvalues * eigenvalues))


def local_id(x: np.ndarray, k: int = 25) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    distances, indices = nn.kneighbors(x)
    values = np.array([participation_ratio(x[idx]) for idx in indices])
    mean_neighbor_distance = distances[:, 1:].mean(axis=1)
    return values, mean_neighbor_distance


def panel_stability_summary(a: np.ndarray, b: np.ndarray) -> dict:
    """Reproduce E8's model-free pair-distance and per-role reliability summaries."""
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    distance_spearman = float(spearmanr(pdist(ac), pdist(bc)).statistic)
    ar = ac - ac.mean(axis=1, keepdims=True)
    br = bc - bc.mean(axis=1, keepdims=True)
    role_reliability = np.sum(ar * br, axis=1) / (
        np.linalg.norm(ar, axis=1) * np.linalg.norm(br, axis=1)
    )
    return {
        "distance_spearman": distance_spearman,
        "mean_role_reliability": float(role_reliability.mean()),
        "min_role_reliability": float(role_reliability.min()),
    }


def role_panel_centroids(
    array_path: Path,
    metadata: pd.DataFrame,
    questions: list[int],
    roles: list[str],
    layer: int = 19,
) -> np.ndarray:
    """Mean the 5 variants x panel questions for each role without loading 16 GB."""
    data = np.load(array_path, mmap_mode="r")
    role_values = metadata["role"].to_numpy()
    question_values = metadata["question_idx"].to_numpy()
    in_panel = np.isin(question_values, questions)
    centroids = np.empty((len(roles), data.shape[-1]), dtype=np.float64)
    for i, role in enumerate(roles):
        idx = np.flatnonzero((role_values == role) & in_panel)
        if len(idx) != 5 * len(questions):
            raise ValueError(f"unexpected cell count for {role}: {len(idx)}")
        centroids[i] = np.asarray(data[idx, layer, :], dtype=np.float32).mean(
            axis=0, dtype=np.float64
        )
    return centroids


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    observed = float(spearmanr(x, y).statistic)
    draws = np.empty(N_BOOT)
    n = len(x)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[i] = spearmanr(x[idx], y[idx]).statistic
    # With no original ties, a permutation Spearman test is just a dot product
    # between centered rank vectors. This is much faster than re-ranking each
    # of the 20,000 permutations and gives the same statistic.
    xr = rankdata(x)
    yr = rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = np.linalg.norm(xr) * np.linalg.norm(yr)
    permuted = np.empty(N_PERM)
    for i in range(N_PERM):
        permuted[i] = float(xr @ yr[rng.permutation(n)] / denom)
    p_two = float((1 + np.sum(np.abs(permuted) >= abs(observed))) / (N_PERM + 1))
    return {
        "rho": observed,
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
        "permutation_p_two_sided": p_two,
    }


def partial_rank_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Spearman-like correlation after linear removal of ranked controls."""
    xr = rankdata(x)
    yr = rankdata(y)
    cr = np.column_stack([rankdata(controls[:, j]) for j in range(controls.shape[1])])
    design = np.column_stack([np.ones(len(x)), cr])
    x_resid = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    y_resid = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def bh_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out.tolist()


def make_role_figure(role_df: pd.DataFrame, stats: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    pairs = [
        ("radius", "panel_reliability_avg", "Distance from the role-cloud center", "Panel reliability", "radius_vs_reliability"),
        ("pca_nmse_avg_d5", "gplvm_benefit_avg_d5", "PCA error at d=5", "GPLVM gain at d=5", "pca_difficulty_vs_benefit"),
        ("local_id_40q_L19", "panel_reliability_avg", "Local thickness", "Panel reliability", "lid_vs_reliability"),
    ]
    for ax, (x_col, y_col, xlab, ylab, key) in zip(axes, pairs):
        ax.scatter(role_df[x_col], role_df[y_col], s=24, alpha=0.72, color="#28536b")
        rho = stats[key]["rho"]
        ci = stats[key]["ci95"]
        ax.set_title(f"Spearman ρ = {rho:.2f}  [{ci[0]:.2f}, {ci[1]:.2f}]")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.2)
    fig.suptitle("Role-level differences mostly track signal strength and baseline difficulty", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig_role_links.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_question_figure(question_df: pd.DataFrame) -> None:
    ordered = question_df.sort_values("eta2_contribution", ascending=False).reset_index(drop=True)
    colors = ordered["panel"].map({"DEV": "#ef8354", "VERIFY": "#38618c"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.1), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].bar(np.arange(len(ordered)), ordered["eta2_contribution"], color=colors)
    axes[0].set_xticks(np.arange(len(ordered)))
    axes[0].set_xticklabels(ordered["question_idx"], rotation=90, fontsize=8)
    axes[0].set_ylabel("Share of total activation variance")
    axes[0].set_xlabel("Question ID (sorted by effect)")
    axes[0].set_title("One DEV question is unusually strong")
    axes[0].grid(axis="y", alpha=0.2)
    for panel, color in [("DEV", "#ef8354"), ("VERIFY", "#38618c")]:
        sub = question_df[question_df["panel"] == panel]
        axes[1].scatter(
            sub["question_shift"] ** 2,
            sub["eta2_contribution"],
            label=panel,
            color=color,
            s=45,
            alpha=0.8,
        )
    q64 = question_df[question_df["question_idx"] == 64].iloc[0]
    axes[1].annotate(
        "question 64",
        (q64["question_shift"] ** 2, q64["eta2_contribution"]),
        xytext=(8, -14), textcoords="offset points", fontsize=9,
    )
    axes[1].set_xlabel("Question shift squared")
    axes[1].set_ylabel("Variance contribution")
    axes[1].set_title("Two experiments measured the same question effect")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_question_panels.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_dimension_figure(dim_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for column, label, color, marker in [
        ("e1b_relative_gain", "Earlier 5-question split", "#7a5195", "o"),
        ("e8_avg_relative_gain", "40-question cross-fit · averaged prompt", "#ef5675", "s"),
        ("e8_last_relative_gain", "40-question cross-fit · last prompt token", "#003f5c", "^"),
    ]:
        ax.plot(dim_df["d"], 100 * dim_df[column], marker=marker, lw=2.4, label=label, color=color)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xticks(dim_df["d"])
    ax.set_xlabel("Number of latent dimensions")
    ax.set_ylabel("GPLVM error reduction vs PCA (%)")
    ax.set_title("The nonlinear gain replicated—only for the averaged-prompt view")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dimension_replication.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_measurement_figure(e7: dict, e8: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    labels = ["Averaged prompt", "Last prompt token"]
    reliability = [
        e8["views"]["prompt_avg"]["p1"]["role_position_reliability"]["mean"],
        e8["views"]["prompt_last"]["p1"]["role_position_reliability"]["mean"],
    ]
    gain = [
        e8["views"]["prompt_avg"]["p2"]["5"]["summary"]["mean_paired_diff_pca_minus_gplvm"],
        e8["views"]["prompt_last"]["p2"]["5"]["summary"]["mean_paired_diff_pca_minus_gplvm"],
    ]
    x = np.arange(2)
    axes[0].bar(x - 0.18, reliability, width=0.36, label="Role reliability", color="#38618c")
    ax2 = axes[0].twinx()
    ax2.bar(x + 0.18, gain, width=0.36, label="GPLVM gain", color="#ef8354")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Mean role reliability")
    ax2.set_ylabel("PCA error minus GPLVM error")
    axes[0].set_title("Pooling choice changes the result")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    values = [
        e7["prompt_L19"]["twonn_handrolled"],
        e7["prompt_L26"]["twonn_handrolled"],
        e7["resp_L19"]["twonn_handrolled"],
        e7["resp_L26"]["twonn_handrolled"],
    ]
    x = np.arange(2)
    axes[1].bar(x - 0.18, values[:2], width=0.36, label="Prompt tokens", color="#38618c")
    axes[1].bar(x + 0.18, values[2:], width=0.36, label="Response tokens", color="#ef8354")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Layer 19", "Layer 26"])
    axes[1].set_ylabel("Estimated intrinsic dimension")
    axes[1].set_title("Token choice matters more than layer here")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_measurement_choices.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rng = np.random.default_rng(SEED)
    e1a = load_json(REPO / "output/e1a_variance_partition/results.json")
    e1b = load_json(REPO / "output/e1b_gplvm_repro/results.json")
    e5 = load_json(REPO / "output/e5_algo_gaps/results.json")
    e7 = load_json(REPO / "output/e7_id_reconciliation/results.json")
    e8 = load_json(REPO / "output/e8_crossfit_40q/results.json")
    old_lid = pd.read_csv(
        REPO / "output/local_id/2026-07-30T20-54-local-intrinsic-dimension/data/local_id_per_role.csv"
    )

    roles = e8["design"]["roles"]
    if roles != sorted(roles) or len(roles) != 276 or len(old_lid) != len(roles):
        raise ValueError("role order is not the shared sorted order")
    if "role" in old_lid.columns:
        # New-format CSV carries role names: merge by role, order-proof.
        old_lid = old_lid.set_index("role").loc[roles].reset_index()
    # Legacy CSVs have no role column; the positional join below then rests on
    # the writer having used the same sorted role order (length-checked above).

    data_dir = REPO / "data/embeddings_roles_resp_40q"
    meta = pd.read_parquet(
        data_dir / "metadata.parquet",
        columns=["role", "question_idx", "instruction_idx", "question"],
    )
    dev_questions = e8["design"]["dev_questions"]
    verify_questions = e8["design"]["verify_questions"]
    dev = role_panel_centroids(data_dir / "prompt_avg.npy", meta, dev_questions, roles)
    verify = role_panel_centroids(data_dir / "prompt_avg.npy", meta, verify_questions, roles)
    q64_centroids = role_panel_centroids(data_dir / "prompt_avg.npy", meta, [64], roles)
    dev_without_q64 = (20.0 * dev - q64_centroids) / 19.0
    q64_leave_out = panel_stability_summary(dev_without_q64, verify)

    midpoint = 0.5 * (dev + verify)
    pca50 = PCA(n_components=50, svd_solver="full", random_state=0).fit(midpoint)
    x_mid = pca50.transform(midpoint)
    x_dev = pca50.transform(dev)
    x_verify = pca50.transform(verify)
    lid_mid, density_mid = local_id(x_mid, k=25)
    lid_dev, _ = local_id(x_dev, k=25)
    lid_verify, _ = local_id(x_verify, k=25)
    radius = np.linalg.norm(x_mid - x_mid.mean(axis=0), axis=1)
    default_index = roles.index("default")
    assistant_axis = x_mid[default_index] - x_mid.mean(axis=0)
    assistant_axis /= np.linalg.norm(assistant_axis)
    axis_projection = (x_mid - x_mid.mean(axis=0)) @ assistant_axis

    role_df = pd.DataFrame({
        "role": roles,
        "local_id_40q_L19": lid_mid,
        "local_id_dev": lid_dev,
        "local_id_verify": lid_verify,
        "mean_neighbor_distance": density_mid,
        "radius": radius,
        "assistant_axis_projection": axis_projection,
        "local_id_original_40q_L19_raw_pca": old_lid["local_id"].to_numpy(),
        "panel_reliability_avg": e8["views"]["prompt_avg"]["p1"]["role_position_reliability"]["values"],
        "panel_reliability_last": e8["views"]["prompt_last"]["p1"]["role_position_reliability"]["values"],
    })
    for d in [2, 3, 5, 8, 12]:
        for view, short in [("prompt_avg", "avg"), ("prompt_last", "last")]:
            per_role = e8["views"][view]["p2"][str(d)]["per_role"]
            role_df[f"gplvm_benefit_{short}_d{d}"] = per_role["paired_diff_pca_minus_gplvm"]
            role_df[f"pca_nmse_{short}_d{d}"] = per_role["nmse_pca_verify"]
            role_df[f"gplvm_nmse_{short}_d{d}"] = per_role["nmse_gplvm_verify"]
    role_df["relative_gplvm_gain_avg_d5"] = (
        role_df["gplvm_benefit_avg_d5"] / role_df["pca_nmse_avg_d5"]
    )

    tests = {
        "lid_vs_reliability": bootstrap_spearman(role_df["local_id_40q_L19"], role_df["panel_reliability_avg"], rng),
        "lid_vs_benefit": bootstrap_spearman(role_df["local_id_40q_L19"], role_df["gplvm_benefit_avg_d5"], rng),
        "reliability_vs_benefit": bootstrap_spearman(role_df["panel_reliability_avg"], role_df["gplvm_benefit_avg_d5"], rng),
        "lid_dev_vs_verify": bootstrap_spearman(role_df["local_id_dev"], role_df["local_id_verify"], rng),
        "raw_pca_lid_vs_centroid_pca_lid": bootstrap_spearman(
            role_df["local_id_original_40q_L19_raw_pca"],
            role_df["local_id_40q_L19"],
            rng,
        ),
        "avg_vs_last_reliability": bootstrap_spearman(role_df["panel_reliability_avg"], role_df["panel_reliability_last"], rng),
        "avg_vs_last_benefit_d5": bootstrap_spearman(role_df["gplvm_benefit_avg_d5"], role_df["gplvm_benefit_last_d5"], rng),
        "radius_vs_reliability": bootstrap_spearman(role_df["radius"], role_df["panel_reliability_avg"], rng),
        "radius_vs_pca_difficulty": bootstrap_spearman(role_df["radius"], role_df["pca_nmse_avg_d5"], rng),
        "pca_difficulty_vs_benefit": bootstrap_spearman(role_df["pca_nmse_avg_d5"], role_df["gplvm_benefit_avg_d5"], rng),
        "reliability_vs_relative_benefit": bootstrap_spearman(role_df["panel_reliability_avg"], role_df["relative_gplvm_gain_avg_d5"], rng),
    }
    adjusted = bh_adjust([v["permutation_p_two_sided"] for v in tests.values()])
    for record, adj in zip(tests.values(), adjusted):
        record["bh_adjusted_p"] = adj

    tests["lid_vs_reliability"]["partial_rank_rho_controlling_radius_density"] = partial_rank_corr(
        role_df["local_id_40q_L19"].to_numpy(),
        role_df["panel_reliability_avg"].to_numpy(),
        role_df[["radius", "mean_neighbor_distance"]].to_numpy(),
    )
    tests["lid_vs_benefit"]["partial_rank_rho_controlling_radius_density"] = partial_rank_corr(
        role_df["local_id_40q_L19"].to_numpy(),
        role_df["gplvm_benefit_avg_d5"].to_numpy(),
        role_df[["radius", "mean_neighbor_distance"]].to_numpy(),
    )
    tests["reliability_vs_benefit"]["partial_rank_rho_controlling_pca_difficulty"] = partial_rank_corr(
        role_df["panel_reliability_avg"].to_numpy(),
        role_df["gplvm_benefit_avg_d5"].to_numpy(),
        role_df[["pca_nmse_avg_d5"]].to_numpy(),
    )
    tests["reliability_vs_relative_benefit"]["partial_rank_rho_controlling_radius_density"] = partial_rank_corr(
        role_df["panel_reliability_avg"].to_numpy(),
        role_df["relative_gplvm_gain_avg_d5"].to_numpy(),
        role_df[["radius", "mean_neighbor_distance"]].to_numpy(),
    )

    q = e1a["40q"]["L19"]
    q_lookup = {int(qid): float(value) for qid, value in zip(q["per_question_ids"], q["per_question_eta2_contrib"])}
    s3 = e8["views"]["prompt_avg"]["s3"]
    question_df = pd.DataFrame({
        "question_idx": s3["question_idx"],
        "question_shift": s3["question_shift_magnitude"],
        "response_char_length": s3["mean_response_char_len"],
    })
    question_df["eta2_contribution"] = question_df["question_idx"].map(q_lookup)
    question_df["panel"] = np.where(question_df["question_idx"].isin(dev_questions), "DEV", "VERIFY")
    question_text = meta[["question_idx", "question"]].drop_duplicates().set_index("question_idx")["question"]
    question_df["question"] = question_df["question_idx"].map(question_text)
    question_shift_rho = float(spearmanr(question_df["question_shift"] ** 2, question_df["eta2_contribution"]).statistic)
    x = (question_df["question_shift"] ** 2).to_numpy()
    y = question_df["eta2_contribution"].to_numpy()
    slope = float(np.dot(x, y) / np.dot(x, x))
    r2_origin = float(1 - np.sum((y - slope * x) ** 2) / np.sum((y - y.mean()) ** 2))
    panel_summary = question_df.groupby("panel").agg(
        n_questions=("question_idx", "size"),
        eta2_total=("eta2_contribution", "sum"),
        mean_shift=("question_shift", "mean"),
        max_shift=("question_shift", "max"),
        mean_response_chars=("response_char_length", "mean"),
    ).reset_index()

    dims = [2, 3, 5, 8, 12]
    dim_rows = []
    for d in dims:
        e1 = e1b["summary"][str(d)]
        e1_pca = e1["models"]["pca"]["mean_normalized_mse"]
        e1_gain = -e1["paired_rbf_gplvm_minus_pca"]["mean"]
        avg = e8["views"]["prompt_avg"]["p2"][str(d)]["summary"]
        last = e8["views"]["prompt_last"]["p2"][str(d)]["summary"]
        dim_rows.append({
            "d": d,
            "e1b_pca_nmse": e1_pca,
            "e1b_gplvm_gain": e1_gain,
            "e1b_relative_gain": e1_gain / e1_pca,
            "e8_avg_pca_nmse": avg["pca_nmse_verify_mean"],
            "e8_avg_gplvm_gain": avg["mean_paired_diff_pca_minus_gplvm"],
            "e8_avg_relative_gain": avg["mean_paired_diff_pca_minus_gplvm"] / avg["pca_nmse_verify_mean"],
            "e8_last_pca_nmse": last["pca_nmse_verify_mean"],
            "e8_last_gplvm_gain": last["mean_paired_diff_pca_minus_gplvm"],
            "e8_last_relative_gain": last["mean_paired_diff_pca_minus_gplvm"] / last["pca_nmse_verify_mean"],
        })
    dim_df = pd.DataFrame(dim_rows)
    curve_rho = float(spearmanr(dim_df["e1b_relative_gain"], dim_df["e8_avg_relative_gain"]).statistic)
    mean_relative_e1b = float(dim_df["e1b_relative_gain"].mean())
    mean_relative_e8 = float(dim_df["e8_avg_relative_gain"].mean())

    q64 = question_df.loc[question_df["question_idx"] == 64].iloc[0]
    role_outliers = {
        "lowest_reliability_avg": role_df.nsmallest(10, "panel_reliability_avg")[["role", "panel_reliability_avg", "local_id_40q_L19", "gplvm_benefit_avg_d5"]].to_dict("records"),
        "highest_local_id": role_df.nlargest(10, "local_id_40q_L19")[["role", "local_id_40q_L19", "panel_reliability_avg", "gplvm_benefit_avg_d5"]].to_dict("records"),
        "highest_gplvm_benefit_d5": role_df.nlargest(10, "gplvm_benefit_avg_d5")[["role", "gplvm_benefit_avg_d5", "panel_reliability_avg", "local_id_40q_L19"]].to_dict("records"),
        "pca_beats_gplvm_d5": role_df.nsmallest(10, "gplvm_benefit_avg_d5")[["role", "gplvm_benefit_avg_d5", "panel_reliability_avg", "local_id_40q_L19"]].to_dict("records"),
    }

    summary = {
        "status": "exploratory cross-experiment synthesis; not preregistered",
        "sources": ["E1a", "E1b", "E5", "E7", "E8", "local-ID run"],
        "same_data_recompute": {
            "view": "prompt_avg",
            "layer": 19,
            "n_questions": 40,
            "n_roles": 276,
            "pca_ambient": 50,
            "local_k": 25,
            "local_id_mean": float(lid_mid.mean()),
            "local_id_cv": float(lid_mid.std() / lid_mid.mean()),
            "local_id_dev_verify_rho": tests["lid_dev_vs_verify"]["rho"],
            "raw_pca_vs_centroid_pca_local_id_rho": tests["raw_pca_lid_vs_centroid_pca_lid"]["rho"],
            "provenance_correction": (
                "The original local-ID manifest says layer 26 because the script "
                "hard-coded that label. Exact value reproduction shows the run "
                "used MP_ROLE_DIR=data/embeddings_roles_resp_40q: 40 questions, "
                "prompt_avg, layer 19."
            ),
        },
        "role_level_tests": tests,
        "question_panel": {
            "shift_squared_vs_eta2_spearman": question_shift_rho,
            "shift_squared_vs_eta2_r2_through_origin": r2_origin,
            "panel_summary": panel_summary.to_dict("records"),
            "question_64": {
                "eta2_contribution": float(q64["eta2_contribution"]),
                "share_of_all_question_eta2": float(q64["eta2_contribution"] / q["question_eta2"]),
                "question_shift": float(q64["question_shift"]),
                "panel": q64["panel"],
                "text": q64["question"],
                "leave_out_stability": q64_leave_out,
            },
        },
        "dimension_curve": {
            "e1b_vs_e8_avg_relative_gain_spearman": curve_rho,
            "mean_relative_gain_e1b": mean_relative_e1b,
            "mean_relative_gain_e8_avg": mean_relative_e8,
            "e8_avg_survives_absolute_panel_gap_check_all_dims": bool(np.all(
                dim_df["e8_avg_gplvm_gain"].to_numpy() >= 0.25 * np.array([
                    abs(e8["views"]["prompt_avg"]["p2"][str(d)]["summary"]["panel_shift_gap_pca"])
                    for d in dims
                ])
            )),
        },
        "measurement_choice": {
            "e8_role_reliability_mean_avg": e8["views"]["prompt_avg"]["p1"]["role_position_reliability"]["mean"],
            "e8_role_reliability_mean_last": e8["views"]["prompt_last"]["p1"]["role_position_reliability"]["mean"],
            "e8_d5_relative_gain_avg": float(dim_df.loc[dim_df.d == 5, "e8_avg_relative_gain"].iloc[0]),
            "e8_d5_relative_gain_last": float(dim_df.loc[dim_df.d == 5, "e8_last_relative_gain"].iloc[0]),
            "e7_twonn_prompt_L19": e7["prompt_L19"]["twonn_handrolled"],
            "e7_twonn_prompt_L26": e7["prompt_L26"]["twonn_handrolled"],
            "e7_twonn_response_L19": e7["resp_L19"]["twonn_handrolled"],
            "e7_twonn_response_L26": e7["resp_L26"]["twonn_handrolled"],
        },
        "e5_context": {
            "response_local_patch_gain_d5": e5["experiment_b_mfa_local_vs_global"]["response"]["paired_diff_global_minus_local"]["mean"],
            "prompt_local_patch_gain_d5": e5["experiment_b_mfa_local_vs_global"]["prompt"]["paired_diff_global_minus_local"]["mean"],
            "interpretation": "smooth global nonlinearity is more robust than a patchwork model",
        },
        "role_outliers": role_outliers,
    }

    role_df.to_csv(OUT / "role_level_join.csv", index=False)
    question_df.to_csv(OUT / "question_level_join.csv", index=False)
    dim_df.to_csv(OUT / "dimension_curve_join.csv", index=False)
    (OUT / "analysis_results.json").write_text(json.dumps(summary, indent=2))

    make_role_figure(role_df, tests)
    make_question_figure(question_df)
    make_dimension_figure(dim_df)
    make_measurement_figure(e7, e8)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
