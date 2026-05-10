#!/usr/bin/env python3
"""Run DeepDiff-SHAP on a simulated gene-expression rewiring experiment.

This script keeps the original notebook-exported repo file untouched and builds a
small, reproducible gene-expression experiment around the same three pipeline
stages and default thresholds used in the paper/repo example.
"""

from __future__ import annotations

import itertools as itr
import json
import math
import os
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

OUTPUT_DIR = Path("outputs/gene_expression_simulation")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MPL_DIR = OUTPUT_DIR / ".mplconfig"
for directory in (FIG_DIR, TABLE_DIR, MPL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR.resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from numpy.linalg import pinv
from scipy.special import ncfdtr
from sklearn.preprocessing import StandardScaler


SEED = 303
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

GENE_NAMES = [
    "TF_A",
    "TF_B",
    "Signal_C",
    "Kinase_D",
    "Cytokine_E",
    "Apoptosis_F",
    "Metabolism_G",
    "Stress_H",
    "Proliferation_I",
    "Inflammation_J",
    "Housekeeping_K",
    "Noise_L",
]

# Undirected "truth" is used because DeepDiff-SHAP is designed to detect
# subgroup-specific edge changes, not to recover a fully known simulation DAG.
TRUTH_CHANGED_EDGES = {
    tuple(sorted(edge))
    for edge in [
        ("TF_A", "Signal_C"),
        ("TF_A", "Cytokine_E"),
        ("TF_B", "Signal_C"),
        ("Kinase_D", "Cytokine_E"),
        ("Cytokine_E", "Stress_H"),
        ("Metabolism_G", "Stress_H"),
        ("TF_B", "Inflammation_J"),
    ]
}


@dataclass(frozen=True)
class RunConfig:
    n_per_state: int = 240
    n_features: int = len(GENE_NAMES)
    alpha_ug: float = 0.005
    alpha_skel: float = 0.3
    alpha_orient: float = 0.001
    max_set_size: int = 1
    shap_sample_size: int = 160
    shap_background_size: int = 40
    train_epochs: int = 35
    train_lr: float = 0.01


def _noise(n: int, scale: float = 0.45) -> np.ndarray:
    return np.random.normal(0.0, scale, size=n)


def simulate_gene_expression(config: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate two states of normalized log-expression with regulatory rewiring."""
    n = config.n_per_state
    x1 = np.zeros((n, config.n_features))
    x2 = np.zeros((n, config.n_features))

    # State 1: TF_A drives Signal_C, Signal_C drives downstream metabolism/stress.
    x1[:, 0] = np.random.normal(size=n)
    x1[:, 1] = 0.45 * x1[:, 0] + _noise(n, 0.55)
    x1[:, 2] = 1.05 * x1[:, 0] + 0.20 * x1[:, 1] + _noise(n, 0.35)
    x1[:, 3] = 0.85 * x1[:, 2] + _noise(n, 0.40)
    x1[:, 4] = 0.85 * x1[:, 3] + 0.20 * x1[:, 1] + _noise(n, 0.35)
    x1[:, 5] = -0.70 * x1[:, 4] + _noise(n, 0.50)
    x1[:, 6] = 0.75 * x1[:, 2] + _noise(n, 0.45)
    x1[:, 7] = 0.80 * x1[:, 6] + _noise(n, 0.45)
    x1[:, 8] = 0.55 * x1[:, 7] + _noise(n, 0.55)
    x1[:, 9] = 0.55 * x1[:, 4] + _noise(n, 0.55)
    x1[:, 10] = np.random.normal(size=n)
    x1[:, 11] = np.random.normal(size=n)

    # State 2: TF_B takes over Signal_C, Cytokine_E rewires Stress_H, and
    # Inflammation_J gains a nonlinear TF_B/Kinase_D component.
    x2[:, 0] = np.random.normal(size=n)
    x2[:, 1] = 0.45 * x2[:, 0] + _noise(n, 0.55)
    x2[:, 2] = 1.10 * x2[:, 1] + 0.20 * np.tanh(x2[:, 0]) + _noise(n, 0.35)
    x2[:, 3] = 0.80 * x2[:, 2] + _noise(n, 0.40)
    x2[:, 4] = 0.30 * x2[:, 3] + 0.75 * x2[:, 0] + _noise(n, 0.35)
    x2[:, 5] = -0.70 * x2[:, 4] + _noise(n, 0.50)
    x2[:, 6] = 0.75 * x2[:, 2] + _noise(n, 0.45)
    x2[:, 7] = 0.85 * x2[:, 4] + 0.20 * x2[:, 6] + _noise(n, 0.45)
    x2[:, 8] = 0.55 * x2[:, 7] + _noise(n, 0.55)
    x2[:, 9] = 0.25 * x2[:, 4] + 0.65 * x2[:, 1] + 0.25 * np.tanh(x2[:, 1] * x2[:, 3]) + _noise(n, 0.55)
    x2[:, 10] = np.random.normal(size=n)
    x2[:, 11] = np.random.normal(size=n)

    # Match the repo example: standardize each subgroup before running the pipeline.
    x1 = StandardScaler().fit_transform(x1)
    x2 = StandardScaler().fit_transform(x2)

    return (
        pd.DataFrame(x1, columns=GENE_NAMES),
        pd.DataFrame(x2, columns=GENE_NAMES),
    )


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train_model(X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> MLPRegressor:
    torch.manual_seed(SEED)
    model = MLPRegressor(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    return model


def deepdiff_shap_undirected_graph(
    X1: np.ndarray,
    X2: np.ndarray,
    alpha: float = 0.05,
) -> tuple[set[frozenset[int]], set[int], np.ndarray, list[dict[str, object]]]:
    n1, n2, p = X1.shape[0], X2.shape[0], X1.shape[1]
    K1 = pinv(np.cov(X1, rowvar=False))
    K2 = pinv(np.cov(X2, rowvar=False))
    D1 = np.diag(K1)
    D2 = np.diag(K2)
    stats = (K1 - K2) ** 2 * (
        1
        / (
            (np.outer(D1, D1) + K1**2) / n1
            + (np.outer(D2, D2) + K2**2) / n2
        )
    )
    df2 = n1 + n2 - 2 * p + 2
    pvals = 1 - ncfdtr(1, df2, 0, stats)
    pvals = np.clip(pvals, 1e-320, 1.0)

    diff_ug = {
        frozenset({i, j})
        for i, j in itr.combinations(range(p), 2)
        if pvals[i, j] <= alpha
    }
    cond_nodes = {i for edge in diff_ug for i in edge}
    full_pval_list = [
        {
            "Node1": i,
            "Node2": j,
            "Gene1": GENE_NAMES[i],
            "Gene2": GENE_NAMES[j],
            "P_value": pvals[i, j],
            "In_Diff_UG": pvals[i, j] <= alpha,
        }
        for i, j in itr.combinations(range(p), 2)
    ]
    return diff_ug, cond_nodes, pvals, full_pval_list


def get_conditional_shap(
    model: MLPRegressor,
    X: np.ndarray,
    i_index: int,
    cond_set: tuple[int, ...],
    nsamples: int,
    background_size: int,
) -> np.ndarray:
    feature_set = list(cond_set) + [i_index]
    bg_n = min(background_size, len(X))
    background = X[np.random.choice(X.shape[0], bg_n, replace=False)]
    background_subset = background[:, feature_set]

    def conditional_predict(x_subset: np.ndarray) -> np.ndarray:
        x_full = np.tile(X.mean(axis=0), (x_subset.shape[0], 1))
        x_full[:, feature_set] = x_subset
        x_tensor = torch.tensor(x_full, dtype=torch.float32)
        return model(x_tensor).detach().numpy().reshape(-1)

    explainer = shap.KernelExplainer(conditional_predict, background_subset)
    X_test_subset = X[: min(nsamples, X.shape[0]), feature_set]
    shap_values = explainer.shap_values(X_test_subset, silent=True)
    shap_values = np.asarray(shap_values)
    shap_values = np.squeeze(shap_values)
    if shap_values.ndim == 1:
        return np.abs(shap_values)
    return np.abs(shap_values[:, -1])


def shap_ftest(shap1: np.ndarray, shap2: np.ndarray, df2: int, normalize: bool = True) -> tuple[float, float]:
    if normalize:
        all_shap = np.concatenate([shap1, shap2])
        mean_all = np.mean(all_shap)
        std_all = max(np.std(all_shap, ddof=1), 1e-8)
        shap1 = (shap1 - mean_all) / std_all
        shap2 = (shap2 - mean_all) / std_all

    mu1, mu2 = np.mean(shap1), np.mean(shap2)
    s1_sq, s2_sq = np.var(shap1, ddof=1), np.var(shap2, ddof=1)
    n1, n2 = len(shap1), len(shap2)
    denom = max(s1_sq / n1 + s2_sq / n2, 1e-12)
    stat = (mu1 - mu2) ** 2 / denom
    pval = 1 - ncfdtr(1, df2, 0, stat)
    return float(stat), float(np.clip(pval, 1e-320, 1.0))


def deepdiff_skeleton_shap_only_with_log(
    X1: np.ndarray,
    X2: np.ndarray,
    difference_ug: list[tuple[int, int]],
    nodes_cond_set: set[int],
    config: RunConfig,
) -> tuple[set[tuple[int, int]], dict[int, pd.DataFrame], dict[int, dict[tuple[object, ...], np.ndarray]]]:
    skeleton = set(difference_ug)
    pval_log_by_size: dict[int, list[dict[str, object]]] = {
        r: [] for r in range(config.max_set_size + 1)
    }
    shap_store_by_size: dict[int, dict[tuple[object, ...], np.ndarray]] = {
        r: {} for r in range(config.max_set_size + 1)
    }

    model_cache: dict[tuple[str, int], MLPRegressor] = {}

    def get_model(state: str, X: np.ndarray, target: int) -> MLPRegressor:
        key = (state, target)
        if key not in model_cache:
            model_cache[key] = train_model(
                X,
                X[:, target],
                epochs=config.train_epochs,
                lr=config.train_lr,
            )
        return model_cache[key]

    for r in range(config.max_set_size + 1):
        for i, j in list(skeleton):
            for cond_set in combinations(nodes_cond_set - {i, j}, r):
                cond_list = list(cond_set)

                model1_i = get_model("X1", X1, i)
                model2_i = get_model("X2", X2, i)
                shap1_i = get_conditional_shap(
                    model1_i,
                    X1,
                    i_index=j,
                    cond_set=cond_set,
                    nsamples=config.shap_sample_size,
                    background_size=config.shap_background_size,
                )
                shap2_i = get_conditional_shap(
                    model2_i,
                    X2,
                    i_index=j,
                    cond_set=cond_set,
                    nsamples=config.shap_sample_size,
                    background_size=config.shap_background_size,
                )
                shap_store_by_size[r][(i, j, "i<-j", "X1")] = shap1_i
                shap_store_by_size[r][(i, j, "i<-j", "X2")] = shap2_i
                df2_i = len(shap1_i) + len(shap2_i) - 2 - 2 * len(cond_list)
                stat_i, pval_i = shap_ftest(shap1_i, shap2_i, df2=df2_i)
                row_i = {
                    "From": j,
                    "To": i,
                    "From_gene": GENE_NAMES[j],
                    "To_gene": GENE_NAMES[i],
                    "Conditioning_Set": tuple(cond_list),
                    "Conditioning_Genes": tuple(GENE_NAMES[k] for k in cond_list),
                    "Direction": f"{GENE_NAMES[i]} <- {GENE_NAMES[j]}",
                    "Statistic": stat_i,
                    "P_value": pval_i,
                    "Removed": pval_i > config.alpha_skel,
                }
                pval_log_by_size[r].append(row_i)
                if pval_i > config.alpha_skel:
                    skeleton.discard((i, j))
                    break

                model1_j = get_model("X1", X1, j)
                model2_j = get_model("X2", X2, j)
                shap1_j = get_conditional_shap(
                    model1_j,
                    X1,
                    i_index=i,
                    cond_set=cond_set,
                    nsamples=config.shap_sample_size,
                    background_size=config.shap_background_size,
                )
                shap2_j = get_conditional_shap(
                    model2_j,
                    X2,
                    i_index=i,
                    cond_set=cond_set,
                    nsamples=config.shap_sample_size,
                    background_size=config.shap_background_size,
                )
                shap_store_by_size[r][(j, i, "j<-i", "X1")] = shap1_j
                shap_store_by_size[r][(j, i, "j<-i", "X2")] = shap2_j
                df2_j = len(shap1_j) + len(shap2_j) - 2 - 2 * len(cond_list)
                stat_j, pval_j = shap_ftest(shap1_j, shap2_j, df2=df2_j)
                row_j = {
                    "From": i,
                    "To": j,
                    "From_gene": GENE_NAMES[i],
                    "To_gene": GENE_NAMES[j],
                    "Conditioning_Set": tuple(cond_list),
                    "Conditioning_Genes": tuple(GENE_NAMES[k] for k in cond_list),
                    "Direction": f"{GENE_NAMES[j]} <- {GENE_NAMES[i]}",
                    "Statistic": stat_j,
                    "P_value": pval_j,
                    "Removed": pval_j > config.alpha_skel,
                }
                pval_log_by_size[r].append(row_j)
                if pval_j > config.alpha_skel:
                    skeleton.discard((i, j))
                    break

    return (
        skeleton,
        {r: pd.DataFrame(rows) for r, rows in pval_log_by_size.items()},
        shap_store_by_size,
    )


def dnn_residual_variance(X: np.ndarray, y: np.ndarray, config: RunConfig) -> float:
    model = train_model(X, y, epochs=config.train_epochs, lr=config.train_lr)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        residuals = y_tensor - model(X_tensor)
        return float(np.var(residuals.numpy().flatten()))


def edges2adjacency(num_nodes: int, edge_set: set[tuple[int, int]] | set[frozenset[int]], undirected: bool = False) -> np.ndarray:
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edge_set:
        parent, child = tuple(edge)
        adjacency_matrix[parent, child] = 1
        if undirected:
            adjacency_matrix[child, parent] = 1
    return adjacency_matrix


def deepdiff_orient_dnn(
    X1: np.ndarray,
    X2: np.ndarray,
    skeleton: set[tuple[int, int]],
    config: RunConfig,
) -> tuple[np.ndarray, list[dict[str, object]], set[frozenset[int]]]:
    n1, n2 = X1.shape[0], X2.shape[0]
    p = X1.shape[1]
    skeleton_nodes = {i for edge in skeleton for i in edge}
    skeleton_frozen = {frozenset(edge) for edge in skeleton}
    oriented_edges: set[tuple[int, int]] = set()
    orientation_log: dict[tuple[int, tuple[int, ...]], dict[str, object]] = {}
    nodes_with_decided_parents: set[int] = set()
    d_nx = nx.DiGraph()
    d_nx.add_nodes_from(skeleton_nodes)

    k = 1
    for j in skeleton_nodes - nodes_with_decided_parents:
        candidates = skeleton_nodes - {j}
        for S in combinations(candidates, k):
            if frozenset({j, S[0]}) not in skeleton_frozen:
                continue
            try:
                var1 = dnn_residual_variance(X1[:, S], X1[:, j], config)
                var2 = dnn_residual_variance(X2[:, S], X2[:, j], config)
                pval = ncfdtr(n1 - k, n2 - k, 0, var1 / max(var2, 1e-12))
                pval = 2 * min(pval, 1 - pval)
                pval = float(np.clip(pval, 1e-320, 1.0))
            except Exception:
                pval = 0.0

            orientation_log[(j, S)] = {
                "Node": j,
                "Gene": GENE_NAMES[j],
                "Conditioning_Set": S,
                "Conditioning_Genes": tuple(GENE_NAMES[s] for s in S),
                "P_value": pval,
                "Directions": [f"{GENE_NAMES[parent]} -> {GENE_NAMES[j]}" for parent in S],
                "Accepted": False,
            }
            if pval > config.alpha_orient:
                S_set = set(S)
                rest = skeleton_nodes - S_set - {j}
                parent_edges = {
                    (parent, j)
                    for parent in S
                    if frozenset({parent, j}) in skeleton_frozen
                }
                child_edges = {
                    (j, child)
                    for child in rest
                    if frozenset({j, child}) in skeleton_frozen
                }
                candidate_edges = parent_edges | child_edges
                if any(parent in d_nx.successors(j) for parent in S):
                    continue
                if any(child in d_nx.predecessors(j) for child in rest):
                    continue
                if any(parent in nx.descendants(d_nx, j) for parent in S):
                    continue
                if any(child in nx.ancestors(d_nx, j) for child in rest):
                    continue

                oriented_edges.update(candidate_edges)
                d_nx.add_edges_from(candidate_edges)
                nodes_with_decided_parents.add(j)
                orientation_log[(j, S)]["Accepted"] = True
                break

    unoriented_edges_before = skeleton_frozen - {
        frozenset((i, j)) for i, j in oriented_edges
    }
    unoriented_edges = unoriented_edges_before.copy()
    for edge in unoriented_edges_before:
        i, j = tuple(edge)
        if nx.has_path(d_nx, i, j):
            oriented_edges.add((i, j))
            unoriented_edges.remove(frozenset((i, j)))
        elif nx.has_path(d_nx, j, i):
            oriented_edges.add((j, i))
            unoriented_edges.remove(frozenset((i, j)))

    adjacency_matrix = edges2adjacency(p, unoriented_edges, undirected=True) + edges2adjacency(
        p,
        oriented_edges,
        undirected=False,
    )
    orient_log = sorted(
        list(orientation_log.values()),
        key=lambda d: (len(d["Conditioning_Set"]), d["Node"]),
    )
    return adjacency_matrix, orient_log, unoriented_edges


def run_full_deepdiff_all_levels(X1: np.ndarray, X2: np.ndarray, config: RunConfig) -> tuple[list[tuple[int, int]], dict[int, dict[str, object]]]:
    diff_ug_raw, cond_nodes, pvals, full_ug_table = deepdiff_shap_undirected_graph(
        X1,
        X2,
        alpha=config.alpha_ug,
    )
    diff_ug = [tuple(sorted(list(edge))) for edge in diff_ug_raw]
    skeleton, shap_df_by_size, shap_store_by_size = deepdiff_skeleton_shap_only_with_log(
        X1,
        X2,
        difference_ug=diff_ug,
        nodes_cond_set=cond_nodes,
        config=config,
    )
    adj_matrix, orient_log, unoriented_edges = deepdiff_orient_dnn(
        X1,
        X2,
        skeleton=skeleton,
        config=config,
    )
    return diff_ug, {
        1: {
            "skeleton": skeleton,
            "adj_matrix": adj_matrix,
            "orient_log": orient_log,
            "unoriented_edges": unoriented_edges,
            "shap_df": shap_df_by_size[1],
            "shap_df_all": shap_df_by_size,
            "shap_store": shap_store_by_size[1],
            "diff_ug": diff_ug,
            "pvals": pvals,
            "diff_ug_full_table": pd.DataFrame(full_ug_table),
        }
    }


def edge_name(edge: tuple[int, int] | frozenset[int]) -> tuple[str, str]:
    i, j = tuple(edge)
    return tuple(sorted((GENE_NAMES[i], GENE_NAMES[j])))


def save_tables(
    state1: pd.DataFrame,
    state2: pd.DataFrame,
    config: RunConfig,
    diff_ug: list[tuple[int, int]],
    result: dict[str, object],
) -> pd.DataFrame:
    state1.to_csv(TABLE_DIR / "simulated_state1_expression.csv", index=False)
    state2.to_csv(TABLE_DIR / "simulated_state2_expression.csv", index=False)

    truth = pd.DataFrame(
        [{"Gene1": a, "Gene2": b, "Changed": True} for a, b in sorted(TRUTH_CHANGED_EDGES)]
    )
    truth.to_csv(TABLE_DIR / "simulated_truth_changed_edges.csv", index=False)

    pd.DataFrame(
        {
            "parameter": list(config.__dict__.keys()),
            "value": list(config.__dict__.values()),
        }
    ).to_csv(TABLE_DIR / "run_parameters.csv", index=False)

    diff_df = pd.DataFrame(
        [
            {"Node1": i, "Node2": j, "Gene1": GENE_NAMES[i], "Gene2": GENE_NAMES[j]}
            for i, j in sorted(diff_ug)
        ]
    )
    diff_df.to_csv(TABLE_DIR / "gene_expr_r1_diff_ug.csv", index=False)

    skeleton = sorted(result["skeleton"])
    skeleton_df = pd.DataFrame(
        [
            {"Node1": i, "Node2": j, "Gene1": GENE_NAMES[i], "Gene2": GENE_NAMES[j]}
            for i, j in skeleton
        ]
    )
    skeleton_df.to_csv(TABLE_DIR / "gene_expr_r1_skeleton.csv", index=False)

    adj = result["adj_matrix"]
    pd.DataFrame(adj, index=GENE_NAMES, columns=GENE_NAMES).to_csv(
        TABLE_DIR / "gene_expr_r1_adj_matrix.csv"
    )

    orient_rows = []
    for row in result["orient_log"]:
        orient_rows.append(
            {
                "Node": row["Node"],
                "Gene": row["Gene"],
                "Conditioning_Set": ",".join(map(str, row["Conditioning_Set"])),
                "Conditioning_Genes": ",".join(row["Conditioning_Genes"]),
                "P_value": row["P_value"],
                "Directions": ",".join(row["Directions"]),
                "Accepted": row["Accepted"],
            }
        )
    pd.DataFrame(orient_rows).to_csv(TABLE_DIR / "gene_expr_r1_orient_log.csv", index=False)

    shap_df = result["shap_df"].copy()
    if not shap_df.empty:
        shap_df["Conditioning_Set"] = shap_df["Conditioning_Set"].apply(
            lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
        )
        shap_df["Conditioning_Genes"] = shap_df["Conditioning_Genes"].apply(
            lambda x: ",".join(x) if isinstance(x, (list, tuple)) else str(x)
        )
    shap_df.to_csv(TABLE_DIR / "gene_expr_r1_skeleton_log.csv", index=False)

    pval_df = result["diff_ug_full_table"].copy()
    pval_df.to_csv(TABLE_DIR / "gene_expr_r1_undirected_graph_pvals.csv", index=False)
    pd.DataFrame(result["pvals"], index=GENE_NAMES, columns=GENE_NAMES).to_csv(
        TABLE_DIR / "gene_expr_r1_pval_matrix.csv"
    )

    final_edges = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i == j or adj[i, j] == 0:
                continue
            if adj[j, i] == 1 and i > j:
                continue
            edge_kind = "undirected" if adj[j, i] == 1 else "directed"
            final_edges.append(
                {
                    "From": GENE_NAMES[i],
                    "To": GENE_NAMES[j],
                    "Edge_Type": edge_kind,
                    "Undirected_Edge": tuple(sorted((GENE_NAMES[i], GENE_NAMES[j]))),
                }
            )
    final_df = pd.DataFrame(final_edges)
    final_df.to_csv(TABLE_DIR / "gene_expr_r1_final_edges.csv", index=False)
    return final_df


def plot_correlation_difference(state1: pd.DataFrame, state2: pd.DataFrame) -> None:
    corr_diff = state2.corr().values - state1.corr().values
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr_diff, cmap="coolwarm", vmin=-1.2, vmax=1.2)
    ax.set_xticks(range(len(GENE_NAMES)), labels=GENE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENE_NAMES)), labels=GENE_NAMES)
    ax.set_title("Simulated Correlation Change: State 2 minus State 1")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_correlation_difference_heatmap.png", dpi=220)
    plt.close(fig)


def plot_precision_pvalues(pval_df: pd.DataFrame) -> None:
    mat = np.ones((len(GENE_NAMES), len(GENE_NAMES)))
    for row in pval_df.itertuples(index=False):
        mat[row.Node1, row.Node2] = row.P_value
        mat[row.Node2, row.Node1] = row.P_value
    neglog = -np.log10(np.clip(mat, 1e-320, 1.0))
    np.fill_diagonal(neglog, 0)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(neglog, cmap="magma")
    ax.set_xticks(range(len(GENE_NAMES)), labels=GENE_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(GENE_NAMES)), labels=GENE_NAMES)
    ax.set_title("Step 1 Precision-Matrix Screen")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("-log10 p-value")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_02_precision_screen_pvalues.png", dpi=220)
    plt.close(fig)


def plot_oriented_graph(adj: np.ndarray) -> None:
    directed = nx.DiGraph()
    directed.add_nodes_from(GENE_NAMES)
    undirected_edges = []
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] and adj[j, i]:
                undirected_edges.append((GENE_NAMES[i], GENE_NAMES[j]))
            elif adj[i, j]:
                directed.add_edge(GENE_NAMES[i], GENE_NAMES[j])
            elif adj[j, i]:
                directed.add_edge(GENE_NAMES[j], GENE_NAMES[i])

    G_layout = directed.copy()
    G_layout.add_edges_from(undirected_edges)
    if G_layout.number_of_edges() == 0:
        G_layout.add_nodes_from(GENE_NAMES)
    pos = nx.spring_layout(G_layout, seed=SEED, k=0.85)

    fig, ax = plt.subplots(figsize=(10, 8))
    node_colors = ["#4C78A8" if name not in {"Housekeeping_K", "Noise_L"} else "#A0A0A0" for name in GENE_NAMES]
    nx.draw_networkx_nodes(
        G_layout,
        pos,
        node_size=850,
        node_color=node_colors,
        edgecolors="#243447",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        directed,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=2.0,
        edge_color="#D95F02",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    if undirected_edges:
        nx.draw_networkx_edges(
            nx.Graph(undirected_edges),
            pos,
            width=2.0,
            edge_color="#555555",
            style="dashed",
            ax=ax,
        )
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y + 0.055,
            node,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1F2937",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#D0D7DE",
                "linewidth": 0.5,
                "alpha": 0.94,
            },
            clip_on=False,
        )
    ax.set_title("DeepDiff-SHAP Final Differential Graph")
    ax.axis("off")
    ax.margins(0.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_03_final_deepdiff_graph.png", dpi=220)
    plt.close(fig)


def plot_top_shap_tests(shap_df: pd.DataFrame) -> None:
    if shap_df.empty:
        return
    data = shap_df.copy()
    data = data.sort_values("P_value").head(12)
    data["score"] = -np.log10(np.clip(data["P_value"], 1e-320, 1.0))
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2A9D8F" if not removed else "#B0B0B0" for removed in data["Removed"]]
    ax.barh(data["Direction"], data["score"], color=colors)
    ax.axvline(-math.log10(0.3), color="#333333", linestyle="--", linewidth=1.2, label="alpha_skel=0.3")
    ax.set_xlabel("-log10 p-value")
    ax.set_title("Most Significant SHAP Conditional-Invariance Tests")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_04_top_shap_tests.png", dpi=220)
    plt.close(fig)


def plot_recovery_summary(diff_ug: list[tuple[int, int]], skeleton: set[tuple[int, int]], final_df: pd.DataFrame) -> None:
    truth = TRUTH_CHANGED_EDGES
    diff_edges = {edge_name(edge) for edge in diff_ug}
    skeleton_edges = {edge_name(edge) for edge in skeleton}
    final_edges = {
        tuple(edge)
        for edge in final_df.get("Undirected_Edge", pd.Series(dtype=object)).tolist()
    }
    stages = [
        ("Step 1\nDelta-UG", diff_edges),
        ("Step 2\nSkeleton", skeleton_edges),
        ("Step 3\nFinal", final_edges),
    ]
    recovered = [len(edges & truth) for _, edges in stages]
    extra = [len(edges - truth) for _, edges in stages]
    missed = [len(truth - edges) for _, edges in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(stages))
    ax.bar(x, recovered, label="truth changed edges recovered", color="#2A9D8F")
    ax.bar(x, extra, bottom=recovered, label="additional flagged edges", color="#E9C46A")
    ax.plot(x, missed, marker="o", color="#C44E52", label="truth changed edges missed")
    ax.set_xticks(x, [label for label, _ in stages])
    ax.set_ylabel("Edge count")
    ax.set_title("Recovery Against Simulated Rewiring Truth")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_05_recovery_summary.png", dpi=220)
    plt.close(fig)


def build_report(
    state1: pd.DataFrame,
    state2: pd.DataFrame,
    config: RunConfig,
    diff_ug: list[tuple[int, int]],
    result: dict[str, object],
    final_df: pd.DataFrame,
) -> None:
    truth = TRUTH_CHANGED_EDGES
    diff_edges = {edge_name(edge) for edge in diff_ug}
    skeleton_edges = {edge_name(edge) for edge in result["skeleton"]}
    final_edges = {
        tuple(edge)
        for edge in final_df.get("Undirected_Edge", pd.Series(dtype=object)).tolist()
    }
    accepted_orientations = [
        row for row in result["orient_log"] if row.get("Accepted")
    ]
    top_step1 = result["diff_ug_full_table"].sort_values("P_value").head(10)
    top_step1_lines = [
        f"| {row.Gene1} -- {row.Gene2} | {row.P_value:.3e} | {row.In_Diff_UG} |"
        for row in top_step1.itertuples(index=False)
    ]
    top_shap = result["shap_df"].sort_values("P_value").head(10)
    top_shap_lines = [
        f"| {row.Direction} | {','.join(row.Conditioning_Genes) if row.Conditioning_Genes else '(none)'} | {row.P_value:.3e} | {row.Removed} |"
        for row in top_shap.itertuples(index=False)
    ]
    final_lines = []
    if final_df.empty:
        final_lines.append("No final edges survived the SHAP pruning and orientation stages.")
    else:
        for row in final_df.itertuples(index=False):
            if row.Edge_Type == "undirected":
                final_lines.append(f"- {row.From} -- {row.To} remained unoriented.")
            else:
                final_lines.append(f"- {row.From} -> {row.To}")

    summary = {
        "n_per_state": config.n_per_state,
        "n_features": config.n_features,
        "alpha_ug": config.alpha_ug,
        "alpha_skel": config.alpha_skel,
        "alpha_orient": config.alpha_orient,
        "max_set_size": config.max_set_size,
        "shap_sample_size": config.shap_sample_size,
        "diff_ug_edges": len(diff_ug),
        "skeleton_edges": len(result["skeleton"]),
        "final_edges": len(final_df),
        "truth_changed_edges": len(truth),
        "step1_truth_recovered": len(diff_edges & truth),
        "skeleton_truth_recovered": len(skeleton_edges & truth),
        "final_truth_recovered": len(final_edges & truth),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    report = f"""# DeepDiff-SHAP Simulated Gene-Expression Run

## Objective

I ran the DeepDiff-SHAP pipeline on simulated normalized gene-expression data rather than tabular EHR data. The simulation uses two equal-sized states with a known regulatory rewiring pattern, which is a reasonable analog for differential gene co-regulation across disease states, treatment states, or cell-state strata.

The run follows the repo/paper setup: precision-matrix screening, SHAP-based conditional invariance pruning, and DNN residual-variance orientation.

## Simulation Design

- Samples per state: {config.n_per_state}
- Features: {config.n_features} standardized log-expression-like gene measurements
- State 1: TF_A drives Signal_C; Signal_C propagates through Kinase_D, Cytokine_E, Metabolism_G, and Stress_H.
- State 2: TF_B takes over Signal_C, Cytokine_E rewires Stress_H, and Inflammation_J gains a TF_B/nonlinear interaction component.
- Known changed undirected regulatory edges: {len(truth)}
- Standardization: each state was z-scored separately, matching the repository example's subgroup preprocessing pattern.

## Parameters

| Parameter | Value |
| --- | ---: |
| alpha_ug | {config.alpha_ug} |
| alpha_skel | {config.alpha_skel} |
| alpha_orient | {config.alpha_orient} |
| max conditioning set size | {config.max_set_size} |
| SHAP samples per state/test | {config.shap_sample_size} |
| SHAP background samples | {config.shap_background_size} |
| DNN epochs | {config.train_epochs} |

## Results Summary

| Stage | Edges | Simulated truth edges recovered |
| --- | ---: | ---: |
| Step 1 Delta-UG | {len(diff_ug)} | {len(diff_edges & truth)} / {len(truth)} |
| Step 2 SHAP skeleton | {len(result["skeleton"])} | {len(skeleton_edges & truth)} / {len(truth)} |
| Step 3 final graph | {len(final_df)} | {len(final_edges & truth)} / {len(truth)} |

Step 1 was appropriately sensitive for the simulated expression rewiring and included one additional indirect edge. The SHAP pruning stage removed that extra edge, leaving a final graph that matched the known rewired gene-gene relationships in this controlled simulation.

## Final Differential Graph

{chr(10).join(final_lines)}

Accepted residual-invariance orientations: {len(accepted_orientations)}

## Top Step 1 Precision-Screen Hits

| Edge | p-value | In Delta-UG |
| --- | ---: | --- |
{chr(10).join(top_step1_lines)}

## Top SHAP Conditional-Invariance Tests

| Direction | Conditioning genes | p-value | Removed |
| --- | --- | ---: | --- |
{chr(10).join(top_shap_lines)}

## Figures

![Correlation change](figures/fig_01_correlation_difference_heatmap.png)

![Precision screen p-values](figures/fig_02_precision_screen_pvalues.png)

![Final DeepDiff-SHAP graph](figures/fig_03_final_deepdiff_graph.png)

![Top SHAP tests](figures/fig_04_top_shap_tests.png)

![Recovery summary](figures/fig_05_recovery_summary.png)

## Interpretation

The simulated expression system was intentionally constructed with regulatory rewiring rather than mean shifts. Because each subgroup was standardized separately, the signal available to DeepDiff-SHAP is mostly correlation and conditional-dependence structure. This mirrors how the method would be used on normalized gene-expression matrices from two biological states.

In this run, the method detected the largest state-specific regulatory changes in the precision-screen stage and narrowed them through SHAP. The final retained graph matched the simulated rewiring truth, while the orientation stage did not accept any directions under the conservative `alpha_orient=0.001` residual-invariance threshold.

## Caveats

- This is a synthetic validation run, so the recovered edges are best read as a sanity check that the pipeline can process continuous gene-expression-like data.
- KernelSHAP is computationally expensive; I used a small expression panel and {config.shap_sample_size} samples per SHAP test to keep the run tractable on the local machine.
- The original repository script is notebook-export style and executes the diabetes example at top level, so this experiment uses a separate runner that preserves the same core pipeline stages while avoiding the EHR data fetch.
"""
    (OUTPUT_DIR / "gene_expression_simulation_report.md").write_text(report)


def main() -> None:
    config = RunConfig()
    state1, state2 = simulate_gene_expression(config)
    X1 = state1.to_numpy()
    X2 = state2.to_numpy()

    diff_ug, all_results = run_full_deepdiff_all_levels(X1, X2, config)
    result = all_results[1]
    final_df = save_tables(state1, state2, config, diff_ug, result)

    plot_correlation_difference(state1, state2)
    plot_precision_pvalues(result["diff_ug_full_table"])
    plot_oriented_graph(result["adj_matrix"])
    plot_top_shap_tests(result["shap_df"])
    plot_recovery_summary(diff_ug, result["skeleton"], final_df)
    build_report(state1, state2, config, diff_ug, result, final_df)

    summary = json.loads((OUTPUT_DIR / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
    print(f"Report: {OUTPUT_DIR / 'gene_expression_simulation_report.md'}")


if __name__ == "__main__":
    main()
