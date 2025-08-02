#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from itertools import combinations
from scipy.stats import f, levene
import shap
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# !pip install ucimlrepo
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler

# Fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# Features and target
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# X is a pandas DataFrame
print(X.columns.tolist())

# # metadata 
# print(cdc_diabetes_health_indicators.metadata) 
  
# # variable information 
# print(cdc_diabetes_health_indicators.variables) 

# Combine for easier slicing
data = pd.concat([X, y], axis=1)

# Ensure target column is called 'Diabetes_binary'
target_col = 'Diabetes_binary'

# Subset by class label
class_0 = data[data[target_col] == 0]
class_1 = data[data[target_col] == 1]

# Drop target column to leave only features
X0_df = class_0.drop(columns=[target_col])
X1_df = class_1.drop(columns=[target_col])

# Convert to numpy arrays
X1 = X0_df.to_numpy()  # Class 0 → call this X1 for consistency with your example
X2 = X1_df.to_numpy()  # Class 1 → call this X2

# Confirm shape and type
print(f"X1 shape (Diabetes = 0): {X1.shape}")
print(f"X2 shape (Diabetes = 1): {X2.shape}")

# Data preprocessing
# Prior to running DeepDiff-SHAP DCI, the data should be at least mean cenetered (along with any other transformations).
scaler = StandardScaler()

X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)



# In[ ]:


# #creating example data with feature dependence (F0-F3, F0-F7, F9-F14, F9-F17)
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# np.random.seed(42)

# n = 200
# p = 20

# # Start with Gaussian noise
# X1 = np.random.randn(n, p)
# X2 = np.random.randn(n, p)

# # Inject strong dependence in opposite directions
# # In X1, X0 depends heavily on X3
# X1[:, 0] = 11 * X1[:, 3] + np.random.normal(0, 0.1, size=n)

# # In X2, X0 depends heavily on X7 instead
# X2[:, 0] = 5 * X2[:, 7] + np.random.normal(0, 0.1, size=n)

# # Inject strong dependence in opposite directions
# # In X1, X0 depends heavily on X3
# X1[:, 9] = 9 * X1[:, 14] + np.random.normal(0, 0.1, size=n)

# # In X2, X0 depends heavily on X7 instead
# X2[:, 9] = 8 * X2[:, 17] + np.random.normal(0, 0.1, size=n)

# scaler = StandardScaler()

# X1 = scaler.fit_transform(X1)
# X2 = scaler.fit_transform(X2)



# In[ ]:


import numpy as np
import networkx as nx
from numpy.linalg import pinv
from scipy.special import ncfdtr
import itertools as itr

def deepdiff_shap_undirected_graph(X1, X2, difference_ug_method='constraint', alpha=0.05, verbose=0):
    """
    Estimates the difference between two undirected graphs directly from two data sets
    using constraint-based method that relies on comparing precision matrices.
    
    Parameters
    ----------
    X1: array, shape = [n_samples, n_features]
        First dataset.    
    X2: array, shape = [n_samples, n_features]
        Second dataset.
    difference_ug_method: str, default = 'constraint'
        Method for computing the undirected difference graph. Only 'constraint' supported.
    alpha: float, default = 0.05
        Significance level parameter for hypothesis testing.
        Higher alpha leads to more edges in the difference undirected graph.
    verbose: int, default = 0
        The verbosity level of logging messages.
        
    Returns
    -------
    difference_ug: set of frozensets
        Set of frozenset edges in the difference undirected graph.
    nodes_cond_set: set
        Nodes to be considered as conditioning sets.
    """
    if difference_ug_method != 'constraint':
        raise ValueError("Only 'constraint' method is supported in this version.")

    if verbose > 0:
        print("Running constraint-based method to get difference undirected graph...")

    n1, n2, p = X1.shape[0], X2.shape[0], X1.shape[1]

    # Step 1: Estimate precision matrices
    K1 = pinv(np.cov(X1, rowvar=False))
    K2 = pinv(np.cov(X2, rowvar=False))
    D1 = np.diag(K1)
    D2 = np.diag(K2)

    # Step 2: Compute test statistic and p-values
    stats = (K1 - K2)**2 * (1/((np.outer(D1, D1) + K1**2)/n1 + (np.outer(D2, D2) + K2**2)/n2))
    df2 = n1 + n2 - 2*p + 2
    pvals = 1 - ncfdtr(1, df2, 0, stats)
    pvals = np.clip(pvals, 1e-320, 1.0)

    # Step 3: Build Δ-UG
    diff_ug = {frozenset({i, j}) for i, j in itr.combinations(range(p), 2) if pvals[i, j] <= alpha}
    cond_nodes = {i for edge in diff_ug for i in edge}

    # Step 4: Create full p-value list
    full_pval_list = [
        {"Node1": i, "Node2": j, "P_value": pvals[i, j], "In_Diff_UG": pvals[i, j] <= alpha}
        for i, j in itr.combinations(range(p), 2)
    ]

    if verbose > 0:
        print(f"alpha threshold = {alpha}")
        print(f"Number of Δ-UG edges: {len(diff_ug)}")
        print("Difference undirected graph:", diff_ug)

    return diff_ug, cond_nodes, pvals, full_pval_list


# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
from itertools import chain, combinations
from scipy.special import ncfdtr

# --- MLP Regressor ---
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

# --- Train DNN ---
def train_model(X, y, epochs=50, lr=0.001):
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

# Get Conditional SHAP Values; fixed to calculate the values using the conditioning sets, rather than just filtering the values by conditioning set 
def get_conditional_shap(model, X, i_index, cond_set, nsamples=10000, background_size=50):
    feature_set = list(cond_set) + [i_index]
    background = X[np.random.choice(X.shape[0], min(background_size, len(X)), replace=False)]
    background_subset = background[:, feature_set]

    def conditional_predict(x_subset):
        x_full = np.tile(X.mean(axis=0), (x_subset.shape[0], 1))
        x_full[:, feature_set] = x_subset
        x_tensor = torch.tensor(x_full, dtype=torch.float32)
        return model(x_tensor).detach().numpy()

    explainer = shap.KernelExplainer(conditional_predict, background_subset)
    X_test_subset = X[:nsamples, feature_set]
    shap_values = explainer.shap_values(X_test_subset)
    return np.abs(np.array(shap_values))[:, -1]


def shap_ftest(shap1, shap2, df2, normalize=True):
    """
    Perform a normalized SHAP comparison using a squared difference F statistic.

    Parameters
    ----------
    shap1 : np.ndarray
        SHAP values for group 1
    shap2 : np.ndarray
        SHAP values for group 2
    df2 : int
        Degrees of freedom (typically: n1 + n2 - 2p + 2)
    normalize : bool, default=True
        If True, normalize SHAP values across both groups using global mean and std

    Returns
    -------
    stat : float
        Test statistic
    pval : float
        P-value from non-central F distribution
    """

    if normalize:
        all_shap = np.concatenate([shap1, shap2])
        mean_all = np.mean(all_shap)
        std_all = np.std(all_shap, ddof=1)
        std_all = max(std_all, 1e-8)  # prevent divide-by-zero
        shap1 = (shap1 - mean_all) / std_all
        shap2 = (shap2 - mean_all) / std_all

    mu1, mu2 = np.mean(shap1), np.mean(shap2)
    s1_sq, s2_sq = np.var(shap1, ddof=1), np.var(shap2, ddof=1)
    n1, n2 = len(shap1), len(shap2)

    stat = (mu1 - mu2) ** 2 / (s1_sq / n1 + s2_sq / n2)
    pval = 1 - ncfdtr(1, df2, 0, stat)

    return stat, pval


def deepdiff_skeleton_shap_only_with_log(
        X1,
        X2,
        difference_ug: list,
        nodes_cond_set: set,
        alpha: float = 0.1,
        max_set_size: int = 2,
        verbose: int = 0,
        shap_sample_size: int = 10000
):
    skeleton = set(difference_ug)
    pval_log_by_size = {r: [] for r in range(max_set_size + 1)}
    shap_store_by_size = {r: {} for r in range(max_set_size + 1)}

    for r in range(max_set_size + 1):  # Conditioning set sizes 0, 1, 2
        for i, j in list(skeleton):  # Only test currently surviving edges
            for cond_set in combinations(nodes_cond_set - {i, j}, r):
                cond_list = list(cond_set)

                # i ~ j + S
                model1_i = train_model(X1, X1[:, i])
                model2_i = train_model(X2, X2[:, i])

                shap1_i = get_conditional_shap(model1_i, X1, i_index=j, cond_set=cond_set, nsamples=shap_sample_size)
                shap2_i = get_conditional_shap(model2_i, X2, i_index=j, cond_set=cond_set, nsamples=shap_sample_size)

                shap_store_by_size[r][(i, j, 'i<-j', 'X1')] = shap1_i
                shap_store_by_size[r][(i, j, 'i<-j', 'X2')] = shap2_i

                df2_i = len(shap1_i) + len(shap2_i) - 2 - 2 * len(cond_list)
                stat_i, pval_i = shap_ftest(shap1_i, shap2_i, df2=df2_i)

                row_i = {
                    "From": j,
                    "To": i,
                    "Conditioning_Set": tuple(cond_list),
                    "Direction": f"{i} <- {j}",
                    "Statistic": stat_i,
                    "P_value": pval_i,
                    "Removed": pval_i > alpha
                }
                pval_log_by_size[r].append(row_i)

                if pval_i > alpha:
                    if verbose == 1:
                        print(
                            f"(r={r}) Removing edge {j}->{i} since p-value={pval_i:.5f} > alpha={alpha:.5f} with cond set {cond_list}"
                        )
                    skeleton.discard((i, j))
                    break
                else:
                    if verbose == 1:
                        print(
                            f"(r={r}) {j}->{i} kept: p-value={pval_i:.5f} < alpha={alpha:.5f} with cond set {cond_list}"
                        )

                # j ~ i + S
                model1_j = train_model(X1, X1[:, j])
                model2_j = train_model(X2, X2[:, j])

                shap1_j = get_conditional_shap(model1_j, X1, i_index=i, cond_set=cond_set, nsamples=shap_sample_size)
                shap2_j = get_conditional_shap(model2_j, X2, i_index=i, cond_set=cond_set, nsamples=shap_sample_size)

                shap_store_by_size[r][(j, i, 'j<-i', 'X1')] = shap1_j
                shap_store_by_size[r][(j, i, 'j<-i', 'X2')] = shap2_j

                df2_j = len(shap1_j) + len(shap2_j) - 2 - 2 * len(cond_list)
                stat_j, pval_j = shap_ftest(shap1_j, shap2_j, df2=df2_j)

                row_j = {
                    "From": i,
                    "To": j,
                    "Conditioning_Set": tuple(cond_list),
                    "Direction": f"{j} <- {i}",
                    "Statistic": stat_j,
                    "P_value": pval_j,
                    "Removed": pval_j > alpha
                }
                pval_log_by_size[r].append(row_j)

                if pval_j > alpha:
                    if verbose == 1:
                        print(
                            f"(r={r}) Removing edge {i}->{j} since p-value={pval_j:.5f} > alpha={alpha:.5f} with cond set {cond_list}"
                        )
                    skeleton.discard((i, j))
                    break
                else:
                    if verbose == 1:
                        print(
                            f"(r={r}) {i}->{j} kept: p-value={pval_j:.5f} < alpha={alpha:.5f} with cond set {cond_list}"
                        )

    return skeleton, {r: pd.DataFrame(pval_log_by_size[r]) for r in pval_log_by_size}, shap_store_by_size


# In[ ]:


import numpy as np
import torch
import torch.nn as nn
import networkx as nx
from itertools import combinations
from scipy.special import ncfdtr

# Simple MLP for regression
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.model(x)

# Estimate residual variance using DNN
def dnn_residual_variance(X, y, epochs=50, lr=0.001):
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

    model.eval()
    with torch.no_grad():
        residuals = y_tensor - model(X_tensor)
        return np.var(residuals.numpy().flatten())

# Convert edges to adjacency matrix
def edges2adjacency(num_nodes, edge_set, undirected=False):
    adjacency_matrix = np.zeros((num_nodes, num_nodes))
    for parent, child in edge_set:
        adjacency_matrix[parent, child] = 1
        if undirected:
            adjacency_matrix[child, parent] = 1
    return adjacency_matrix

# Main function: orientation
def deepdiff_orient_dnn(
        X1, 
        X2,
        skeleton: set,
        nodes_cond_set: set,  # kept for interface compatibility
        alpha: float = 0.1,
        verbose: int = 0
):
    n1, n2 = X1.shape[0], X2.shape[0]
    p = X1.shape[1]

    skeleton_nodes = {i for edge in skeleton for i in edge}
    skeleton_frozen = {frozenset(edge) for edge in skeleton}

    oriented_edges = set()
    orientation_log = {}
    nodes_with_decided_parents = set()
    d_nx = nx.DiGraph()
    d_nx.add_nodes_from(skeleton_nodes)

    # Conditioning set size = 1 only
    k = 1
    if verbose > 0:
        print(f"--- Conditioning sets of size {k} ---")

    for j in skeleton_nodes - nodes_with_decided_parents:
        candidates = skeleton_nodes - {j}
        for S in combinations(candidates, k):
            if frozenset({j, S[0]}) not in skeleton_frozen:
                continue

            try:
                var1 = dnn_residual_variance(X1[:, S], X1[:, j])
                var2 = dnn_residual_variance(X2[:, S], X2[:, j])
                pval = ncfdtr(n1 - k, n2 - k, 0, var1 / var2)
                pval = 2 * min(pval, 1 - pval)
            except:
                pval = 0.0

            orientation_log[(j, S)] = {
                "Node": j,
                "Conditioning_Set": S,
                "P_value": pval,
                "Directions": [f"{p} → {j}" for p in S],
                "Accepted": False
            }

            if pval > alpha:
                S_set = set(S)
                rest = skeleton_nodes - S_set - {j}

                parent_edges = {(p, j) for p in S if frozenset({p, j}) in skeleton_frozen}
                child_edges = {(j, c) for c in rest if frozenset({j, c}) in skeleton_frozen}
                candidate_edges = parent_edges | child_edges

                # Cycle/contradiction checks
                if any(p in d_nx.successors(j) for p in S):
                    continue
                if any(c in d_nx.predecessors(j) for c in rest):
                    continue
                if any(p in nx.descendants(d_nx, j) for p in S):
                    continue
                if any(c in nx.ancestors(d_nx, j) for c in rest):
                    continue

                oriented_edges.update(candidate_edges)
                d_nx.add_edges_from(candidate_edges)
                nodes_with_decided_parents.add(j)
                orientation_log[(j, S)]["Accepted"] = True

                if verbose > 0:
                    print(f"Adding {candidate_edges}")
                break  # move to next j

    # graph traversal to orient remaining edges if possible
    unoriented_edges_before = skeleton_frozen - {frozenset((i, j)) for i, j in oriented_edges}
    unoriented_edges = unoriented_edges_before.copy()

    for edge in unoriented_edges_before:
        i, j = tuple(edge)
        if list(nx.all_simple_paths(d_nx, source=i, target=j)):
            oriented_edges.add((i, j))
            unoriented_edges.remove(frozenset((i, j)))
            if verbose > 0:
                print(f"Oriented ({i}, {j}) as ({i}, {j}) with graph traversal")
        elif list(nx.all_simple_paths(d_nx, source=j, target=i)):
            oriented_edges.add((j, i))
            unoriented_edges.remove(frozenset((i, j)))
            if verbose > 0:
                print(f"Oriented ({i}, {j}) as ({j}, {i}) with graph traversal")

    adjacency_matrix = edges2adjacency(p, unoriented_edges, undirected=True) + \
                       edges2adjacency(p, oriented_edges, undirected=False)

    orient_log = sorted(list(orientation_log.values()), key=lambda d: (len(d['Conditioning_Set']), d['Node']))
    return adjacency_matrix, orient_log, unoriented_edges


# In[ ]:


def run_full_deepdiff_all_levels(
    X1,
    X2,
    alpha_ug=0.00000005,
    alpha_skel=0.1,
    alpha_orient=0.05,
    max_set_size=1,
    verbose=1,
    shap_sample_size=1000
):
    if verbose:
        print("Step 1: Estimating Δ-UG (difference undirected graph)...")

    diff_ug_raw, cond_nodes, pvals, full_ug_table = deepdiff_shap_undirected_graph(X1, X2, alpha=alpha_ug)
    diff_ug = [tuple(sorted(list(edge))) for edge in diff_ug_raw]
    if verbose:
        print("Undirected difference DAG:", diff_ug)

    if verbose:
        print("Step 2: Pruning to skeletons via SHAP invariance...")

    skeleton, shap_df_by_size, shap_store_by_size = deepdiff_skeleton_shap_only_with_log(
        X1, X2,
        difference_ug=diff_ug,
        nodes_cond_set=cond_nodes,
        alpha=alpha_skel,
        max_set_size=max_set_size,
        verbose=verbose,
        shap_sample_size=shap_sample_size
    )

    results_by_r = {}

    # Just do orientation at r = 1
    r = 1
    if verbose:
        print(f"\n--- Step 3: Orientation for conditioning set size r = {r} ---")
        print("Skeleton edges:", skeleton)

    adj_matrix, orient_log, unoriented_edges = deepdiff_orient_dnn(
        X1, X2,
        skeleton=skeleton,
        nodes_cond_set=cond_nodes,
        alpha=alpha_orient,
        verbose=verbose  # max_set_size no longer needed
    )

    results_by_r[r] = {
        "skeleton": skeleton,
        "adj_matrix": adj_matrix,
        "orient_log": orient_log,
        "unoriented_edges": unoriented_edges,
        "shap_df": shap_df_by_size[r],
        "shap_store": shap_store_by_size[r],
        "diff_ug": diff_ug,
        "pvals": pvals, 
        "diff_ug_full_table": pd.DataFrame(full_ug_table)
    }

    return diff_ug, results_by_r


# In[ ]:


diff_ug, all_results = run_full_deepdiff_all_levels(
    X1, X2,
    alpha_ug=0.005,
    alpha_skel=0.3,
    alpha_orient=0.001,
    max_set_size=1,   # This still controls SHAP skeleton pruning
    verbose=1
)

# # Only orientation for r = 1 is available
# r1 = all_results[1]

# # Access components
# adj_r1 = r1["adj_matrix"]
# log_r1 = r1["orient_log"]
# skeleton_r1 = r1["skeleton"]
# shap_df_r1 = r1["shap_df"]
# shap_store_r1 = r1["shap_store"]
# unoriented_edges_r1 = r1["unoriented_edges"]


# In[ ]:


# r1["shap_df"]


# In[ ]:


import pandas as pd
import numpy as np

def flatten_orient_log(log):
    return [
        {
            "Node": e["Node"],
            "Conditioning_Set": ",".join(map(str, sorted(e["Conditioning_Set"]))),
            "P_value": e["P_value"],
            "Directions": ",".join(e["Directions"]),
            "Accepted": e["Accepted"]
        }
        for e in log
    ]

def save_deepdiff_shap_result(result_dict, prefix):
    # 1. Save orient_log
    if "orient_log" in result_dict:
        orient_log = flatten_orient_log(result_dict["orient_log"])
        df_log = pd.DataFrame(orient_log)
        df_log.to_csv(f"{prefix}_orient_log.csv", index=False)

    # 2. Save skeleton
    if "skeleton" in result_dict:
        skeleton = list(result_dict["skeleton"])
        df_skel = pd.DataFrame(skeleton, columns=["Node1", "Node2"])
        df_skel.to_csv(f"{prefix}_skeleton.csv", index=False)

    # 3. Save adjacency matrix
    if "adj_matrix" in result_dict:
        adj = result_dict["adj_matrix"]
        np.savetxt(f"{prefix}_adj_matrix.csv", adj, delimiter=",", fmt="%d")

    # 4. Save SHAP skeleton pruning log
    if "shap_df" in result_dict and isinstance(result_dict["shap_df"], pd.DataFrame):
        df_skeleton_log = result_dict["shap_df"].copy()
        df_skeleton_log["Conditioning_Set"] = df_skeleton_log["Conditioning_Set"].apply(
            lambda x: ",".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
        )
        df_skeleton_log.to_csv(f"{prefix}_skeleton_log.csv", index=False)

    # 5. Save full p-value matrix and Δ-UG edge summary
    if "pvals" in result_dict and isinstance(result_dict["pvals"], np.ndarray):
        pvals = result_dict["pvals"]
        n = pvals.shape[0]
        if "diff_ug" in result_dict:
            ug_set = set(map(tuple, map(sorted, result_dict["diff_ug"])))
        else:
            ug_set = set()

        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "Node1": i,
                    "Node2": j,
                    "P_value": pvals[i, j],
                    "In_Diff_UG": (i, j) in ug_set
                })
        df_pvals = pd.DataFrame(rows)
        df_pvals.to_csv(f"{prefix}_undirected_graph_pvals.csv", index=False)
        np.savetxt(f"{prefix}_pval_matrix.csv", pvals, delimiter=",")


# In[ ]:


save_deepdiff_shap_result(all_results[1], prefix="r1")


# In[ ]:


# import networkx as nx
# import matplotlib.pyplot as plt

# def plot_oriented_graph(adj_matrix, r, node_labels=None, layout='spring', spacing=2.0, figsize=(12, 10)):
#     G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

#     if layout == 'spring':
#         pos = nx.spring_layout(G, seed=42, k=spacing, scale=spacing*1.5)
#     elif layout == 'circular':
#         pos = nx.circular_layout(G)
#     elif layout == 'kamada_kawai':
#         pos = nx.kamada_kawai_layout(G)
#     else:
#         pos = nx.spring_layout(G, seed=42, k=spacing)

#     plt.figure(figsize=figsize)
#     nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
#     nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=15, edge_color='gray')

#     if node_labels:
#         nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#     else:
#         nx.draw_networkx_labels(G, pos, font_size=10)

#     plt.title(f"Oriented Graph (Conditioning Set Size r = {r})")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()


# In[ ]:


# for r, result in all_results.items():
#     print(f"Showing graph for conditioning set size r = {r}")
#     plot_oriented_graph(result["adj_matrix"], r)


# In[ ]:




