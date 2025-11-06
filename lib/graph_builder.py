# SPDX-License-Identifier: MIT

"""
Graph builder for RWR attention.

Constructs sparse similarity graphs W from query and key embeddings,
normalizes to transition matrices P, and supports reversible chain
symmetrization for detailed balance.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def build_sparse_W(
    Q: torch.Tensor,
    K: torch.Tensor,
    topk: int = 32,
    window: int = 128,
    threshold: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sparse similarity graph W from Q and K via tiled cosine similarity.

    For each query, keeps local window edges plus topk global neighbors above
    threshold. This ensures graph connectivity while maintaining sparsity.

    Args:
        Q: Query tensor [B, H, N, D] (batch, heads, seq_len, head_dim)
        K: Key tensor [B, H, N, D]
        topk: Number of top-k neighbors to keep per query (default: 32)
        window: Local window half-width (default: 128)
        threshold: Minimum similarity to include edge (default: 0.0)

    Returns:
        indices: Sparse edge indices [B, H, E, 2] where E is num edges
        values: Sparse edge values [B, H, E]
        shape: Original shape (N, N) for reconstruction
    """
    B, H, N, D = Q.shape
    device = Q.device
    dtype = Q.dtype

    # Normalize for cosine similarity
    Q_norm = F.normalize(Q, p=2, dim=-1)  # [B, H, N, D]
    K_norm = F.normalize(K, p=2, dim=-1)  # [B, H, N, D]

    # For memory efficiency, we'll build edges in blocks
    # For now, use a simple full similarity computation with masking
    # TODO: Implement proper tiling for very long sequences (>4k)

    # Compute full similarity matrix (batch-friendly)
    sim = torch.matmul(Q_norm, K_norm.transpose(-2, -1))  # [B, H, N, N]

    # Create local window mask (±window around diagonal)
    idx = torch.arange(N, device=device)
    local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= window  # [N, N]
    local_mask = local_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

    # Find topk global neighbors per query
    topk_vals, topk_idx = torch.topk(sim, k=min(topk, N), dim=-1)  # [B, H, N, k]

    # Create edge list
    all_indices = []
    all_values = []

    for b in range(B):
        for h in range(H):
            edges_i = []
            edges_j = []
            edges_v = []

            # Add local window edges
            i_local, j_local = torch.where(local_mask[0, 0])
            for idx_pair in range(len(i_local)):
                i, j = i_local[idx_pair].item(), j_local[idx_pair].item()
                val = sim[b, h, i, j].item()
                if val >= threshold:
                    edges_i.append(i)
                    edges_j.append(j)
                    edges_v.append(val)

            # Add topk global edges (may overlap with local, we'll deduplicate)
            for i in range(N):
                for k_idx in range(min(topk, N)):
                    j = topk_idx[b, h, i, k_idx].item()
                    val = topk_vals[b, h, i, k_idx].item()
                    if val >= threshold:
                        # Simple dedup check (not efficient, but works)
                        edge_exists = False
                        for ei, ej in zip(edges_i, edges_j):
                            if ei == i and ej == j:
                                edge_exists = True
                                break
                        if not edge_exists:
                            edges_i.append(i)
                            edges_j.append(j)
                            edges_v.append(val)

            # Convert to tensors
            if len(edges_i) > 0:
                indices = torch.tensor(
                    [[ei, ej] for ei, ej in zip(edges_i, edges_j)],
                    dtype=torch.long,
                    device=device,
                )
                values = torch.tensor(edges_v, dtype=dtype, device=device)
            else:
                # Fallback: at least keep diagonal
                indices = torch.stack([idx, idx], dim=-1)  # [N, 2]
                values = torch.ones(N, dtype=dtype, device=device)

            all_indices.append(indices)
            all_values.append(values)

    # For now, return list format (can be converted to sparse tensor by caller)
    # This is a simplified implementation - full version would use block-sparse
    return all_indices, all_values, (N, N)


def normalize_to_P(
    indices_list: list,
    values_list: list,
    shape: Tuple[int, int],
    eps: float = 1e-8,
) -> Tuple[list, list, torch.Tensor]:
    """
    Row-normalize sparse W to transition matrix P = D^{-1} W.

    Args:
        indices_list: List of edge indices per (batch, head)
        values_list: List of edge values per (batch, head)
        shape: Graph shape (N, N)
        eps: Numerical stability epsilon

    Returns:
        indices_list: Same indices (unchanged)
        p_values_list: Row-normalized values
        row_sums: Row sums D for each graph [len(indices_list), N]
    """
    N = shape[0]
    p_values_list = []
    all_row_sums = []

    for indices, values in zip(indices_list, values_list):
        # Compute row sums
        row_sums = torch.zeros(N, dtype=values.dtype, device=values.device)
        for idx, val in zip(indices, values):
            i = idx[0].item()
            row_sums[i] += val

        # Clamp to avoid division by zero
        row_sums = row_sums.clamp_min(eps)

        # Normalize values
        p_values = []
        for idx, val in zip(indices, values):
            i = idx[0].item()
            p_values.append(val / row_sums[i])

        p_values = torch.stack(p_values)
        p_values_list.append(p_values)
        all_row_sums.append(row_sums)

    row_sums_tensor = torch.stack(all_row_sums)  # [B*H, N]

    return indices_list, p_values_list, row_sums_tensor


def reversible(
    indices_list: list,
    p_values_list: list,
    row_sums: torch.Tensor,
    shape: Tuple[int, int],
    eps: float = 1e-8,
) -> Tuple[list, list]:
    """
    Compute reversible transition matrix P_rev = 0.5 * (P + D^{-1} P^T D).

    This enforces detailed balance w.r.t. stationary distribution π ∝ D·1,
    improving stability and smoothing asymmetries.

    Args:
        indices_list: Edge indices per graph
        p_values_list: Edge values (row-normalized P)
        row_sums: Row sums D per graph [num_graphs, N]
        shape: Graph shape (N, N)
        eps: Numerical stability epsilon

    Returns:
        indices_list_rev: Updated indices (may include new reverse edges)
        p_values_list_rev: Reversible-symmetrized values
    """
    N = shape[0]
    p_rev_indices_list = []
    p_rev_values_list = []

    for graph_idx, (indices, p_vals) in enumerate(zip(indices_list, p_values_list)):
        D = row_sums[graph_idx]  # [N]

        # Build adjacency map for forward edges
        adj_map = {}  # (i, j) -> value
        for idx, val in zip(indices, p_vals):
            i, j = idx[0].item(), idx[1].item()
            adj_map[(i, j)] = val.item()

        # Compute P^T D (transpose with D scaling)
        # P^T D: if (i,j) in P, then (j,i) in P^T D with value P[i,j] * D[i]
        pt_d_map = {}  # (j, i) -> P[i,j] * D[i]
        for (i, j), val in adj_map.items():
            pt_d_map[(j, i)] = val * D[i].item()

        # D^{-1} P^T D: divide by D[j]
        d_inv_pt_d_map = {}  # (j, i) -> P[i,j] * D[i] / D[j]
        for (j, i), val in pt_d_map.items():
            d_inv_pt_d_map[(j, i)] = val / (D[j].item() + eps)

        # P_rev = 0.5 * (P + D^{-1} P^T D)
        # Merge both edge sets
        all_edges = set(adj_map.keys()) | set(d_inv_pt_d_map.keys())

        rev_indices = []
        rev_values = []
        for i, j in all_edges:
            p_ij = adj_map.get((i, j), 0.0)
            d_inv_pt_d_ij = d_inv_pt_d_map.get((i, j), 0.0)
            p_rev_ij = 0.5 * (p_ij + d_inv_pt_d_ij)

            rev_indices.append([i, j])
            rev_values.append(p_rev_ij)

        # Convert to tensors
        rev_indices = torch.tensor(rev_indices, dtype=torch.long, device=p_vals.device)
        rev_values = torch.tensor(rev_values, dtype=p_vals.dtype, device=p_vals.device)

        p_rev_indices_list.append(rev_indices)
        p_rev_values_list.append(rev_values)

    return p_rev_indices_list, p_rev_values_list


def sparse_mm_batch(
    R: torch.Tensor,
    indices_list: list,
    values_list: list,
    shape: Tuple[int, int],
) -> torch.Tensor:
    """
    Batched sparse matrix multiplication: R @ P.

    Args:
        R: Dense tensor [B*H, M, N] where M is number of query rows
        indices_list: Sparse indices per graph (length B*H)
        values_list: Sparse values per graph (length B*H)
        shape: Sparse matrix shape (N, N)

    Returns:
        Result of R @ P as dense tensor [B*H, M, N]
    """
    BH, M, N = R.shape
    device = R.device
    dtype = R.dtype

    result = torch.zeros_like(R)

    for graph_idx in range(BH):
        indices = indices_list[graph_idx]
        values = values_list[graph_idx]

        # Build sparse P for this graph
        # R[graph_idx] @ P -> for each row r in R, compute r · P_col[j]
        for idx, val in zip(indices, values):
            i, j = idx[0].item(), idx[1].item()
            # P[i, j] = val
            # result[:, j] += R[:, i] * val
            result[graph_idx, :, j] += R[graph_idx, :, i] * val

    return result
