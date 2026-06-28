"""
Inference helper for TabTransformer + DeepHit model, callable from R via reticulate.

- Loads a pickled PyTorch model (architecture + weights).
- Uses metadata to validate feature shapes and durations.
- Exposes simple functions: load_model, predict_from_arrays, predict_from_dataframe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from einops import rearrange, repeat
from torch.serialization import add_safe_globals

import tab_transformer_pytorch
from tab_transformer_pytorch.tab_transformer_pytorch import (
    Attention as InnerAttention,
    FeedForward as InnerFeedForward,
    GEGLU as InnerGEGLU,
    MLP as InnerMLP,
    PreNorm as InnerPreNorm,
    Transformer as InnerTransformer,
    TabTransformer as BaseTabTransformer,
)

# Keep CPU usage modest inside Shiny
torch.set_num_threads(1)

MODEL_CACHE: Dict[Path, Any] = {}
META_CACHE: Dict[Path, Dict[str, Any]] = {}


def _patch_tab_transformer() -> None:
    """Patch inner classes so pickled models load correctly."""
    setattr(tab_transformer_pytorch, "Transformer", InnerTransformer)
    setattr(tab_transformer_pytorch, "PreNorm", InnerPreNorm)
    setattr(tab_transformer_pytorch, "Attention", InnerAttention)
    setattr(tab_transformer_pytorch, "FeedForward", InnerFeedForward)
    setattr(tab_transformer_pytorch, "GEGLU", InnerGEGLU)
    setattr(tab_transformer_pytorch, "MLP", InnerMLP)


def _register_unpickle_targets() -> None:
    """
    Make sure the custom classes are discoverable under the names used at training time.
    Some pickles reference __main__.TabTransformerSharedCauseSpecificNet, so we bind
    the classes into __main__ as well.
    """
    import __main__

    __main__.TabTransformerwithEmbedding = TabTransformerwithEmbedding
    __main__.TabTransformerSharedCauseSpecificNet = TabTransformerSharedCauseSpecificNet
    add_safe_globals([TabTransformerwithEmbedding, TabTransformerSharedCauseSpecificNet])


def exists(val):
    return val is not None


class TabTransformerwithEmbedding(BaseTabTransformer):
    """Same as training: return embedding when requested."""

    def forward(self, x_categ, x_cont, return_attn=False, return_embedding=False):
        xs = []

        assert x_categ.shape[-1] == self.num_categories
        if self.num_unique_categories > 0:
            x_categ = x_categ + self.categories_offset
            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(
                    self.shared_category_embed, "n d -> b n d", b=categ_embed.shape[0]
                )
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim=-1)

            x, attns = self.transformer(categ_embed, return_attn=True)
            flat_categ = rearrange(x, "b ... -> b (...)")
            xs.append(flat_categ)

        if self.num_continuous > 0:
            if exists(getattr(self, "continuous_mean_std", None)):
                mean, std = self.continuous_mean_std.unbind(dim=-1)
                x_cont = (x_cont - mean) / std
            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim=-1)
        x = self.embedding_proj(x)

        if return_embedding:
            if return_attn:
                return x, attns
            return x

        logits = self.mlp(x)
        if return_attn:
            return logits, attns
        return logits


class TabTransformerSharedCauseSpecificNet(nn.Module):
    """Shared trunk + risk-specific heads (DeepHit style)."""

    def __init__(
        self,
        tab_net,
        embed_dim,
        num_nodes_shared,
        num_nodes_indiv,
        num_risks,
        out_features,
        dropout=0.1,
    ):
        super().__init__()
        self.tab = TabTransformerwithEmbedding(**tab_net)
        self.shared_mlp = tt.practical.MLPVanilla(
            embed_dim,
            num_nodes_shared[:-1],
            num_nodes_shared[-1],
            batch_norm=True,
            dropout=dropout,
        )
        self.risk_nets = nn.ModuleList(
            [
                tt.practical.MLPVanilla(
                    num_nodes_shared[-1],
                    num_nodes_indiv,
                    out_features,
                    batch_norm=True,
                    dropout=dropout,
                )
                for _ in range(num_risks)
            ]
        )

    def forward(self, x_cat, x_cont):
        x_embed = self.tab(x_cat, x_cont, return_embedding=True)
        shared_out = self.shared_mlp(x_embed)
        out = [risk_net(shared_out) for risk_net in self.risk_nets]
        return torch.stack(out, dim=1)


def _load_meta(meta_path: str | Path) -> Dict[str, Any]:
    meta_path = Path(meta_path)
    if meta_path in META_CACHE:
        return META_CACHE[meta_path]
    with meta_path.open() as f:
        meta = json.load(f)
    META_CACHE[meta_path] = meta
    return meta


def load_model(model_path: str) -> None:
    """Load and cache the pickled model (architecture + weights)."""
    _patch_tab_transformer()
    _register_unpickle_targets()
    mpath = Path(model_path)
    if mpath in MODEL_CACHE:
        return
    net = torch.load(mpath, map_location="cpu", weights_only=False)
    net.eval()
    MODEL_CACHE[mpath] = net


def _prepare_arrays(
    x_cat: np.ndarray, x_cont: np.ndarray, expected_cats: int, expected_cont: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x_cat.ndim != 2:
        raise ValueError(f"x_cat must be 2D, got shape {x_cat.shape}")
    if x_cont.ndim != 2:
        raise ValueError(f"x_cont must be 2D, got shape {x_cont.shape}")
    if x_cat.shape[1] != expected_cats:
        raise ValueError(
            f"Categorical feature count mismatch: data {x_cat.shape[1]}, expected {expected_cats}"
        )
    if x_cont.shape[1] != expected_cont:
        raise ValueError(
            f"Continuous feature count mismatch: data {x_cont.shape[1]}, expected {expected_cont}"
        )
    x_cat_np = np.asarray(x_cat, dtype=np.int64)
    x_cont_np = np.asarray(x_cont, dtype=np.float32)
    return torch.tensor(x_cat_np), torch.tensor(x_cont_np)


def predict_from_arrays(
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    model_path: str,
    meta_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Run inference from already-separated categorical and continuous arrays."""
    meta = _load_meta(meta_path) if meta_path else None
    expected_cats = len(meta["categ_idx"]) if meta else x_cat.shape[1]
    expected_cont = len(meta["cont_idx"]) if meta else x_cont.shape[1]
    duration_labels: List[str] = meta.get("duration_labels", []) if meta else []
    risk_names: List[str] = meta.get("risk_names", []) if meta else []

    load_model(model_path)
    net = MODEL_CACHE[Path(model_path)]
    x_cat_t, x_cont_t = _prepare_arrays(x_cat, x_cont, expected_cats, expected_cont)

    with torch.no_grad():
        logits = net(x_cat_t, x_cont_t)
        prob = F.softmax(logits, dim=2).cpu().numpy()  # shape [B, R, T]
    surv = np.maximum(1.0 - prob.cumsum(axis=2), 0.0)

    return {
        "prob": prob,
        "surv": surv,
        "duration_labels": duration_labels if duration_labels else list(range(prob.shape[2])),
        "risk_names": risk_names if risk_names else [f"risk_{i}" for i in range(prob.shape[1])],
    }


def predict_from_dataframe(
    df: pd.DataFrame,
    model_path: str,
    meta_path: str,
) -> Dict[str, Any]:
    """Run inference from a DataFrame, using metadata to slice/reorder features."""
    meta = _load_meta(meta_path)
    feature_names: List[str] = meta["feature_names"]
    categ_idx: List[int] = meta["categ_idx"]
    cont_idx: List[int] = meta["cont_idx"]

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_ordered = df[feature_names]
    x_cat = df_ordered.iloc[:, categ_idx].to_numpy(dtype=np.int64, copy=False)
    x_cont = df_ordered.iloc[:, cont_idx].to_numpy(dtype=np.float32, copy=False)

    return predict_from_arrays(
        x_cat=x_cat,
        x_cont=x_cont,
        model_path=model_path,
        meta_path=meta_path,
    )


# ---------------------------------------------------------------------------
# Permutation importance (test-case level) — adapted from Permutation.py
# ---------------------------------------------------------------------------


def _get_prob_at_bin(
    net: nn.Module,
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    risk_idx: int,
    time_bin: int,
    expected_cats: int,
    expected_cont: int,
) -> float:
    """Run model and return prob[0, risk_idx, time_bin] as a scalar."""
    x_cat_t, x_cont_t = _prepare_arrays(x_cat, x_cont, expected_cats, expected_cont)
    with torch.no_grad():
        logits = net(x_cat_t, x_cont_t)
        prob = F.softmax(logits, dim=2).cpu().numpy()
    return float(prob[0, risk_idx, time_bin])


def _get_cumulative_event_prob(
    net: nn.Module,
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    risk_idx: int,
    expected_cats: int,
    expected_cont: int,
) -> float:
    """Run model and return cumulative event probability (1 - survival at last bin)."""
    x_cat_t, x_cont_t = _prepare_arrays(x_cat, x_cont, expected_cats, expected_cont)
    with torch.no_grad():
        logits = net(x_cat_t, x_cont_t)
        prob = F.softmax(logits, dim=2).cpu().numpy()
    cum_prob = prob[0, risk_idx, :].sum()
    return float(cum_prob)


def permutation_importance_single_row(
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    ref_cat: np.ndarray,
    ref_cont: np.ndarray,
    model_path: str,
    meta_path: str | Path,
    risk_idx: int = 0,
    time_bin: int = -1,
    features_to_permute: List[str] = None,  # list of feature names; None = all
    non_top_defaults: Dict[str, float] = None,  # dict of {feature_name: value} for fixed features
) -> Dict[str, float]:
    """
    Compute permutation (sensitivity) feature importance for a single test case.

    For each feature, replace its value with a random sample from the reference
    distribution, re-run prediction, and record |perturbed_pred - baseline_pred|.

    Parameters
    ----------
    x_cat : np.ndarray
        Categorical features for the test case, shape (1, n_cat).
    x_cont : np.ndarray
        Continuous features for the test case, shape (1, n_cont).
    ref_cat : np.ndarray
        Reference categorical data to sample replacement values, shape (n_ref, n_cat).
    ref_cont : np.ndarray
        Reference continuous data, shape (n_ref, n_cont).
    model_path : str
        Path to the PyTorch model file.
    meta_path : str | Path
        Path to model_meta.json.
    risk_idx : int
        Risk head index: 0 = AF, 1 = All cause death.
    time_bin : int
        Time bin index for the probability. -1 means last bin. -2 means overall
        (cumulative event probability across all bins).
    features_to_permute : List[str] or None
        Feature names to permute. If None, all features are permuted (backward-compatible).
        When provided, only those features are permuted; others are skipped.
        Returns exactly len(features_to_permute) keys.
    non_top_defaults : Dict[str, float] or None
        Fixed values for features NOT in features_to_permute. Currently unused in the
        loop (test row already contains correct baseline values), but accepted for
        API symmetry with permutation_importance_from_dataframe.

    Returns
    -------
    Dict[str, float]
        {feature_name: importance} for each permuted feature.
        len == len(features_to_permute) when features_to_permute is provided, else 78.
    """
    meta = _load_meta(meta_path)
    feature_names: List[str] = meta["feature_names"]
    categ_idx: List[int] = meta["categ_idx"]
    cont_idx: List[int] = meta["cont_idx"]
    cat_names: List[str] = [feature_names[i] for i in categ_idx]
    cont_names: List[str] = [feature_names[i] for i in cont_idx]
    expected_cats = len(categ_idx)
    expected_cont = len(cont_idx)

    load_model(model_path)
    net = MODEL_CACHE[Path(model_path)]

    if x_cat.shape[0] != 1 or x_cont.shape[0] != 1:
        raise ValueError("x_cat and x_cont must have exactly one row (single test case)")

    num_risks = len(meta.get("risk_names", [])) or 2
    if risk_idx < 0 or risk_idx >= num_risks:
        raise ValueError(f"risk_idx must be in [0, {num_risks - 1}], got {risk_idx}")

    n_cat = x_cat.shape[1]
    n_cont = x_cont.shape[1]
    n_ref = ref_cat.shape[0]
    if n_ref < 1:
        raise ValueError("ref_cat and ref_cont must have at least one row")
    if ref_cont.shape[0] != n_ref:
        raise ValueError("ref_cat and ref_cont must have same number of rows")

    # Resolve target: -2 = overall (cumulative), -1 = last bin, else specific bin
    use_overall = time_bin == -2
    if use_overall:
        base_pred = _get_cumulative_event_prob(
            net, x_cat, x_cont, risk_idx, expected_cats, expected_cont
        )
    else:
        n_bins = len(meta.get("duration_labels", [])) or 10
        tb = time_bin if time_bin >= 0 else n_bins - 1
        tb = max(0, min(tb, n_bins - 1))
        base_pred = _get_prob_at_bin(
            net, x_cat, x_cont, risk_idx, tb, expected_cats, expected_cont
        )

    def get_pred(x_c, x_ct):
        if use_overall:
            return _get_cumulative_event_prob(
                net, x_c, x_ct, risk_idx, expected_cats, expected_cont
            )
        return _get_prob_at_bin(
            net, x_c, x_ct, risk_idx, tb, expected_cats, expected_cont
        )

    importance: Dict[str, float] = {}

    # Permute categorical features
    for i in range(n_cat):
        feat_name = cat_names[i]
        if features_to_permute is not None and feat_name not in features_to_permute:
            continue  # skip non-top features
        x_cat_perm = x_cat.copy()
        r = np.random.randint(0, n_ref)
        x_cat_perm[0, i] = ref_cat[r, i]
        new_pred = get_pred(x_cat_perm, x_cont)
        importance[feat_name] = abs(new_pred - base_pred)

    # Permute continuous features
    for j in range(n_cont):
        feat_name = cont_names[j]
        if features_to_permute is not None and feat_name not in features_to_permute:
            continue  # skip non-top features
        x_cont_perm = x_cont.copy()
        r = np.random.randint(0, n_ref)
        x_cont_perm[0, j] = ref_cont[r, j]
        new_pred = get_pred(x_cat, x_cont_perm)
        importance[feat_name] = abs(new_pred - base_pred)

    return importance


def permutation_importance_from_dataframe(
    df_test_row: pd.DataFrame,
    df_ref: pd.DataFrame,
    model_path: str,
    meta_path: str | Path,
    risk_idx: int = 0,
    time_bin: int = -1,
    features_to_permute: List[str] = None,
    non_top_defaults: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Wrapper: compute permutation importance from DataFrames.

    Uses metadata to extract and order features. Exclude the test row from
    df_ref if it is contained (e.g. df_ref = df[df.index != test_idx]).

    Parameters
    ----------
    features_to_permute : List[str] or None
        Feature names to permute. None = all 78 (backward-compatible).
        When provided (e.g. TOP_FEATURES in Manual mode), only those features
        are permuted and the returned dict has exactly len(features_to_permute) keys.
    non_top_defaults : Dict[str, float] or None
        Fixed values for features not in features_to_permute. Defaults to empty dict.
        Passed through to permutation_importance_single_row for API symmetry.
    """
    meta = _load_meta(meta_path)
    feature_names: List[str] = meta["feature_names"]
    categ_idx: List[int] = meta["categ_idx"]
    cont_idx: List[int] = meta["cont_idx"]

    if non_top_defaults is None:
        non_top_defaults = {}

    missing_test = [c for c in feature_names if c not in df_test_row.columns]
    missing_ref = [c for c in feature_names if c not in df_ref.columns]
    if missing_test:
        raise ValueError(f"df_test_row missing columns: {missing_test}")
    if missing_ref:
        raise ValueError(f"df_ref missing columns: {missing_ref}")

    if len(df_test_row) != 1:
        raise ValueError("df_test_row must have exactly one row")
    if len(df_ref) < 1:
        raise ValueError("df_ref must have at least one row")

    df_test_ord = df_test_row[feature_names]
    df_ref_ord = df_ref[feature_names]

    x_cat = df_test_ord.iloc[:, categ_idx].to_numpy(dtype=np.int64)
    x_cont = df_test_ord.iloc[:, cont_idx].to_numpy(dtype=np.float32)
    ref_cat = df_ref_ord.iloc[:, categ_idx].to_numpy(dtype=np.int64)
    ref_cont = df_ref_ord.iloc[:, cont_idx].to_numpy(dtype=np.float32)

    return permutation_importance_single_row(
        x_cat=x_cat,
        x_cont=x_cont,
        ref_cat=ref_cat,
        ref_cont=ref_cont,
        model_path=model_path,
        meta_path=meta_path,
        risk_idx=risk_idx,
        time_bin=time_bin,
        features_to_permute=features_to_permute,
        non_top_defaults=non_top_defaults,
    )


# ---------------------------------------------------------------------------
# Conformal Prediction — AF Lower Prediction Bound
# ---------------------------------------------------------------------------

# Module-level cache so tau is only loaded once per session
_CP_TAU_CACHE: Dict[str, Any] = {}


def load_cp_tau(tau_path: str = "cp_outputs/tau_hat_af_demo.json") -> Dict[str, Any]:
    """
    Load the calibrated tau_hat from JSON.
    Cached after first load — safe to call repeatedly.
    """
    if tau_path in _CP_TAU_CACHE:
        return _CP_TAU_CACHE[tau_path]
    with open(tau_path) as f:
        tau_obj = json.load(f)
    _CP_TAU_CACHE[tau_path] = tau_obj
    return tau_obj


def predict_af_lower_bound(
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    model_path: str,
    meta_path: str | Path,
    tau_path: str = "cp_outputs/tau_hat_af_demo.json",
) -> Dict[str, Any]:
    """
    Compute the AF Conformal Prediction lower bound for one or more patients.

    The lower bound L satisfies (on the calibration set):
        P(true AF time >= L | AF event occurs) >= 1 - alpha

    Parameters
    ----------
    x_cat      : int array  [n_patients, 61]
    x_cont     : float array [n_patients, 17]
    model_path : path to .pt file
    meta_path  : path to model_meta.json
    tau_path   : path to tau_hat JSON produced by run_cp_synthetic.py

    Returns
    -------
    dict with keys:
        lower_bound_bin   : int array [n_patients]  — bin index of lower bound
        lower_bound_year  : float array [n_patients] — year value of lower bound
        tau_hat           : float  — the calibrated tau used
        alpha             : float  — miscoverage level (e.g. 0.10)
        target_coverage   : float  — 1 - alpha (e.g. 0.90)
        af_cif            : float array [n_patients, n_bins] — AF CIF curve
        t_grid            : float array [n_bins] — year values for each bin
    """
    meta = _load_meta(meta_path)
    feature_names: List[str] = meta["feature_names"]
    categ_idx: List[int]     = meta["categ_idx"]
    cont_idx: List[int]      = meta["cont_idx"]
    expected_cats = len(categ_idx)
    expected_cont = len(cont_idx)

    # Load model and tau
    load_model(model_path)
    net = MODEL_CACHE[Path(model_path)]
    tau_obj = load_cp_tau(tau_path)
    tau_hat = float(tau_obj["af"])

    # Validate inputs
    x_cat_t, x_cont_t = _prepare_arrays(x_cat, x_cont, expected_cats, expected_cont)

    # Forward pass
    with torch.no_grad():
        logits = net(x_cat_t, x_cont_t)                    # [B, 2, n_bins]
        prob   = F.softmax(logits, dim=2).cpu().numpy()    # [B, 2, n_bins]

    n_bins = prob.shape[2]

    # AF CIF = cumulative sum of AF probability (risk index 0)
    af_cif = prob[:, 0, :].cumsum(axis=1)                  # [B, n_bins]
    af_cif = np.clip(af_cif, 0.0, 1.0)

    # Build t_grid: evenly spaced 0–15 years matching training range
    t_grid = np.linspace(0.0, 15.0, n_bins)

    # Lower bound: first bin where AF CIF >= tau_hat
    n_patients = af_cif.shape[0]
    lb_bin  = np.full(n_patients, n_bins - 1, dtype=int)
    lb_year = np.full(n_patients, t_grid[-1], dtype=float)

    for i in range(n_patients):
        idx = np.where(af_cif[i, :] >= tau_hat)[0]
        if len(idx) > 0:
            lb_bin[i]  = int(idx[0])
            lb_year[i] = float(t_grid[idx[0]])

    return {
        "lower_bound_bin":  lb_bin.tolist(),
        "lower_bound_year": lb_year.tolist(),
        "tau_hat":          tau_hat,
        "alpha":            float(tau_obj.get("alpha", 0.10)),
        "target_coverage":  float(tau_obj.get("target_coverage", 0.90)),
        "af_cif":           af_cif.tolist(),
        "t_grid":           t_grid.tolist(),
    }


# ---------------------------------------------------------------------------
# Conformal Prediction — Aalen-Johansen recalibrated Upper Prediction Bound
# ---------------------------------------------------------------------------
# New method (replaces the LPB lower bound). For each cause it returns Lhat:
# a year by which the event is predicted to occur with (1 - gamma) confidence.
# Calibration artifact (delta + time_grid) is produced by ucttrp_aj_upb.py.

_CP_RECALIB_CACHE: Dict[str, Any] = {}


def load_cp_recalib(
    recalib_path: str = "cp_outputs/ucttrp_aj_upb_recalibration.npz",
) -> Dict[str, Any]:
    """
    Load the Aalen-Johansen recalibration artifact. Cached after first load.

    The .npz contains:
        delta     : [K, T]  mean predicted CIF - AJ marginal CIF (per cause, per bin)
        time_grid : [T]     year value of each discrete-time bin
        aj, mean_pred : diagnostics (unused at inference)
    """
    if recalib_path in _CP_RECALIB_CACHE:
        return _CP_RECALIB_CACHE[recalib_path]
    z = np.load(recalib_path)
    obj = {
        "delta":     z["delta"].astype(np.float64),
        "time_grid": z["time_grid"].astype(np.float64),
    }
    _CP_RECALIB_CACHE[recalib_path] = obj
    return obj


def _deephit_cif(net, x_cat_t, x_cont_t) -> np.ndarray:
    """
    DeepHit cumulative incidence function matching pycox's predict_cif:
    joint softmax over all (cause x time) logits plus an implicit survival
    slot, then cumulative sum over time per cause.

    Returns CIF with shape [K, T, N]  (cause, time, sample).
    """
    with torch.no_grad():
        logits = net(x_cat_t, x_cont_t)                    # [B, K, T]
        B, K, T = logits.shape
        flat = logits.reshape(B, K * T)
        pad  = torch.zeros(B, 1, dtype=flat.dtype)
        pmf  = F.softmax(torch.cat([flat, pad], dim=1), dim=1)[:, : K * T]
        cif  = pmf.reshape(B, K, T).cumsum(dim=2).cpu().numpy()   # [B, K, T]
    cif = np.transpose(cif, (1, 2, 0))                     # → [K, T, N]
    return np.clip(cif, 0.0, 1.0).astype(np.float64)


def _enforce_valid_cifs(cif: np.ndarray) -> np.ndarray:
    """cif: [K, T, N]. Enforce nonnegative, monotone in time, sum_k CIF_k <= 1."""
    cif = np.maximum(cif, 0.0)
    cif = np.maximum.accumulate(cif, axis=1)
    total = cif.sum(axis=0, keepdims=True)
    cif = cif / np.maximum(total, 1.0)
    cif = np.maximum.accumulate(cif, axis=1)
    return np.clip(cif, 0.0, 1.0)


def _upper_bound_from_cif(cif_event, time_grid, gamma=0.1, eps=1e-8):
    """
    cif_event: [T, N]. Lhat_i = first time where
    CIF_i(t) / CIF_i(tmax) >= 1 - gamma; otherwise the final bin (tmax).
    Returns (lhat_year [N], lhat_bin [N]).
    """
    final_raw  = cif_event[-1, :]
    cond_cdf   = cif_event / np.maximum(final_raw, eps)[None, :]
    target     = 1.0 - gamma
    n          = cond_cdf.shape[1]
    last_bin   = len(time_grid) - 1

    lhat_year = np.full(n, time_grid[-1], dtype=float)
    lhat_bin  = np.full(n, last_bin, dtype=int)

    for i in range(n):
        if final_raw[i] <= eps:
            continue
        idx = np.where(cond_cdf[:, i] >= target)[0]
        if len(idx) > 0:
            lhat_year[i] = float(time_grid[idx[0]])
            lhat_bin[i]  = int(idx[0])

    return lhat_year, lhat_bin


def predict_event_upper_bounds(
    x_cat: np.ndarray,
    x_cont: np.ndarray,
    model_path: str,
    meta_path: str | Path,
    recalib_path: str = "cp_outputs/ucttrp_aj_upb_recalibration.npz",
    gamma: float = 0.1,
) -> Dict[str, Any]:
    """
    AJ-recalibrated conformal Upper Prediction Bound (UPB) per cause.

    For each patient and cause k, Lhat is a year by which the event is
    predicted to occur with (1 - gamma) confidence:
        af    -> cause index 0
        death -> cause index 1  (All cause death)

    Returns a dict with shared keys (gamma, target_coverage, t_grid) and,
    for each available cause, a sub-dict keyed 'af' / 'death' with:
        upper_bound_year : [n_patients]
        upper_bound_bin  : [n_patients]
    """
    meta = _load_meta(meta_path)
    categ_idx = meta["categ_idx"]
    cont_idx  = meta["cont_idx"]

    load_model(model_path)
    net = MODEL_CACHE[Path(model_path)]

    recal     = load_cp_recalib(recalib_path)
    delta     = recal["delta"]        # [K, T]
    time_grid = recal["time_grid"]    # [T]

    x_cat_t, x_cont_t = _prepare_arrays(x_cat, x_cont, len(categ_idx), len(cont_idx))

    cif_raw = _deephit_cif(net, x_cat_t, x_cont_t)   # [K, T, N]
    K, T_model = cif_raw.shape[0], cif_raw.shape[1]

    if delta.shape[1] != T_model:
        raise ValueError(
            f"Recalibration delta has {delta.shape[1]} time bins but the model "
            f"outputs {T_model}; they must match (use the dur30 model)."
        )

    cif_cal = _enforce_valid_cifs(cif_raw - delta[:, :, None])

    out: Dict[str, Any] = {
        "gamma":           float(gamma),
        "target_coverage": float(1.0 - gamma),
        "t_grid":          time_grid.tolist(),
    }
    for name, k in {"af": 0, "death": 1}.items():
        if k >= K:
            continue
        lhat_year, lhat_bin = _upper_bound_from_cif(cif_cal[k], time_grid, gamma=gamma)
        out[name] = {
            "upper_bound_year": lhat_year.tolist(),
            "upper_bound_bin":  lhat_bin.tolist(),
        }
    return out


__all__ = [
    "load_model",
    "predict_from_arrays",
    "predict_from_dataframe",
    "permutation_importance_single_row",
    "permutation_importance_from_dataframe",
    "load_cp_tau",
    "predict_af_lower_bound",
    "load_cp_recalib",
    "predict_event_upper_bounds",
]
