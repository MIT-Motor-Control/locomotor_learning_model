"""Reanalyse the original S5 numerical grid without running simulations.

This ports the C/E comparison from the reviewer-response ``fall_reward_analysis``
sources. It preserves the bounded exponential fit, grid order/tie breaking,
condition-specific leave-one-subject-out fitting, and fitted-curve noise floor.
Unlike the historical code, a failed exponential fit is an explicit error.
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def fit_energy_exponential(values: np.ndarray) -> np.ndarray:
    """Fit A+B*exp(-t/tau), with the original S5 bounds and initialization."""
    y = np.asarray(values, float)
    if y.shape != (2700,):
        raise ValueError("S5 exponential fitting requires 2700 one-Hz samples")
    finite = np.isfinite(y)
    if finite.sum() < 10:
        raise ValueError("S5 exponential fitting requires at least ten finite samples")
    t = np.arange(2700, dtype=float)

    def model(time, a, b, tau):
        return a + b * np.exp(-time / tau)

    v = y[finite]
    parameters, _ = curve_fit(
        model,
        t[finite],
        v,
        p0=[v[-1], v[0] - v[-1], 300.0],
        bounds=([-200, -500, 5.0], [200, 500, 8100.0]),
        maxfev=10000,
    )
    fitted = model(t, *parameters)
    if not np.all(np.isfinite(fitted)):
        raise ValueError("S5 exponential fitting produced nonfinite values")
    return fitted


def _rmse(first, second):
    difference = np.asarray(first) - np.asarray(second)
    finite = difference[np.isfinite(difference)]
    if not finite.size:
        raise ValueError("S5 RMSE has no finite paired samples")
    return float(np.sqrt(np.mean(finite**2)))


def _paired(rows, speed):
    selected = [r for r in rows if speed == "pooled" or r["delta_mps"] == speed]
    difference = np.array([r["rmse_C"] - r["rmse_E"] for r in selected])
    nonzero = int(np.count_nonzero(difference))
    return {
        "delta_mps": speed,
        "n": len(selected),
        "n_effective": nonzero,
        "n_zero_differences": len(selected) - nonzero,
        "mean_difference_C_minus_E": float(difference.mean()),
        "median_difference_C_minus_E": float(np.median(difference)),
        "wilcoxon_two_sided_p": float(stats.wilcoxon(difference).pvalue) if nonzero else 1.0,
        "method": "scipy.wilcoxon, two-sided, zero_method=wilcox, method=auto",
    }


def analyze_s5_grid(data_root: Path, experimental: dict, participant_ids: dict) -> dict:
    """Refit every cached cell, reselect C/E, and recompute all 26 LOSO folds.

    ``experimental`` contains the original S5 smoothed/onset-normalized participant
    matrices, keyed by speed. Every grid cell and seed is validated before use.
    Saved selection and summary results are not used in this computation.
    """
    base = Path(data_root) / "extensions/s5"
    config = json.loads((base / "grid_config.json").read_text())
    speeds = config["speeds"]
    axes = [config[k] for k in ("learning_rates", "symmetry_weights", "fall_weights")]
    if len(axes[2]) != len(set(axes[2])) or 0.0 not in axes[2]:
        raise ValueError("S5 fall-weight grid must have distinct values including zero")
    cells, mechanical, sources = {}, [], ["extensions/s5/grid_config.json"]
    for speed, lr, sym, weight in product(speeds, *axes):
        relative = f"extensions/s5/cells/sp{speed:g}_lr{lr:.3e}_sym{sym:g}_w{weight:g}.npz"
        with np.load(Path(data_root) / relative, allow_pickle=False) as cell:
            for name, expected in (
                ("speed", speed),
                ("learning_rate", lr),
                ("symmetry_weight", sym),
                ("fall_weight", weight),
            ):
                if not np.isclose(float(cell[name]), expected, rtol=1e-12, atol=0):
                    raise ValueError(f"S5 cached cell metadata differs: {relative}: {name}")
            seeds, failed = np.asarray(cell["ok_seeds"]), np.asarray(cell["failed_seeds"])
            if sorted(np.r_[seeds, failed].tolist()) != sorted(config["seeds"]):
                raise ValueError(f"S5 cached cell has missing or duplicate seeds: {relative}")
            energy = np.asarray(cell["energy_pct"], float)
            penalty, height = (
                np.asarray(cell["mean_penalty"], float),
                np.asarray(cell["min_com"], float),
            )
            if energy.shape != (seeds.size, 2700) or not seeds.size:
                raise ValueError(f"S5 cached cell has invalid energy trajectories: {relative}")
            if penalty.shape != seeds.shape or height.shape != seeds.shape:
                raise ValueError(f"S5 cached cell has invalid mechanical records: {relative}")
            if not np.all(np.isfinite(energy)) or not np.all(np.isfinite([penalty, height])):
                raise ValueError(f"S5 cached cell contains nonfinite values: {relative}")
            curve = fit_energy_exponential(np.mean(energy, axis=0))
            key = (speed, lr, sym, weight)
            cells[key] = curve
            mechanical.append(
                {
                    "delta_mps": speed,
                    "learning_rate": lr,
                    "symmetry_weight": sym,
                    "fall_weight": weight,
                    "successful_seeds": int(seeds.size),
                    "failed_seeds": int(failed.size),
                    "mean_penalty": float(penalty.mean()),
                    "min_com_height_over_seeds": float(height.min()),
                    "max_seed_min_com_height": float(height.max()),
                    "min_seed_mean_penalty": float(penalty.min()),
                    "max_seed_mean_penalty": float(penalty.max()),
                }
            )
        sources.append(relative)
    selections, folds = [], []
    for speed in speeds:
        matrix = np.asarray(experimental[speed], float)
        ids = participant_ids[speed]
        if matrix.ndim != 2 or matrix.shape != (len(ids), 2700) or len(ids) < 2:
            raise ValueError("S5 participant matrix and identifiers differ")
        keys = [key for key in cells if key[0] == speed]
        zero_keys = [key for key in keys if key[-1] == 0]
        group = fit_energy_exponential(np.nanmean(matrix, axis=0))
        for model, options in (("C", keys), ("E", zero_keys)):
            chosen = min(options, key=lambda key: _rmse(cells[key], group))
            selections.append(
                {
                    "delta_mps": speed,
                    "model": model,
                    "learning_rate": chosen[1],
                    "symmetry_weight": chosen[2],
                    "fall_weight": chosen[3],
                    "group_fit_rmse": _rmse(cells[chosen], group),
                }
            )
        for i, participant in enumerate(ids):
            held = fit_energy_exponential(matrix[i])
            train = fit_energy_exponential(np.nanmean(np.delete(matrix, i, axis=0), axis=0))
            row = {
                "delta_mps": speed,
                "subject_id": participant,
                "training_participants": len(ids) - 1,
                "noise_floor_rmse": _rmse(train, held),
            }
            for model, options in (("C", keys), ("E", zero_keys)):
                chosen = min(options, key=lambda key: _rmse(cells[key], train))
                row.update(
                    {
                        f"rmse_{model}": _rmse(cells[chosen], held),
                        f"learning_rate_{model}": chosen[1],
                        f"symmetry_weight_{model}": chosen[2],
                        f"fall_weight_{model}": chosen[3],
                    }
                )
            folds.append(row)
    summaries = []
    for speed in (*speeds, "pooled"):
        rows = [r for r in folds if speed == "pooled" or r["delta_mps"] == speed]
        for label, column in (
            ("C", "rmse_C"),
            ("E", "rmse_E"),
            ("noise_floor", "noise_floor_rmse"),
        ):
            values = np.array([r[column] for r in rows])
            summaries.append(
                {
                    "delta_mps": speed,
                    "model": label,
                    "n": len(rows),
                    "mean_rmse": float(values.mean()),
                    "median_rmse": float(np.median(values)),
                }
            )
    tests = [_paired(folds, speed) for speed in (*speeds, "pooled")]
    ordered = sorted(tests[:-1], key=lambda row: row["wilcoxon_two_sided_p"])
    running = 0.0
    for i, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * row["wilcoxon_two_sided_p"]))
        row["holm_p_within_three_conditions"] = running
    tests[-1]["holm_p_within_three_conditions"] = "not_in_family"
    # Sweep weight at each full-data C learning-rate/symmetry pair; do not refit
    # those two parameters during the mechanical comparison.
    mechanical_slices = [
        row
        for selected in selections
        if selected["model"] == "C"
        for row in mechanical
        if all(row[k] == selected[k] for k in ("delta_mps", "learning_rate", "symmetry_weight"))
    ]
    return {
        "tables": {
            "s5_loso_folds": folds,
            "s5_loso_summary": summaries,
            "s5_loso_tests": tests,
            "s5_recomputed_selection": selections,
            "s5_mechanical_cells": mechanical,
            "s5_mechanical_weight_slices": mechanical_slices,
        },
        "sources": sources,
        "details": {
            "cached_cells_refitted": len(cells),
            "loso_folds": len(folds),
            "simulation_rerun": False,
            "fit_fallbacks": 0,
            "primary_metric": "RMSE between bounded exponential fits over 2700 seconds",
            "noise_floor": "RMSE between the training-group and held-participant exponential fits",
            "mechanical_resolution": "Per-seed mean penalty and minimum COM height; individual step penalties are unavailable",
        },
    }
