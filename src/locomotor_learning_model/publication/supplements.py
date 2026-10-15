"""Portable numerical reproductions of Figure 2B and supplementary analyses.

The published fitted curves and the original reviewer-response numerical caches
are inputs, never digitized illustrations. Replotting a cache is explicitly
distinguished from rerunning an analysis. Source algorithms are documented in
each renderer; no imports from the historical handoff/private archive are needed.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats
from scipy.io import loadmat
from scipy.optimize import OptimizeWarning, curve_fit

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SPEEDS = (0.4, 0.7, 1.0)
PARAMETERS = "original/2d_sweep_trial50_param_result.mat"
ENERGY = "original/data/cosmed/processed/processedEnergyData.mat"
S1_PREFIX = "regenerated/figures/comment1_loso_cv"
STYLE = {
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
}


def _require(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required publication input is missing: {relative}")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Empty source table: {path.name}")
    return rows


def _write_csv(root: Path, name: str, rows: list[dict]) -> Path:
    if not rows:
        raise ValueError(f"Refusing to write an empty source table: {name}")
    path = root / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _save(fig, root: Path, name: str) -> list[Path]:
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _record(identifier, outputs, sources, mode, caveats=(), status="passed", **details):
    return {
        "id": identifier,
        "status": status,
        "outputs": [str(p) for p in outputs],
        "caveats": list(caveats),
        "details": {"mode": mode, "sources": list(sources), **details},
    }


def matlab_movmean(x: np.ndarray, window: int) -> np.ndarray:
    """MATLAB movmean shrink endpoints, including its asymmetric even window.

    This is the original manuscript transform, which differs from S5's
    reviewer-response moving average for even window sizes.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("movmean requires a nonempty finite one-dimensional signal")
    if window < 1 or int(window) != window:
        raise ValueError("window must be a positive integer")
    i = np.arange(x.size)
    lo = np.maximum(0, i - window // 2)
    hi = np.minimum(x.size, i + window - window // 2)
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    return (cumulative[hi] - cumulative[lo]) / (hi - lo)


def stored_fit_metrics(y: np.ndarray, normalized_fit: np.ndarray) -> tuple[float, float]:
    """Recover the least-squares scale and R² of an already stored fit shape.

    No exponential coefficients or learning parameters are refitted.
    """
    y, g = np.asarray(y, float), np.asarray(normalized_fit, float)
    if y.ndim != 1 or y.shape != g.shape or not y.size:
        raise ValueError("Observed signal and stored fit must be equal-length vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(g)):
        raise ValueError("Observed signal and stored fit must be finite")
    u = 1.0 + g / 100.0
    denominator = float(np.dot(u, u))
    sst = float(np.sum((y - y.mean()) ** 2))
    if denominator == 0 or sst == 0:
        raise ValueError("Stored fit has zero scale or observed signal has zero variance")
    scale = float(np.dot(y, u) / denominator)
    return 1.0 - float(np.sum((y - scale * u) ** 2)) / sst, scale


def _load_published(root: Path):
    p = loadmat(
        _require(root, PARAMETERS),
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["static_params", "exp_subj_num"],
    )
    e = loadmat(
        _require(root, ENERGY),
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["split_energy", "tied1_energy"],
    )
    n = np.asarray(p["exp_subj_num"], dtype=int)
    if not np.array_equal(n, [8, 9, 9]) or p["static_params"].shape != (3, 9):
        raise ValueError("Unexpected published subject layout; expected 8/9/9 in a 3×9 array")
    return p["static_params"], n, e


def _figure2(root: Path, output: Path) -> dict:
    """Port of revision/fig2b_timeseries.py, VARIANT=split SUBJ=7 AXIS=pct."""
    parameters, nsub, energy = _load_published(root)
    condition, subject = 1, 7  # Historical SUBJ is zero-based: participant 8, condition 2.
    raw_exp = np.asarray(energy["split_energy"].condition_2, float)
    st = parameters[condition, subject]
    exp_fit = np.asarray(st.exp_energy, float)
    exp_smoothed = matlab_movmean(raw_exp[subject], 180)
    _, exp_scale = stored_fit_metrics(exp_smoothed, exp_fit)
    exp_percent = (exp_smoothed - exp_scale) / exp_scale * 100
    model_smoothed = matlab_movmean(np.asarray(st.model_energy_raw), 100)
    model_fit = np.asarray(st.model_energy, float)
    _, model_scale = stored_fit_metrics(model_smoothed, model_fit[: model_smoothed.size])
    model_percent = (model_smoothed - model_scale) / model_scale * 100
    # The historical handoff script truncates every model participant to the shortest rollout
    # in this condition before it selects SUBJ=7. Preserve that exact convention.
    model_length = min(
        np.asarray(parameters[condition, s].model_energy_raw).size for s in range(nsub[condition])
    )
    model_percent, model_fit = model_percent[:model_length], model_fit[:model_length]
    series = [
        ("experiment", exp_percent, exp_fit, "#990000"),
        ("model", model_percent, model_fit, "#000099"),
    ]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True, layout="constrained")
    lower = min(min(y.min(), f.min()) for _, y, f, _ in series) - 4
    upper = max(max(y.max(), f.max()) for _, y, f, _ in series) + 4
    for ax, (name, values, fit, color) in zip(axes, series):
        percent = np.linspace(0, 100, values.size)
        ax.plot(percent, values, color=color, alpha=0.28, lw=0.8, label="Smoothed signal")
        ax.plot(percent, fit, color=color, lw=2, label="Stored fit")
        ax.set(xlabel="Exposure (%)", title=name.capitalize(), xlim=(0, 100), ylim=(lower, upper))
        rows.extend(
            {
                "source": name,
                "condition": 2,
                "delta_mps": 0.7,
                "subject": 8,
                "sample_index": i,
                "exposure_pct": float(x),
                "smoothed_energy_pct": float(y),
                "stored_fit_pct": float(f),
            }
            for i, (x, y, f) in enumerate(zip(percent, values, fit))
        )
    axes[0].set_ylabel("Energy change (%)")
    axes[1].legend(frameon=False, fontsize=7)
    outputs = [_write_csv(output, "figure2b_timeseries", rows)]
    outputs += _save(fig, output, "figure2b_timeseries")
    return _record(
        "figure2",
        outputs,
        [PARAMETERS, ENERGY],
        "replot_stored_fits",
        [
            "Figure 2B only; conceptual panels are not numerical analyses.",
            "Historical handoff participant 8 is retained; its approximately −52% fitted plateau does not match the final PDF representative plot (approximately −25% experiment and −22% model). The final participant/source remains unresolved.",
            "Experimental time and model strides each span 0–100% of their own exposure.",
            "Model rollout is truncated to the shortest of all nine condition-2 rollouts.",
        ],
        status="partial",
        condition=2,
        subject_one_based=8,
        model_samples=int(model_length),
        experiment_samples=int(exp_percent.size),
    )


def _s1_from_tables(prefix: Path, output: Path, mode: str, sources: list[str]) -> dict:
    rows = _read_csv(Path(f"{prefix}_persubject.csv"))
    saved_test = _read_csv(Path(f"{prefix}_test.csv"))[0]
    saved_stats = _read_csv(Path(f"{prefix}_stats.csv"))
    cs = np.array([float(r["cond_specific"]) for r in rows])
    shared = np.array([float(r["common_size_matched"]) for r in rows])
    cond = np.array([int(r["condition"]) for r in rows])
    if len(rows) != 26 or not np.all(np.isfinite([cs, shared])):
        raise ValueError("S1 requires finite paired errors for all 26 participants")
    # The final manuscript reports MATLAB signrank's default approximation for
    # n=26. Display that value; retain the historical styled plot's exact test
    # separately so that the two statistical methods cannot be confused.
    p_exact = float(stats.wilcoxon(cs, shared, method="exact").pvalue)
    p_matlab = float(saved_test["wilcoxon_p_condspec_vs_commonsize"])
    summary = [
        {
            "median_condition_specific": float(np.median(cs)),
            "median_common_size_matched": float(np.median(shared)),
            "median_ratio_common_over_specific": float(np.median(shared) / np.median(cs)),
            "participants_better": int(np.sum(cs < shared)),
            "participants": len(rows),
            "wilcoxon_two_sided_exact_scipy": p_exact,
            "wilcoxon_two_sided_matlab_signrank_default": p_matlab,
            "analysis_mode": mode,
        }
    ]
    outputs = [
        _write_csv(output, "s1_loso_persubject", rows),
        _write_csv(output, "s1_loso_stats", saved_stats),
        _write_csv(output, "s1_loso_matlab_test", [saved_test]),
        _write_csv(output, "s1_loso_summary", summary),
    ]
    fig, ax = plt.subplots(figsize=(3.6, 3.7), layout="constrained")
    boxes = ax.boxplot(
        [cs, shared],
        positions=[0, 1],
        widths=0.46,
        showfliers=False,
        patch_artist=True,
        zorder=3,
        medianprops={"color": "0.15", "lw": 1.8},
    )
    for patch, color in zip(boxes["boxes"], ["#EDEDF5", "#F5EDED"]):
        patch.set_facecolor(color)
    colors = [plt.cm.Purples(v) for v in (0.50, 0.70, 0.92)]
    for c, color in enumerate(colors, 1):
        indices = np.flatnonzero(cond == c)
        for i in indices:
            ax.plot([0.03, 0.97], [cs[i], shared[i]], color=color, lw=0.7, alpha=0.75)
        ax.scatter(
            np.full(len(indices), 0.03),
            cs[indices],
            color=color,
            s=9,
            zorder=4,
            label=f"{SPEEDS[c - 1]:.1f} m/s",
        )
        ax.scatter(np.full(len(indices), 0.97), shared[indices], color=color, s=9, zorder=4)
    ax.set(
        xticks=[0, 1],
        xticklabels=["Condition-\nspecific", "Shared\n(size-matched)"],
        xlim=(-0.45, 1.45),
        ylabel="Leave-one-subject-out held-out RMSE (%)",
    )
    ax.text(
        0.03,
        0.97,
        f"Median {np.median(cs):.1f} vs {np.median(shared):.1f}\n"
        f"{np.sum(cs < shared)}/{len(rows)} participants better\n"
        f"Wilcoxon p = {p_matlab:.1e} (MATLAB)",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        fontsize=7,
        handlelength=1,
        columnspacing=0.8,
    )
    outputs += _save(fig, output, "s1_loso")
    return _record(
        "s1",
        outputs,
        sources,
        mode,
        [
            "Uses the regenerated trial-average grid, not the original best-trial grid.",
            "Figure displays the manuscript's MATLAB signrank approximation; the historical SciPy exact value is exported separately.",
            "Cached replot does not rerun LOSO sampling."
            if mode == "saved_reference_replot"
            else "Fresh LOSO is evaluated on a stored RMSE grid; simulations are not rerun.",
        ],
        **summary[0],
    )


def _s1(root: Path, output: Path) -> dict:
    sources = [f"{S1_PREFIX}_{suffix}.csv" for suffix in ("persubject", "stats", "test")]
    for source in sources:
        _require(root, source)
    return _s1_from_tables(root / S1_PREFIX, output, "saved_reference_replot", sources)


def recompute_s1_matlab(
    data_root: Path, output_root: Path, matlab_executable: str = "matlab"
) -> dict:
    """Rerun original MATLAB LOSO analysis using the supplied trial-average grid.

    Requires MATLAB and Statistics and Machine Learning Toolbox; Curve Fitting
    Toolbox is unnecessary. Input MAT files and reference CSVs are never changed.
    """
    data_root, output_root = Path(data_root), Path(output_root)
    source = "regenerated/2d_sweep_trial50_param_result_trialavg.mat"
    grid = _require(data_root, source).resolve()
    executable = shutil.which(str(matlab_executable))
    if executable is None:
        raise FileNotFoundError(f"MATLAB executable is unavailable: {matlab_executable}")
    vendor = Path(__file__).with_name("vendor")
    tables = (output_root / "tables" / "s1_matlab").resolve()
    tables.mkdir(parents=True, exist_ok=True)
    quote = lambda value: str(value).replace("'", "''")
    command = f"addpath('{quote(vendor)}'); publication_s1_loso('{quote(grid)}','{quote(tables)}');"
    result = subprocess.run(
        [executable, "-batch", command], capture_output=True, text=True, timeout=600, check=False
    )
    log = output_root / "tables" / "s1_matlab.log"
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"MATLAB LOSO failed (exit {result.returncode}); see {log}")
    with plt.rc_context(STYLE):
        record = _s1_from_tables(
            tables / "comment1_loso_cv", output_root, "fresh_matlab_loso_on_cached_grid", [source]
        )
    record["outputs"] += [str(log)] + [str(p) for p in sorted(tables.glob("*.csv"))]
    return record


def _s2(root: Path, output: Path) -> dict:
    """Original comment_2 scalar statistics and numerical trajectory caches.

    Student-t intervals are recomputed from the original participant summaries.
    Cached model means and GRF-derived trajectories are replotted without raw
    GRF processing or new model simulations.
    """
    base = "extensions/s2"
    sources = [
        f"{base}/{name}"
        for name in (
            "per_subject_temporal.csv",
            "analysis_summary.json",
            "group_trajectories.npz",
            "model_temporal.npz",
        )
    ]
    paths = [_require(root, source) for source in sources]
    participants = _read_csv(paths[0])
    summary = json.loads(paths[1].read_text(encoding="utf-8"))
    with np.load(paths[2], allow_pickle=False) as loaded:
        group = {key: loaded[key] for key in loaded.files}
    with np.load(paths[3], allow_pickle=False) as loaded:
        model = {key: loaded[key] for key in loaded.files}
    rows = []
    maximum_reference_error = 0.0
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5), layout="constrained")
    for metric_index, (metric, column, label) in enumerate(
        (
            ("sla", "sla_adapted_fp", "Step length asymmetry"),
            ("sta", "sta_adapted", "Step time asymmetry"),
        )
    ):
        data_means, lower, upper, model_means = [], [], [], []
        for speed in SPEEDS:
            values = np.array(
                [float(row[column]) for row in participants if float(row["speed_diff_ms"]) == speed]
            )
            if len(values) < 2 or not np.all(np.isfinite(values)):
                raise ValueError(f"S2 {metric}: missing participant values at {speed} m/s")
            mean = float(values.mean())
            half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
            reference = summary["statistics"][
                "sla_by_condition" if metric == "sla" else "by_condition"
            ][str(speed)]
            reference_prefix = "" if metric == "sla" else "sta_"
            saved = [
                reference[f"{reference_prefix}adapted_mean"],
                *reference[f"{reference_prefix}adapted_ci95"],
            ]
            error = float(np.max(np.abs(np.array([mean, mean - half, mean + half]) - saved)))
            maximum_reference_error = max(error, maximum_reference_error)
            if error > 1e-10:
                raise ValueError(
                    "S2 participant statistics do not match the original analysis cache"
                )
            prediction = float(model[f"scalar_{metric}_adapted_{speed}"])
            rows.append(
                {
                    "metric": metric,
                    "delta_mps": speed,
                    "n": len(values),
                    "experimental_mean": mean,
                    "experimental_ci95_low": mean - half,
                    "experimental_ci95_high": mean + half,
                    "model_mean": prediction,
                    "model_learning_rate": float(model[f"scalar_learning_rate_{speed}"]),
                    "model_symmetry_weight": float(model[f"scalar_symmetry_weight_{speed}"]),
                }
            )
            data_means.append(mean)
            lower.append(mean - half)
            upper.append(mean + half)
            model_means.append(prediction)
        data_means = np.array(data_means)
        axes[metric_index, 0].errorbar(
            SPEEDS,
            data_means,
            yerr=[data_means - lower, np.array(upper) - data_means],
            color=".35",
            fmt="o-",
            capsize=3,
            lw=1.6,
        )
        axes[metric_index, 1].plot(SPEEDS, model_means, "o--", color="#000099", lw=1.6)
        if metric == "sla":
            for weight in (0.0, 0.5):
                values = [
                    summary["model_symmetry_sweep"][f"speed{s}_psi{weight}"]["sla_adapted"]
                    for s in SPEEDS
                ]
                axes[metric_index, 1].plot(
                    SPEEDS,
                    values,
                    "--",
                    color="#000099",
                    alpha=0.3,
                    label=f"Symmetry weight {weight:g}",
                )
            axes[metric_index, 1].legend(frameon=False, fontsize=7)
        lo = min(*lower, *model_means)
        hi = max(*upper, *model_means)
        pad = 0.12 * (hi - lo) or 0.01
        for col, ax in enumerate(axes[metric_index]):
            ax.axhline(0, color=".8", lw=0.6, zorder=0)
            ax.set(
                xlabel="Speed difference (m/s)",
                ylabel=label,
                title=("Experimental data" if col == 0 else "Model"),
                xticks=SPEEDS,
                ylim=(lo - pad, hi + pad),
            )
    outputs = [
        _write_csv(output, "s2_adapted", rows),
        _write_csv(output, "s2_participants", participants),
    ]
    speed_values = np.array([float(row["speed_diff_ms"]) for row in participants])
    adapted_sta = np.array([float(row["sta_adapted"]) for row in participants])
    early_sta = np.array([float(row["sta_early"]) for row in participants])
    slope = stats.linregress(speed_values, adapted_sta)
    paired = stats.ttest_rel(adapted_sta, early_sta)
    change = adapted_sta - early_sta
    half = float(stats.t.ppf(0.975, len(change) - 1) * stats.sem(change))
    statistics = [
        {
            "comparison": "adapted_STA_vs_speed",
            "n": len(participants),
            "method": "linear regression",
            "estimate": float(slope.slope),
            "ci95_low": "",
            "ci95_high": "",
            "p_two_sided": float(slope.pvalue),
        },
        {
            "comparison": "adapted_STA_minus_early_STA",
            "n": len(participants),
            "method": "paired t test",
            "estimate": float(change.mean()),
            "ci95_low": float(change.mean() - half),
            "ci95_high": float(change.mean() + half),
            "p_two_sided": float(paired.pvalue),
        },
    ]
    if not np.isclose(slope.pvalue, summary["statistics"]["sta_vs_speed"]["p"], rtol=1e-10):
        raise ValueError("S2 STA regression differs from the original source")
    if not np.isclose(
        paired.pvalue, summary["statistics"]["sta_adapted_vs_early"]["p"], rtol=1e-10
    ):
        raise ValueError("S2 STA adaptation test differs from the original source")
    outputs.append(_write_csv(output, "s2_sta_statistics", statistics))
    sweep_rows = []
    for speed in SPEEDS:
        for weight in (0.0, 0.5, 1.0):
            sweep_rows.append(
                {
                    "delta_mps": speed,
                    "symmetry_weight": weight,
                    **summary["model_symmetry_sweep"][f"speed{speed}_psi{weight}"],
                }
            )
    outputs.append(_write_csv(output, "s2_symmetry_sweep", sweep_rows))
    outputs += _save(fig, output, "s2_adapted")
    fig, axes = plt.subplots(2, 2, figsize=(7, 5), layout="constrained")
    time_rows = []
    colors = ["#2C7BB6", "#FDAE61", "#D7191C"]
    for r, (metric, label) in enumerate(
        (("sla", "Step length asymmetry"), ("sta", "Step time asymmetry"))
    ):
        limits = []
        for c, source in enumerate(("experiment", "model")):
            z = group if c == 0 else model
            edges = np.asarray(z["edges"], float)
            minutes = (edges[:-1] + edges[1:]) / 120
            for speed, color in zip(SPEEDS, colors):
                mean = np.asarray(
                    z[f"{metric}_{speed}_mean" if c == 0 else f"{metric}_mean_{speed}"], float
                )
                sem = np.asarray(z[f"{metric}_{speed}_sem"], float) if c == 0 else None
                if mean.shape != minutes.shape or not np.all(np.isfinite(mean)):
                    raise ValueError("S2 trajectory cache has invalid dimensions or values")
                axes[r, c].plot(
                    minutes, mean, "-" if c == 0 else "--", color=color, label=f"{speed:.1f} m/s"
                )
                if sem is not None:
                    axes[r, c].fill_between(
                        minutes, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18
                    )
                    limits.extend([np.min(mean - 1.96 * sem), np.max(mean + 1.96 * sem)])
                else:
                    limits.extend([mean.min(), mean.max()])
                time_rows.extend(
                    {
                        "metric": metric,
                        "source": source,
                        "delta_mps": speed,
                        "bin_start_s": float(edges[i]),
                        "bin_end_s": float(edges[i + 1]),
                        "time_min": float(time),
                        "mean": float(value),
                        "sem": float(sem[i]) if sem is not None else "",
                    }
                    for i, (time, value) in enumerate(zip(minutes, mean))
                )
            axes[r, c].set(
                xlabel="Time (min)", ylabel=label, title="Experimental data" if c == 0 else "Model"
            )
        pad = 0.06 * (max(limits) - min(limits))
        for ax in axes[r]:
            ax.set_ylim(min(limits) - pad, max(limits) + pad)
    axes[0, 0].legend(frameon=False, fontsize=7)
    outputs.append(_write_csv(output, "s2_timeseries", time_rows))
    outputs += _save(fig, output, "s2_timeseries")
    return _record(
        "s2",
        outputs,
        sources,
        "recomputed_summary_and_cached_trajectories",
        [
            "Participant summaries and model trajectories come from original comment_2 computations; no SVG digitization.",
            "Adapted intervals are Student-t 95% CIs; time-course shading is 1.96 × SEM.",
            "S2 uses 20 participants (7/7/6 by condition) and shared model parameters alpha=6e-5, psi=1 selected by the energy-only Model A; these differ from the main individual condition-specific fits.",
            "A nonsignificant STA adaptation test is not an equivalence test establishing zero adaptation.",
            "Raw force/marker preprocessing and model simulations are not rerun.",
        ],
        participants=len(participants),
        max_reference_error=maximum_reference_error,
    )


def _s3(root: Path, output: Path) -> dict:
    source = _require(root, ENERGY)
    energy = loadmat(
        source,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=["split_energy", "tied1_energy"],
    )
    experimental = []
    means = []
    for condition, speed in enumerate(SPEEDS, 1):
        split = np.asarray(getattr(energy["split_energy"], f"condition_{condition}"), float)
        tied = np.asarray(getattr(energy["tied1_energy"], f"condition_{condition}"), float)
        baseline = tied[:, -180:].mean(axis=1)
        if split.shape[0] != tied.shape[0] or np.any(baseline == 0):
            raise ValueError("S3 split/tied participants differ or tied baseline is zero")
        adapted = split[:, -180:].mean(axis=1)
        percent = (adapted - baseline) / baseline * 100
        if not np.all(np.isfinite(percent)):
            raise ValueError("S3 experimental energy contains nonfinite values")
        means.append(float(percent.mean()))
        experimental.extend(
            {
                "condition": condition,
                "delta_mps": speed,
                "subject": i + 1,
                "tied_last180_mean": float(baseline[i]),
                "split_last180_mean": float(adapted[i]),
                "energy_change_pct": float(value),
            }
            for i, value in enumerate(percent)
        )
    trajectory_speeds = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    trajectory_energy = np.array([3.161, 3.183, 3.138, 3.046, 2.968])
    trajectory_pct = (trajectory_energy - 3.147) / 3.147 * 100
    trajectory = [
        {
            "delta_mps": float(speed),
            "tied_energy": 3.147,
            "split_energy": float(value),
            "energy_change_pct": float(pct),
        }
        for speed, value, pct in zip(trajectory_speeds, trajectory_energy, trajectory_pct)
    ]
    outputs = [
        _write_csv(output, "s3a_trajectory_optimization", trajectory),
        _write_csv(output, "s3c_experimental", experimental),
        _write_csv(
            output,
            "s3c_group_means",
            [
                {"delta_mps": speed, "mean_energy_change_pct": mean}
                for speed, mean in zip(SPEEDS, means)
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.9), layout="constrained")
    for ax, letter, title, x, y, color in (
        (axes[0], "A", "Trajectory optimization", trajectory_speeds, trajectory_pct, "#009900"),
        (axes[1], "C", "Experimental data", SPEEDS, means, "#990000"),
    ):
        ax.axhline(0, color=".75", lw=0.7, ls=":")
        ax.plot(x, y, "o-", color=color, lw=1.8)
        ax.set(
            xlabel="Speed difference (m/s)",
            ylabel="Energy change (%)",
            title=f"{letter}  {title}",
            xticks=x,
        )
    axes[0].set_ylim(-7, 2.5)
    axes[1].set_ylim(-15, 27)
    fig.suptitle("S3 partial reproduction — panel B unavailable", fontsize=9)
    outputs += _save(fig, output, "s3_ac_partial")
    return _record(
        "s3",
        outputs,
        [ENERGY, "trajectory-optimization constants in supplements.py"],
        "experimental_reanalysis_and_reported_trajectory_values",
        [
            "Panel B is unavailable: raw regenerated model sweeps required for its tied baseline are missing.",
            "Panel A uses five reported OpenSim/Moco outputs; trajectory optimization is not rerun.",
            "The stale handoff S3 model CSV is deliberately not used.",
        ],
        status="partial",
        reproduced_panels=["A", "C"],
        missing_panels=["B"],
        experimental_means=means,
    )


def reviewer_energy_percent(
    raw: np.ndarray, window: int = 180, onset_window: int = 30
) -> np.ndarray:
    """Original S5 NaN-aware smoothing and 30-s smoothed-onset normalization.

    With window=180 the historical centered implementation includes 181 points
    in the interior (i-90 through i+90). Preserve this rather than silently using
    MATLAB's 180-point manuscript convention.
    """
    raw = np.asarray(raw, float)
    if raw.ndim != 1 or not raw.size or window < 1 or onset_window < 1:
        raise ValueError("Invalid S5 energy signal or smoothing window")
    finite = np.isfinite(raw)
    cs = np.concatenate(([0.0], np.cumsum(np.where(finite, raw, 0.0))))
    counts = np.concatenate(([0], np.cumsum(finite)))
    i = np.arange(raw.size)
    lo, hi = np.maximum(0, i - window // 2), np.minimum(raw.size, i + window // 2 + 1)
    n = counts[hi] - counts[lo]
    smoothed = np.divide(cs[hi] - cs[lo], n, out=np.full(raw.size, np.nan), where=n > 0)
    onset = float(np.nanmean(smoothed[:onset_window]))
    if not np.isfinite(onset) or abs(onset) < 1e-6:
        raise ValueError("S5 energy has an invalid smoothed-onset baseline")
    return (smoothed - onset) / onset * 100


def _s5(root: Path, output: Path) -> dict:
    from .s5_analysis import analyze_s5_grid

    base = "extensions/s5"
    selection_source = f"{base}/analysis_summary.json"
    energy_source = "extensions/experimental_csv/split_energy.csv"
    selection = json.loads(_require(root, selection_source).read_text())["selection"]
    raw_rows = _read_csv(_require(root, energy_source))
    time_columns = [key for key in raw_rows[0] if key.startswith("t") and key[1:].isdigit()]
    if len(time_columns) != 2700:
        raise ValueError("S5 requires 2700 ordered experimental energy samples")
    if [int(key[1:]) for key in time_columns] != sorted(int(key[1:]) for key in time_columns):
        raise ValueError("S5 experimental time columns are not ordered")
    sources = [selection_source, energy_source]
    rows, selections, subject_rows = [], [], []
    experimental, participant_ids = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, speed in zip(axes, SPEEDS):
        condition_rows = [row for row in raw_rows if float(row["speed_diff_ms"]) == speed]
        if len(condition_rows) != (8 if speed == 0.4 else 9):
            raise ValueError(f"S5 has an unexpected energy cohort at {speed} m/s")
        em = np.vstack(
            [
                reviewer_energy_percent(np.array([float(row[k]) for k in time_columns]))
                for row in condition_rows
            ]
        )
        experimental[speed] = em
        participant_ids[speed] = [row["subject_id"] for row in condition_rows]
        mean = np.nanmean(em, axis=0)
        sem = np.nanstd(em, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(em), axis=0))
        minutes = np.arange(2700) / 60
        predictions = {}
        for model_name, label in (("C", "fall_weight_free"), ("E", "fall_weight_zero")):
            parameters = selection[model_name]["per_speed"][str(speed)]
            lr, sym, weight = (float(parameters[k]) for k in ("lr", "sym", "w"))
            if model_name == "E" and weight != 0:
                raise ValueError("S5 no-fall selection has a nonzero fall weight")
            relative = f"{base}/cells/sp{speed:g}_lr{lr:.3e}_sym{sym:g}_w{weight:g}.npz"
            sources.append(relative)
            with np.load(_require(root, relative), allow_pickle=False) as cell:
                for key, expected in (
                    ("speed", speed),
                    ("learning_rate", lr),
                    ("symmetry_weight", sym),
                    ("fall_weight", weight),
                ):
                    if not np.isclose(float(cell[key]), expected, rtol=1e-12, atol=0):
                        raise ValueError(f"S5 cell metadata does not match selected {key}")
                values = np.asarray(cell["energy_pct"], float)
                successful = int(cell["ok_seeds"].size)
                failed = int(cell["failed_seeds"].size)
                if values.ndim != 2 or values.shape != (successful, 2700) or not successful:
                    raise ValueError("S5 cell has no valid complete energy trajectories")
                predictions[label] = np.nanmean(values, axis=0)
                if not np.all(np.isfinite(predictions[label])):
                    raise ValueError("S5 selected model mean has nonfinite values")
            selections.append(
                {
                    "delta_mps": speed,
                    "model": model_name,
                    "label": label,
                    "learning_rate": lr,
                    "symmetry_weight": sym,
                    "fall_weight": weight,
                    "successful_seeds": successful,
                    "failed_seeds": failed,
                }
            )
        rows.extend(
            {
                "delta_mps": speed,
                "time_s": i,
                "time_min": float(minutes[i]),
                "experiment_mean_pct": float(mean[i]),
                "experiment_sem_pct": float(sem[i]),
                "model_C_fall_weight_free_pct": float(predictions["fall_weight_free"][i]),
                "model_E_fall_weight_zero_pct": float(predictions["fall_weight_zero"][i]),
            }
            for i in range(2700)
        )
        for i, row in enumerate(condition_rows):
            subject_rows.extend(
                {
                    "delta_mps": speed,
                    "subject_id": row["subject_id"],
                    "time_s": j,
                    "energy_change_pct": float(value),
                }
                for j, value in enumerate(em[i])
            )
        ax.fill_between(minutes, mean - sem, mean + sem, color="#D9D9D9")
        ax.plot(minutes, mean, color="black", lw=1.8, label="Experiment")
        ax.plot(
            minutes,
            predictions["fall_weight_zero"],
            "--",
            color="#8B0000",
            lw=1.7,
            label="Model without fall penalty",
        )
        ax.plot(
            minutes,
            predictions["fall_weight_free"],
            color="#0000CD",
            lw=1.7,
            label="Model with fall penalty",
        )
        weight = selection["C"]["per_speed"][str(speed)]["w"]
        ax.set(
            xlabel="Time (min)",
            title=f"{speed:.1f} m/s  (w = {weight:g})",
            xlim=(0, 45),
            xticks=[0, 15, 30, 45],
        )
    axes[0].set_ylabel("Energy change (%)")
    axes[1].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=3, fontsize=8
    )
    outputs = [
        _write_csv(output, "s5_energy_timeseries", rows),
        _write_csv(output, "s5_selected_models", selections),
        _write_csv(output, "s5_experimental_subjects", subject_rows),
    ]
    outputs += _save(fig, output, "s5_energy")
    analysis = analyze_s5_grid(root, experimental, participant_ids)
    for name, values in analysis["tables"].items():
        outputs.append(_write_csv(output, name, values))
    for fresh in analysis["tables"]["s5_recomputed_selection"]:
        saved = next(
            row
            for row in selections
            if row["delta_mps"] == fresh["delta_mps"] and row["model"] == fresh["model"]
        )
        for key in ("learning_rate", "symmetry_weight", "fall_weight"):
            if saved[key] != fresh[key]:
                raise ValueError(f"S5 fresh selection disagrees with archived selection: {key}")
    sources.extend(analysis["sources"])
    return _record(
        "s5",
        outputs,
        list(dict.fromkeys(sources)),
        "experimental_reanalysis_and_original_cached_model_trajectories",
        [
            "Original comment_4 current cells are used; no SVG digitization or obsolete buggy-interpolation cells.",
            "Model C versus E allows learning rate and symmetry to refit; this is not a fixed-parameter fall-term toggle.",
            "All cached model means are refitted and C/E model selection and LOSO are recomputed; simulations are not rerun.",
            "Historical S5 smoothing includes 181 interior samples for its window=180 setting.",
            "The noise floor compares fitted exponential curves, not raw measured trajectories.",
            "Saved per-seed aggregate mechanical metrics cannot establish that every step has a nonzero penalty gradient.",
        ],
        selections=selections,
        analysis=analysis["details"],
    )


def compare_exponential_timescales(values: np.ndarray) -> dict:
    """Repeat the historical single/double-timescale diagnostic on one signal.

    This is the original unbounded SciPy diagnostic, not a refit of the MATLAB
    headline R². Preserve its initialization and nonnegative reported gain, but
    also export the unclipped result and optimizer covariance-warning state.
    """
    y = np.asarray(values, float)
    if y.ndim != 1 or y.size < 6 or not np.all(np.isfinite(y)) or np.var(y) == 0:
        raise ValueError("Exponential comparison requires a finite, nonconstant vector")
    x = np.arange(1, y.size + 1, dtype=float)

    def single(t, a, b, c):
        return a * np.exp(b * t) + c

    def double(t, a, b, c, d, g):
        return a * np.exp(b * t) + c * np.exp(d * t) + g

    with (
        warnings.catch_warnings(record=True) as caught,
        np.errstate(over="ignore", invalid="ignore"),
    ):
        warnings.simplefilter("always", OptimizeWarning)
        first, _ = curve_fit(
            single, x, y, p0=[y[0] - y[-1], -1 / (0.3 * y.size), y[-1]], maxfev=200000
        )
        second, _ = curve_fit(
            double,
            x,
            y,
            p0=[first[0] / 2, first[1], first[0] / 2, first[1] * 6, first[2]],
            maxfev=400000,
        )
    sst = np.sum((y - y.mean()) ** 2)
    r1 = float(1 - np.sum((y - single(x, *first)) ** 2) / sst)
    r2 = float(1 - np.sum((y - double(x, *second)) ** 2) / sst)
    if not np.all(np.isfinite([r1, r2])):
        raise ValueError("Exponential comparison produced nonfinite R²")
    return {
        "r2_single_refit": r1,
        "r2_double_refit_raw": r2,
        "r2_double_refit_reported": max(r1, r2),
        "delta_r2_reported": max(0.0, r2 - r1),
        "covariance_warning": any(issubclass(w.category, OptimizeWarning) for w in caught),
    }


def _r2(root: Path, output: Path) -> dict:
    parameters, nsub, energy = _load_published(root)
    rows, timescales, rmse_rows = [], [], []
    for condition, speed in enumerate(SPEEDS):
        raw = np.asarray(getattr(energy["split_energy"], f"condition_{condition + 1}"), float)
        for subject in range(nsub[condition]):
            st = parameters[condition, subject]
            calculated_rmse = float(
                np.sqrt(np.mean((np.asarray(st.model_energy) - np.asarray(st.exp_energy)) ** 2))
            )
            if not np.isclose(calculated_rmse, float(st.rmse), rtol=1e-10, atol=1e-10):
                raise ValueError(
                    "Stored inverse-fit RMSE disagrees with its stored exponential curves"
                )
            rmse_rows.append(
                {
                    "condition": condition + 1,
                    "delta_mps": speed,
                    "subject": subject + 1,
                    "rmse_stored": float(st.rmse),
                    "rmse_recomputed": calculated_rmse,
                    "units": "percentage points relative to split onset",
                }
            )
            ye = matlab_movmean(raw[subject], 180)
            r2e, _ = stored_fit_metrics(ye, np.asarray(st.exp_energy))
            ym = matlab_movmean(np.asarray(st.model_energy_raw), 100)
            gm = np.asarray(st.model_energy)[: ym.size]
            for source, observed in (("experiment", ye), ("model", ym)):
                timescales.append(
                    {
                        "condition": condition + 1,
                        "delta_mps": speed,
                        "subject": subject + 1,
                        "source": source,
                        **compare_exponential_timescales(observed),
                    }
                )
            r2m, scale = stored_fit_metrics(ym, gm)
            a, b, c, d = np.asarray(st.coeff, float).ravel()
            coefficient_scale = float(a * np.exp(b) + c * np.exp(d))
            fit = scale * (1 + gm / 100)
            smooth_r2 = []
            for window in (300, 500):
                observed = matlab_movmean(np.asarray(st.model_energy_raw), window)
                smooth_r2.append(
                    1 - np.sum((observed - fit) ** 2) / np.sum((observed - observed.mean()) ** 2)
                )
            rows.append(
                {
                    "condition": condition + 1,
                    "delta_mps": speed,
                    "subject": subject + 1,
                    "r2_experimental": float(r2e),
                    "r2_model": float(r2m),
                    "model_scale_leastsq": scale,
                    "model_scale_from_coeff": coefficient_scale,
                    "scale_check_pct_diff": float(
                        abs(scale - coefficient_scale) / coefficient_scale * 100
                    ),
                    "d_coeff_model": float(d),
                    "r2_model_smooth300": float(smooth_r2[0]),
                    "r2_model_smooth500": float(smooth_r2[1]),
                }
            )
    summary = []
    for speed in (*SPEEDS, "pooled"):
        selected = [r for r in rows if speed == "pooled" or r["delta_mps"] == speed]
        for key in ("r2_experimental", "r2_model"):
            values = np.array([r[key] for r in selected])
            summary.append(
                {
                    "delta_mps": speed,
                    "metric": key,
                    "n": len(values),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )
    sources = [PARAMETERS, ENERGY]
    reference = root / "reference_tables/expfit_r2.csv"
    verified = False
    if reference.is_file():
        ref_rows = _read_csv(reference)
        if len(ref_rows) != len(rows):
            raise ValueError("R² reference has a different participant count")
        for actual, expected in zip(rows, ref_rows):
            for key in (
                "condition",
                "subject",
                "delta_mps",
                "r2_experimental",
                "r2_model",
                "r2_model_smooth300",
                "r2_model_smooth500",
            ):
                if not np.isclose(actual[key], float(expected[key]), atol=5.01e-5, rtol=0):
                    raise ValueError(f"R² differs from published reference for {key}")
        verified = True
        sources.append("reference_tables/expfit_r2.csv")
    outputs = [_write_csv(output, "r2_expfit", rows), _write_csv(output, "r2_summary", summary)]
    rmse_summary = []
    for speed in SPEEDS:
        values = np.array(
            [row["rmse_recomputed"] for row in rmse_rows if row["delta_mps"] == speed]
        )
        rmse_summary.append(
            {
                "delta_mps": speed,
                "n": len(values),
                "mean_rmse": float(values.mean()),
                "sd_rmse": float(values.std(ddof=1)),
                "units": "percentage points relative to split onset",
            }
        )
    outputs += [
        _write_csv(output, "inverse_fit_rmse_subjects", rmse_rows),
        _write_csv(output, "inverse_fit_rmse_summary", rmse_summary),
    ]
    timescale_summary = []
    for speed in SPEEDS:
        for source in ("experiment", "model"):
            chosen = [r for r in timescales if r["delta_mps"] == speed and r["source"] == source]
            timescale_summary.append(
                {
                    "delta_mps": speed,
                    "source": source,
                    "n": len(chosen),
                    "mean_delta_r2": float(np.mean([r["delta_r2_reported"] for r in chosen])),
                    "mean_single_r2": float(np.mean([r["r2_single_refit"] for r in chosen])),
                    "mean_double_r2": float(
                        np.mean([r["r2_double_refit_reported"] for r in chosen])
                    ),
                }
            )
    outputs += [
        _write_csv(output, "r2_timescale_comparison", timescales),
        _write_csv(output, "r2_timescale_summary", timescale_summary),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), layout="constrained")
    for ax, key, title, color in (
        (axes[0], "r2_experimental", "Experimental data", "#990000"),
        (axes[1], "r2_model", "Model", "#000099"),
    ):
        for i, speed in enumerate(SPEEDS):
            values = np.array([r[key] for r in rows if r["delta_mps"] == speed])
            ax.scatter(np.linspace(i - 0.1, i + 0.1, len(values)), values, color=color, s=14)
            ax.plot([i - 0.18, i + 0.18], [values.mean()] * 2, color="black")
        ax.set(
            title=title,
            ylabel="Stored-fit R²",
            xlabel="Speed difference (m/s)",
            xticks=range(3),
            xticklabels=[f"{s:.1f}" for s in SPEEDS],
        )
    outputs += _save(fig, output, "r2_expfit")
    return _record(
        "r2",
        outputs,
        sources,
        "recomputed_metrics_of_stored_fits",
        [
            "Stored normalized exponential shapes are rescaled by least squares; curves are not refitted.",
            "300/500-window columns score the same fit against more smoothed model signals.",
            "Separate timescale-comparison tables repeat the original unbounded SciPy single/double exponential diagnostic; these are distinct from the MATLAB headline fits.",
            "The 0.011–0.055 gain range spans model and experiment condition means; experiment-only gains are approximately 0.015–0.027.",
        ],
        reference_verified=verified,
        max_coefficient_scale_disagreement_pct=max(r["scale_check_pct_diff"] for r in rows),
    )


def render_supplements(
    data_root: Path, output_root: Path, figures: list[str] | None = None
) -> list[dict]:
    """Render selected available supplements, exporting every plotted value.

    Unknown selections and missing required data fail explicitly. S3's known
    missing panel is reported as partial rather than replaced with stale values.
    """
    renderers = {"figure2": _figure2, "s1": _s1, "s2": _s2, "s3": _s3, "s5": _s5, "r2": _r2}
    selected = list(renderers) if figures is None else list(dict.fromkeys(figures))
    unknown = set(selected) - renderers.keys()
    if unknown:
        raise ValueError(f"Unknown supplementary figure selection: {', '.join(sorted(unknown))}")
    with plt.rc_context(STYLE):
        return [renderers[name](Path(data_root), Path(output_root)) for name in selected]
