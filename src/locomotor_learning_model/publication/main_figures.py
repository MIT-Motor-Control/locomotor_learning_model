"""Portable cached-data reproduction of main figures and landscape slices."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.stats import ttest_rel

SPEEDS = (0.4, 0.7, 1.0)
OPT_SPEEDS = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
OPT_ENERGY = (np.array([3.161, 3.183, 3.138, 3.046, 2.968]) - 3.147) / 3.147 * 100
OPT_SLA = np.array([11.268, 16.814, 23.186, 29.740, 42.391])


def _mat(path: Path, **kwargs) -> dict:
    with path.open("rb") as stream:
        return loadmat(stream, struct_as_record=False, squeeze_me=True, **kwargs)


def _csv(root: Path, name: str, rows: list[dict]) -> str:
    if not rows:
        raise ValueError(f"Refusing to export an empty source table: {name}")
    path = root / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path.resolve())


def _plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(root: Path, name: str, figure) -> list[str]:
    from matplotlib import pyplot as plt

    folder = root / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = folder / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=180)
        paths.append(str(path.resolve()))
    plt.close(figure)
    return paths


def _record(identifier, status="passed", outputs=None, caveats=None, details=None):
    return {
        "id": identifier,
        "status": status,
        "outputs": outputs or [],
        "caveats": caveats or [],
        "details": details or {},
    }


def matlab_box_summary(values) -> dict:
    """Match MATLAB boxplot midpoint percentiles and its 1.5-IQR whiskers."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Boxplots need finite, nonempty one-dimensional participant data")
    q1, median, q3 = np.percentile(values, [25, 50, 75], method="hazen")
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    return {
        "n": len(values),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "whisker_low": float(min(q1, values[values >= low].min())),
        "whisker_high": float(max(q3, values[values <= high].max())),
        "hidden_outliers": int(np.sum((values < low) | (values > high))),
    }


def _boxes(axis, groups, positions, widths=0.5) -> list[dict]:
    summaries = [matlab_box_summary(group) for group in groups]
    axis.bxp(
        [
            {
                "q1": s["q1"],
                "med": s["median"],
                "q3": s["q3"],
                "whislo": s["whisker_low"],
                "whishi": s["whisker_high"],
            }
            for s in summaries
        ],
        positions=positions,
        widths=widths,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#990000", "alpha": 0.5},
        medianprops={"color": "black"},
    )
    return summaries


def figure1_tables(data_root: Path) -> dict[str, list[dict]]:
    """Preserve the handoff's windows and condition-local participant ordering."""
    original = Path(data_root) / "original"
    energy = _mat(original / "data/cosmed/processed/processedEnergyData.mat")
    sla = _mat(original / "processedSLAData_23Subj.mat")
    energy_rows, sla_rows, time_rows, test_rows = [], [], [], []
    for condition, (speed, ne, ns) in enumerate(zip(SPEEDS, (8, 9, 9), (7, 8, 8)), 1):
        split = np.atleast_2d(getattr(energy["split_energy"], f"condition_{condition}"))
        tied = np.atleast_2d(getattr(energy["tied1_energy"], f"condition_{condition}"))
        if split.shape != (ne, 2700) or tied.shape != (ne, 360):
            raise ValueError(f"Unexpected metabolic cohort/shape for condition {condition}")
        if not np.isfinite(split).all() or not np.isfinite(tied).all():
            raise ValueError("Energy arrays contain non-finite values")
        baseline = tied[:, -180:].mean(axis=1)
        if np.any(baseline == 0):
            raise ValueError("A tied-belt metabolic baseline is zero")
        for index in range(ne):
            for label, window in (
                ("short_6_min", slice(180, 360)),
                ("long_45_min", slice(-180, None)),
            ):
                value = float(split[index, window].mean())
                energy_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "exposure": label,
                        "tied_baseline_w_kg": float(baseline[index]),
                        "split_w_kg": value,
                        "energy_change_pct": float(
                            (value - baseline[index]) / baseline[index] * 100
                        ),
                    }
                )
        times = (3, 9, 18, 27, 36, 45)
        time_values = np.stack(
            [split[:, t * 60 - 180 : t * 60].mean(axis=1) for t in times], axis=1
        )
        for index in range(ne):
            for column, minute in enumerate(times):
                time_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "time_min": minute,
                        "metabolic_cost_w_kg": float(time_values[index, column]),
                    }
                )
        for column in range(1, len(times)):
            test = ttest_rel(time_values[:, column], time_values[:, column - 1])
            test_rows.append(
                {
                    "condition": condition,
                    "delta_mps": speed,
                    "n": ne,
                    "previous_min": times[column - 1],
                    "current_min": times[column],
                    "t_statistic": float(test.statistic),
                    "df": ne - 1,
                    "p_two_sided": float(test.pvalue),
                    "significant_p_lt_005": bool(test.pvalue < 0.05),
                    "correction": "none (submitted Methods)",
                }
            )
        split_sla = np.atleast_1d(sla[f"split_sla_cond{condition}"])
        tied_sla = np.atleast_1d(sla[f"tied1_sla_cond{condition}"])
        if sum(np.asarray(v).size > 0 for v in split_sla) != ns:
            raise ValueError(f"Unexpected SLA cohort for condition {condition}")
        indices = np.atleast_1d(sla.get(f"short_idx_cond{condition}", np.full(ns, np.nan)))
        for index in range(ns):
            s, t = np.asarray(split_sla[index]).ravel(), np.asarray(tied_sla[index]).ravel()
            end = int(np.ceil(6 / 45 * len(s)))
            if end < 100 or len(t) < 100 or not np.isfinite(s).all() or not np.isfinite(t).all():
                raise ValueError("SLA record cannot support the submitted 100-stride windows")
            baseline_sla = float(t[-100:].mean() * 100)
            for label, values in (
                ("short_6_min_fractional_stride", s[end - 100 : end]),
                ("long_45_min", s[-100:]),
            ):
                value = float(values.mean() * 100)
                sla_rows.append(
                    {
                        "condition": condition,
                        "delta_mps": speed,
                        "participant_index": index + 1,
                        "exposure": label,
                        "tied_sla_pct": baseline_sla,
                        "split_sla_pct": value,
                        "sla_change_percentage_points": value - baseline_sla,
                        "short_end_stride": end,
                        "stored_short_index_not_used": float(indices[index]),
                    }
                )
    optimization = [
        {
            "delta_mps": float(speed),
            "tied_energy_w_kg": 3.147,
            "energy_change_pct": float(e),
            "sla_change_percentage_points": float(s),
            "source": "Hard-coded submitted OpenSim/Moco outputs; raw solutions unavailable",
        }
        for speed, e, s in zip(OPT_SPEEDS, OPT_ENERGY, OPT_SLA)
    ]
    return {
        "energy_participants": energy_rows,
        "sla_participants": sla_rows,
        "energy_timeseries": time_rows,
        "paired_tests": test_rows,
        "optimization": optimization,
    }


def _sanchez(data_root: Path) -> list[dict]:
    folder = data_root / "original/data/sanchez data"
    energy = _mat(folder / "sanchez_energy.mat")["MetabolicPower"]
    sla = _mat(folder / "sanchez_sla.mat")["StepLengthAsymmetry"]
    rows = []
    for participant in range(1, 16):
        e, s = getattr(energy, f"P{participant}"), getattr(sla, f"P{participant}")
        es, ss = np.asarray(e.Split).ravel(), np.asarray(s.Split).ravel()
        eb, sb = (
            float(np.asarray(e.Base).ravel()[-100:].mean()),
            float(np.asarray(s.Base).ravel()[-100:].mean() * 100),
        )
        end = int(np.floor(6 / 45 * len(es) + 0.5))
        window = slice(end - 100, end) if end > 100 else slice(0, 100)
        if max(window.stop, 100) > min(len(es), len(ss)) or eb == 0:
            raise ValueError("Sanchez input does not support the submitted windows")
        for label, w in (("short_6_min", window), ("long_45_min", slice(-100, None))):
            rows.append(
                {
                    "participant": f"P{participant}",
                    "delta_mps": 1.0,
                    "exposure": label,
                    "energy_change_pct": float((es[w].mean() - eb) / eb * 100),
                    "sla_change_percentage_points": float(ss[w].mean() * 100 - sb),
                }
            )
    return rows


def _figure1(data_root: Path, output_root: Path) -> dict:
    tables = figure1_tables(data_root)
    outputs = [_csv(output_root, f"figure1_{name}", rows) for name, rows in tables.items()]
    caveats = [
        "Participant indices identify ordering within each condition, not global participant IDs.",
        (
            "Short SLA uses ceil(6/45 × total strides), preserving the submitted script's approximation; "
            "stored short_idx variables are not used."
        ),
        "Optimization values are author-supplied constants; OpenSim solutions are not included.",
        "Fig 1E tests are two-sided paired t-tests without multiplicity correction, as submitted.",
    ]
    missing = [
        name
        for name in ("sanchez_energy.mat", "sanchez_sla.mat")
        if not (data_root / "original/data/sanchez data" / name).is_file()
    ]
    sanchez = [] if missing else _sanchez(data_root)
    if sanchez:
        outputs.append(_csv(output_root, "figure1_sanchez_participants", sanchez))
    else:
        caveats.append("Sanchez comparison inputs are missing; Fig 1C/D are explicitly partial.")
    plt = _plotting()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    summaries = []
    for row, (name, column, ylabel) in enumerate(
        (
            ("energy_participants", "energy_change_pct", "Energy change vs tied (%)"),
            (
                "sla_participants",
                "sla_change_percentage_points",
                "SLA change vs tied (percentage points)",
            ),
        )
    ):
        opt = OPT_ENERGY if row == 0 else OPT_SLA
        x = np.linspace(0.2, 1, 500)
        axes[row, 0].scatter(
            OPT_SPEEDS, opt, facecolors="#e4f0e4", edgecolors="black", linewidths=0.5
        )
        axes[row, 0].plot(
            x, np.polyval(np.polyfit(OPT_SPEEDS, opt, 2 if row == 0 else 1), x), color="#009900"
        )
        axes[row, 0].set_title("Trajectory optimization")
        axes[row, 0].set(xlim=(0.1, 1.1), ylim=(-10, 5) if row == 0 else (0, 50))
        for col, exposure in enumerate(("short", "long"), 1):
            groups = [
                np.array(
                    [
                        r[column]
                        for r in tables[name]
                        if r["delta_mps"] == speed and r["exposure"].startswith(exposure)
                    ]
                )
                for speed in SPEEDS
            ]
            boxes = _boxes(axes[row, col], groups, SPEEDS, widths=0.075)
            means = [float(v.mean()) for v in groups]
            xx = np.linspace(0.4, 1, 500)
            axes[row, col].plot(xx, np.polyval(np.polyfit(SPEEDS, means, 2), xx), color="#990000")
            for speed, values, box in zip(SPEEDS, groups, boxes):
                summaries.append(
                    {
                        "panel": "1C" if row == 0 else "1D",
                        "exposure": exposure,
                        "delta_mps": speed,
                        "n": len(values),
                        "mean": float(values.mean()),
                        "sd": float(values.std(ddof=1)),
                        "units": ylabel,
                        **box,
                    }
                )
            if sanchez:
                value = np.mean([r[column] for r in sanchez if r["exposure"].startswith(exposure)])
                axes[row, col].scatter(
                    [1], [value], color="#000099", label="Sánchez 2021", zorder=4
                )
                axes[row, col].legend(frameon=False)
            axes[row, col].set_title(f"Experiment: {'6' if col == 1 else '45'} min")
            axes[row, col].set(xlim=(0.3, 1.1), ylim=(-45, 80) if row == 0 else (-30, 10))
        for axis in axes[row]:
            axis.set_xlabel("Belt-speed difference (m/s)")
            axis.set_ylabel(ylabel)
    fig.suptitle("Figure 1C/D" + (" — PARTIAL: Sánchez comparison unavailable" if missing else ""))
    fig.tight_layout()
    outputs += _save(output_root, "figure1_cd", fig)
    outputs.append(_csv(output_root, "figure1_cd_summary", summaries))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    times = (3, 9, 18, 27, 36, 45)
    box_rows = []
    for axis, speed in zip(axes, SPEEDS):
        groups = [
            [
                r["metabolic_cost_w_kg"]
                for r in tables["energy_timeseries"]
                if r["delta_mps"] == speed and r["time_min"] == t
            ]
            for t in times
        ]
        boxes = _boxes(axis, groups, range(1, 7))
        box_rows.extend(
            {"delta_mps": speed, "time_min": minute, **box} for minute, box in zip(times, boxes)
        )
        axis.set_xticks(range(1, 7), times)
        for index, test in enumerate(
            [r for r in tables["paired_tests"] if r["delta_mps"] == speed]
        ):
            if test["significant_p_lt_005"]:
                axis.text(
                    index + 1.5, max(max(v) for v in groups) + 0.2, "*", ha="center", fontsize=15
                )
        axis.set_title(f"{speed:g} m/s")
        axis.set_xlabel("Time (min)")
    axes[0].set_ylabel("Metabolic cost (W/kg)")
    axes[0].set_ylim(top=max(r["metabolic_cost_w_kg"] for r in tables["energy_timeseries"]) + 0.7)
    fig.suptitle("Figure 1E — three-minute means and adjacent paired tests")
    fig.tight_layout()
    outputs += _save(output_root, "figure1_e", fig)
    outputs.append(_csv(output_root, "figure1_e_box_summary", box_rows))
    return _record(
        "figure1",
        "partial" if missing else "passed",
        outputs,
        caveats,
        {
            "energy_n": [8, 9, 9],
            "sla_n": [7, 8, 8],
            "missing_sanchez_files": missing,
            "panels": {
                "1C": "partial" if missing else "passed",
                "1D": "partial" if missing else "passed",
                "1E": "passed",
            },
        },
    )


def fall_filename(speed: float, rate: float, trial: int) -> str:
    return (
        f"model_split45_trans20_slow{1 - speed / 2:g}_fast{1 + speed / 2:g}_noise0.01_"
        f"learning{rate:g}_energysym1_swingcost1_trial{trial}.mat"
    )


def fall_tables(data_root: Path) -> tuple[list[dict], list[dict]]:
    """Require every submitted trial; never change the denominator by skipping files."""
    folder = Path(data_root) / "fall_probability"
    speeds, rates = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), (0.01, 0.02, 0.03, 0.04, 0.05)
    expected = [
        (speed, rate, trial, folder / fall_filename(speed, rate, trial))
        for speed in speeds
        for rate in rates
        for trial in range(1, 51)
    ]
    missing = [p.name for _, _, _, p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Figure 3 needs all 1,750 trials: {len(missing)} missing; first: {missing[0]}"
        )
    trials, summary = [], []
    for speed, rate, trial, path in expected:
        value = _mat(path, variable_names=["fall_flag", "time"])
        flag = np.asarray(value["fall_flag"]).ravel()
        if len(flag) != 2 or flag[0] not in (0, 1):
            raise ValueError(f"Invalid fall_flag in {path.name}")
        fell = bool(flag[0])
        fallback = bool(fell and flag[1] == 0)
        stride = int(np.asarray(value["time"]).size if fallback else flag[1]) if fell else None
        trials.append(
            {
                "delta_mps": speed,
                "learning_rate": rate,
                "symmetry_weight": 0,
                "trial": trial,
                "fell": fell,
                "fall_stride": stride,
                "stride_from_time_length": fallback,
                "source_file": f"fall_probability/{path.name}",
            }
        )
    for speed in speeds:
        for rate in rates:
            values = [r for r in trials if r["delta_mps"] == speed and r["learning_rate"] == rate]
            strides = [r["fall_stride"] for r in values if r["fell"]]
            summary.append(
                {
                    "delta_mps": speed,
                    "learning_rate": rate,
                    "symmetry_weight": 0,
                    "falls": len(strides),
                    "trials": len(values),
                    "fall_probability": len(strides) / len(values),
                    "fall_probability_pct": 100 * len(strides) / len(values),
                    "mean_strides_until_fall": float(np.mean(strides)) if strides else "",
                }
            )
    return trials, summary


def fit_exp1(x, y) -> dict:
    """Unweighted nonlinear least squares for the submitted ``a*exp(b*x)`` model."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.shape != y.shape
        or x.ndim != 1
        or len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("exp1 needs at least three finite paired observations")
    if np.any(y <= 0) or np.ptp(x) == 0:
        raise ValueError("exp1 requires positive mean fall strides and distinct speeds")

    def residual(parameters):
        return np.exp(parameters[0] + parameters[1] * x) - y

    def jacobian(parameters):
        prediction = np.exp(parameters[0] + parameters[1] * x)
        return np.column_stack((prediction, x * prediction))

    initial_b, initial_log_a = np.polyfit(x, np.log(y), 1)
    candidates = [
        least_squares(
            residual,
            [initial_log_a, initial_b],
            jac=jacobian,
            xtol=1e-13,
            ftol=1e-13,
            gtol=1e-13,
            max_nfev=10000,
        )
    ]
    for b in (-10.0, -2.0, 0.0, 2.0, 10.0):
        candidates.append(
            least_squares(
                residual,
                [np.log(y.mean()) - b * x.mean(), b],
                jac=jacobian,
                xtol=1e-13,
                ftol=1e-13,
                gtol=1e-13,
                max_nfev=10000,
            )
        )
    successful = [r for r in candidates if r.success and np.isfinite(r.cost)]
    if not successful:
        raise ValueError("The exponential least-squares fit did not converge")
    result = min(successful, key=lambda r: r.cost)
    sse = float(np.sum(result.fun**2))
    return {
        "a": float(np.exp(result.x[0])),
        "b": float(result.x[1]),
        "sse": sse,
        "rmse": float(np.sqrt(sse / len(x))),
        "rmse_df_adjusted": float(np.sqrt(sse / (len(x) - 2))),
        "n": len(x),
        "residuals": result.fun.tolist(),
        "objective": "Unweighted sum of squared residuals of the per-speed mean fall strides",
        "model": "a * exp(b * delta_mps)",
    }


def _figure3(data_root: Path, output_root: Path) -> dict:
    trials, summary = fall_tables(data_root)
    outputs = [
        _csv(output_root, "figure3_trials", trials),
        _csv(output_root, "figure3_c_counts", summary),
    ]
    plt = _plotting()
    fig, axis = plt.subplots(figsize=(5.2, 3.8))
    for rate, color in zip((0.01, 0.02, 0.03, 0.04, 0.05), plt.cm.Reds(np.linspace(0.35, 0.95, 5))):
        values = [r for r in summary if r["learning_rate"] == rate]
        axis.plot(
            [r["delta_mps"] for r in values],
            [r["fall_probability"] for r in values],
            "o-",
            color=color,
            label=f"α = {rate:g}",
        )
    axis.set(
        xlabel="Belt-speed difference (m/s)",
        ylabel="Fall probability",
        xlim=(0.35, 1.05),
        ylim=(-0.05, 1.05),
        title="Figure 3C — 50 trials per cell; ψ = 0",
    )
    axis.legend(frameon=False)
    outputs += _save(output_root, "figure3_c", fig)

    examples = []
    for speed in (0.9, 1.0):
        for trial in range(1, 51):
            row = next(
                r
                for r in trials
                if r["delta_mps"] == speed and r["learning_rate"] == 0.05 and r["trial"] == trial
            )
            value = _mat(
                data_root / row["source_file"], variable_names=["fall_flag", "min_vert_pos"]
            )
            flag = np.asarray(value["fall_flag"]).ravel()
            kind = (
                "fall" if flag[0] == 1 and flag[1] > 1000 else "no_fall" if flag[0] == 0 else None
            )
            if kind and kind not in [r["kind"] for r in examples]:
                height = np.asarray(value["min_vert_pos"])[:, 1]
                if kind == "fall":
                    height = np.append(height, 0)
                examples.append(
                    {"kind": kind, "source_file": row["source_file"], "heights": height}
                )
    caveats = [
        "Cached trial reconstruction; not a fresh stochastic rerun.",
        (
            "The final manuscript states a 20-second transition, but the handoff generator and audited "
            "cached speed profiles encode 25 seconds despite trans20 filenames. The generator and cached "
            "profiles also divide physical belt speeds by 2, whereas the manuscript describes division "
            "by sqrt(g*leg_length). These provenance discrepancies require author clarification."
        ),
        (
            "The representative fall trace appends a terminal zero, following the original plotting script; "
            "the cached height field contains the per-stance minimum, rather than a separately stored end-stance sample."
        ),
        (
            "Figure 3D preserves the original 0-2000-stride axis; all falling trials, including points above "
            "that display range, remain in the source table and exponential-fit inputs."
        ),
        (
            "Figure 3D preserves the source exp1 model and unweighted nonlinear least-squares objective on cell means. "
            "A portable SciPy solver is used; the original MATLAB Curve Fitting solver is unavailable and byte-identical coefficients are not claimed."
        ),
    ]
    panels = {"3C": "passed", "3D": "passed", "3B": "passed" if len(examples) == 2 else "blocked"}
    if len(examples) == 2:
        example_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
        for axis, kind in zip(axes, ("fall", "no_fall")):
            example = next(v for v in examples if v["kind"] == kind)
            height = example["heights"]
            for index, value in enumerate(height, 1):
                example_rows.append(
                    {
                        "example": kind,
                        "stride": index,
                        "min_vertical_height": float(value),
                        "source_file": example["source_file"],
                        "appended_terminal_zero": bool(kind == "fall" and index == len(height)),
                    }
                )
            axis.plot(np.arange(1, len(height) + 1), height, color="#990000")
            axis.set(
                xlabel="Stride",
                ylabel="Minimum vertical height",
                title=kind.replace("_", " "),
                ylim=(0, 1.1),
                xlim=(0, 2500) if kind == "fall" else (0, 2000),
            )
        fig.suptitle("Figure 3B — original representative-trial selection")
        fig.tight_layout()
        outputs += [_csv(output_root, "figure3_b_traces", example_rows)] + _save(
            output_root, "figure3_b", fig
        )
    else:
        caveats.append("Original representative-trial selection did not find both examples.")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), sharey=True)
    rows_d, fit_rows, curve_rows = [], [], []
    for axis, rate, color in zip(axes, (0.02, 0.05), ("#d98e8e", "#660000")):
        selected = [r for r in trials if r["learning_rate"] == rate and r["fell"]]
        axis.scatter(
            [r["delta_mps"] for r in selected],
            [r["fall_stride"] for r in selected],
            s=8,
            alpha=0.2,
            color=color,
        )
        means = [r for r in summary if r["learning_rate"] == rate]
        observed = [r for r in means if r["falls"]]
        fit = fit_exp1(
            [r["delta_mps"] for r in observed], [r["mean_strides_until_fall"] for r in observed]
        )
        fit_rows.append(
            dict(
                learning_rate=rate,
                **{key: value for key, value in fit.items() if key != "residuals"},
            )
        )
        for row in means:
            fitted = float(fit["a"] * np.exp(fit["b"] * row["delta_mps"]))
            rows_d.append(
                {
                    "delta_mps": row["delta_mps"],
                    "learning_rate": rate,
                    "falls": row["falls"],
                    "mean_strides_until_fall": row["mean_strides_until_fall"],
                    "fitted_mean_strides": fitted,
                    "residual_predicted_minus_observed": fitted - row["mean_strides_until_fall"]
                    if row["falls"]
                    else "",
                }
            )
        x_fit = np.linspace(0.4, 1.0, 100)
        y_fit = fit["a"] * np.exp(fit["b"] * x_fit)
        curve_rows.extend(
            {"learning_rate": rate, "delta_mps": float(x), "fitted_mean_strides": float(y)}
            for x, y in zip(x_fit, y_fit)
        )
        axis.plot(x_fit, y_fit, color=color)
        axis.set(
            xlabel="Belt-speed difference (m/s)",
            ylabel="Strides until fall",
            title=f"α = {rate:g}",
            xlim=(0.35, 1.05),
            ylim=(0, 2000),
        )
    fig.suptitle("Figure 3D — original exponential model fitted to the seven cell means")
    fig.tight_layout()
    outputs += [
        _csv(output_root, "figure3_d_fit_inputs", rows_d),
        _csv(output_root, "figure3_d_fit_coefficients", fit_rows),
        _csv(output_root, "figure3_d_fit_curves", curve_rows),
    ] + _save(output_root, "figure3_d", fig)
    return _record(
        "figure3",
        "passed" if len(examples) == 2 else "partial",
        outputs,
        caveats,
        {
            "required_trials": 1750,
            "available_trials": len(trials),
            "cells": len(summary),
            "panels": panels,
            "zero_stride_fallbacks": sum(r["stride_from_time_length"] for r in trials),
        },
    )


def landscape_data(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = _mat(Path(data_root) / "original/iso_fall_risk_heat_map.mat")
    alpha = np.asarray(value["learning_pool"], dtype=float).ravel()
    cost = np.asarray(value["sym_cost_pool"], dtype=float).ravel()
    order = np.argsort(1 - cost)
    psi = (1 - cost)[order]
    rates = np.asarray(value["fall_rate_matrix_result"], dtype=float)[:, order, :]
    if (
        rates.shape != (len(alpha), len(psi), 3)
        or not np.isfinite(rates).all()
        or np.any((rates < 0) | (rates > 100))
    ):
        raise ValueError("Landscape is incomplete or contains invalid fall probabilities")
    return alpha, psi, rates


def landscape_tables(data_root: Path) -> dict[str, list[dict]]:
    alpha, psi, rates = landscape_data(data_root)
    grid = [
        {
            "delta_mps": speed,
            "learning_rate": float(a),
            "symmetry_weight": float(p),
            "fall_probability_pct": float(rates[i, j, c]),
        }
        for c, speed in enumerate(SPEEDS)
        for i, a in enumerate(alpha)
        for j, p in enumerate(psi)
    ]
    value = _mat(Path(data_root) / "original/2d_sweep_trial50_param_result.mat")
    n = np.asarray(value["exp_subj_num"]).ravel().astype(int)
    parameters = np.asarray(value["static_params"])
    participants, ellipses, risks = [], [], []
    x, y = np.meshgrid(psi, alpha)
    for c, speed in enumerate(SPEEDS):
        lr = np.array([parameters[c, s].learning_rate for s in range(n[c])], dtype=float)
        sym = 1 - np.array([parameters[c, s].sym_cost for s in range(n[c])], dtype=float)
        if not np.isfinite(lr).all() or not np.isfinite(sym).all():
            raise ValueError("Non-finite fitted parameters")
        cx, cy = float(np.median(sym)), float(np.median(lr))
        rx, ry = float(sym.std(ddof=1) / np.sqrt(n[c])), float(lr.std(ddof=1) / np.sqrt(n[c]))
        if rx == 0 or ry == 0:
            raise ValueError(
                "A zero-width ellipse cannot reproduce the original grid inclusion rule"
            )
        ellipses.append(
            {
                "delta_mps": speed,
                "n": int(n[c]),
                "center_symmetry_weight": cx,
                "center_learning_rate": cy,
                "sem_symmetry_weight": rx,
                "sem_learning_rate": ry,
            }
        )
        for s, (a, p) in enumerate(zip(lr, sym), 1):
            participants.append(
                {
                    "delta_mps": speed,
                    "participant_index": s,
                    "learning_rate": float(a),
                    "symmetry_weight": float(p),
                }
            )
        for prior, ellipse in enumerate(ellipses):
            mask = (
                (x - ellipse["center_symmetry_weight"]) / ellipse["sem_symmetry_weight"]
            ) ** 2 + (
                (y - ellipse["center_learning_rate"]) / ellipse["sem_learning_rate"]
            ) ** 2 <= 1
            mean = float(rates[:, :, c][mask].mean()) if mask.any() else float("nan")
            risks.append(
                {
                    "landscape_delta_mps": speed,
                    "ellipse_delta_mps": SPEEDS[prior],
                    "included_grid_nodes": int(mask.sum()),
                    "mean_fall_probability_pct": mean,
                    "displayed_integer_pct": int(np.floor(mean + 0.5)) if np.isfinite(mean) else "",
                }
            )
    return {
        "landscape": grid,
        "participant_parameters": participants,
        "ellipses": ellipses,
        "ellipse_risk": risks,
    }


def _figure4(data_root: Path, output_root: Path) -> dict:
    tables = landscape_tables(data_root)
    outputs = [_csv(output_root, f"figure4_{key}", rows) for key, rows in tables.items()]
    alpha, psi, rates = landscape_data(data_root)
    plt = _plotting()
    from matplotlib.patches import Ellipse

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True, layout="constrained")
    for c, (axis, speed) in enumerate(zip(axes, SPEEDS)):
        display = axis.pcolormesh(
            psi, alpha, rates[:, :, c] / 100, shading="nearest", vmin=0, vmax=1, cmap="viridis"
        )
        for prior in range(c + 1):
            row = tables["ellipses"][prior]
            risk = next(
                r
                for r in tables["ellipse_risk"]
                if r["landscape_delta_mps"] == speed and r["ellipse_delta_mps"] == SPEEDS[prior]
            )
            cx, cy = row["center_symmetry_weight"], row["center_learning_rate"]
            axis.add_patch(
                Ellipse(
                    (cx, cy),
                    2 * row["sem_symmetry_weight"],
                    2 * row["sem_learning_rate"],
                    facecolor="#b00000" if prior == c else "gray",
                    alpha=0.6,
                    edgecolor="black",
                )
            )
            axis.plot(cx, cy, "o", color="black", markersize=2)
            axis.annotate(
                f"{risk['displayed_integer_pct'] / 100:.2f}",
                (cx, cy),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set(
            xlabel="Symmetry weight ψ", title=f"{speed:g} m/s", xlim=(0, 1), ylim=(0.005, 0.05)
        )
    axes[0].set_ylabel("Learning rate α")
    fig.colorbar(display, ax=axes, label="Fall probability", fraction=0.025, pad=0.03)
    fig.suptitle("Figure 4 — original cached landscape and inferred parameter regions")
    outputs += _save(output_root, "figure4", fig)
    return _record(
        "figure4",
        "passed",
        outputs,
        [
            "Original cached landscape; raw trial counts, seeds and simulation revision are absent from its MAT metadata.",
            "Ellipse values are arithmetic means of included grid nodes, not significance tests.",
            "The manuscript risk-comparison p<.05 claim cannot be reproduced from the supplied analysis; no p-value is invented.",
            "Styling is portable; original manual Figma arrows and MATLAB colormap are not reproduced.",
        ],
        {
            "grid_shape": list(rates.shape),
            "participant_n": [8, 9, 9],
            "current_ellipse_displayed_pct": [
                r["displayed_integer_pct"]
                for r in tables["ellipse_risk"]
                if r["landscape_delta_mps"] == r["ellipse_delta_mps"]
            ],
        },
    )


def _s4(data_root: Path, output_root: Path) -> dict:
    alpha, psi, rates = landscape_data(data_root)
    if rates.shape != (20, 20, 3):
        raise ValueError("S4 requires the original 20 × 20 × 3 landscape")
    ai, pi = [0, 2, 6, 15], [0, 9, 19]
    rows, summary = [], []
    plt = _plotting()
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharey=True)
    for c, speed in enumerate(SPEEDS):
        for row, (indices, axis_values, fixed_name, swept_name) in enumerate(
            (
                (ai, psi, "learning_rate", "symmetry_weight"),
                (pi, alpha, "symmetry_weight", "learning_rate"),
            )
        ):
            for index in indices:
                fixed = float(alpha[index] if row == 0 else psi[index])
                values = rates[index, :, c] if row == 0 else rates[:, index, c]
                axes[row, c].plot(
                    axis_values,
                    values,
                    "o-",
                    markersize=2,
                    label=f"{'α' if row == 0 else 'ψ'}={fixed:.4g}",
                )
                for x, value in zip(axis_values, values):
                    rows.append(
                        {
                            "delta_mps": speed,
                            "sweep": "psi_sweep" if row == 0 else "alpha_sweep",
                            "fixed_parameter": fixed_name,
                            "fixed_value": fixed,
                            "swept_parameter": swept_name,
                            "swept_value": float(x),
                            "fall_probability_pct": float(value),
                        }
                    )
                summary.append(
                    {
                        "sweep": "psi_sweep" if row == 0 else "alpha_sweep",
                        "delta_mps": speed,
                        "held_fixed": "alpha" if row == 0 else "psi",
                        "fixed_value": fixed,
                        "fall_pct_at_low_end": float(values[0]),
                        "fall_pct_at_high_end": float(values[-1]),
                        "end_to_end_change_pp": float(values[-1] - values[0]),
                    }
                )
            axes[row, c].set(
                xlabel="Symmetry weight ψ" if row == 0 else "Learning rate α",
                ylim=(-4, 104),
                title=f"{speed:g} m/s",
            )
            axes[row, c].legend(frameon=False, fontsize=7)
    axes[0, 0].set_ylabel("Fall probability (%)")
    axes[1, 0].set_ylabel("Fall probability (%)")
    fig.suptitle("Supplementary Figure 4 — slices of the original cached landscape")
    fig.tight_layout()
    outputs = [
        _csv(output_root, "s4_plotted_points", rows),
        _csv(output_root, "s4_endpoint_summary", summary),
    ]
    outputs += _save(output_root, "s4", fig)
    return _record(
        "s4",
        "passed",
        outputs,
        [
            "The submitted script's exact grid indices are preserved; actual parameter values are exported.",
            "Cached probabilities are reproduced without assuming unrecorded trial counts or seeds.",
        ],
        {"plotted_points": len(rows), "summary_rows": len(summary)},
    )


def render_main(data_root: Path, output_root: Path, figures: list[str] | None = None) -> list[dict]:
    """Render supported figures and explicit partials, reporting missing inputs."""
    data_root, output_root = Path(data_root), Path(output_root)
    functions = {"figure1": _figure1, "figure3": _figure3, "figure4": _figure4, "s4": _s4}
    selected = list(functions) if figures is None else list(figures)
    unknown = set(selected) - functions.keys()
    if unknown:
        raise ValueError(f"Unknown main figure selection: {', '.join(sorted(unknown))}")
    records = []
    for name in selected:
        try:
            records.append(functions[name](data_root, output_root))
        except FileNotFoundError as error:
            records.append(_record(name, "blocked", caveats=[f"{type(error).__name__}: {error}"]))
    return records
