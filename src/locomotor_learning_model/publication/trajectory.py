"""Evaluate the supplied muscle-output subset using the original energy equations.

This reads saved OpenSim outputs; it does not run trajectory optimization. The
available conditions differ from the five rounded values plotted in the paper.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

MUSCLES = tuple(
    f"{name}_{side}"
    for side in ("r", "l")
    for name in (
        "bflh",
        "bfsh",
        "glmax2",
        "psoas",
        "recfem",
        "vasint",
        "gasmed",
        "soleus",
        "tibant",
        "ehl",
        "fdl",
        "fdb",
    )
)
SLOW_TWITCH = np.tile(
    [0.5425, 0.529, 0.55, 0.5, 0.3865, 0.543, 0.566, 0.803, 0.7, 0.75, 0.6, 0.6], 2
)
SOURCE_FILES = {
    0.0: "1_1_Outputs.sto",
    0.4: "0-8_1-2_Outputs.sto",
    0.7: "0-65_1-35_Outputs.sto",
    1.0: "0-5_1-5_Outputs.sto",
}


def read_storage(path: Path) -> tuple[list[str], np.ndarray]:
    """Read an OpenSim text storage table, rejecting missing or ambiguous data."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip() == "endheader":
                break
        else:
            raise ValueError(f"OpenSim storage header is incomplete: {path.name}")
        columns = stream.readline().split()
        values = np.loadtxt(stream, ndmin=2)
    if (
        not columns
        or columns[0] != "time"
        or len(set(columns)) != len(columns)
        or values.shape[1] != len(columns)
        or len(values) < 2
        or not np.isfinite(values).all()
        or np.any(np.diff(values[:, 0]) <= 0)
    ):
        raise ValueError(f"Invalid OpenSim storage table: {path.name}")
    return columns, values


def source_energy_cost(storage: Path, model: Path, average_speed: float = 1.0) -> dict:
    """Port computeOsimEnergy.m, retaining sample means and its muscle ordering."""
    if not np.isfinite(average_speed) or average_speed <= 0:
        raise ValueError("Average treadmill speed must be positive and finite")
    columns, values = read_storage(storage)
    elements = [
        e for e in ET.parse(model).getroot().iter() if e.find("max_isometric_force") is not None
    ]
    if tuple(e.get("name") for e in elements) != MUSCLES:
        raise ValueError("The energy model requires the source's ordered 24-muscle model")
    force, length, maximum_velocity = (
        np.array([float(e.findtext(key, "nan")) for e in elements])
        for key in ("max_isometric_force", "optimal_fiber_length", "max_contraction_velocity")
    )
    if any(not np.isfinite(x).all() or np.any(x <= 0) for x in (force, length, maximum_velocity)):
        raise ValueError("Invalid muscle force, length or velocity parameters")
    # The MATLAB code reads Vmax from the final muscle; all supplied values are 12.
    if not np.all(maximum_velocity == maximum_velocity[-1]):
        raise ValueError("The original energy calculation requires a shared maximum velocity")

    def outputs(quantity):
        expected = [f"/forceset/{muscle}|{quantity}" for muscle in MUSCLES]
        if any(name not in columns for name in expected):
            raise ValueError(f"Missing OpenSim muscle outputs: {quantity}")
        return values[:, [columns.index(name) for name in expected]]

    activation = outputs("activation")
    velocity = outputs("fiber_velocity")
    active_force = outputs("active_fiber_force")
    fiber_length = outputs("fiber_length")
    multiplier = outputs("active_force_length_multiplier")
    if np.any((activation < 0) | (activation > 1)) or np.any(fiber_length <= 0):
        raise ValueError("Invalid muscle activation or fiber length")
    vn = velocity / length
    vn_short = 0.5 * (vn - np.sqrt(vn**2 + 0.001**2))
    vn_long = 0.5 * (vn + np.sqrt(vn**2 + 0.001**2))
    mass = force / 600000 * length * 1059.7
    fast_twitch = 1 - SLOW_TWITCH
    a_slow, a_fast = 40 / maximum_velocity[-1], 153 / maximum_velocity[-1]
    h_am = (128 * fast_twitch + 25) * np.where(fiber_length < length, 1, 0.4 + 0.6 * multiplier)
    h_sl = 4 * a_slow * vn_long - a_slow * vn_short * SLOW_TWITCH - a_fast * vn_short * fast_twitch
    # Preserve the original cap expression, including its subtraction of 100.
    h_sl = np.where(
        -a_slow * vn_short > 100, 4 * a_slow * vn_long - 100 - a_fast * vn_short * fast_twitch, h_sl
    )
    h_sl *= np.where(fiber_length < length, 1, multiplier)
    heat = np.maximum(1.5 * mass * (h_am * activation**0.6 + h_sl * activation**2), mass)
    energy = np.maximum(heat - active_force * velocity, mass).sum(axis=1)
    cost = float(energy.mean() / 75.337 / average_speed)
    if not np.isfinite(cost):
        raise ValueError("Nonfinite OpenSim source energy result")
    return {
        "samples": len(values),
        "duration_s": float(values[-1, 0] - values[0, 0]),
        "body_mass_kg": 75.337,
        "average_speed_mps": average_speed,
        "net_cost_j_kg_m": cost,
        "aggregation": "arithmetic sample mean, matching computeOsimEnergy.m",
    }


def trajectory_source_tables(data_root: Path) -> list[dict]:
    """Return available-source checks, separately from the published constants."""
    folder = Path(data_root) / "original/data/trajectory optimization"
    paths = [folder / "model_13d24m_sbt.osim", *(folder / name for name in SOURCE_FILES.values())]
    if not all(path.is_file() for path in paths):
        return []
    rows = [
        {
            "delta_mps": speed,
            "source": f"original/data/trajectory optimization/{name}",
            **source_energy_cost(folder / name, paths[0]),
        }
        for speed, name in SOURCE_FILES.items()
    ]
    baseline = rows[0]["net_cost_j_kg_m"]
    for row in rows:
        row["energy_change_pct"] = (row["net_cost_j_kg_m"] - baseline) / baseline * 100
        row["role"] = (
            "Independent saved-source subset; not substituted for the five published constants"
        )
    return rows
