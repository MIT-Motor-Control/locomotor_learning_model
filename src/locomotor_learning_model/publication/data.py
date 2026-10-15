"""Import publication inputs without changing the author handoff or source archives."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from .reporting import sha256


def prepare_data(
    bundle: Path,
    destination: Path,
    fall_archive: Path | None = None,
    *,
    private_archive: Path | None = None,
) -> dict:
    """Copy curated inputs, recover embedded energy, and record portable provenance.

    ``fall_archive`` is the older directory containing individual fall-probability
    MAT files; ``private_archive`` is the research repository containing
    ``reviewer_comments``. Neither source is modified. Conflicting copies fail
    before any inputs are copied. Existing destination inputs are never replaced
    with different content.
    """
    bundle, destination = Path(bundle).resolve(), Path(destination).resolve()
    if not (bundle / "code/original_data").is_dir():
        raise FileNotFoundError(f"Author handoff not found: {bundle / 'code/original_data'}")
    roots = {"bundle": bundle}
    for name, source in (
        ("bundle", bundle),
        ("fall_archive", fall_archive),
        ("private_archive", private_archive),
    ):
        if source is not None:
            source = Path(source).resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Source archive does not exist: {name}")
            if destination.is_relative_to(source) or source.is_relative_to(destination):
                raise ValueError(
                    "The prepared-data destination must be outside and not contain any source archive"
                )
            roots[name] = source

    plan: dict[str, dict] = {}

    def add(path: Path, relative: str, source: str, transformation: str = "Copied byte-for-byte"):
        if not path.is_file():
            return
        if not path.resolve().is_relative_to(roots[source.split(":", 1)[0]]):
            raise ValueError(f"Scientific input escapes its source archive: {source}")
        digest = sha256(path)
        if relative in plan:
            if plan[relative]["sha256"] != digest:
                raise ValueError(f"Conflicting scientific input: {relative} ({source})")
            plan[relative]["corroborating_sources"].append(source)
            return
        plan[relative] = {
            "path": relative,
            "sha256": digest,
            "size": path.stat().st_size,
            "source": source,
            "transformation": transformation,
            "corroborating_sources": [],
            "_source_path": path,
        }

    original = bundle / "code/original_data"
    for path in sorted(original.glob("*.mat")):
        add(path, f"original/{path.name}", f"bundle:/code/original_data/{path.name}")
    for relative in (
        "data/cosmed/processed/processedEnergyData.mat",
        "data/sanchez data/sanchez_energy.mat",
        "data/sanchez data/sanchez_sla.mat",
    ):
        add(original / relative, f"original/{relative}", f"bundle:/code/original_data/{relative}")
    for path in sorted((original / "Submitted Data").glob("*.csv")):
        add(
            path,
            f"original/Submitted Data/{path.name}",
            f"bundle:/code/original_data/Submitted Data/{path.name}",
        )
    for path in sorted((bundle / "code/verify").glob("*.mat")):
        add(path, f"regenerated/{path.name}", f"bundle:/code/verify/{path.name}")
    for path in sorted((bundle / "code/verify/data/2d sweep fine").glob("*.mat")):
        add(
            path,
            f"regenerated/data/2d sweep fine/{path.name}",
            f"bundle:/code/verify/data/2d sweep fine/{path.name}",
        )
    for path in sorted((bundle / "code/verify/figures").glob("comment1_loso_cv_*.csv")):
        add(path, f"regenerated/figures/{path.name}", f"bundle:/code/verify/figures/{path.name}")
    for path in sorted((bundle / "results").glob("*.csv")):
        add(
            path,
            f"reference_tables/{path.name}",
            f"bundle:/results/{path.name}",
            "Author reference table; not assumed to represent the final plotted analysis",
        )

    for folder, prefix in (
        (original / "data/fall probability", "bundle:/code/original_data/data/fall probability"),
        (Path(fall_archive) if fall_archive else None, "fall_archive:"),
    ):
        if folder is not None:
            for path in sorted(folder.glob("*.mat")):
                add(path, f"fall_probability/{path.name}", f"{prefix}/{path.name}")

    if private_archive is not None:
        private_archive = Path(private_archive)
        for comment, extension in (("comment_2", "s2"), ("comment_4", "s5")):
            result = private_archive / "reviewer_comments" / comment / "results"
            for path in sorted(result.iterdir()) if result.is_dir() else []:
                if path.is_file() and path.suffix in {".npz", ".csv", ".json"}:
                    add(
                        path,
                        f"extensions/{extension}/{path.name}",
                        f"private_archive:/reviewer_comments/{comment}/results/{path.name}",
                    )
            if extension == "s5":
                for path in sorted((result / "cells").glob("*.npz")):
                    add(
                        path,
                        f"extensions/s5/cells/{path.name}",
                        f"private_archive:/reviewer_comments/comment_4/results/cells/{path.name}",
                    )
        for path in sorted((private_archive / "Data").glob("*.csv")):
            add(
                path,
                f"extensions/experimental_csv/{path.name}",
                f"private_archive:/Data/{path.name}",
            )

    energy_relative = "original/data/cosmed/processed/processedEnergyData.mat"
    if energy_relative not in plan:
        embedded = original / "2d_sweep_trial50_param_result.mat"
        if embedded.is_file():
            from scipy.io import loadmat, savemat

            fields = ("split_energy", "tied1_energy", "tied2_energy")
            values = loadmat(embedded, variable_names=list(fields))
            if all(field in values for field in fields):
                stream = io.BytesIO()
                savemat(stream, {field: values[field] for field in fields}, do_compression=True)
                payload = stream.getvalue()
                header = (
                    b"MATLAB 5.0 MAT-file, recovered publication energy fields; see manifest.json"
                )
                payload = header.ljust(116, b" ") + payload[116:]
                plan[energy_relative] = {
                    "path": energy_relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size": len(payload),
                    "source": "bundle:/code/original_data/2d_sweep_trial50_param_result.mat",
                    "transformation": "Extracted exact split_energy, tied1_energy, tied2_energy MATLAB structs; "
                    "no numerical processing; deterministic MAT header",
                    "source_sha256": sha256(embedded),
                    "corroborating_sources": [],
                    "_payload": payload,
                }

    if destination.exists():
        unlisted = [
            p.relative_to(destination).as_posix()
            for p in destination.rglob("*")
            if p.is_file()
            and p.relative_to(destination).as_posix() != "manifest.json"
            and p.relative_to(destination).as_posix() not in plan
        ]
        if unlisted:
            raise ValueError(
                "Existing prepared inputs are absent from this import plan; repeat all original source "
                f"arguments or use a fresh destination. First unlisted input: {min(unlisted)}"
            )
    for relative, item in plan.items():
        existing = destination / relative
        if not existing.resolve().is_relative_to(destination):
            raise ValueError(f"Prepared-data path escapes its destination: {relative}")
        if existing.exists() and (not existing.is_file() or sha256(existing) != item["sha256"]):
            raise ValueError(f"Refusing to replace a different existing input: {relative}")
    for relative, item in plan.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            if "_payload" in item:
                target.write_bytes(item["_payload"])
            else:
                shutil.copy2(item["_source_path"], target)

    expected = [
        energy_relative,
        "original/processedSLAData_23Subj.mat",
        "original/data/sanchez data/sanchez_energy.mat",
        "original/data/sanchez data/sanchez_sla.mat",
        "original/iso_fall_risk_heat_map.mat",
    ]
    rollout_metadata = {
        "path": "regenerated/data/2d sweep fine",
        "required_trials_per_cell": 50,
        "available_files": sum(k.startswith("regenerated/data/2d sweep fine/") for k in plan),
        "required_unique_cells": None,
        "required_files": None,
        "note": "Final S3 requires the complete 50-trial library at each unique selected published-fit cell; "
        "best-trial fitted trajectories do not substitute for those rollouts. "
        "File presence alone does not verify the required cell/trial identities.",
    }
    parameter_file = original / "2d_sweep_trial50_param_result.mat"
    if parameter_file.is_file():
        import numpy as np
        from scipy.io import loadmat

        fields = ["static_params", "exp_subj_num", "learning_pool", "sym_cost_pool"]
        values = loadmat(
            parameter_file, variable_names=fields, struct_as_record=False, squeeze_me=True
        )
        if all(field in values for field in fields):
            params, n = values["static_params"], np.asarray(values["exp_subj_num"]).astype(int)
            lp, sp = np.asarray(values["learning_pool"]), np.asarray(values["sym_cost_pool"])
            cells = {
                (
                    c,
                    int(np.argmin(abs(lp - params[c, s].learning_rate))),
                    int(np.argmin(abs(sp - params[c, s].sym_cost))),
                )
                for c in range(len(n))
                for s in range(n[c])
            }
            rollout_metadata.update(
                required_unique_cells=len(cells),
                required_files=50 * len(cells),
                condition_unique_cells=[sum(c == i for c, _, _ in cells) for i in range(len(n))],
                expectation_source="Original published static_params nearest learning_pool and sym_cost_pool grid cells; 50 trials per cell",
            )
    manifest = {
        "schema_version": 1,
        "prepared_at": datetime.now(UTC).isoformat(),
        "purpose": "Cached numerical reproduction of the submitted publication; not a raw simulation rerun",
        "files": [
            {key: value for key, value in plan[relative].items() if not key.startswith("_")}
            for relative in sorted(plan)
        ],
        "missing_inputs": [relative for relative in expected if relative not in plan],
        "missing_analyses": {"s3_rollouts": rollout_metadata},
        "counts": {
            "files": len(plan),
            "fall_probability_files": sum(k.startswith("fall_probability/") for k in plan),
        },
        "caveats": [
            "Original and regenerated realizations are retained separately.",
            "Raw COSMED and OpenSim optimization inputs are not supplied by this import.",
            "Reference tables may describe earlier analyses; use per-figure exported tables for plotted data.",
            "No regenerated S3 rollout subset is synthesized or inferred from fitted parameters.",
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination / "manifest.json")
    return manifest
