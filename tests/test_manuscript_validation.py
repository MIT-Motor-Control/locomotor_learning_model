import csv
import json

import pytest

from locomotor_learning_model.publication import manuscript
from locomotor_learning_model.publication.reporting import sha256, verify_data, write_report


def reproduction(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"schema_version": 1, "files": []}))
    results = tmp_path / "results"
    (results / "tables").mkdir(parents=True)
    (results / "tables/value.csv").write_text("value\n1\n")
    write_report(results, [], verify_data(data))
    return data, results


def test_claim_comparison_preserves_discrepancy_and_rounding():
    assert (
        manuscript.compare_claim("rmse", 4, [0.37148], [0.33], 0.005, "source")["status"]
        == "mismatch"
    )
    assert (
        manuscript.compare_claim("rmse", 5, [8.676], [8.68], 0.005, "source")["status"] == "matched"
    )
    assert (
        manuscript.compare_claim("bad", 1, [float("nan")], [1], 0.1, "source")["status"]
        == "mismatch"
    )
    assert manuscript.compare_claim("missing", 1, [], [1], 0.1, "source")["status"] == "mismatch"


def test_verify_reproduction_refuses_altered_table(tmp_path):
    data, results = reproduction(tmp_path)
    manuscript.verify_reproduction(data, results)
    (results / "tables/value.csv").write_text("value\n999\n")
    with pytest.raises(ValueError, match="changed reproduction artifact"):
        manuscript.verify_reproduction(data, results)


def test_verify_reproduction_refuses_stale_code(tmp_path, monkeypatch):
    data, results = reproduction(tmp_path)
    monkeypatch.setattr(manuscript, "source_hashes", lambda: {"changed.py": "changed"})
    with pytest.raises(ValueError, match="code changed"):
        manuscript.verify_reproduction(data, results)


def test_validation_refuses_unreviewed_pdf(tmp_path):
    pdf = tmp_path / "different.pdf"
    pdf.write_bytes(b"different manuscript")
    with pytest.raises(ValueError, match="differs from the audited"):
        manuscript.validate_manuscript(
            pdf, tmp_path / "data", tmp_path / "results", tmp_path / "out"
        )


def test_missing_analyses_are_unverified_not_matched(tmp_path, monkeypatch):
    data, results = reproduction(tmp_path)
    pdf = tmp_path / "test.pdf"
    pdf.write_bytes(b"test manuscript")
    monkeypatch.setattr(manuscript, "MANUSCRIPT_SHA256", sha256(pdf))
    output = tmp_path / "validation"
    report = manuscript.validate_manuscript(pdf, data, results, output)
    assert report["status"] == "incomplete"
    assert all(item["status"] == "unverified" for item in report["claims"])
    assert (output / "report.md").is_file()
    with pytest.raises(ValueError, match="empty"):
        manuscript.validate_manuscript(pdf, data, results, output)


@pytest.mark.parametrize(
    "p_values,t_values,expected",
    [
        ([0.01, 0.01, 0.5, 0.5, 0.5], [-1] * 5, "matched"),
        ([0.5, 0.01, 0.5, 0.5, 0.5], [-1] * 5, "mismatch"),
        ([0.01, 0.01, 0.5, 0.5, 0.5], [1] * 5, "mismatch"),
        ([0.5] * 5, [-1] * 5, "mismatch"),
    ],
)
def test_significant_decreases_must_match_every_interval(tmp_path, p_values, t_values, expected):
    data, results = reproduction(tmp_path)
    table = results / "tables/figure1_paired_tests.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["delta_mps", "current_min", "p_two_sided", "t_statistic"]
        )
        writer.writeheader()
        writer.writerows(
            {"delta_mps": 0.7, "current_min": t, "p_two_sided": p, "t_statistic": stat}
            for t, p, stat in zip([9, 18, 27, 36, 45], p_values, t_values)
        )
    report = write_report(results, [], verify_data(data))
    claims = manuscript.numerical_claims(results, report)
    claim = next(item for item in claims if "Figure 1E" in item["claim"] and "0.7" in item["claim"])
    assert claim["status"] == expected


@pytest.mark.parametrize("ratios,expected", [([1.9, 2.2, 2.6], "matched"), ([1.9, 1.5, 2.6], "mismatch"), ([1.9, 2.2], "mismatch")])
def test_grid_advantage_requires_all_three_increasing_comparisons(tmp_path, ratios, expected):
    data, results = reproduction(tmp_path)
    table = results / "tables/s1_grid_sensitivity_summary.csv"
    table.write_text("median_ratio_common_over_specific\n" + "\n".join(map(str, ratios)) + "\n")
    report = write_report(results, [], verify_data(data))
    claims = manuscript.numerical_claims(results, report)
    claim = next(item for item in claims if item["claim"].startswith("S1 advantage"))
    assert claim["status"] == expected
