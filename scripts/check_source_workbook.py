"""Read-only numerical validation of the artifact-tool source workbook.

Usage: python scripts/check_source_workbook.py REPRODUCTION_DIR OUTPUT.xlsx
Requires openpyxl only for reading the saved workbook. This script never authors
or modifies a workbook. It verifies every imported cell, source checksum, table
header, and both S1 test-method cells against the numerical CSVs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from openpyxl import load_workbook


def typed(value):
    if value == "":
        return None
    if value in ("True", "False"):
        return value == "True"
    try:
        return float(value)
    except ValueError:
        return value


def equal(actual, expected):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, (float, int)):
        return isinstance(actual, (float, int)) and math.isclose(
            actual, expected, rel_tol=2e-14, abs_tol=1e-14
        )
    return actual == expected


def check(input_root: Path, output: Path) -> dict:
    metadata = json.loads(output.with_suffix(".validation.json").read_text())
    workbook = load_workbook(output, read_only=True, data_only=False)
    cells = 0
    hashes = {}

    def source(name, digest):
        path = input_root / "tables" / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert hashes[name] == digest, f"CSV changed after workbook build: {name}"
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    for table in metadata["tables"]:
        rows = source(table["csv"], table["sourceSha256"])
        assert len(rows) == table["rows"], f"Row count differs: {table['csv']}"
        sheet = workbook[table["sheet"]]
        headers = [
            cell.value
            for cell in next(
                sheet.iter_rows(
                    min_row=table["headerRow"],
                    max_row=table["headerRow"],
                    max_col=table["columns"],
                )
            )
        ]
        assert headers == table["displayHeaders"], f"Headers differ: {table['sheet']}"
        actual_rows = sheet.iter_rows(
            min_row=table["firstDataRow"],
            max_row=table["firstDataRow"] + len(rows) - 1,
            max_col=table["columns"],
            values_only=True,
        )
        for i, (original, actual) in enumerate(zip(rows, actual_rows)):
            for j, field in enumerate(table["originalHeaders"]):
                expected = typed(original[field])
                assert equal(actual[j], expected), (
                    f"Value differs: {table['sheet']} source row {i + 1}, {field}: "
                    f"{actual[j]!r} != {expected!r}"
                )
                cells += 1
    for item in metadata["extraSourceCells"]:
        rows = source(item["csv"], item["sourceSha256"])
        value = workbook[item["sheet"]][item["cell"]].value
        expected = typed(rows[item["row"]][item["field"]])
        assert equal(value, expected), f"Test-method value differs: {item}"
        cells += 1
    for sheet in workbook:
        for row in sheet.iter_rows():
            for cell in row:
                assert cell.data_type != "e", f"Spreadsheet error: {sheet.title}!{cell.coordinate}"
    assert "Sanchez" in workbook["Source guide"]["D8"].value
    assert "S3B" in workbook["Source guide"]["D13"].value
    result = {
        "sheets": len(workbook.sheetnames),
        "tables": len(metadata["tables"]),
        "verified_cells": cells,
        "source_csvs": len(hashes),
        "status": "passed",
    }
    workbook.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python scripts/check_source_workbook.py REPRODUCTION_DIR OUTPUT.xlsx"
        )
    print(json.dumps(check(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
