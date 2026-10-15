/**
 * Editor-requested numerical source workbook, built from publication CSVs.
 * Usage: node scripts/build_source_workbook.mjs <reproduction-dir> <output.xlsx>
 * Requires @oai/artifact-tool in the Node environment. For a Codex runtime,
 * ARTIFACT_TOOL_WORKSPACE may name a temporary directory whose node_modules
 * symlink points at load_workspace_dependencies' bundled node_modules.
 * WORKBOOK_RENDER_SHEETS optionally limits previews to comma-separated sheet
 * names after the first complete visual review; workbook contents are unchanged.
 * Scientific calculations remain in the Python/MATLAB reproduction pipeline.
 */
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";

const runtimeWorkspace = process.env.ARTIFACT_TOOL_WORKSPACE;
const resolvePackage = createRequire(runtimeWorkspace
  ? path.join(path.resolve(runtimeWorkspace), "package.json") : import.meta.url);
const { Workbook, SpreadsheetFile } = await import(
  pathToFileURL(resolvePackage.resolve("@oai/artifact-tool")).href);

const [inputArgument, outputArgument] = process.argv.slice(2);
if (!inputArgument || !outputArgument) {
  throw new Error("Usage: node scripts/build_source_workbook.mjs <reproduction-dir> <output.xlsx>");
}
const input = path.resolve(inputArgument);
const output = path.resolve(outputArgument);
const previewDir = path.join(path.dirname(output), ".source-workbook-previews");
const report = JSON.parse(await fs.readFile(path.join(input, "report.json"), "utf8"));
const workbook = Workbook.create();
const navy = "#253D5B", gray = "#56616E", line = "#D5DEE8";
const tables = [];
const views = [];
const extraSourceCells = [];

const columnLetter = (index) => {
  let result = "";
  for (let n = index + 1; n > 0; n = Math.floor((n - 1) / 26)) {
    result = String.fromCharCode(65 + (n - 1) % 26) + result;
  }
  return result;
};

const numeric = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?$/;
function typed(value) {
  if (value == null || value === "") return null;
  if (typeof value !== "string") return value;
  if (numeric.test(value)) return Number(value);
  if (value === "True" || value === "False") return value === "True";
  return value;
}

async function source(name) {
  const bytes = await fs.readFile(path.join(input, "tables", name));
  const imported = await Workbook.fromCSV(bytes.toString("utf8"), { sheetName: "Import" });
  const values = imported.worksheets.getItemAt(0).getUsedRange().values;
  if (values.length < 2) throw new Error(`Source is empty: ${name}`);
  const headers = values[0].map(String);
  return { name, headers, rows: values.slice(1).map(row => row.map(typed)),
    sha256: createHash("sha256").update(bytes).digest("hex") };
}

function sheet(name, title, note, sourceNote, columns = 8) {
  const result = workbook.worksheets.add(name);
  result.showGridLines = false;
  result.getRange(`A1:${columnLetter(columns - 1)}7`).format.font = { name: "Arial", size: 10, color: "#202B37" };
  result.getRange(`A1:${columnLetter(columns - 1)}7`).format.verticalAlignment = "center";
  result.getRange("A2").values = [[title]];
  result.getRange("A2").format.font = { name: "Arial", size: 15, bold: true, color: navy };
  result.getRange(`A3:${columnLetter(columns - 1)}3`).format.borders = {
    bottom: { style: "thin", color: line },
  };
  result.getRange("A4").values = [[note]];
  result.getRange("A4").format.font = { name: "Arial", size: 10, color: gray };
  result.getRange("A5").values = [[sourceNote]];
  result.getRange("A5").format.font = { name: "Arial", size: 10, color: gray, italic: true };
  result.getRange("A2").format.rowHeight = 26;
  result.getRange("A3:A6").format.rowHeight = 19;
  return result;
}

const displayHeader = (name) => ({
  delta_mps: "Speed difference (m/s)",
  energy_change_pct: "Energy change (%)",
  fall_probability: "Fall probability (0–1)",
  fall_probability_pct: "Fall probability (%)",
  sla_change_percentage_points: "SLA change (percentage points)",
  tied_baseline_w_kg: "Tied baseline (W/kg)",
  split_w_kg: "Split energy (W/kg)",
  metabolic_cost_w_kg: "Metabolic cost (W/kg)",
  cond_specific: "Condition-specific RMSE (%)",
  common_full: "Common full-pool RMSE (%)",
  common_size_matched: "Common size-matched RMSE (%)",
  tied_last180_mean: "Tied last 180 s (W/kg)",
  split_last180_mean: "Split last 180 s (W/kg)",
  tied_sla_pct: "Tied SLA (%)", split_sla_pct: "Split SLA (%)",
  tied_energy_w_kg: "Tied energy (W/kg)",
  learning_rate: "Learning rate", symmetry_weight: "Symmetry weight",
  source_file: "Source trial file", mean: "Mean", sd: "Standard deviation",
  p_two_sided: "Two-sided p-value", t_statistic: "t statistic",
  significant_p_lt_005: "p < 0.05",
  mean_heldout_RMSE: "Mean held-out RMSE (%)",
  median_heldout_RMSE: "Median held-out RMSE (%)",
}[name] ?? name.replaceAll("_", " "));

function putTable(ws, sourceData, row, name, overrides = {}) {
  const columns = sourceData.headers.length;
  const lastColumn = columnLetter(columns - 1);
  const lastRow = row + sourceData.rows.length;
  const displayed = sourceData.headers.map(key => overrides[key] ?? displayHeader(key));
  ws.getRange(`A${row}:${lastColumn}${row}`).values = [displayed];
  ws.getRange(`A${row + 1}:${lastColumn}${lastRow}`).values = sourceData.rows;
  const range = ws.getRange(`A${row}:${lastColumn}${lastRow}`);
  range.format.font = { name: "Arial", size: 10, color: "#202B37" };
  range.format.rowHeight = 19;
  range.format.verticalAlignment = "center";
  range.format.horizontalAlignment = "left";
  range.format.wrapText = false;
  const header = ws.getRange(`A${row}:${lastColumn}${row}`);
  header.format.fill = navy;
  header.format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
  header.format.horizontalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 35;
  header.format.borders = { insideVertical: { style: "thin", color: "#FFFFFF" } };
  sourceData.headers.forEach((key, i) => {
    const letter = columnLetter(i);
    const field = ws.getRange(`${letter}${row + 1}:${letter}${lastRow}`);
    const isNumeric = sourceData.rows.every(r => r[i] == null || typeof r[i] === "number");
    const isInteger = isNumeric && sourceData.rows.every(r => r[i] == null || Number.isInteger(r[i]));
    const width = key === "source_file" ? 144 : key === "exposure" ? 34
      : key === "source" ? 91 : key === "units" ? 34 : key === "model" ? 29
        : key === "correction" ? 29
        : Math.min(28, Math.max(13, displayed[i].length * .61));
    ws.getRange(`${letter}${row}:${letter}${lastRow}`).format.columnWidth = width;
    if (isNumeric) {
      field.format.horizontalAlignment = "right";
      field.setNumberFormat(isInteger ? "0" : /(^p_|wilcoxon)/.test(key)
        ? "0.000E+00" : /learning_rate/.test(key) ? "0.00000" : "0.0000");
    }
  });
  const table = ws.tables.add(`A${row}:${lastColumn}${lastRow}`, true, name);
  table.showFilterButton = true;
  table.showBandedColumns = false;
  tables.push({ sheet: ws.name, csv: sourceData.name, sourceSha256: sourceData.sha256,
    headerRow: row, firstDataRow: row + 1, rows: sourceData.rows.length,
    columns, originalHeaders: sourceData.headers, displayHeaders: displayed });
  // Verify each imported value before exporting; independent XLSX/CSV verification
  // is performed by the small read-only Python checker distributed alongside this builder.
  const copied = ws.getRange(`A${row + 1}:${lastColumn}${lastRow}`).values;
  if (JSON.stringify(copied) !== JSON.stringify(sourceData.rows)) {
    throw new Error(`Workbook changed imported source values: ${sourceData.name}`);
  }
  return lastRow;
}

const guide = sheet("Source guide", "Numerical source data", "PBIOLOGY-D-26-00535R2. Available records for the editor-requested panels.",
  `Reproduction run: ${report.created_at}. Underlying numerical files remain in the companion tables folder.`, 5);
guide.tabColor = navy;
guide.getRange("A7:E13").values = [
  ["Panel", "Records included", "Coverage", "Missing source", "Interpretation"],
  ["1C / 1D", "52 energy and 46 SLA participant records", "Partial", "Sanchez comparison energy and SLA files", "Current experiment and reported optimization outputs included"],
  ["1E", "156 energy timepoints and 15 paired tests", "Partial", "Sanchez comparison series", "Current experiment only. Tests are uncorrected as in submitted Methods"],
  ["3C", "35 parameter cells and 1,750 trial records", "Available", "None for this panel", "Fall counts are recomputed from 50 trials per cell"],
  ["S1", "26 paired participant errors and original test summary", "Available", "None for cached replot", "Regenerated trial-average grid. MATLAB recomputation is separate"],
  ["S3A", "5 reported trajectory-optimization outputs", "Available values", "Raw OpenSim/Moco optimization solutions", "Reported outputs are provided. Optimization was not rerun"],
  ["S3B / S3C", "26 experimental participant records for S3C", "Partial", "S3B regenerated model rollout files", "S3B values are unavailable. They are not replaced with stale results"],
];
guide.getRange("A7:E13").format.font = { name: "Arial", size: 10, color: "#202B37" };
guide.getRange("A7:E13").format.verticalAlignment = "center";
guide.getRange("A7:E13").format.wrapText = true;
guide.getRange("A7:E7").format.fill = navy;
guide.getRange("A7:E7").format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
guide.getRange("A7:E7").format.horizontalAlignment = "center";
guide.getRange("A7:E7").format.rowHeight = 25;
guide.getRange("A8:E13").format.rowHeight = 43;
[15, 40, 19, 39, 65].forEach((width, i) => {
  guide.getRange(`${columnLetter(i)}7:${columnLetter(i)}13`).format.columnWidth = width;
});
guide.getRange("C8:C13").conditionalFormats.add("containsText", {
  text: "Partial", format: { fill: "#FFF0CC", font: { color: "#8A5600" } },
});
guide.getRange("A16").values = [["Values retain full numerical precision. Displayed decimals are formatting only."]];
guide.getRange("A17").values = [["Percent-change columns are already in percent units. A stored value of 20 means 20%, not 0.20%."]];
guide.getRange("A18").values = [["Participant indices are condition-local. They do not establish a cross-modality participant match."]];
guide.getRange("A19").values = [["This workbook contains processed numerical sources. It is not a replacement for raw experimental records."]];
guide.getRange("A16:E19").format.font = { name: "Arial", size: 10, color: gray };
views.push({ sheetName: guide.name, range: "A1:E20", filename: "source-guide" });

const definitions = [
  ["Fig1CD summary", "Figure 1C–D group summaries", "Mean and SD across participants. Sanchez comparison data are missing.",
    "figure1_cd_summary.csv", "Figure1Summary"],
  ["Fig1C energy", "Figure 1C participant energy", "Current experiment only. Short exposure: 6 min. Long exposure: 45 min. Sanchez data are missing.",
    "figure1_energy_participants.csv", "Figure1Energy"],
  ["Fig1D SLA", "Figure 1D participant step-length asymmetry", "SLA is in percent units. Change is in percentage points. Sanchez data are missing.",
    "figure1_sla_participants.csv", "Figure1SLA"],
  ["Fig1E timepoints", "Figure 1E participant energy timepoints", "Current experiment only. Means use the submitted time windows. Sanchez comparison data are missing.",
    "figure1_energy_timeseries.csv", "Figure1Timepoints"],
  ["Trajectory outputs", "Trajectory optimization values for 1C–D and S3A", "Five reported OpenSim/Moco outputs. Raw solutions and a rerun are unavailable.",
    "figure1_optimization.csv", "TrajectoryValues"],
  ["Fig3C counts", "Figure 3C fall probability by parameter cell", "Fall probability = falls / trials. Mean strides until fall is blank when no trial fell.",
    "figure3_c_counts.csv", "FallCounts"],
  ["Fig3C trials", "Figure 3C individual simulation trials", "One row per trial. A blank fall stride denotes a non-fall trial.",
    "figure3_trials.csv", "FallTrials"],
  ["S1 cross-validation", "S1 held-out participant prediction errors", "Saved regenerated trial-average analysis. All errors are percentages.",
    "s1_loso_persubject.csv", "LOSOParticipants"],
  ["S3C experiment", "S3C experimental energy relative to tied walking", "Last 180 s of split and tied exposure. Panel S3B is unavailable because model rollout files are missing.",
    "s3c_experimental.csv", "S3Experimental"],
];

for (const [name, title, note, csvName, tableName] of definitions) {
  const data = await source(csvName);
  const ws = sheet(name, title, note, `Source: tables/${csvName}`, data.headers.length);
  const end = putTable(ws, data, 7, tableName);
  if (data.rows.length > 25) ws.freezePanes.freezeRows(7);
  views.push({ sheetName: name, range: `A1:${columnLetter(data.headers.length - 1)}${Math.min(end + 1, 19)}`,
    filename: name.toLowerCase().replaceAll(" ", "-") });
  if (name === "Fig1E timepoints") {
    const tests = await source("figure1_paired_tests.csv");
    ws.getRange(`A${end + 3}`).values = [["Consecutive-window paired tests. Two-sided. No multiplicity correction in submitted Methods."]];
    ws.getRange(`A${end + 4}`).values = [[`Source: tables/${tests.name}`]];
    const last = putTable(ws, tests, end + 6, "EnergyPairedTests");
    views.push({ sheetName: name, range: `A${end + 3}:J${last}`, filename: "fig1e-paired-tests" });
  }
  if (name === "S1 cross-validation") {
    const statsData = await source("s1_loso_stats.csv");
    ws.getRange(`A${end + 3}`).values = [["Model summary. Source: tables/s1_loso_stats.csv"]];
    const last = putTable(ws, statsData, end + 5, "LOSOModelSummary");
    const p = await source("s1_loso_summary.csv");
    const row = Object.fromEntries(p.headers.map((h, i) => [h, p.rows[0][i]]));
    ws.getRange(`A${last + 3}:C${last + 6}`).values = [
      ["Wilcoxon method", "Two-sided p-value", "Where used"],
      ["SciPy exact", row.wilcoxon_two_sided_exact_scipy, "Final styled S1 figure"],
      ["MATLAB signrank default", row.wilcoxon_two_sided_matlab_signrank_default, "Original MATLAB test CSV"],
      ["Analysis mode", "Saved reference replot", "Optional MATLAB command reruns LOSO"],
    ];
    ws.getRange(`B${last + 4}:B${last + 5}`).setNumberFormat("0.000000E+00");
    ws.getRange(`A${last + 3}:C${last + 3}`).format.fill = navy;
    ws.getRange(`A${last + 3}:C${last + 3}`).format.font = { name: "Arial", size: 10, bold: true, color: "#FFFFFF" };
    ws.getRange(`A${last + 3}:C${last + 6}`).format.rowHeight = 23;
    ws.getRange(`B7:B${last + 9}`).format.columnWidth = 27;
    ws.getRange(`C7:C${last + 9}`).format.columnWidth = 43;
    ws.getRange(`A${last + 3}:C${last + 3}`).format.horizontalAlignment = "center";
    for (const [field, cell] of [["wilcoxon_two_sided_exact_scipy", `B${last + 4}`],
      ["wilcoxon_two_sided_matlab_signrank_default", `B${last + 5}`]]) {
      extraSourceCells.push({ sheet: ws.name, cell, csv: p.name, field, row: 0, sourceSha256: p.sha256 });
    }
    ws.getRange(`A${last + 8}`).values = [["Source: tables/s1_loso_summary.csv. Different test methods explain the two p-values."]];
    views.push({ sheetName: name, range: `A${end + 3}:E${last + 9}`, filename: "s1-test-methods" });
  }
  if (name === "Fig1E timepoints") {
    ws.getRange("A4").values = [["Current experiment only. Sanchez comparison data are missing."]];
  }
  if (name === "S3C experiment") {
    const means = await source("s3c_group_means.csv");
    ws.getRange(`A${end + 3}`).values = [["S3C plotted means. Source: tables/s3c_group_means.csv"]];
    const last = putTable(ws, means, end + 5, "S3GroupMeans", {
      mean_energy_change_pct: "Mean energy change (%)",
    });
    views.push({ sheetName: name, range: `A${end + 3}:F${last + 1}`, filename: "s3c-group-means" });
  }
}

// Independent relationships that matter to the scientific interpretation.
const fallCounts = await source("figure3_c_counts.csv");
const fallTrials = await source("figure3_trials.csv");
if (fallCounts.rows.length !== 35 || fallTrials.rows.length !== 1750) {
  throw new Error("Figure 3C expected 35 cells with 50 trials each");
}
const f = Object.fromEntries(fallCounts.headers.map((h, i) => [h, i]));
for (const row of fallCounts.rows) {
  if (row[f.trials] !== 50 || Math.abs(row[f.falls] / row[f.trials] - row[f.fall_probability]) > 1e-12) {
    throw new Error("Inconsistent Figure 3C probability numerator or denominator");
  }
}
const s3a = await source("s3a_trajectory_optimization.csv");
const optimization = await source("figure1_optimization.csv");
for (let i = 0; i < 5; i++) {
  if (s3a.rows[i][0] !== optimization.rows[i][0] || Math.abs(s3a.rows[i][3] - optimization.rows[i][2]) > 1e-12) {
    throw new Error("S3A and Figure 1 optimization outputs disagree");
  }
}
// Add the tied/split absolute energy columns needed to support S3A without
// duplicating the same five percent-change records in another worksheet.
const optimizationSheet = workbook.worksheets.getItem("Trajectory outputs");
const absoluteData = { ...s3a, name: "s3a_trajectory_optimization.csv",
  headers: ["delta_mps", "tied_energy", "split_energy"], rows: s3a.rows.map(row => row.slice(0, 3)) };
optimizationSheet.getRange("A16").values = [["S3A absolute energy values. Source: tables/s3a_trajectory_optimization.csv"]];
putTable(optimizationSheet, absoluteData, 18, "TrajectoryAbsoluteEnergy", {
  tied_energy: "Tied energy (W/kg)", split_energy: "Split energy (W/kg)",
});
views.push({ sheetName: optimizationSheet.name, range: "A15:F24", filename: "s3a-absolute-energy" });

workbook.recalculate();
const inspection = await workbook.inspect({ kind: "table", range: "'Source guide'!A7:E13",
  include: "values,formulas", tableMaxRows: 7, tableMaxCols: 5, maxChars: 2400 });
const errors = await workbook.inspect({ kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 20 }, summary: "Source workbook error scan" });
console.log(inspection.ndjson);
console.log(errors.ndjson);
await fs.mkdir(previewDir, { recursive: true });
const renderSelection = process.env.WORKBOOK_RENDER_SHEETS?.split(",");
const renderedViews = views.filter(view => !renderSelection || renderSelection.includes(view.sheetName));
for (const view of renderedViews) {
  const preview = await workbook.render({ sheetName: view.sheetName, range: view.range, scale: 1.3, format: "png" });
  await fs.writeFile(path.join(previewDir, view.filename + ".png"), new Uint8Array(await preview.arrayBuffer()));
  console.log(`Rendered ${view.sheetName} ${view.range}`);
}
await fs.mkdir(path.dirname(output), { recursive: true });
const artifact = await SpreadsheetFile.exportXlsx(workbook);
await artifact.save(output);
await fs.writeFile(output.replace(/\.xlsx$/i, ".validation.json"), JSON.stringify({
  sourceReportCreatedAt: report.created_at, inputManifestSha256: report.inputs?.manifest_sha256,
  tables, extraSourceCells, previews: renderedViews, note: "All plotted values remain numerical CSV sources; this workbook adds no scientific refitting.",
}, null, 2) + "\n");
console.log(`Saved ${output}`);
