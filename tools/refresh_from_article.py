#!/usr/bin/env python3
"""Refresh an explicit source/aggregate allowlist from the final article folder.

Uses the standard library, does not run analyses, and never writes to the article.
Only previously generated release files may be removed. Publication is separate.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tempfile

RELEASE = Path(__file__).resolve().parents[1]
FROZEN = "results/application_reanalysis_20260921"
MATCHED = "results/recovered_integration_20261001"
CORE = (
    "replay.py", "replay_common.py", "frozen_calculations.py",
    "biological_calculations.py", "matched_calculations.py",
    "geometry_calculations.py", "figure_calculations.py", "render_figures.py",
    "requirements-replay.txt", "tolerances.json", "expected_figure_values.csv",
)
SCRIPTS = (
    "build_figures.py", "build_supplement_tables.py", "build_revision_tables.py",
    "build_interpretation_assets.py", "verify_supplements.py",
    "calculate_program_sensitivity.py", "build_sensitivity_table.py",
    "build_frozen_readout_table.py",
)
REFERENCE = (
    "figure_values.csv", "interpretation_display_values.csv",
    "figure3_composition_context.csv", "figure3_composition_context.json",
    "program_sensitivity_summary.json",
    "frozen_readout_table_sources.json",
)
SENSITIVITY = (
    "marker_sensitivity_full_precision.csv", "marker_sensitivity_group_weights.csv",
    "marker_sensitivity_existing_reference_checks.csv", "marker_sensitivity_manifest.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def js(data):
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


def strip_non_scientific_metadata(text):
    """Remove provenance-dictionary authorship fields without changing formulas."""
    offsets, edits = [0], []
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode()))
    encoded = text.encode()
    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value.endswith("_authorship"):
                start = offsets[key.lineno - 1] + key.col_offset
                end = offsets[value.end_lineno - 1] + value.end_col_offset
                cursor = end
                while cursor < len(encoded) and chr(encoded[cursor]).isspace():
                    cursor += 1
                if encoded[cursor:cursor + 1] != b",":
                    raise ValueError("Expected a separated metadata dictionary field")
                edits.append((start, cursor + 1))
    for start, end in sorted(edits, reverse=True):
        encoded = encoded[:start] + encoded[end:]
    result = encoded.decode()
    ast.parse(result)
    return result


def adapt_script(name, text):
    text = strip_non_scientific_metadata(text)
    # A standalone release never falls back to the original repository hierarchy.
    text = text.replace("ROOT = PROJECT.parents[1]", "ROOT = PROJECT")
    if name == "build_supplement_tables.py":
        old = "def read_json(path):\n    return json.loads(tracked(path).read_text())"
        new = """def read_json(path):
    relative = str(Path(path).relative_to(ROOT))
    metadata = PROJECT / 'aggregate_metadata.json'
    SOURCES['aggregate_metadata.json'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return json.loads(metadata.read_text())['records'][relative]"""
        if old not in text:
            raise ValueError("Supplementary formatter JSON interface changed; inspect before refreshing")
        text = text.replace(old, new)
        start = text.index("    # Receipt inventory validates")
        end = text.index("    csv_out('matched_run_inventory',rows)", start)
        replacement = """    # Field-selected aggregate inventory; no historical receipt files required.
    metadata = PROJECT / 'aggregate_metadata.json'
    SOURCES['aggregate_metadata.json'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    rows = json.loads(metadata.read_text())['matched_run_inventory']
    assert len(rows) == 90
    assert sum(r['method'] == 'scRL' for r in rows) == 72
"""
        text = text[:start] + replacement + text[end:]
    ast.parse(text)
    return text


def check_csv(relative, data):
    reader = csv.reader(io.StringIO(data.decode()))
    fields = next(reader, [])
    forbidden = {"cell_id", "cellid", "barcode", "cell_barcode", "donor_id", "subject_id"}
    if forbidden.intersection(field.lower() for field in fields):
        raise ValueError("Per-cell or participant identifier column is outside release scope: " + relative)


def refresh(source, allow_incomplete=False, repository_url=None):
    source = source.resolve()
    if source == RELEASE or source.is_relative_to(RELEASE):
        raise ValueError("The canonical article must be outside the release")
    required = [source / "scripts" / name for name in SCRIPTS]
    required += [source / "sensitivity_data" / name for name in SENSITIVITY]
    required += [source / "evidence" / name for name in REFERENCE]
    required += [source / "supplement_tables/s20_marker_sensitivity.csv",
                 source / "supplement_tables/s20_marker_sensitivity.tex",
                 source / "supplement_tables/s21_additional_frozen_readouts.csv",
                 source / "supplement_tables/s21_additional_frozen_readouts.tex"]
    missing = [str(p.relative_to(source)) for p in required if not p.is_file()]
    if missing and not allow_incomplete:
        raise FileNotFoundError("Final article additions are missing: " + ", ".join(missing))

    outputs, copies = {}, []

    main_source = source / "main.tex"
    title_match = re.search(r"\\title\{([^{}]+)\}", main_source.read_text())
    if not title_match:
        raise ValueError("Cannot identify the article title in main.tex")
    article_title = title_match.group(1)
    previous_article = RELEASE / "ARTICLE.json"
    previous_title = json.loads(previous_article.read_text())["title"] if previous_article.exists() else "Cell composition and transcriptional programs contextualize model-derived fate readouts across hematopoietic conditions"
    changed_documents = {}
    for relative in ("README.md", "PROVENANCE.md", ".zenodo.json"):
        original = (RELEASE / relative).read_text()
        updated = original.replace(previous_title, article_title)
        if updated != original:
            changed_documents[relative] = updated
    if repository_url:
        if not re.fullmatch(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?", repository_url):
            raise ValueError("Supply the verified public GitHub repository URL")
        original = (RELEASE / "CITATION.cff").read_text()
        updated = re.sub(r"^repository-code:.*\n?", "", original, flags=re.MULTILINE)
        updated += "repository-code: " + json.dumps(repository_url.rstrip("/")) + "\n"
        if updated != original:
            changed_documents["CITATION.cff"] = updated
    if changed_documents:
        backup = Path(tempfile.mkdtemp(prefix="hematopoietic-release-documents-"))
        for relative, updated in changed_documents.items():
            shutil.copy2(RELEASE / relative, backup / relative)
            (RELEASE / relative).write_text(updated)

    def add(relative, data, original=None, transform="byte-preserved copy"):
        if relative.endswith(".csv"):
            check_csv(relative, data)
        outputs[relative] = data
        if original is not None:
            raw = original.read_bytes()
            copies.append({"source_path": str(original.relative_to(source)),
                           "release_path": relative, "source_sha256": digest(raw),
                           "released_sha256": digest(data), "transformation": transform})

    def copy(original, relative=None, transform=None):
        if not original.is_file():
            if allow_incomplete:
                return
            raise FileNotFoundError(original)
        data = original.read_bytes()
        if transform:
            data = transform(data.decode()).encode()
        add(relative or str(original.relative_to(source)), data, original,
            "release-local paths and scientific metadata only; formulas unchanged" if transform else "byte-preserved copy")

    add("ARTICLE.json", js({"title": article_title, "author": "Zeyu Fu",
                            "orcid": "0009-0001-8329-0108",
                            "correspondence": ["fuzeyu99@tmmu.edu.cn", "fuzeyu99@126.com"],
                            "source_main_sha256": digest(main_source.read_bytes()),
                            "scope": "Identification of the associated manuscript; no manuscript text included"}))

    for name in SCRIPTS:
        copy(source / "scripts" / name, transform=lambda text, n=name: adapt_script(n, text))
    for name in CORE:
        copy(source / "analysis_replay" / name,
             transform=(lambda text: adapt_script("figure_calculations.py", text)) if name == "figure_calculations.py" else None)
    for original in sorted((source / "analysis_replay/vendor").rglob("*")):
        if original.is_file() and (original.suffix == ".py" or original.name == "LiVAE-LICENSE"):
            copy(original)
    for original in sorted((source / "analysis_replay/expected").rglob("*.csv")):
        copy(original)
    for original in sorted((source / "source_data").rglob("*.csv")):
        copy(original)
    for original in sorted((source / "supplement_tables").glob("s[0-9][0-9]*")):
        if original.suffix in (".csv", ".tex"):
            copy(original)
    copy(source / "supplement_tables/matched_run_inventory.csv")
    for name in REFERENCE:
        copy(source / "evidence" / name, "reference_data/" + name)
    for name in SENSITIVITY:
        copy(source / "sensitivity_data" / name)
    for upstream in ("LiVAE", "scRL", "scfocus"):
        copy(source / "provenance_support/upstream_metadata" / upstream / "LICENSE",
             "LICENSES/" + upstream + ".txt")

    records = {}
    protocol_fields = ("cohort", "accession", "n_input", "n_cells", "n_genes", "conditions",
                       "root", "starts", "ends", "program_genes", "program_annotation_is_provisional")
    for cohort in ("ucb", "dapp1", "sleep", "marrow"):
        relative = FROZEN + "/" + cohort + "/protocol.json"
        original = source / "source_data" / relative
        full = json.loads(original.read_text())
        records[relative] = {key: full[key] for key in protocol_fields if key in full}
        copies.append({"source_path": str(original.relative_to(source)),
                       "release_path": "aggregate_metadata.json#records/" + relative,
                       "source_sha256": digest(original.read_bytes()),
                       "released_sha256": digest(js(records[relative])),
                       "transformation": "scientific cohort/protocol fields only; local source paths omitted"})
    training_fields = ("training_cells", "optimization_validation_cells", "actual_epochs", "best_epoch",
                       "initial_validation_loss", "best_validation_loss", "relative_optimization_loss_reduction",
                       "mean_15NN_overlap_with_frozen_PCA", "median_15NN_overlap_with_frozen_PCA")
    for cohort in ("ucb", "dapp1", "sleep"):
        relative = MATCHED + "/native_representation/" + cohort + "/receipt.json"
        original = source / "source_data" / relative
        full = json.loads(original.read_text())
        records[relative] = {key: full[key] for key in training_fields}
        copies.append({"source_path": str(original.relative_to(source)),
                       "release_path": "aggregate_metadata.json#records/" + relative,
                       "source_sha256": digest(original.read_bytes()),
                       "released_sha256": digest(js(records[relative])),
                       "transformation": "aggregate optimization diagnostics only"})
    inventory_file = source / "supplement_tables/matched_run_inventory.csv"
    inventory = list(csv.DictReader(io.StringIO(inventory_file.read_text())))
    for item in inventory:
        if item["method"].lower() != "scfocus":
            continue
        relative = MATCHED + "/paired_downstream/" + item["arm"] + "/" + item["cohort"] + "/" + item["run"] + "/receipt.json"
        original = source / "source_data" / relative
        full = json.loads(original.read_text())
        records[relative] = {"actual_episodes": full["actual_episodes"]}
        copies.append({"source_path": str(original.relative_to(source)),
                       "release_path": "aggregate_metadata.json#records/" + relative,
                       "source_sha256": digest(original.read_bytes()),
                       "released_sha256": digest(js(records[relative])),
                       "transformation": "scFocus ensemble episode counts only"})
    add("aggregate_metadata.json", js({"scope": "Aggregate inputs for the existing table formatter; no per-cell identifiers or full execution records",
                                       "records": records, "matched_run_inventory": inventory}))

    acquisition_file = source / "analysis_replay/acquisition_manifest.json"
    acquisition = json.loads(acquisition_file.read_text())
    allowed_source_fields = ("cohort", "accession", "file", "sha256", "acquisition_url", "public_source", "role")
    sources = [{key: row[key] for key in allowed_source_fields if key in row} for row in acquisition["sources"]]
    add("provenance/source_acquisition.json", js({"scope": "Raw source provenance only; no files downloaded or raw data included", "sources": sources}),
        acquisition_file, "source accession, filename, checksum, acquisition URL and role fields only")

    ports_file = source / "analysis_replay/calculation_ports.json"
    ports = json.loads(ports_file.read_text())
    for row in ports:
        release_path = "analysis_replay/" + row["portable_file"]
        row["article_portable_sha256"] = row["portable_sha256"]
        row["portable_sha256"] = digest(outputs[release_path])
    add("analysis_replay/calculation_ports.json", js(ports), ports_file,
        "retain historical source attribution; record released utility hashes separately")
    add("provenance/source_copies.json", js({"schema_version": 1, "source_project": source.name,
        "scope": "Manuscript-specific software and aggregate data; no per-cell payloads or internal work records",
        "missing_final_inputs": missing, "files": copies}))

    manifest_path = RELEASE / "RELEASE_MANIFEST.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for relative in set(previous.get("generated_files", [])) - set(outputs):
        candidate = RELEASE / relative
        if candidate.is_file() and candidate.resolve().is_relative_to(RELEASE):
            candidate.unlink()
    for relative, data in outputs.items():
        path = RELEASE / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("Release output cannot be a symlink: " + relative)
        path.write_bytes(data)
    files = []
    for path in sorted(RELEASE.rglob("*")):
        relative = path.relative_to(RELEASE)
        if path.is_file() and path != manifest_path and not any(part in (".git", "__pycache__") for part in relative.parts) and path.suffix != ".pyc":
            files.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())})
    manifest_path.write_bytes(js({"schema_version": 1, "scope": "Lean manuscript-specific source release",
                                 "generated_files": sorted(outputs), "files": files,
                                 "missing_final_inputs": missing}))
    return {"status": "incomplete" if missing else "pass", "files": len(files),
            "bytes": sum(row["bytes"] for row in files), "missing_final_inputs": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Prepare a local draft while final sensitivity assets are being written")
    parser.add_argument("--repository-url", help="Add an externally verified GitHub URL to CITATION.cff")
    args = parser.parse_args()
    print(json.dumps(refresh(args.source, args.allow_incomplete, args.repository_url), indent=2))


if __name__ == "__main__":
    main()
