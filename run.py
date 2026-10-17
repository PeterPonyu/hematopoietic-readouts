#!/usr/bin/env python3
"""Portable, standard-library entry points for the manuscript-specific release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify():
    record = json.loads((HERE / "RELEASE_MANIFEST.json").read_text())
    if record.get("schema_version") != 1 or record.get("missing_final_inputs"):
        raise ValueError("Release is incomplete; refresh from the final article before distribution")
    expected, errors = set(), []
    for item in record["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in expected:
            raise ValueError("Unsafe or repeated release path: " + str(relative))
        expected.add(str(relative))
        target = HERE / relative
        if target.is_symlink() or not target.resolve().is_relative_to(HERE):
            errors.append("Unsafe release path: " + str(relative))
        elif not target.is_file():
            errors.append("Missing " + str(relative))
        elif target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
            errors.append("Content differs: " + str(relative))
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob("*") if path.is_file()
              and not any(part in (".git", "__pycache__") for part in path.relative_to(HERE).parts)
              and path.suffix != ".pyc"}
    errors += ["Unlisted file: " + name for name in sorted(actual - expected - {"RELEASE_MANIFEST.json"})]
    if errors:
        raise ValueError("\n".join(errors))
    return {"status": "pass", "files_checked": len(expected),
            "bytes": sum(item["bytes"] for item in record["files"]),
            "scope": "Closed inventory, size and checksum agreement; no analysis executed"}


def execute(arguments):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, "-I", *map(str, arguments)], check=True, env=environment)


def bundle_root(path):
    root = path.resolve()
    if not (root / "SUPPLEMENTARY_MANIFEST.json").is_file():
        raise FileNotFoundError("Missing SUPPLEMENTARY_MANIFEST.json. Extract Supplementary Code 1 and Data 1–8 into their common project folder and pass that folder with --bundle.")
    execute([HERE / "scripts/verify_supplements.py", "--root", root])
    return root


def new_output(path):
    root = path.resolve()
    if root == HERE or root.is_relative_to(HERE):
        raise ValueError("Choose an output directory outside this source repository")
    if root.exists():
        raise FileExistsError("Use a new output directory: " + str(root))
    root.parent.mkdir(parents=True, exist_ok=True)
    return root


def compare_csv(actual, expected, atol=1e-10):
    with actual.open(newline="") as stream:
        got = list(csv.reader(stream))
    with expected.open(newline="") as stream:
        wanted = list(csv.reader(stream))
    if len(got) != len(wanted):
        raise ValueError("Different row count: " + actual.name)
    for row_index, (left, right) in enumerate(zip(got, wanted)):
        if len(left) != len(right):
            raise ValueError("Different column count: " + actual.name)
        for column, (value, reference) in enumerate(zip(left, right)):
            if value == reference:
                continue
            try:
                numeric, expected_numeric = float(value), float(reference)
            except ValueError:
                raise ValueError(f"Different field in {actual.name}, record {row_index}, column {column + 1}") from None
            if not ((math.isnan(numeric) and math.isnan(expected_numeric)) or
                    math.isclose(numeric, expected_numeric, rel_tol=1e-10, abs_tol=atol)):
                raise ValueError(f"Different number in {actual.name}, record {row_index}, column {column + 1}")
    return max(0, len(got) - 1)


def displays(output, figures):
    verify()
    output = new_output(output)
    output.mkdir()
    receipt = {"status": "in_progress", "release_manifest_sha256": sha256(HERE / "RELEASE_MANIFEST.json")}
    try:
        for directory in ("scripts", "source_data", "supplement_tables", "sensitivity_data"):
            shutil.copytree(HERE / directory, output / directory)
        shutil.copy2(HERE / "aggregate_metadata.json", output / "aggregate_metadata.json")
        execute([output / "scripts/build_supplement_tables.py"])
        execute([output / "scripts/build_sensitivity_table.py", "--root", output,
                 "--input-dir", output / "sensitivity_data", "--output-root", output])
        execute([output / "scripts/build_frozen_readout_table.py", "--root", output,
                 "--output-root", output])
        if figures:
            execute([output / "scripts/build_figures.py"])
            execute([output / "scripts/build_interpretation_assets.py"])
        compared = []
        for expected in sorted((HERE / "supplement_tables").glob("*")):
            actual = output / "supplement_tables" / expected.name
            if expected.suffix in (".csv", ".tex"):
                if actual.read_bytes() != expected.read_bytes():
                    raise ValueError("Formatted table bytes differ: " + expected.name)
                compared.append(expected.name)
        receipt.update({"status": "passed", "table_files_compared": compared,
                        "formatted_tables": "S1–S13 and S20–S21; retained S14–S19 copied as aggregate references",
                        "scientific_analysis_recomputed": False})
        if figures:
            rows = {name: compare_csv(output / "evidence" / name, HERE / "reference_data" / name)
                    for name in ("figure_values.csv", "figure3_composition_context.csv", "interpretation_display_values.csv")}
            receipt.update({"figures": 8, "display_rows_compared": rows,
                            "additional_formatted_tables": "S18–S19 regenerated from aggregate tables"})
    except BaseException as exc:
        receipt.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        save(output / "display_receipt.json", receipt)
    return receipt


def replay(bundle, output, cohorts, checkpoints, figures, denied):
    verify()
    root = bundle_root(bundle)
    output = new_output(output)
    with tempfile.TemporaryDirectory(prefix="hematopoietic-replay-") as folder:
        stage = Path(folder) / "analysis_replay"
        shutil.copytree(HERE / "analysis_replay", stage)
        for name in ("inputs", "archival"):
            (stage / name).symlink_to(root / "analysis_replay" / name, target_is_directory=True)
        shutil.copy2(root / "analysis_replay/support_manifest.json", stage / "support_manifest.json")
        command = [stage / "replay.py", "--input-root", stage, "--output-root", output, "--cohorts", *cohorts]
        if checkpoints:
            command.append("--checkpoints")
        for path in denied:
            command += ["--deny-read-root", path.resolve()]
        execute(command)
        if figures:
            execute([stage / "render_figures.py", "--replay-output", output])
    receipt = {"status": "passed", "scope": "Repository replay code executed with separately supplied retained inputs",
               "release_manifest_sha256": sha256(HERE / "RELEASE_MANIFEST.json"),
               "supplementary_manifest_sha256": sha256(root / "SUPPLEMENTARY_MANIFEST.json"),
               "scientific_receipt_sha256": sha256(output / "replay_receipt.json"),
               "figures_requested": figures, "checkpoints_requested": checkpoints}
    save(output / "source_release_replay_receipt.json", receipt)
    return receipt


def sensitivity(bundle, output):
    verify()
    root = bundle_root(bundle)
    output = new_output(output)
    output.mkdir()
    receipt = {"status": "in_progress", "release_manifest_sha256": sha256(HERE / "RELEASE_MANIFEST.json"),
               "supplementary_manifest_sha256": sha256(root / "SUPPLEMENTARY_MANIFEST.json")}
    try:
        execute([HERE / "scripts/calculate_program_sensitivity.py", "--root", root,
                 "--output-dir", output / "sensitivity_data"])
        execute([HERE / "scripts/build_sensitivity_table.py", "--root", root,
                 "--input-dir", output / "sensitivity_data", "--output-root", output])
        compared = {name: compare_csv(output / "sensitivity_data" / name, HERE / "sensitivity_data" / name)
                    for name in ("marker_sensitivity_full_precision.csv", "marker_sensitivity_group_weights.csv",
                                 "marker_sensitivity_existing_reference_checks.csv")}
        compare_csv(output / "supplement_tables/s20_marker_sensitivity.csv",
                    HERE / "supplement_tables/s20_marker_sensitivity.csv")
        receipt.update({"status": "passed", "aggregate_rows_compared": compared,
                        "scope": "Single-marker and leave-one-marker-out summaries under frozen groups; no reclustering, training or independent annotation"})
    except BaseException as exc:
        receipt.update({"status": "failed", "error": str(exc)})
        raise
    finally:
        save(output / "source_release_sensitivity_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Check this source release inventory and checksums")
    supplied = commands.add_parser("verify-supplements", help="Check the separate scientific supplements")
    supplied.add_argument("--bundle", type=Path, required=True)
    for name in ("tables", "display"):
        command = commands.add_parser(name, help="Regenerate aggregate tables" if name == "tables" else "Regenerate aggregate tables and eight figures")
        command.add_argument("--output-root", type=Path, required=True)
    scientific = commands.add_parser("replay", help="Recompute retained-artifact summaries with separately supplied inputs")
    scientific.add_argument("--bundle", type=Path, required=True)
    scientific.add_argument("--output-root", type=Path, required=True)
    scientific.add_argument("--cohorts", nargs="+", required=True,
                            choices=("ucb", "dapp1", "sleep", "marrow", "dapp1_spliced"))
    scientific.add_argument("--checkpoints", action="store_true")
    scientific.add_argument("--figures", action="store_true")
    scientific.add_argument("--deny-read-root", type=Path, action="append", default=[])
    extra = commands.add_parser("sensitivity", help="Recompute marker sensitivity with separately supplied frozen inputs")
    extra.add_argument("--bundle", type=Path, required=True)
    extra.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify()
    elif args.command == "verify-supplements":
        verify()
        result = {"status": "pass", "bundle": str(bundle_root(args.bundle)), "scope": "Supplement integrity only"}
    elif args.command in ("tables", "display"):
        result = displays(args.output_root, args.command == "display")
    elif args.command == "replay":
        result = replay(args.bundle, args.output_root, args.cohorts, args.checkpoints, args.figures, args.deny_read_root)
    else:
        result = sensitivity(args.bundle, args.output_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
