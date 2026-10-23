#!/usr/bin/env python3
"""Generate source-data figures and descriptive biological/model comparisons."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import adjusted_rand_score

from replay_common import save_json, sha256

COLORS = ["#266c91", "#b96b37", "#528867", "#8b6aa5", "#ba5253", "#6a9599", "#8c8353", "#a17188", "#64788d", "#878787", "#96a44d", "#735344"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
TITLES = {"irradiation": "Irradiation: LSK state and lineage readouts", "dapp1": "Dapp1 knockout: composition and within-state changes", "sleep": "Sleep deprivation: marrow composition and programs", "ucb": "TPO-cultured cord blood: megakaryocyte development", "marrow": "Human marrow progenitors: sorted lineage reference"}


def ordered_conditions(cohort, cells):
    candidates = {"irradiation": [f"d{x}" for x in [0, 2, 5, 8, 11, 14, 21, 30]], "dapp1": ["wt", "ko"], "sleep": ["CT", "SD"], "ucb": ["D0", "D4", "D7", "D11", "D14"], "marrow": ["HSC", "MPP", "MLP", "CMP", "MEP", "GMP", "PreBNK"]}[cohort]
    present = list(cells.condition.unique())
    return [x for x in candidates if x in present] + [x for x in present if x not in candidates]


def analyze(cohort, folder, dest, require_complete):
    protocol = json.loads((folder / "protocol.json").read_text())
    cells = pd.read_csv(folder / "cells.csv.gz", index_col=0)
    cells.index = cells.index.astype(str)
    pack = np.load(folder / "input.npz")
    if not np.array_equal(pack["cell_id"], cells.index):
        raise ValueError("Input barcodes differ")
    expressions = pd.read_csv(folder / "marker_log1p.csv.gz", index_col=0)
    if not np.array_equal(expressions.index.astype(str), cells.index):
        raise ValueError("Expression barcode order differs")
    conditions = ordered_conditions(cohort, cells)
    programs = [x for x in cells if x.startswith("program_") and x != "program_annotation"]
    group = "cell_type" if cohort in ["irradiation", "marrow"] else "leiden"
    cells[group] = cells[group].astype(str)
    counts = pd.crosstab(cells.condition, cells[group]).reindex(conditions)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    counts.to_csv(dest / "composition_counts.csv")
    proportions.to_csv(dest / "composition_fractions.csv")
    condition_programs = cells.groupby("condition", observed=True)[programs].mean().reindex(conditions)
    condition_programs.to_csv(dest / "condition_program_means.csv")
    expressions.assign(condition=cells.condition).groupby("condition").mean().reindex(conditions).to_csv(dest / "condition_gene_means.csv")
    within = cells.groupby(["condition", group], observed=True)[programs].mean()
    within.to_csv(dest / "within_group_program_means.csv")
    effects = []
    if cohort in ["dapp1", "sleep"]:
        c0, c1 = conditions
        eligible = counts.columns[(counts.loc[c0] >= 20) & (counts.loc[c1] >= 20)]
        weights = counts[eligible].sum(); weights /= weights.sum()
        for program in programs:
            delta = within.loc[c1, program].reindex(eligible) - within.loc[c0, program].reindex(eligible)
            effects.append({"program": program, "contrast": f"{c1}-{c0}", "unadjusted_delta": condition_programs.loc[c1, program] - condition_programs.loc[c0, program], "composition_standardized_delta": (delta * weights).sum(), "matched_groups": len(eligible), "fraction_cells_covered": counts[eligible].values.sum() / len(cells), "scale": "mean gene-wise standardized log1p(counts per 1e4)"})
        pd.DataFrame(effects).to_csv(dest / "composition_standardized_effects.csv", index=False)
    receipts, learned, foci = [], {}, {}
    for p in sorted(folder.glob("*/receipt.json")):
        r = json.loads(p.read_text())
        if r["n_cells"] != len(cells) or r["input_sha256"] != sha256(folder / "input.npz") or r["output_sha256"] != sha256(p.parent / "outputs.npz"):
            raise ValueError(f"Integrity mismatch: {p}")
        output = dict(np.load(p.parent / "outputs.npz"))
        for key in ["value"] if r["method"] == "scRL" else ["raw_weights", "normalized_weights", "entropy"]:
            if not np.isfinite(output[key]).all() or len(output[key]) != len(cells):
                raise ValueError(f"Invalid output {p} {key}")
        receipts.append(r)
        if r["method"] == "scRL":
            learned[(r["seed"], r["lineage"], r["reward"])] = output
        else:
            if not np.allclose(output["normalized_weights"].sum(axis=1), 1):
                raise ValueError("Focus normalization failed")
            foci[r["seed"]] = output
    expected = 3 + len(protocol["ends"]) * 9
    complete = len(receipts) == expected
    if require_complete and not complete:
        raise ValueError(f"{cohort}: {len(receipts)}/{expected} complete runs")
    model_rows, state_rows, stability, associations = [], [], [], []
    for (seed, lineage, reward), output in learned.items():
        values, pt = output["value"], output["pseudotime"]
        percentiles = (rankdata(values) - .5) / len(values)
        selected = values >= np.quantile(values, .9)
        model_rows.append({"seed": seed, "lineage": lineage, "reward": reward, "top_decile_mean_pseudotime": pt[selected].mean(), "q_pseudotime_spearman": spearmanr(values, pt).statistic})
        frame = cells[["condition", group, "cell_type"]].copy() if group != "cell_type" else cells[["condition", "cell_type"]].copy()
        frame["value"] = values; frame["value_percentile"] = percentiles
        frame["grid_pseudotime"] = pt
        for condition in conditions:
            for population, mask in [("all", cells.condition == condition), ("starting_population", (cells.condition == condition) & cells.cell_type.isin(protocol["starts"]))]:
                if not mask.any():
                    continue
                state_rows.append({"seed": seed, "lineage": lineage, "reward": reward, "condition": condition, "population": population, "n_cells": int(mask.sum()), "median_value_percentile": np.median(percentiles[mask]), "mean_value": np.mean(values[mask])})
        for program in programs:
            associations.append({"seed": seed, "lineage": lineage, "reward": reward, "program": program, "spearman": spearmanr(values, cells[program]).statistic, "interpretation": "descriptive same-data association; not independent fate accuracy"})
    for lineage in protocol["ends"]:
        for reward in ["decision", "neutral", "contribution"]:
            for a, b in itertools.combinations([0, 1, 2], 2):
                if (a, lineage, reward) in learned and (b, lineage, reward) in learned:
                    stability.append({"method": "scRL", "lineage": lineage, "reward": reward, "seed_a": a, "seed_b": b, "metric": "cell_value_spearman", "value": spearmanr(learned[(a, lineage, reward)]["value"], learned[(b, lineage, reward)]["value"]).statistic})
    for a, b in itertools.combinations(sorted(foci), 2):
        stability.append({"method": "scfocus", "lineage": "all", "reward": "unsupervised", "seed_a": a, "seed_b": b, "metric": "assignment_ARI", "value": adjusted_rand_score(foci[a]["assignment"], foci[b]["assignment"])})
    for seed, f in foci.items():
        frame = pd.DataFrame(f["normalized_weights"], index=cells.index, columns=[f"branch_{i}" for i in range(f["normalized_weights"].shape[1])])
        frame["entropy"] = f["entropy"]; frame["branch"] = f["assignment"]
        frame["condition"] = cells.condition
        frame.groupby("condition").mean().reindex(conditions).to_csv(dest / f"scfocus_condition_means_seed{seed}.csv")
    model_df, state_df, stability_df = pd.DataFrame(model_rows), pd.DataFrame(state_rows), pd.DataFrame(stability)
    model_df.to_csv(dest / "reward_timing.csv", index=False)
    state_df.to_csv(dest / "state_values_by_condition.csv", index=False)
    stability_df.to_csv(dest / "seed_stability.csv", index=False)
    pd.DataFrame(associations).to_csv(dest / "value_program_associations.csv", index=False)
    summary = {"cohort": cohort, "accession": protocol["accession"], "n_cells": len(cells), "n_genes": protocol["n_genes"], "n_completed_runs": len(receipts), "expected_runs": expected, "complete": complete, "conditions": protocol["conditions"], "effects": effects, "focus_branches": {str(k): v["normalized_weights"].shape[1] for k, v in foci.items()}, "seed_stability": stability, "no_biological_replication_inference": True}
    if cohort == "irradiation":
        summary["accession"] = None
        summary["historical_protocol_accession"] = protocol["accession"]
        summary["source_identity"] = "author-owned historical local irradiation series; historical protocol accession is not a full-series public identity claim"
    save_json(dest / "summary.json", summary)
    return summary

