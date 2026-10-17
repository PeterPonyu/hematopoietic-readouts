#!/usr/bin/env python3
"""Link measured marker expression to newly trained outputs without cell p-values."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from replay_common import save_json
from frozen_calculations import ordered_conditions, COLORS, TITLES


def analyze(cohort, folder, dest):
    cells = pd.read_csv(folder / "cells.csv.gz", index_col=0)
    exp = pd.read_csv(folder / "marker_log1p.csv.gz", index_col=0)
    assert cells.index.equals(exp.index)
    protocol = json.loads((folder / "protocol.json").read_text())
    conditions = ordered_conditions(cohort, cells)
    group = "cell_type" if cohort in ["irradiation", "marrow"] else "leiden"
    cells[group] = cells[group].astype(str)
    programs = [p for p in cells if p.startswith("program_") and p != "program_annotation"]
    keys = cells[["condition", group]]
    mean = exp.join(keys).groupby(["condition", group], observed=True).mean()
    detection = (exp > 0).astype(float).join(keys).groupby(["condition", group], observed=True).mean()
    mean.to_csv(dest / "within_group_gene_means.csv")
    detection.to_csv(dest / "within_group_gene_detection.csv")
    (exp > 0).astype(float).assign(condition=cells.condition).groupby("condition").mean().to_csv(dest / "condition_gene_detection.csv")
    profiles, focus_effects, matched = [], [], []
    ref = np.load(folder / "scfocus_seed0/outputs.npz")["assignment"]
    # Integer branch IDs have no cross-seed meaning. Match to seed 0 by cell
    # overlap, explicitly retaining unmatched branches when branch counts differ.
    for seed in [0, 1, 2]:
        output = np.load(folder / f"scfocus_seed{seed}/outputs.npz")
        assignment = output["assignment"]
        weights = output["normalized_weights"]
        tab = pd.crosstab(ref, assignment)
        ii, jj = linear_sum_assignment(-tab.values)
        mapping = {tab.columns[j]: tab.index[i] for i, j in zip(ii, jj) if tab.iloc[i, j] > 0}
        for branch in range(weights.shape[1]):
            mask = assignment == branch
            row = {"seed": seed, "branch": branch, "matched_seed0_branch": mapping.get(branch, -1), "n_assigned": int(mask.sum())}
            row.update(cells.loc[mask, programs].mean().to_dict())
            lineage_scores = {k: v for k, v in row.items() if k in programs and k not in ["program_interferon", "program_inflammatory", "program_cycling"]}
            row["dominant_program"] = max(lineage_scores, key=lineage_scores.get).removeprefix("program_") if mask.any() else "unassigned"
            profiles.append(row)
            for condition in conditions:
                cm = cells.condition == condition
                focus_effects.append({"seed": seed, "branch": branch, "matched_seed0_branch": mapping.get(branch, -1), "condition": condition, "mean_weight": float(weights[cm, branch].mean()), "assigned_fraction": float((assignment[cm] == branch).mean()), "dominant_program": row["dominant_program"]})
        for a, b in mapping.items():
            intersection = int(((ref == b) & (assignment == a)).sum())
            union = int(((ref == b) | (assignment == a)).sum())
            matched.append({"seed": seed, "branch": int(a), "seed0_branch": int(b), "jaccard": intersection / union})
    pd.DataFrame(profiles).to_csv(dest / "scfocus_branch_programs.csv", index=False)
    pd.DataFrame(focus_effects).to_csv(dest / "scfocus_matched_condition_readouts.csv", index=False)
    pd.DataFrame(matched).to_csv(dest / "scfocus_branch_matches.csv", index=False)
    # Fixed marker selection, decided by the experimental question rather than
    # the genes with the largest observed differences.
    marker_sets = {"irradiation": ["Hlf", "Hoxa9", "Klf1", "Gata1", "Mpo", "Cebpe", "Isg15", "Ifit1"],
                   "dapp1": ["Dapp1", "Hlf", "Hoxa9", "Mpo", "Elane", "Ctsg", "Cebpe", "Nfkbia"],
                   "sleep": ["Isg15", "Ifit1", "Ifit3", "Irf7", "Il1b", "Tnf", "S100a8", "S100a9"],
                   "ucb": ["CD34", "HLF", "GATA2", "PF4", "PPBP", "ITGA2B", "GP9", "GP1BB"],
                   "marrow": ["HLF", "GATA2", "KLF1", "GATA1", "MPO", "ELANE", "DNTT", "EBF1"]}
    lookup = {g.upper(): g for g in exp}
    genes = [lookup[g.upper()] for g in marker_sets[cohort] if g.upper() in lookup]
    avg = exp[genes].assign(condition=cells.condition).groupby("condition").mean().reindex(conditions)
    detect = (exp[genes] > 0).astype(float).assign(condition=cells.condition).groupby("condition").mean().reindex(conditions)
    key_programs = {"dapp1": ["primitive", "neutrophil", "inflammatory"], "sleep": ["interferon", "inflammatory", "neutrophil"], "ucb": ["primitive", "megakaryocyte", "cycling"]}[cohort]
    # Fixed starting population is a descriptive subset, not a held-out test.
    start_mask = cells.cell_type.isin(protocol["starts"])
    within_start = cells[start_mask].groupby("condition")[[f"program_{p}" for p in key_programs]].mean().reindex(conditions)
    within_start.to_csv(dest / "starting_population_programs.csv")
