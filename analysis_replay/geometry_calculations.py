#!/usr/bin/env python3
"""Descriptive endpoint-proximity controls on frozen application inputs.

This is not a fate-prediction benchmark: endpoint labels, marker programs and
representations are shared with the original analysis. No training is repeated.
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from replay_common import save_json, sha256
PROGRAM = {"erythroid": "erythroid", "myeloid": "neutrophil", "megakaryocyte": "megakaryocyte"}
LABELS = {"irradiation": "Irradiation", "ucb": "TPO culture", "dapp1": "Dapp1 KO", "sleep": "Sleep loss"}


def analyze(OUT, DEST, COHORTS):
    ROOT=OUT
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST / "receipt.json").exists():
        raise FileExistsError("Completed reference exists; retain it and use its saved tables")
    started = time.time()
    rows, sources, checks = [], [], []
    for cohort in COHORTS:
        folder = OUT / cohort
        for name in ["input.npz", "cells.csv.gz", "protocol.json"]:
            sources.append({"path": str((folder / name).relative_to(ROOT)), "sha256": sha256(folder / name)})
        cells = pd.read_csv(folder / "cells.csv.gz", index_col=0)
        input_hash = sha256(folder / "input.npz")
        pack = np.load(folder / "input.npz")
        protocol = json.loads((folder / "protocol.json").read_text())
        assert np.array_equal(pack["cell_id"], cells.index.astype(str))
        for lineage, labels in protocol["ends"].items():
            endpoint = cells.cell_type.isin(labels).to_numpy()
            starts = cells.cell_type.isin(protocol["starts"]).to_numpy()
            assert endpoint.sum() >= 15 and not (endpoint & starts).any()
            target = cells["program_" + PROGRAM[lineage]].to_numpy()
            values, selected_k = {}, 15
            for name, key in [("UMAP distance", "embedding"), ("PCA30 distance", "latent")]:
                x = pack[key].astype(np.float64)
                model = NearestNeighbors(n_neighbors=selected_k + 1, metric="euclidean", n_jobs=1).fit(x[endpoint])
                distances, _ = model.kneighbors(x)
                # A terminal cell excludes its own zero-distance match. Tied
                # coordinates are harmless: deleting any zero gives the same mean.
                score = -distances[:, :selected_k].mean(axis=1)
                score[endpoint] = -distances[endpoint, 1:selected_k + 1].mean(axis=1)
                assert np.isfinite(score).all()
                # Independent direct-distance check on deterministically spaced cells.
                idx = np.linspace(0, len(x) - 1, 19, dtype=int)
                exact = np.sort(cdist(x[idx], x[endpoint]), axis=1)
                expected = np.array([-d[1:16].mean() if endpoint[i] else -d[:15].mean() for i, d in zip(idx, exact)])
                np.testing.assert_allclose(score[idx], expected, rtol=1e-10, atol=1e-10)
                checks.append({"cohort": cohort, "lineage": lineage, "space": name, "direct_distance_cells": len(idx), "max_abs_error": float(np.max(np.abs(score[idx] - expected)))})
                values[name] = score
            for seed in [0, 1, 2]:
                path = folder / f"scRL_seed{seed}_{lineage}_neutral" / "outputs.npz"
                receipt = json.loads(path.with_name("receipt.json").read_text())
                assert receipt["input_sha256"] == input_hash
                assert receipt["output_sha256"] == sha256(path)
                sources.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
                values[f"scRL seed {seed}"] = np.load(path)["value"]
            masks = {"all": np.ones(len(cells), dtype=bool), "nonterminal": ~endpoint, "starting": starts}
            for population, mask in masks.items():
                for method, score in values.items():
                    rho = float(spearmanr(score[mask], target[mask]).statistic)
                    assert np.isfinite(rho)
                    rows.append({"cohort": cohort, "lineage": lineage, "program": PROGRAM[lineage], "population": population,
                                 "method": method, "n_cells": int(mask.sum()), "n_endpoint": int(endpoint.sum()), "spearman": rho})
            pd.DataFrame({"cell_id": cells.index, "endpoint": endpoint, "starting": starts,
                          "program": target, **values}).to_csv(DEST / f"{cohort}_{lineage}_cells.csv.gz", index=False)
            print(cohort, lineage, "reference complete", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(DEST / "associations.csv", index=False)
    summary = []
    for keys, group in result.groupby(["cohort", "lineage", "population"], sort=False):
        rl = group[group.method.str.startswith("scRL")].spearman
        ref = group[~group.method.str.startswith("scRL")].set_index("method").spearman
        summary.append(dict(zip(["cohort", "lineage", "population"], keys), n_cells=int(group.n_cells.iloc[0]),
                            scrl_min=rl.min(), scrl_median=rl.median(), scrl_max=rl.max(), umap=ref["UMAP distance"], pca=ref["PCA30 distance"]))
    pd.DataFrame(summary).to_csv(DEST / "summary.csv", index=False)
    save_json(DEST / "receipt.json", {"status": "completed", "elapsed_s": time.time() - started,
        "definition": "Negative mean Euclidean distance to 15 nearest endpoint cells; exclude self for endpoint queries; no coordinate rescaling",
        "evaluation": "Same-data Spearman with fixed marker programs; all, nonterminal, and starting cells; seeds are algorithm repeats",
        "n_tasks": len(summary) // 3, "n_metric_rows": len(result), "n_new_training_runs": 0,
        "driver_sha256": sha256(__file__), "sources": sources, "checks": checks,
        "outputs": [{"file": p.name, "sha256": sha256(p)} for p in sorted(DEST.glob("*.csv*"))]})
