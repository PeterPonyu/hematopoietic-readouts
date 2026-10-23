#!/usr/bin/env python3
"""Endpoint-free diffusion geometry for cohorts without informative fate tasks."""
from datetime import datetime, timezone
import importlib.metadata
import json
import resource
import time
import palantir
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from hematopoietic_external import OUT, load, sha, write_json, package_sources

for cohort in ['ucb','sleep']:
    cells,pack,_=load(cohort)
    dest=OUT/cohort/'diffusion/seed0';dest.mkdir(parents=True,exist_ok=False)
    start=time.time()
    diffusion=palantir.utils.run_diffusion_maps(pd.DataFrame(pack['latent'].astype(float),index=cells.index),
                                               n_components=10,knn=30,alpha=0,seed=0)
    space=palantir.utils.determine_multiscale_space(diffusion)
    assert np.array_equal(space.index,cells.index) and np.isfinite(space).all().all()
    np.savez_compressed(dest/'outputs.npz',cell_id=cells.index.to_numpy(dtype=str),
                        inference_space=minmax_scale(space.to_numpy()))
    receipt={'status':'pass','cohort':cohort,'seed':0,'purpose':'geometric control only; no fate fit',
             'started_utc':datetime.now(timezone.utc).isoformat(),
             'input_sha256':sha(OUT.parent/'application_reanalysis_20260921'/cohort/'input.npz'),
             'output_sha256':sha(dest/'outputs.npz'),'driver_sha256':sha(__file__),
             'source_files':package_sources(palantir),'package_version':importlib.metadata.version('palantir'),
             'multiscale_dimensions':space.shape[1],'component_scaling':'min-max, matching Palantir inference',
             'elapsed_seconds':time.time()-start,'peak_rss_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    write_json(dest/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
