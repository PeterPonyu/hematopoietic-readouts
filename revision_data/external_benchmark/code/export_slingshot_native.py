#!/usr/bin/env python3
"""Preserve native R doubles directly from saved fits; never refit or round them."""
import os,subprocess,json
import numpy as np
from hematopoietic_external import ROOT,OUT,COHORTS,load,sha,write_json

script=ROOT/'scripts/benchmarks/export_slingshot_native.R'
library=ROOT/'.rlib/hematopoietic-benchmark-20261021'
environment=os.environ.copy()
environment['LD_LIBRARY_PATH']='/home/zeyufu/miniconda3/lib'+(':'+environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
reviews=[];failed=[]
for cohort in COHORTS:
    folder=OUT/cohort/'slingshot/seed0_upstream_R'
    receipt=json.loads((folder/'receipt.json').read_text())
    if receipt['status']=='running':
        print(json.dumps({'cohort':cohort,'status':'fit_still_running'}),flush=True)
        continue
    if receipt['status']=='fail':
        failed.append({'cohort':cohort,'reason':receipt.get('error'),'receipt_sha256':sha(folder/'receipt.json')})
        continue
    assert receipt['status']=='pass'
    assert receipt['outputs']['fit.rds']==sha(folder/'fit.rds')
    if (folder/'native_export_receipt.json').exists():
        previous=json.loads((folder/'native_export_receipt.json').read_text())
        assert previous['source_fit_sha256']==sha(folder/'fit.rds')
        assert previous['native_output_sha256']==sha(folder/'native_outputs.npz')
        reviews.append(previous);continue
    subprocess.run(['Rscript',str(script),str(folder),str(library)],env=environment,check=True)
    shape=tuple(map(int,(folder/'native_shape.txt').read_text().strip().split(',')))
    cells,_,_=load(cohort)
    cell_id=np.asarray((folder/'native_cell_ids.txt').read_text().splitlines(),dtype=str)
    assert np.array_equal(cell_id,cells.index)
    pt=np.fromfile(folder/'native_pseudotime.f64',dtype='<f8').reshape(shape)
    weights=np.fromfile(folder/'native_weights.f64',dtype='<f8').reshape(shape)
    original=np.load(folder/'outputs.npz')
    assert shape==original['pseudotime'].shape==original['curve_weights'].shape
    assert np.isfinite(weights).all() and (weights>=0).all()
    mask=np.isfinite(pt)&(weights>0)
    assert np.all(mask.sum(axis=0)>1)
    assert len(json.loads((folder/'lineages.json').read_text())['lineages'])==shape[1]
    # CSV is a readable companion; native doubles define the new statistics.
    np.testing.assert_allclose(pt,original['pseudotime'],atol=1e-10,rtol=1e-13,equal_nan=True)
    np.testing.assert_allclose(weights,original['curve_weights'],atol=1e-13,rtol=1e-13,equal_nan=True)
    np.savez_compressed(folder/'native_outputs.npz',cell_id=cell_id,pseudotime=pt,curve_weights=weights,
                        cluster_categories=original['cluster_categories'],evaluated=mask)
    result={'status':'pass','cohort':cohort,'shape':shape,'source_fit_sha256':sha(folder/'fit.rds'),
            'native_output_sha256':sha(folder/'native_outputs.npz'),
            'R_export_source_sha256':sha(script),'python_export_source_sha256':sha(__file__),
            'numeric_transport':'little-endian native R float64; no CSV rounding in reported statistics',
            'mask_rule':'finite native pseudotime and positive native curve weight',
            'no_refitting':True,'no_weight_normalization_or_score_filling':True}
    write_json(folder/'native_export_receipt.json',result);reviews.append(result)
    print(json.dumps(result),flush=True)
status='partial' if len(reviews)+len(failed)!=len(COHORTS) else ('pass_with_failed_fits' if failed else 'pass')
write_json(OUT/'slingshot_native_export_review.json',{'status':status,'cohorts':reviews,'failed':failed})
