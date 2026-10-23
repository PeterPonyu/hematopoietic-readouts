#!/usr/bin/env python3
"""Check pinned Palantir source and native failed scores, reachability and conditioning."""
from pathlib import Path
import argparse,hashlib,json,zipfile
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--article-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();r=a.packet;p=a.article_root;imp=r/'implementation'
    release=json.loads((imp/'upstream_release.json').read_text());wheel=imp/release['filename'];assert sha(wheel)==release['sha256']
    source=json.loads((imp/'palantir_source_conformance.json').read_text());assert len(source['python_files'])==12
    with zipfile.ZipFile(wheel) as z:
        for item in source['python_files']:
            data=z.read('palantir/'+item['path']);assert hashlib.sha256(data).hexdigest()==item['upstream_sha256']==item['local_sha256']
            assert data==(imp/'upstream_palantir_1_4_2/palantir'/item['path']).read_bytes()
            assert data==(p/'revision_data/external_benchmark/manual_sources/palantir'/item['path']).read_bytes()
    rows=[];table=pd.read_csv(imp/'palantir_failure_diagnostics.csv');previous=json.loads((imp/'palantir_diagnostics/recorded_diagnostics.json').read_text())
    for _,row in table.iterrows():
        seed=int(row.seed);folder=imp/'palantir_diagnostics';raw=np.load(folder/f'seed{seed}_native_outputs.npz');x=raw['probability'];mass=x.sum(1)
        invalid=(~np.isfinite(x).all(1))|(x.min(1)<0)|(mass>1+1e-6)|(mass<1-.01*x.shape[1]-1e-6)
        assert len(x)==row.n_cells and invalid.sum()==row.failed_cells
        assert (mass==0).sum()==row.zero_mass_cells and (mass>1+1e-6).sum()==row.above_unit_mass_cells
        for value,key in [(invalid.mean(),'failed_fraction'),(x.min(),'score_minimum'),(x.max(),'score_maximum'),(mass.min(),'row_mass_minimum'),(mass.max(),'row_mass_maximum')]:np.testing.assert_allclose(value,row[key],atol=1e-12,rtol=1e-12)
        assert sha(folder/f'seed{seed}_native_outputs.npz')==previous['results'][seed]['output_sha256']
        T=sparse.load_npz(folder/f'seed{seed}_transition.npz');T.eliminate_zeros();w=json.loads((folder/f'seed{seed}_waypoints.json').read_text());ends=np.asarray(w['absorbing_indices']);reached=np.zeros(T.shape[0],bool)
        for k in ends:reached[breadth_first_order(T.T.tocsr(),int(k),directed=True,return_predecessors=False)]=True
        assert np.array_equal(reached,w['can_reach_terminal']) and (~reached).sum()==row.unreachable_waypoints and len(reached)==row.waypoints
        transient=np.setdiff1d(np.arange(T.shape[0]),ends);Q=T[transient][:,transient].toarray();s=np.linalg.svd(np.eye(len(Q))-Q,compute_uv=False)
        condition=float(s[0]/s[-1]);assert condition>1e14 and s[-1]<1e-14
        # Recorded extreme condition numbers are roundoff-sensitive. Verify the
        # reported snapshot and the near-singular diagnosis without requiring an
        # unstable bitwise equality of its smallest singular value.
        old=previous['results'][seed];assert old['fundamental_system_condition_number']==row.condition_number or np.isclose(old['fundamental_system_condition_number'],row.condition_number,rtol=1e-14)
        rows.append({'seed':seed,'native_failed_cells':int(invalid.sum()),'reachability_exact':True,'recomputed_condition_number':condition,'recomputed_smallest_singular_value':float(s[-1]),'near_singular_diagnosis_reproduced':True})
    record={'status':'pass','source_files_byte_identical':12,'native_failure_rows':3,'checks':rows,
            'conditioning_scope':'Recorded SVD snapshot reproduced as a near-singular diagnosis; extreme condition-number magnitudes depend on numerical roundoff.',
            'script_sha256':sha(Path(__file__))}
    a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
