#!/usr/bin/env python3
"""Matched-population, signed associations; no biological accuracy inference."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from hematopoietic_external import ROOT, OUT, INPUT, COHORTS, load, sha, write_json

PROGRAM={'erythroid':'erythroid','myeloid':'neutrophil','megakaryocyte':'megakaryocyte'}
GEOMETRY=INPUT/'geometry_reference'


def select_run(cohort,method,seed,inventory):
    candidates=[];failed=[]
    for path in sorted((OUT/cohort/method).glob(f'seed{seed}*/receipt.json')):
        record=json.loads(path.read_text())
        expected=f'seed{seed}'
        if method=='palantir' and cohort=='dapp1' and seed==0:expected='seed0_cutoff_checked'
        if method=='fateid' and cohort=='irradiation':expected=f'seed{seed}_bounded'
        if method=='slingshot':expected='seed0_upstream_R'
        diagnostic=path.parent.name!=expected
        inventory.append({'cohort':cohort,'method':method,'seed':seed,'run':str(path.parent.relative_to(OUT)),
                          'role':'technical_diagnostic' if diagnostic else 'primary_attempt',
                          'status':record['status'],'error':record.get('error'),
                          'elapsed_seconds':record.get('elapsed_seconds'),
                          'peak_rss_MiB':max([record[k] for k in ['peak_rss_MiB','r_peak_rss_MiB'] if record.get(k) is not None],default=np.nan),
                          'receipt_sha256':sha(path)})
        if not diagnostic and record['status']=='pass':candidates.append((path.parent,record))
        elif not diagnostic and record['status']=='fail':failed.append((path.parent,record))
        elif not diagnostic and record['status'] not in ['cancelled','fail']:
            raise ValueError(f'Run is not finished: {path}')
    if not candidates and failed:
        # A completed failed method stays in the denominator. Its fate scores
        # remain unavailable; finite pre-inference geometry can be used separately.
        folder,record=failed[-1]
        data=dict(np.load(folder/'outputs.npz',allow_pickle=False)) if (folder/'outputs.npz').exists() else None
        return folder,record,data
    if len(candidates)!=1:
        raise ValueError(f'Expected exactly one valid run, found {len(candidates)}: {cohort}/{method}/{seed}')
    folder,record=candidates[0]
    assert record['outputs']['outputs.npz']==sha(folder/'outputs.npz')
    output=folder/'outputs.npz'
    if method=='slingshot':
        native=json.loads((folder/'native_export_receipt.json').read_text())
        assert native['status']=='pass' and native['source_fit_sha256']==sha(folder/'fit.rds')
        output=folder/'native_outputs.npz'
        assert native['native_output_sha256']==sha(output)
    with np.load(output,allow_pickle=False) as saved:
        # The earliest retained-port run serialized gene names as object strings.
        # The receipt's gene-axis hash binds them; no pickle loading is needed.
        data={k:saved[k] for k in saved.files if k!='genes'}
    return folder,record,data


def correlation(score,target,mask):
    x=score[mask];y=target[mask]
    if len(x)<3:return np.nan,'insufficient_cells'
    if not np.isfinite(x).all():return np.nan,'nonfinite_score'
    if np.ptp(x)==0:return np.nan,'constant_score'
    if np.ptp(y)==0:return np.nan,'constant_program'
    return float(spearmanr(x,y).statistic),'defined'


def proximity(space,endpoint):
    assert endpoint.sum()>=16
    model=NearestNeighbors(n_neighbors=16,metric='euclidean',n_jobs=1).fit(space[endpoint])
    distances,_=model.kneighbors(space)
    score=-distances[:,:15].mean(axis=1)
    score[endpoint]=-distances[endpoint,1:16].mean(axis=1)
    indices=np.linspace(0,len(space)-1,19,dtype=int)
    exact=np.sort(cdist(space[indices],space[endpoint]),axis=1)
    expected=np.array([-d[1:16].mean() if endpoint[i] else -d[:15].mean() for i,d in zip(indices,exact)])
    np.testing.assert_allclose(score[indices],expected,atol=1e-10,rtol=1e-10)
    return score,float(np.max(np.abs(score[indices]-expected)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='analysis_v1')
    args=parser.parse_args();dest=OUT/args.name;dest.mkdir(exist_ok=False)
    registration=json.loads((OUT/'registration.json').read_text())
    old_receipt=json.loads((GEOMETRY/'receipt.json').read_text())
    old_hashes={x['file']:x['sha256'] for x in old_receipt['outputs']}
    rows=[];trajectory=[];coverage=[];inventory=[];sources={};distance_checks=[];applicability=[]
    for cohort in COHORTS:
        cells,pack,protocol=load(cohort);config=registration['cohorts'][cohort]
        for name,key in [('input.npz','input_sha256'),('cells.csv.gz','cells_sha256'),('protocol.json','protocol_sha256')]:
            assert sha(INPUT/cohort/name)==config[key]
            sources[str((INPUT/cohort/name).relative_to(ROOT))]=config[key]
        terminal=cells.cell_type.isin(list(config['terminal_representatives'])).to_numpy()
        masks={'all':np.ones(len(cells),bool),'shared_nonterminal':~terminal,
               'starting':cells.cell_type.isin(protocol['starts']).to_numpy()}
        mask_hashes={k:hashlib.sha256('\n'.join(cells.index[v]).encode()).hexdigest() for k,v in masks.items()}
        fits={};failed_fits={};geometry_data={}
        if config['probability_comparison']:
            for method in ['palantir','cellrank','fateid']:
                for seed in [0,1,2]:
                    folder,receipt,data=select_run(cohort,method,seed,inventory)
                    if receipt['status']!='pass':
                        failed_fits[method,seed]=receipt.get('error','invalid_native_output')
                        if method=='palantir' and data is not None:
                            assert np.array_equal(data['cell_id'],cells.index)
                            geometry_data[seed]=data
                            sources[str((folder/'outputs.npz').relative_to(ROOT))]=sha(folder/'outputs.npz')
                        continue
                    assert np.array_equal(data['cell_id'],cells.index)
                    assert np.isfinite(data['probability']).all() and data['evaluated'].all()
                    assert receipt['input_sha256']==config['input_sha256']
                    assert set(data['component_names'])==set(config['terminal_representatives'])
                    fits[method,seed]=data
                    if method=='palantir':geometry_data[seed]=data
                    sources[str((folder/'outputs.npz').relative_to(ROOT))]=sha(folder/'outputs.npz')
            spaces={seed:geometry_data[seed]['inference_space'] for seed in [0,1,2]}
            for space in spaces.values():assert len(space)==len(cells) and np.isfinite(space).all()
        else:
            folder=OUT/cohort/'diffusion/seed0'
            receipt=json.loads((folder/'receipt.json').read_text());data=dict(np.load(folder/'outputs.npz'))
            assert receipt['status']=='pass' and receipt['output_sha256']==sha(folder/'outputs.npz')
            assert np.array_equal(data['cell_id'],cells.index)
            spaces={0:data['inference_space']}
            sources[str((folder/'outputs.npz').relative_to(ROOT))]=sha(folder/'outputs.npz')
        task_values={}
        for task,labels in protocol['ends'].items():
            target=cells['program_'+PROGRAM[task]].to_numpy()
            frozen_path=GEOMETRY/f'{cohort}_{task}_cells.csv.gz'
            assert sha(frozen_path)==old_hashes[frozen_path.name]
            frozen=pd.read_csv(frozen_path,float_precision='round_trip')
            assert np.array_equal(frozen.cell_id.astype(str),cells.index)
            np.testing.assert_allclose(frozen.program,target,atol=1e-14,rtol=0)
            sources[str(frozen_path.relative_to(ROOT))]=sha(frozen_path)
            values={('PCA30 proximity',-1):frozen['PCA30 distance'].to_numpy(),
                    ('UMAP proximity',-1):frozen['UMAP distance'].to_numpy()}
            for seed in [0,1,2]:
                path=INPUT/cohort/f'scRL_seed{seed}_{task}_neutral/outputs.npz'
                receipt=json.loads(path.with_name('receipt.json').read_text())
                assert receipt['input_sha256']==config['input_sha256'] and receipt['output_sha256']==sha(path)
                score=np.load(path)['value']
                np.testing.assert_allclose(score,frozen[f'scRL seed {seed}'],atol=1e-14,rtol=0)
                values['scRL value',seed]=score;sources[str(path.relative_to(ROOT))]=sha(path)
            endpoint=cells.cell_type.isin(labels).to_numpy()
            for seed,space in spaces.items():
                score,error=proximity(space,endpoint)
                values['Diffusion proximity',seed]=score
                distance_checks.append({'cohort':cohort,'task':task,'seed':seed,'n_direct_checks':19,'max_error':error})
            for (method,seed),data in fits.items():
                columns=[list(data['component_names']).index(k) for k in labels]
                values[{'palantir':'Palantir','cellrank':'CellRank 2','fateid':'FateID Python'}[method],seed]=data['probability'][:,columns].sum(axis=1)
            for method,seed in failed_fits:
                values[{'palantir':'Palantir','cellrank':'CellRank 2','fateid':'FateID Python'}[method],seed]=np.full(len(cells),np.nan)
            task_values[task]=values
            support=pd.DataFrame({'cell_id':cells.index,'cell_type':cells.cell_type.to_numpy(),
                                  'leiden':cells.leiden.astype(str).to_numpy(),'program':target,
                                  'mask_all':masks['all'],'mask_shared_nonterminal':masks['shared_nonterminal'],
                                  'mask_starting':masks['starting'],'mask_endpoint':endpoint})
            for (method,seed),score in values.items():support[f'{method}|seed={seed}']=score
            support_path=dest/f'{cohort}_{task}_evaluation.csv.gz'
            support.to_csv(support_path,index=False,float_format='%.17g')
            decoded=pd.read_csv(support_path,float_precision='round_trip')
            assert np.array_equal(decoded.cell_id.astype(str),cells.index)
            for (method,seed),score in values.items():
                assert np.array_equal(decoded[f'{method}|seed={seed}'].to_numpy(),score,equal_nan=True)
            for population,mask in masks.items():
                for (method,seed),score in values.items():
                    assert score.shape==(len(cells),)
                    rho,status=correlation(score,target,mask)
                    if not np.isfinite(score).any():status='failed_fit_no_valid_score'
                    rows.append({'cohort':cohort,'task':task,'program':PROGRAM[task],'population':population,
                                 'method':method,'seed':seed,'n_cells':int(mask.sum()),'n_endpoint':int(endpoint.sum()),
                                 'n_all_terminal':int(terminal.sum()),'mask_sha256':mask_hashes[population],
                                 'spearman':rho,'status':status})
        folder,receipt,data=select_run(cohort,'slingshot',0,inventory)
        if receipt['status']=='pass':
            assert np.array_equal(data['cell_id'],cells.index)
            lineages=json.loads((folder/'lineages.json').read_text())
            sources[str((folder/'native_outputs.npz').relative_to(ROOT))]=sha(folder/'native_outputs.npz')
            cluster_categories=lineages['cluster_categories']
            for j,lineage in enumerate(lineages['lineages']):
                assigned=data['evaluated'][:,j];end_cluster=cluster_categories[lineage[-1]]
                tasks=[task for task,labels in protocol['ends'].items()
                       if any(str(cells.loc[config['terminal_representatives'][label],'leiden'])==end_cluster for label in labels)]
                coverage.append({'cohort':cohort,'fit_status':'pass','lineage':j,'cluster_path':' -> '.join(cluster_categories[k] for k in lineage),
                                 'end_cluster':end_cluster,'matched_tasks':';'.join(tasks),
                                 'n_cohort':len(cells),'n_assigned':int(assigned.sum()),'fraction_assigned':float(assigned.mean()),
                                 'n_assigned_nonterminal':int((assigned&~terminal).sum()),
                                 'fraction_nonterminal_covered':float((assigned&~terminal).sum()/(~terminal).sum()),
                                 'n_unassigned':int((~assigned).sum())})
                for task in tasks:
                    target=cells['program_'+PROGRAM[task]].to_numpy()
                    values={**task_values[task],('Slingshot pseudotime',0):data['pseudotime'][:,j]}
                    # Compare one fixed seed-zero diffusion geometry on each deterministic
                    # lineage mask. Retain all three scRL fits; fate scores remain a separate family.
                    values={k:v for k,v in values.items() if k[0] in ['scRL value','PCA30 proximity','UMAP proximity','Slingshot pseudotime'] or k==('Diffusion proximity',0)}
                    for population,base in masks.items():
                        mask=assigned&base
                        mh=hashlib.sha256('\n'.join(cells.index[mask]).encode()).hexdigest()
                        for (method,seed),score in values.items():
                            rho,status=correlation(score,target,mask)
                            trajectory.append({'cohort':cohort,'task':task,'program':PROGRAM[task],'lineage':j,
                                               'end_cluster':end_cluster,'population':population,'method':method,'seed':seed,
                                               'n_cells':int(mask.sum()),'mask_sha256':mh,'spearman':rho,'status':status})
        else:
            coverage.append({'cohort':cohort,'fit_status':'fail','lineage':np.nan,
                             'n_cohort':len(cells),'n_assigned':np.nan,'fraction_assigned':np.nan,
                             'n_assigned_nonterminal':np.nan,'fraction_nonterminal_covered':np.nan,
                             'n_unassigned':np.nan,'matched_tasks':'','cluster_path':'','end_cluster':'',
                             'reason':receipt.get('error','unavailable curve output')})
        for method in ['palantir','cellrank','fateid','slingshot']:
            applicable=method=='slingshot' or config['probability_comparison']
            planned=1 if method=='slingshot' else (3 if applicable else 0)
            valid=sum(x['cohort']==cohort and x['method']==method and x['status']=='pass' and x['role']=='primary_attempt' for x in inventory)
            applicability.append({'cohort':cohort,'n_cells':len(cells),'method':method,'applicable':applicable,
                                  'planned_runs':planned,'valid_runs':valid,
                                  'reason':'lineage ordering and coverage' if method=='slingshot' else
                                  ('competing configured broad fates' if applicable else config['single_fate_gate'])})
        print(cohort,'analysis checked',flush=True)
    result=pd.DataFrame(rows);traj=pd.DataFrame(trajectory)
    original=pd.read_csv(GEOMETRY/'associations.csv')
    reference_preservation=[]
    for _,row in result.loc[result.population.isin(['all','starting'])|
                             (result.cohort.isin(['ucb','sleep'])&result.population.eq('shared_nonterminal'))].iterrows():
        old_name={'PCA30 proximity':'PCA30 distance','UMAP proximity':'UMAP distance'}.get(row.method)
        if row.method=='scRL value':old_name=f'scRL seed {int(row.seed)}'
        if old_name is None:continue
        old_population='nonterminal' if row.population=='shared_nonterminal' else row.population
        prior=original.loc[original.cohort.eq(row.cohort)&original.lineage.eq(row.task)&
                           original.population.eq(old_population)&original.method.eq(old_name)]
        assert len(prior)==1
        error=abs(float(prior.spearman.iloc[0])-row.spearman)
        assert int(prior.n_cells.iloc[0])==int(row.n_cells) and error<1e-12
        reference_preservation.append({'cohort':row.cohort,'task':row.task,'population':row.population,
                                       'method':row.method,'seed':int(row.seed),'absolute_error':error})
    result.to_csv(dest/'associations.csv',index=False);traj.to_csv(dest/'trajectory_associations.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'lineage_coverage.csv',index=False)
    pd.DataFrame(inventory).to_csv(dest/'run_inventory.csv',index=False)
    pd.DataFrame(applicability).to_csv(dest/'applicability.csv',index=False)
    summaries=[]
    for key,group in result.groupby(['cohort','task','population','method'],sort=False):
        defined=group.loc[group.status=='defined','spearman']
        summaries.append(dict(zip(['cohort','task','population','method'],key),
                              n_cells=int(group.n_cells.iloc[0]),n_runs=len(group),n_defined=len(defined),
                              median=float(defined.median()),minimum=float(defined.min()),maximum=float(defined.max()),
                              all_defined=bool(len(defined)==len(group)),
                              statuses=';'.join(sorted(set(group.status)))))
    summary=pd.DataFrame(summaries);summary.to_csv(dest/'summary.csv',index=False)
    differences=[]
    for key,group in summary.groupby(['cohort','task','population'],sort=False):
        ref=group.set_index('method')
        for method in ['scRL value','Palantir','CellRank 2','FateID Python']:
            if method not in ref.index:continue
            for reference in ['PCA30 proximity','UMAP proximity','Diffusion proximity']:
                differences.append(dict(zip(['cohort','task','population'],key),method=method,reference=reference,
                                        n_cells=int(ref.loc[method,'n_cells']),
                                        delta_rho=float(ref.loc[method,'median']-ref.loc[reference,'median'])))
    pd.DataFrame(differences).to_csv(dest/'reference_differences.csv',index=False)
    verification={'status':'pass','completed_utc':datetime.now(timezone.utc).isoformat(),
                  'registered_cohorts':4,'registered_external_methods':4,
                  'valid_probability_runs':sum(x['status']=='pass' and x['method']!='slingshot' and x['role']=='primary_attempt' for x in inventory),
                  'valid_slingshot_runs':sum(x['status']=='pass' and x['method']=='slingshot' and x['role']=='primary_attempt' for x in inventory),
                  'all_attempts':inventory,'distance_checks':distance_checks,
                  'historical_reference_checks':reference_preservation,
                  'association_rows':len(result),'trajectory_rows':len(traj),'sources':sources,
                  'analyzer_sha256':sha(__file__),'registration_sha256':sha(OUT/'registration.json'),
                  'evaluation':'Signed same-data program association; all, union-excluded nonterminal and frozen starting masks. No biological p-values or accuracy claim.',
                  'diffusion_reference':'15-nearest full-target Euclidean distance in min-max-scaled multiscale space. Three seeds on probability cohorts; seed zero on single-broad-target cohorts.',
                  'missing_values':'Undefined associations remain NA, with explicit status; not replaced by zero.',
                  'manuscript_updated':False}
    verification['outputs']={p.name:sha(p) for p in dest.glob('*.csv*')}
    write_json(dest/'verification.json',verification)
    print(json.dumps({k:verification[k] for k in ['status','valid_probability_runs','valid_slingshot_runs','association_rows','trajectory_rows']}))


if __name__=='__main__':main()
