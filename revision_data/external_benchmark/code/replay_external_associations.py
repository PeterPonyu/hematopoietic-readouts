#!/usr/bin/env python3
"""Verify saved external-method summaries using average ranks, without refitting.

Requires only NumPy and pandas. `--folder` holds the evaluation tables and
verification.json; `--raw-root` holds the registered cohort/method run tree.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def rank_correlation(x,y,mask):
    x=np.asarray(x)[mask];y=np.asarray(y)[mask]
    if len(x)<3:return np.nan,'insufficient_cells'
    if not np.isfinite(x).all():return np.nan,'nonfinite_score'
    if np.ptp(x)==0:return np.nan,'constant_score'
    if np.ptp(y)==0:return np.nan,'constant_program'
    rx=pd.Series(x).rank(method='average').to_numpy()
    ry=pd.Series(y).rank(method='average').to_numpy()
    return float(np.corrcoef(rx,ry)[0,1]),'defined'


def close(actual,expected,tol=1e-12):
    if np.isnan(expected):assert np.isnan(actual)
    else:assert np.isfinite(actual) and abs(actual-expected)<=tol,(actual,expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--raw-root',type=Path,required=True)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();folder=args.folder
    manifest=json.loads((folder/'verification.json').read_text())
    assert manifest['status']=='pass'
    for name,expected in manifest['outputs'].items():assert digest(folder/name)==expected,name
    assert digest(args.registration)==manifest['registration_sha256']
    registration=json.loads(args.registration.read_text())
    associations=pd.read_csv(folder/'associations.csv')
    trajectory=pd.read_csv(folder/'trajectory_associations.csv')
    coverage=pd.read_csv(folder/'lineage_coverage.csv').fillna('')
    inventory=pd.read_csv(folder/'run_inventory.csv').fillna('')
    fixtures={};n_association=0;n_trajectory=0;mask_checks=0;raw_checks=0;direct_checks=0;max_error=0.
    for path in sorted(folder.glob('*_evaluation.csv.gz')):
        cohort,task=path.name.removesuffix('_evaluation.csv.gz').split('_',1)
        frame=pd.read_csv(path,float_precision='round_trip',dtype={'cell_id':str,'cell_type':str,'leiden':str})
        config=registration['cohorts'][cohort]
        assert len(frame)==config['n_cells'] and frame.cell_id.is_unique
        axis_hash=hashlib.sha256('\n'.join(frame.cell_id).encode()).hexdigest()
        assert axis_hash==config['cell_axis_sha256']
        terminal=frame.cell_type.isin(config['terminal_representatives']).to_numpy()
        assert np.array_equal(frame.mask_shared_nonterminal,~terminal)
        assert np.array_equal(frame.mask_starting,frame.cell_type.isin(config['starting_labels']))
        assert np.array_equal(frame.mask_endpoint,frame.cell_type.isin(config['tasks'][task]))
        assert frame.mask_all.all();mask_checks+=4
        fixtures[cohort,task]=frame
        columns=[name for name in frame if '|seed=' in name]
        for population in ['all','shared_nonterminal','starting']:
            mask=frame['mask_'+population].to_numpy(bool)
            mask_hash=hashlib.sha256('\n'.join(frame.loc[mask,'cell_id']).encode()).hexdigest()
            for column in columns:
                method,seed=column.split('|seed=');seed=int(seed)
                table=associations.loc[associations.cohort.eq(cohort)&associations.task.eq(task)&
                      associations.population.eq(population)&associations.method.eq(method)&associations.seed.eq(seed)]
                assert len(table)==1;row=table.iloc[0]
                assert int(row.n_cells)==int(mask.sum()) and row.mask_sha256==mask_hash
                rho,status=rank_correlation(frame[column],frame.program,mask)
                if row.status=='failed_fit_no_valid_score':assert not np.isfinite(frame[column]).any()
                else:assert status==row.status,(cohort,task,method,status,row.status)
                close(rho,row.spearman)
                if np.isfinite(rho):max_error=max(max_error,abs(rho-row.spearman))
                n_association+=1
        for method,label in [('palantir','Palantir'),('cellrank','CellRank 2'),('fateid','FateID Python')]:
            for seed in [0,1,2]:
                column=f'{label}|seed={seed}'
                if column not in frame:continue
                selected=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq(method)&
                        inventory.seed.eq(seed)&inventory.status.eq('pass')&inventory.role.eq('primary_attempt')]
                if not len(selected):
                    assert frame[column].isna().all();continue
                assert len(selected)==1;run=args.raw_root/selected.iloc[0].run
                receipt=json.loads((run/'receipt.json').read_text())
                assert receipt['outputs']['outputs.npz']==digest(run/'outputs.npz')
                raw=np.load(run/'outputs.npz',allow_pickle=False)
                assert np.array_equal(raw['cell_id'],frame.cell_id)
                prob=raw['probability'];assert prob.shape==(len(frame),len(raw['component_names']))
                assert np.isfinite(prob).all() and prob.min()>=-1e-8
                mass=prob.sum(axis=1)
                if method=='palantir':
                    assert mass.max()<=1+1e-6 and mass.min()>=1-.01*prob.shape[1]-1e-6
                else:
                    np.testing.assert_allclose(mass,1,atol=1e-6,rtol=0)
                    for j,component in enumerate(raw['component_names']):
                        np.testing.assert_allclose(prob[frame.cell_type.eq(component),j],1,atol=1e-8,rtol=0)
                components=[list(raw['component_names']).index(name) for name in config['tasks'][task]]
                score=prob[:,components].sum(axis=1)
                assert np.array_equal(score,frame[column].to_numpy())
                raw_checks+=1
        # Direct distances independently validate saved diffusion proximity.
        for column in [c for c in columns if c.startswith('Diffusion proximity|')]:
            seed=int(column.split('|seed=')[1])
            if config['probability_comparison']:
                run=args.raw_root/cohort/'palantir'/('seed0_cutoff_checked' if cohort=='dapp1' and seed==0 else f'seed{seed}')
            else:run=args.raw_root/cohort/'diffusion'/f'seed{seed}'
            raw=np.load(run/'outputs.npz',allow_pickle=False)
            assert np.array_equal(raw['cell_id'],frame.cell_id)
            x=raw['inference_space'];ends=frame.mask_endpoint.to_numpy(bool)
            for i in np.linspace(0,len(frame)-1,19,dtype=int):
                distances=np.sort(np.sqrt(np.square(x[ends].astype(float)-x[i].astype(float)).sum(axis=1)))
                value=-distances[1:16].mean() if ends[i] else -distances[:15].mean()
                close(value,frame[column].iloc[i],tol=1e-9);direct_checks+=1
    assert n_association==len(associations)
    native={}
    for cohort in registration['cohorts']:
        selected=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq('slingshot')&
                inventory.status.eq('pass')&inventory.role.eq('primary_attempt')]
        if not len(selected):
            failures=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq('slingshot')&
                     inventory.status.eq('fail')&inventory.role.eq('primary_attempt')]
            absent=coverage.loc[coverage.cohort.eq(cohort)]
            assert len(failures) and len(absent)==1 and absent.fit_status.iloc[0]=='fail'
            assert absent.n_assigned.iloc[0]=='' and absent.fraction_nonterminal_covered.iloc[0]==''
            assert not trajectory.cohort.eq(cohort).any()
            continue
        assert len(selected)==1;run=args.raw_root/selected.iloc[0].run
        receipt=json.loads((run/'native_export_receipt.json').read_text())
        assert receipt['status']=='pass' and receipt['native_output_sha256']==digest(run/'native_outputs.npz')
        assert receipt['source_fit_sha256']==digest(run/'fit.rds')
        native[cohort]=dict(np.load(run/'native_outputs.npz',allow_pickle=False))
        raw=native[cohort];assigned=np.isfinite(raw['pseudotime'])&(raw['curve_weights']>0)
        assert np.array_equal(assigned,raw['evaluated'])
        frame=next(v for (c,t),v in fixtures.items() if c==cohort)
        assert np.array_equal(raw['cell_id'],frame.cell_id)
        for _,row in coverage.loc[coverage.cohort.eq(cohort)].iterrows():
            mask=assigned[:,int(row.lineage)]
            assert int(mask.sum())==int(row.n_assigned) and int((~mask).sum())==int(row.n_unassigned)
            sub=mask&frame.mask_shared_nonterminal.to_numpy(bool)
            assert int(sub.sum())==int(row.n_assigned_nonterminal)
            close(float(sub.sum()/frame.mask_shared_nonterminal.sum()),float(row.fraction_nonterminal_covered))
    for _,row in trajectory.iterrows():
        frame=fixtures[row.cohort,row.task];raw=native[row.cohort]
        mask=raw['evaluated'][:,int(row.lineage)]&frame['mask_'+row.population].to_numpy(bool)
        assert int(mask.sum())==int(row.n_cells)
        assert hashlib.sha256('\n'.join(frame.loc[mask,'cell_id']).encode()).hexdigest()==row.mask_sha256
        score=raw['pseudotime'][:,int(row.lineage)] if row.method=='Slingshot pseudotime' else frame[f'{row.method}|seed={int(row.seed)}'].to_numpy()
        rho,status=rank_correlation(score,frame.program,mask)
        assert status==row.status;close(rho,row.spearman)
        if np.isfinite(rho):max_error=max(max_error,abs(rho-row.spearman))
        n_trajectory+=1
    assert n_trajectory==len(trajectory)
    result={'status':'pass','association_rows_verified':n_association,'trajectory_rows_verified':n_trajectory,
            'raw_probability_aggregations_verified':raw_checks,'mask_definition_checks':mask_checks,
            'independent_direct_distance_cells':direct_checks,'maximum_rank_correlation_error':max_error,
            'rank_method':'pandas average ranks followed by NumPy Pearson correlation; comparison against the SciPy-based summary',
            'method_fitting_repeated':False,'script_sha256':digest(__file__)}
    target=args.output or folder/'replay_verification.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
