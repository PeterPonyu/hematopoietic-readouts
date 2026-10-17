#!/usr/bin/env python3
"""Portable saved-analysis replay. Explicit cohort selection; never fits models."""
import argparse, importlib.metadata, itertools, json, os, platform, sys, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
import pandas as pd
from scipy import sparse
from replay_common import save_json, sha256

APPLICATIONS=['ucb','dapp1','sleep']
ALLOWED=APPLICATIONS+['marrow','dapp1_spliced']
def installed_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return None
FROZEN=Path('results/application_reanalysis_20260921')
MATCHED=Path('results/recovered_integration_20261001/native_analysis')
CHECKS=[]


def check_arrays(name,actual,expected,policy,exact=False):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape:raise AssertionError(f'{name}: shapes {a.shape} != {b.shape}')
    numeric=np.issubdtype(a.dtype,np.number) and np.issubdtype(b.dtype,np.number)
    record={'comparison':name,'shape':list(a.shape),'exact_required':bool(exact or not numeric)}
    if exact or not numeric:
        equal=np.array_equal(a,b,equal_nan=True) if numeric else np.array_equal(a,b)
        record.update(passed=bool(equal));CHECKS.append(record)
        if not equal:raise AssertionError(f'{name}: exact equality failed')
    else:
        nan=np.array_equal(np.isnan(a),np.isnan(b)); finite=np.isfinite(a)&np.isfinite(b)
        record.update(passed=bool(nan and np.allclose(a,b,equal_nan=True,**policy)),max_abs_error=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.0,rtol=policy['rtol'],atol=policy['atol'])
        CHECKS.append(record)
        if not record['passed']:raise AssertionError(f'{name}: numerical policy failed, max_abs_error={record["max_abs_error"]}')


def compare_csv(a,b,policy,cohorts=None):
    x=pd.read_csv(a);y=pd.read_csv(b)
    if cohorts and 'cohort' in y:y=y[y.cohort.isin(cohorts)].reset_index(drop=True)
    if list(x.columns)!=list(y.columns):raise AssertionError(f'{a.name}: column mismatch')
    # The historical calculations define deterministic row ordering. Do not sort
    # to hide missing/duplicated/reordered records.
    for col in y:
        exact=not pd.api.types.is_numeric_dtype(y[col]) or pd.api.types.is_integer_dtype(y[col])
        check_arrays(f'{b.relative_to(b.parents[3])}:{col}',x[col].to_numpy(),y[col].to_numpy(),policy,exact)


def expression(folder,policy):
    pack=np.load(folder/'expression_support.npz',allow_pickle=False);cells=pd.read_csv(folder/'cells.csv.gz');exp=pd.read_csv(folder/'marker_log1p.csv.gz',index_col=0);meta=json.loads((folder/'expression_support.json').read_text());protocol=json.loads((folder/'protocol.json').read_text())
    check_arrays(f'{folder.name}: expression cell IDs',pack['cell_id'],cells.cell_id.astype(str),policy,True)
    check_arrays(f'{folder.name}: expression table IDs',pack['cell_id'],exp.index.astype(str),policy,True)
    check_arrays(f'{folder.name}: full-axis marker positions',pack['full_retained_genes'][pack['marker_gene_indices']],pack['marker_genes'],policy,True)
    assert len(pack['full_retained_genes'])==meta['retained_gene_count']==protocol['n_genes']
    counts=pack['marker_counts'];totals=pack['library_counts_all_retained_genes'];assert np.issubdtype(counts.dtype,np.integer) and np.issubdtype(totals.dtype,np.integer);assert (totals>0).all() and (counts.sum(axis=1)<=totals).all()
    # Match Scanpy sparse normalization: float32 division by library/10000.
    factor=totals.astype(np.float32)/np.float32(10000)
    normalized=counts.astype(np.float32)/factor[:,None]
    measured=np.log1p(normalized)
    check_arrays(f'{folder.name}: marker log1p',measured,exp.to_numpy(),policy)
    expression_frame=pd.DataFrame(measured,columns=pack['marker_genes'])
    for name,genes in protocol['program_genes'].items():
        if len(genes)<2:continue
        e=expression_frame[genes];scores=((e-e.mean())/e.std(ddof=0).replace(0,1)).mean(axis=1)
        check_arrays(f'{folder.name}: program_{name}',scores,cells['program_'+name],policy)
    return {'cohort':folder.name,'cells':len(cells),'full_retained_genes':len(pack['full_retained_genes']),'marker_genes':len(pack['marker_genes']),'full_nonmarker_matrix_included':False,'denominator':'sum over every retained gene; not over HVGs/markers'}


def observational_extras(c,folder,dest):
    from frozen_calculations import ordered_conditions
    cells=pd.read_csv(folder/'cells.csv.gz',index_col=0);conditions=ordered_conditions('dapp1' if c=='dapp1_spliced' else c,cells);programs=[n for n in cells if n.startswith('program_') and n!='program_annotation'];lineage=[n for n in programs if n not in ['program_interferon','program_inflammatory','program_cycling']]
    # prepare.py calculated these cluster means before CSV export, in float32.
    cells[lineage]=cells[lineage].astype(np.float32)
    cells.leiden=cells.leiden.astype(str)
    order=json.loads((folder/'expression_support.json').read_text())['frozen_leiden_category_order']
    cells.groupby('leiden',observed=True)[lineage].mean().reindex(order).to_csv(dest/'cluster_programs.csv')
    if c not in APPLICATIONS:return
    groups={'ucb':{'Prim-like':['primitive'],'Mk-like':['megakaryocyte'],'Ery-like':['erythroid']},'dapp1':{'cluster 6 (granulocytic)':['6']},'sleep':{'Prim-like':['primitive'],'Neu-like':['neutrophil'],'Mono-like':['monocyte']}}[c]
    col='leiden' if c=='dapp1' else 'cell_type'
    frame=pd.DataFrame(index=conditions)
    for name,labels in groups.items():frame[name]=[cells.loc[cells.condition==cond,col].astype(str).isin(labels).mean() for cond in conditions]
    if c=='dapp1':frame['other clusters']=1-frame['cluster 6 (granulocytic)']
    frame.to_csv(dest/'thesis_composition.csv')


def sensitivity(folder,dest):
    cells=pd.read_csv(folder/'cells.csv.gz',index_col=0);cells.leiden=cells.leiden.astype(str);programs=[p for p in cells if p.startswith('program_') and p!='program_annotation'];counts=pd.crosstab(cells.condition,cells.leiden);within=cells.groupby(['condition','leiden'],observed=True)[programs].mean();means=cells.groupby('condition')[programs].mean();eligible=counts.columns[(counts.loc['wt']>=20)&(counts.loc['ko']>=20)];weights=counts[eligible].sum();weights/=weights.sum();rows=[]
    for p in programs:rows.append({'program':p,'contrast':'ko-wt','unadjusted_delta':means.loc['ko',p]-means.loc['wt',p],'composition_standardized_delta':((within.loc['ko',p].reindex(eligible)-within.loc['wt',p].reindex(eligible))*weights).sum(),'matched_groups':len(eligible),'fraction_cells_covered':counts[eligible].values.sum()/len(cells),'scale':'mean gene-wise standardized log1p(counts per 1e4)'})
    pd.DataFrame(rows).to_csv(dest/'composition_standardized_effects.csv',index=False)


def checkpoint(root,out,cohorts,policy):
    import torch
    sys.path.insert(0,str(root/'vendor'));from livae.module import VAE
    torch.set_num_threads(2);records=[]
    for c in cohorts:
        folder=root/'inputs/encoders'/c;axis=np.load(folder/'frozen_axis.npz',allow_pickle=False);p=np.load(folder/'frozen_counts.npz',allow_pickle=False);x=sparse.csr_matrix((p['data'],p['indices'],p['indptr']),shape=p['shape']);saved=np.load(folder/'input.npz',allow_pickle=False)
        check_arrays(f'{c}: checkpoint cell IDs',axis['cell_id'],saved['cell_id'],policy,True)
        check_arrays(f'{c}: complete disjoint optimization split',np.sort(np.r_[axis['training_rows'],axis['validation_rows']]),np.arange(len(axis['cell_id'])),policy,True)
        original_receipt=json.loads((folder/'receipt.json').read_text())
        assert sha256(folder/'representation.pt')==original_receipt['checkpoint_sha256']
        assert sha256(folder/'frozen_axis.npz')==original_receipt['frozen_axis_sha256']
        assert sha256(folder/'input.npz')==original_receipt['new_input_sha256']
        state=torch.load(folder/'representation.pt',map_location='cpu',weights_only=True);model=VAE(state['state_dim'],state['hidden_dim'],state['latent_dim'],state['i_dim']).float();model.load_state_dict(state['model_state_dict'],strict=True);model.eval();latent=[];bottleneck=[]
        with torch.no_grad():
            for start in range(0,x.shape[0],1024):
                counts=torch.from_numpy(x[start:start+1024].toarray().astype(np.float32));qm=model.encoder.nn(torch.log1p(counts/counts.sum(1,keepdim=True).clamp_min(1)*1e4))[:,:state['latent_dim']];latent.append(qm.numpy());bottleneck.append(model.latent_encoder(qm).numpy())
        latent=np.concatenate(latent);bottleneck=np.concatenate(bottleneck)
        check_arrays(f'{c}: deterministic encoder posterior mean',latent,saved['latent'],policy)
        check_arrays(f'{c}: deterministic encoder bottleneck',bottleneck,saved['bottleneck'],policy)
        np.savez_compressed(out/f'{c}_checkpoint_export.npz',cell_id=axis['cell_id'],latent=latent,bottleneck=bottleneck)
        records.append({'cohort':c,'cells':x.shape[0],'features':x.shape[1],'checkpoint_epoch':state['epoch'],'checkpoint_sha256':sha256(folder/'representation.pt'),'scope':'posterior-mean and bottleneck export only; saved UMAP reused, no projection refit, no downstream checkpoint inference or training'})
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--cohorts',choices=ALLOWED,nargs='+',required=True);p.add_argument('--checkpoints',action='store_true');p.add_argument('--deny-read-root',type=Path,action='append',default=[]);args=p.parse_args();root=args.input_root.resolve();out=args.output_root.resolve()
    if root!=HERE:raise ValueError('Input root must be the directory containing this replay.py; code and inputs are a single bound bundle')
    if out.exists():raise FileExistsError('Use a new output directory; historical or previous replay outputs are never overwritten')
    if len(set(args.cohorts))!=len(args.cohorts):raise ValueError('Duplicate cohort')
    out.mkdir(parents=True)
    receipt={'status':'in_progress','cohorts':args.cohorts}
    try:
        policy=json.loads((root/'tolerances.json').read_text());summary_policy={k:policy['saved_summary'][k] for k in ['rtol','atol']};expression_policy={k:policy['expression_float32'][k] for k in ['rtol','atol']};checkpoint_policy={k:policy['encoder_float32'][k] for k in ['rtol','atol']}
        denied=[str(p.resolve()) for p in args.deny_read_root]
        def audit(event,values):
            if event=='open' and isinstance(values[0],(str,bytes)):
                path=os.path.abspath(os.fsdecode(values[0]))
                if any(path==base or path.startswith(base+os.sep) for base in denied):raise PermissionError('Isolation audit denied source access: '+path)
        if denied:sys.addaudithook(audit)
        receipt={'status':'running','cohorts':args.cohorts,'policies':policy,'policy_sha256':sha256(root/'tolerances.json'),'runtime':{'python':platform.python_version(),'platform':platform.platform(),'packages':{name:installed_version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','torch']}},'isolation':{'cwd':str(Path.cwd()),'input_root':str(root),'denied_read_roots':denied,'guard':'Python audit hook on file opens; not an OS namespace; no repository/home-data dependency used'},'no_training':True,'raw_preparation_reproduced':False,'new_biological_validation':False}
        manifest=json.loads((root/'support_manifest.json').read_text());verified=0
        for entry in manifest['files']:
            f=root/entry['path'];assert f.is_file() and sha256(f)==entry['sha256'],entry['path'];verified+=1
        receipt['manifest_files_verified']=verified;receipt['support_manifest_sha256']=sha256(root/'support_manifest.json');receipt['code_sha256']={str(f.relative_to(root)):sha256(f) for f in sorted(root.rglob('*.py'))};receipt['expression']=[]
        from frozen_calculations import analyze as frozen
        from biological_calculations import analyze as biological
        for c in args.cohorts:
            folder=root/'inputs/frozen'/c;dest=out/'source_data'/FROZEN/c;dest.mkdir(parents=True)
            receipt['expression'].append(expression(folder,expression_policy))
            if c=='dapp1_spliced':sensitivity(folder,dest)
            else:
                frozen(c,folder,dest,c!='marrow')
                if c!='marrow':biological(c,folder,dest)
            observational_extras(c,folder,dest)
            print('replayed frozen',c,flush=True)
        apps=[c for c in args.cohorts if c in APPLICATIONS]
        if apps:
            for c in apps:
                original=pd.read_csv(root/'inputs/frozen'/c/'cells.csv.gz')
                for arm in ['PCA','LiVAE']:
                    paired=pd.read_csv(root/'inputs/matched'/arm/c/'cells.csv.gz')
                    for column in ['cell_id','condition','leiden','cell_type']:
                        check_arrays(f'{c}/{arm}: matched frozen {column}',paired[column].astype(str),original[column].astype(str),summary_policy,True)
                    for column in [n for n in original if n.startswith('program_') and n!='program_annotation']:
                        check_arrays(f'{c}/{arm}: matched frozen {column}',paired[column],original[column],summary_policy)
            from matched_calculations import analyze as matched
            matched(root/'inputs/matched',root/'inputs/encoders',out/'source_data'/MATCHED,apps)
        geom=[c for c in args.cohorts if c!='dapp1_spliced']
        if geom:
            from geometry_calculations import analyze as geometry
            geometry(root/'inputs/frozen',out/'source_data'/FROZEN/'geometry_reference',geom)
        compared=[]
        for expected in sorted((root/'expected').rglob('*.csv')):
            relative=expected.relative_to(root/'expected');actual=out/'source_data'/relative
            needed=(relative.parts[1]=='recovered_integration_20261001' and bool(apps)) or relative.parts[2] in args.cohorts or (relative.parts[2]=='geometry_reference' and bool(geom))
            if needed:
                if not actual.is_file():raise AssertionError('Missing required regenerated table: '+str(relative))
                compare_csv(actual,expected,summary_policy,args.cohorts);compared.append(str(relative))
        receipt['summary_tables_compared']=compared
        if args.checkpoints:receipt['checkpoints']=checkpoint(root,out,apps,checkpoint_policy)
        receipt['status']='passed';receipt['tiers']={'saved_per_cell_to_summaries':'passed','expression_count_support_to_programs':'passed','encoder_checkpoint_export':'passed' if args.checkpoints and apps else 'not_executed','raw_count_to_model_retraining':'not_executed','downstream_checkpoint_inference':'not_executed','public_download_access':'not_executed','figure_generation':'separate article build step'}
    except Exception as exc:
        receipt['status']='failed';receipt['failure']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};raise
    finally:
        receipt['output_files']=[{'path':str(f.relative_to(out)),'sha256':sha256(f),'bytes':f.stat().st_size} for f in sorted(out.rglob('*')) if f.is_file() and f.name!='replay_receipt.json']
        receipt['comparisons']=CHECKS;receipt['n_comparisons']=len(CHECKS);save_json(out/'replay_receipt.json',receipt)
if __name__=='__main__':main()
