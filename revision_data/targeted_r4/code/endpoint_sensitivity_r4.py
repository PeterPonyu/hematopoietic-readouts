#!/usr/bin/env python3
"""Fixed stricter Dapp1 endpoint and nonoverlapping-program sensitivity."""
from pathlib import Path
import argparse,hashlib,json,os,time,traceback
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/hematopoietic-r4-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/hematopoietic-r4-mpl')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from component_and_partition_r4 import proximity,expression,mh,sha,save
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/targeted_review_20261021_r4'
OLD=ROOT/'results/external_method_benchmark_20261021';INPUT=ROOT/'results/application_reanalysis_20260921/dapp1'
DEST=OUT/'endpoint'


def prepare():
    DEST.mkdir(exist_ok=False)
    cells=pd.read_csv(INPUT/'cells.csv.gz',index_col=0);pack=np.load(INPUT/'input.npz',allow_pickle=False)
    source,logs=expression('dapp1');assert np.array_equal(source['cell_id'],cells.index)
    genes=list(source['marker_genes']);original=cells.cell_type.eq('erythroid').to_numpy()
    strict=original&(source['marker_counts'][:,genes.index('Klf1')]>0)&(source['marker_counts'][:,genes.index('Gata1')]>0)
    assert strict.sum()>=16
    positions=np.flatnonzero(strict);x=pack['latent'];distance=np.square(x[positions].astype(float)-x[positions].mean(0)).sum(1)
    index=positions[np.lexsort((cells.index.to_numpy()[positions],distance))[0]]
    config=json.loads((OLD/'registration.json').read_text())['cohorts']['dapp1'];previous=config['terminal_representatives']['erythroid']
    config['terminal_representatives']['erythroid']=str(cells.index[index])
    protocol=json.loads((INPUT/'protocol.json').read_text());split=[g for g in protocol['program_genes']['erythroid'] if g not in ['Klf1','Gata1']]
    a=logs[split];score=((a-a.mean())/a.std(ddof=0).replace(0,1)).mean(1).to_numpy()
    masks=~cells.cell_type.isin(config['terminal_representatives']).to_numpy()
    frame=pd.DataFrame({'cell_id':cells.index,'condition':cells.condition.to_numpy(),'cell_type':cells.cell_type.to_numpy(),
        'original_endpoint':original,'strict_endpoint':strict,'primary_mask':masks,
        'program_original':cells.program_erythroid.to_numpy(),'program_marker_split':score})
    frame.to_csv(DEST/'evaluation.csv.gz',index=False,float_format='%.17g')
    np.savez_compressed(DEST/'definition_counts.npz',cell_id=source['cell_id'],genes=np.asarray(['Klf1','Gata1']),
                        counts=source['marker_counts'][:,[genes.index('Klf1'),genes.index('Gata1')]],strict_endpoint=strict,
                        split_genes=np.asarray(split),split_log1p=logs[split].to_numpy())
    save(DEST/'configuration.json',{'cohort':'dapp1','n_original_endpoint':int(original.sum()),'n_strict_endpoint':int(strict.sum()),
        'strict_condition_counts':cells.loc[strict,'condition'].value_counts().to_dict(),'config':config,
        'original_representative':previous,'strict_representative':str(cells.index[index]),'representative_changed':previous!=str(cells.index[index]),
        'split_evaluation_genes':split,'primary_mask_sha256':mh(cells.index,masks),'primary_n':int(masks.sum()),
        'selection_rule':'Original erythroid membership and co-detected integer Klf1/Gata1; fixed before new inference and associations.',
        'source_cell_sha256':sha(INPUT/'cells.csv.gz'),'driver_sha256':sha(__file__)})
    print('strict endpoint',strict.sum(),'representative changed',previous!=str(cells.index[index]),flush=True)


def fit(seed):
    import palantir
    started=time.time();dest=DEST/f'seed{seed}';dest.mkdir(exist_ok=False)
    config=json.loads((DEST/'configuration.json').read_text())['config'];cells=pd.read_csv(INPUT/'cells.csv.gz',index_col=0)
    old=OLD/'dapp1/palantir'/('seed0_cutoff_checked' if seed==0 else f'seed{seed}')/'outputs.npz';data=np.load(old,allow_pickle=False)
    assert np.array_equal(data['cell_id'],cells.index)
    record={'seed':seed,'status':'in_progress','diffusion_input_sha256':sha(old),'configuration_sha256':sha(DEST/'configuration.json'),'driver_sha256':sha(__file__)}
    try:
        latent=pd.DataFrame(data['diffusion_space'],index=cells.index)
        terminal=list(config['terminal_representatives'].values())
        result=palantir.core.run_palantir(latent,config['root_cell'],terminal_states=terminal,knn=30,num_waypoints=1200,n_jobs=2,
            scale_components=True,use_early_cell_as_start=True,max_iterations=25,seed=seed)
        names=list(config['terminal_representatives']);prob=result.branch_probs.loc[cells.index,terminal].to_numpy()
        np.savez_compressed(dest/'outputs.npz',cell_id=data['cell_id'],component_names=np.asarray(names),probability=prob,
                            pseudotime=result.pseudotime.reindex(cells.index).to_numpy(),evaluated=np.ones(len(cells),bool))
        mass=prob.sum(1);assert np.isfinite(prob).all() and prob.min()>=-1e-8 and np.all(mass<=1+1e-6) and np.all(mass>=1-.01*len(names)-1e-6)
        record.update(status='pass',row_mass_minimum=float(mass.min()),row_mass_maximum=float(mass.max()),output_sha256=sha(dest/'outputs.npz'))
    except Exception as error:
        record.update(status='fail',error=repr(error));(dest/'traceback.txt').write_text(traceback.format_exc())
    record['elapsed_seconds']=time.time()-started;save(dest/'receipt.json',record);print(json.dumps(record),flush=True)


def analyze():
    frame=pd.read_csv(DEST/'evaluation.csv.gz',float_precision='round_trip');pack=np.load(INPUT/'input.npz')
    mask=frame.primary_mask.to_numpy(bool);rows=[]
    for definition in ['original','strict']:
        endpoint=frame[definition+'_endpoint'].to_numpy(bool)
        values={('PCA30 proximity',-1):proximity(pack['latent'],endpoint),('UMAP proximity',-1):proximity(pack['embedding'],endpoint)}
        for seed in [0,1,2]:
            old=OLD/'dapp1/palantir'/('seed0_cutoff_checked' if seed==0 else f'seed{seed}')/'outputs.npz';space=np.load(old,allow_pickle=False)
            values['Diffusion proximity',seed]=proximity(space['inference_space'],endpoint)
            if definition=='original':data=space;valid=True
            else:
                r=json.loads((DEST/f'seed{seed}/receipt.json').read_text());valid=r['status']=='pass'
                data=np.load(DEST/f'seed{seed}/outputs.npz',allow_pickle=False) if valid else None
            values['Palantir, local variant',seed]=data['probability'][:,list(data['component_names']).index('erythroid')] if valid else np.full(len(frame),np.nan)
        for (method,seed),value in values.items():
            frame[f'{definition}|{method}|seed={seed}']=value
            for program in ['original','marker_split']:
                y=frame['program_'+program].to_numpy()
                statistic=float(spearmanr(value[mask],y[mask]).statistic) if np.isfinite(value).all() else np.nan
                rows.append({'definition':definition,'program':program,'method':method,'seed':seed,'n_cells':int(mask.sum()),'n_endpoint':int(endpoint.sum()),'mask_sha256':mh(frame.cell_id,mask),'spearman':statistic,'status':'defined' if np.isfinite(statistic) else 'failed_fit_no_valid_score'})
    pd.DataFrame(rows).to_csv(DEST/'associations.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['definition','program','method'],sort=False).agg(n_cells=('n_cells','first'),n_endpoint=('n_endpoint','first'),n_runs=('seed','size'),n_defined=('spearman','count'),median=('spearman','median'),minimum=('spearman','min'),maximum=('spearman','max')).reset_index()
    summary.to_csv(DEST/'summary.csv',index=False);frame.to_csv(DEST/'evaluation.csv.gz',index=False,float_format='%.17g')
    save(DEST/'analysis_receipt.json',{'status':'pass','association_rows':len(rows),'method_scores_repaired':False,'program_split':'Remove Klf1/Gata1 from the original erythroid evaluation program; same original 8298 nonterminal cells.','driver_sha256':sha(__file__),'outputs':{f.name:sha(f) for f in DEST.iterdir() if f.is_file() and f.name!='analysis_receipt.json'}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','fit','analyze']);parser.add_argument('--seed',type=int);a=parser.parse_args()
    if a.operation=='prepare':prepare()
    elif a.operation=='fit':fit(a.seed)
    else:analyze()
