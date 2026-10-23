#!/usr/bin/env python3
"""Review-informed component matching and small fixed partition/scaling controls."""
from pathlib import Path
import argparse,hashlib,json,os
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/hematopoietic-r4-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/hematopoietic-r4-mpl')
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[2];P=ROOT/'paper/scientific-reports-hematopoietic-readouts'
OLD=ROOT/'results/external_method_benchmark_20261021';OUT=ROOT/'results/targeted_review_20261021_r4'
INPUT=ROOT/'results/application_reanalysis_20260921'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def mh(ids,mask):return hashlib.sha256('\n'.join(np.asarray(ids)[mask]).encode()).hexdigest()


def rho(x,y,mask):
    a=np.asarray(x)[mask];b=np.asarray(y)[mask]
    if len(a)<3:return np.nan,'insufficient_cells'
    if not np.isfinite(a).all():return np.nan,'failed_fit_no_valid_score'
    if np.ptp(a)==0 or np.ptp(b)==0:return np.nan,'constant'
    return float(spearmanr(a,b).statistic),'defined'


def proximity(x,ends):
    x=np.asarray(x,dtype=np.float64)
    assert ends.sum()>=16
    d,_=NearestNeighbors(n_neighbors=16,n_jobs=1).fit(x[ends]).kneighbors(x)
    score=-d[:,:15].mean(1);score[ends]=-d[ends,1:16].mean(1)
    ii=np.linspace(0,len(x)-1,19,dtype=int);dd=np.sort(cdist(x[ii],x[ends]),axis=1)
    exact=np.asarray([-v[1:16].mean() if ends[k] else -v[:15].mean() for k,v in zip(ii,dd)])
    np.testing.assert_allclose(score[ii],exact,atol=1e-10,rtol=1e-10)
    return score


def expression(cohort):
    folder=P/'analysis_replay/inputs/frozen'/cohort;z=np.load(folder/'expression_support.npz',allow_pickle=False)
    divisor=z['library_counts_all_retained_genes'].astype(np.float32)/np.float32(10000)
    log=np.log1p(z['marker_counts'].astype(np.float32)/divisor[:,None])
    return z,pd.DataFrame(log,columns=z['marker_genes'])


def run_components():
    dest=OUT/'components';dest.mkdir(exist_ok=False)
    reg=json.loads((OLD/'registration.json').read_text());follow=json.loads((OUT/'registration.json').read_text())
    inventory=pd.read_csv(OLD/'analysis_v2/run_inventory.csv');rows=[];traj=[];audit=[]
    for cohort in ['dapp1','irradiation','ucb','sleep']:
        cells=pd.read_csv(INPUT/cohort/'cells.csv.gz',index_col=0);pack=np.load(INPUT/cohort/'input.npz',allow_pickle=False)
        assert np.array_equal(cells.index,pack['cell_id']);config=reg['cohorts'][cohort]
        terminal=cells.cell_type.isin(config['terminal_representatives']).to_numpy()
        masks={'all':np.ones(len(cells),bool),'shared_nonterminal':~terminal,'starting':cells.cell_type.isin(config['starting_labels']).to_numpy()}
        names=follow['component_evaluation']['fate_components'].get(cohort,{})
        if not names:
            names={label:('megakaryocyte' if label=='megakaryocyte' else label)
                   for label in config['terminal_representatives']}
        space_fits={};prob_fits={}
        if config['probability_comparison']:
            for method in ['palantir','cellrank','fateid']:
                for seed in [0,1,2]:
                    entry=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq(method)&inventory.seed.eq(seed)&inventory.role.eq('primary_attempt')].iloc[0]
                    path=OLD/entry.run/'outputs.npz';data=np.load(path,allow_pickle=False)
                    assert np.array_equal(data['cell_id'],cells.index)
                    if method=='palantir':space_fits[seed]=data['inference_space']
                    prob_fits[method,seed]=(data,entry.status)
        else:
            data=np.load(OLD/cohort/'diffusion/seed0/outputs.npz',allow_pickle=False);space_fits[0]=data['inference_space']
        if cohort=='irradiation':
            support,logs=expression(cohort);genes=pd.read_csv(P/'source_matrix/irradiation/genes.csv').gene.tolist()
            requested=follow['component_evaluation']['irradiation_dendritic_genes'];found=[g for g in requested if g in genes]
            assert len(found)>=2;positions=[genes.index(g) for g in found];counts=np.zeros((len(cells),len(found)),dtype=np.int64)
            manifest=json.loads((P/'source_matrix/irradiation/manifest.json').read_text())
            for block in manifest['blocks']:
                f=P/'source_matrix/irradiation'/block['path'];matrix=sparse.load_npz(f)
                a,b=block['row_start'],block['row_stop'];assert np.array_equal(np.asarray(matrix.sum(1)).ravel(),support['library_counts_all_retained_genes'][a:b])
                counts[a:b]=matrix[:,positions].toarray()
            log=np.log1p(counts.astype(np.float32)/(support['library_counts_all_retained_genes'].astype(np.float32)/np.float32(10000))[:,None])
            z=(log-log.mean(0))/np.where(log.std(0,ddof=0)>0,log.std(0,ddof=0),1)
            cells['program_dendritic']=z.mean(1)
            np.savez_compressed(dest/'irradiation_DC_expression.npz',cell_id=np.asarray(cells.index,dtype=str),genes=np.asarray(found),counts=counts,library_counts=support['library_counts_all_retained_genes'],log1p=log,score=cells.program_dendritic.to_numpy())
            for j,g in enumerate(found):audit.append({'cohort':cohort,'program':'dendritic','gene':g,'detected_cells':int((counts[:,j]>0).sum()),'detected_fraction':float((counts[:,j]>0).mean()),'requested':';'.join(requested),'missing':';'.join(g for g in requested if g not in found)})
        task_values={}
        for label,program in names.items():
            target=cells['program_'+program].to_numpy();ends=cells.cell_type.eq(label).to_numpy()
            values={('PCA30 component proximity',-1):proximity(pack['latent'],ends),
                    ('UMAP component proximity',-1):proximity(pack['embedding'],ends)}
            for seed,space in space_fits.items():values['Diffusion component proximity',seed]=proximity(space,ends)
            for (method,seed),(data,status) in prob_fits.items():
                key={'palantir':'Palantir, local variant','cellrank':'CellRank 2, DPT kernel','fateid':'FateID Python'}[method]
                values[key,seed]=data['probability'][:,list(data['component_names']).index(label)] if status=='pass' else np.full(len(cells),np.nan)
            frame=pd.DataFrame({'cell_id':cells.index,'cell_type':cells.cell_type.to_numpy(),'program':target,'endpoint':ends,**{'mask_'+k:v for k,v in masks.items()}})
            for (method,seed),value in values.items():
                frame[f'{method}|seed={seed}']=value
                for population,mask in masks.items():
                    value_rho,status=rho(value,target,mask)
                    rows.append({'cohort':cohort,'component':label,'program':program,'method':method,'seed':seed,'population':population,'n_cells':int(mask.sum()),'n_endpoint':int(ends.sum()),'mask_sha256':mh(cells.index,mask),'spearman':value_rho,'status':status})
            frame.to_csv(dest/f'{cohort}_{label}_evaluation.csv.gz',index=False,float_format='%.17g')
            task_values[label]=(target,values)
        selected=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq('slingshot')&inventory.role.eq('primary_attempt')]
        if selected.status.iloc[0]=='pass':
            folder=OLD/selected.run.iloc[0];native=np.load(folder/'native_outputs.npz',allow_pickle=False)
            lineage=json.loads((folder/'lineages.json').read_text());cats=lineage['cluster_categories']
            assert np.array_equal(native['cell_id'],cells.index)
            for j,path in enumerate(lineage['lineages']):
                end=cats[path[-1]]
                labels=[label for label,rep in config['terminal_representatives'].items() if str(cells.loc[rep,'leiden'])==str(end)]
                for label in labels:
                    target,values=task_values[label]
                    current={k:v for k,v in values.items() if k[0] in ['PCA30 component proximity','UMAP component proximity'] or k==('Diffusion component proximity',0)}
                    current['Slingshot pseudotime',0]=native['pseudotime'][:,j]
                    for population,base in masks.items():
                        mask=native['evaluated'][:,j]&base
                        for (method,seed),value in current.items():
                            value_rho,status=rho(value,target,mask)
                            traj.append({'cohort':cohort,'component':label,'program':names[label],'lineage':j,'end_cluster':end,'method':method,'seed':seed,'population':population,'n_cells':int(mask.sum()),'coverage':float(mask.sum()/base.sum()),'mask_sha256':mh(cells.index,mask),'spearman':value_rho,'status':status})
    pd.DataFrame(rows).to_csv(dest/'associations.csv',index=False)
    pd.DataFrame(traj).to_csv(dest/'trajectory_associations.csv',index=False)
    pd.DataFrame(audit).to_csv(dest/'program_audit.csv',index=False)
    group=['cohort','component','program','population','method']
    summary=pd.DataFrame(rows).groupby(group,sort=False).agg(n_cells=('n_cells','first'),n_runs=('seed','size'),n_defined=('spearman','count'),median=('spearman','median'),minimum=('spearman','min'),maximum=('spearman','max')).reset_index()
    summary.to_csv(dest/'summary.csv',index=False)
    save(dest/'receipt.json',{'status':'pass','association_rows':len(rows),'trajectory_rows':len(traj),'direct_distance_checks':19*len({(r['cohort'],r['component'],r['method'],r['seed']) for r in rows if 'proximity' in r['method']}),
        'primary_population':'same r3 union-excluded cells; component labels and expression programs matched','historical_fits_refitted':False,'driver_sha256':sha(__file__),'registration_sha256':sha(OUT/'registration.json'),
        'outputs':{f.name:sha(f) for f in dest.iterdir() if f.is_file()}})


def run_partitions():
    import scanpy as sc
    import anndata as ad
    dest=OUT/'partitions';dest.mkdir(exist_ok=False);rows=[];groups=[]
    for cohort,programs in [('dapp1',['neutrophil','cycling']),('sleep',['neutrophil'])]:
        cells=pd.read_csv(INPUT/cohort/'cells.csv.gz');pack=np.load(INPUT/cohort/'input.npz',allow_pickle=False)
        assert np.array_equal(cells.cell_id,pack['cell_id']);support,logs=expression(cohort)
        assert np.array_equal(support['cell_id'],cells.cell_id)
        a,b=('wt','ko') if cohort=='dapp1' else ('CT','SD');cond=cells.condition.to_numpy()
        protocol=json.loads((INPUT/cohort/'protocol.json').read_text())
        data=ad.AnnData(np.zeros((len(cells),1),dtype=np.float32));data.obsm['X_pca']=pack['latent']
        sc.pp.neighbors(data,n_neighbors=15,use_rep='X_pca',random_state=0)
        partitions={'historical':cells.leiden.astype(str).to_numpy()}
        for resolution in [.5,1.5]:
            sc.tl.leiden(data,resolution=resolution,random_state=0,flavor='igraph',n_iterations=2,directed=False,key_added='partition')
            partitions[f'fresh_{resolution}']=data.obs.partition.astype(str).to_numpy()
        fixture=pd.DataFrame({'cell_id':cells.cell_id,'condition':cond,**{'partition_'+k:v for k,v in partitions.items()}})
        for program in programs:
            genes=protocol['program_genes'][program];log=logs[genes].to_numpy().astype(float)
            reference=log[cond==a];mean=reference.mean(0);sd=reference.std(0,ddof=0);assert (sd>0).all()
            scores={'cohort_reference':cells['program_'+program].to_numpy(),'condition_A_reference':((log-mean)/sd).mean(1)}
            for scaling,score in scores.items():fixture[f'{program}|{scaling}']=score
            for part,labels in partitions.items():
                counts=pd.crosstab(labels,cond);eligible=counts.index[(counts[a]>=20)&(counts[b]>=20)]
                weights=counts.loc[eligible,[a,b]].sum(1).astype(float);weights/=weights.sum();covered=np.isin(labels,eligible)
                assert len(eligible)>0 and abs(weights.sum()-1)<1e-12
                for scaling,score in scores.items():
                    frame=pd.DataFrame({'group':labels,'condition':cond,'score':score})
                    means=frame.groupby(['group','condition']).score.mean().unstack();delta=means.loc[eligible,b]-means.loc[eligible,a]
                    fixed=float(np.dot(weights,delta));raw=float(score[cond==b].mean()-score[cond==a].mean())
                    raw_eligible=float(score[(cond==b)&covered].mean()-score[(cond==a)&covered].mean())
                    rows.append({'cohort':cohort,'program':program,'partition':part,'scaling':scaling,'n_groups':len(counts),'n_eligible':len(eligible),'n_cells':len(cells),'n_covered':int(covered.sum()),'coverage':float(covered.mean()),'coverage_A':float(covered[cond==a].mean()),'coverage_B':float(covered[cond==b].mean()),'raw_all':raw,'raw_eligible':raw_eligible,'fixed':fixed,'eligible_mask_sha256':mh(cells.cell_id,covered)})
                    for label in eligible:groups.append({'cohort':cohort,'program':program,'partition':part,'scaling':scaling,'group':label,'n_A':int(counts.loc[label,a]),'n_B':int(counts.loc[label,b]),'weight':float(weights.loc[label]),'mean_A':float(means.loc[label,a]),'mean_B':float(means.loc[label,b]),'delta':float(delta.loc[label])})
        fixture.to_csv(dest/f'{cohort}_evaluation.csv.gz',index=False,float_format='%.17g')
        sparse.save_npz(dest/f'{cohort}_connectivities.npz',data.obsp['connectivities'])
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False);pd.DataFrame(groups).to_csv(dest/'group_values.csv',index=False)
    save(dest/'receipt.json',{'status':'pass','summary_rows':len(rows),'group_rows':len(groups),'graph':'15-NN retained count-derived PCA30, seed0','Leiden':{'resolutions':[.5,1.5],'flavor':'igraph','iterations':2,'random_state':0,'directed':False},'coverage_rule':'at least 20 cells in each condition; report all and eligible raw contrasts separately','driver_sha256':sha(__file__),'outputs':{f.name:sha(f) for f in dest.iterdir() if f.is_file()}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['components','partitions']);a=parser.parse_args()
    (run_components if a.operation=='components' else run_partitions)()
