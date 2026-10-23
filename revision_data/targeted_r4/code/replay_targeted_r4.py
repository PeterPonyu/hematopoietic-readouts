#!/usr/bin/env python3
"""Recalculate follow-up summaries from supplied arrays without model fitting."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mask_sha(ids,m):return hashlib.sha256('\n'.join(np.asarray(ids)[m]).encode()).hexdigest()
def close(a,b,tol=1e-10):
    if np.isnan(a) and np.isnan(b):return
    assert np.isfinite(a) and np.isfinite(b) and abs(a-b)<=tol,(a,b)
def correlation(x,y,m):
    a=np.asarray(x)[m];b=np.asarray(y)[m]
    if not np.isfinite(a).all() or len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return float(np.corrcoef(pd.Series(a).rank().to_numpy(),pd.Series(b).rank().to_numpy())[0,1])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True)
    parser.add_argument('--article-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();r=args.packet;p=args.article_root;old=p/'revision_data/external_benchmark'
    registry=json.loads((old/'protocol/registration.json').read_text())
    count={'component_associations':0,'trajectory_associations':0,'endpoint_associations':0,'partition_summaries':0,
           'component_raw_probability_columns':0,'direct_distance_cells':0,'training_checkpoints':0,'evaluation_episodes':0}
    fixtures={};cells={};checks=[]
    for cohort in registry['cohorts']:
        c=pd.read_csv(p/'analysis_replay/inputs/frozen'/cohort/'cells.csv.gz',dtype={'cell_id':str});cells[cohort]=c
        config=registry['cohorts'][cohort];terminal=c.cell_type.isin(config['terminal_representatives']).to_numpy()
        for file in sorted((r/'components').glob(cohort+'_*_evaluation.csv.gz')):
            component=file.name[len(cohort)+1:].removesuffix('_evaluation.csv.gz')
            f=pd.read_csv(file,float_precision='round_trip',dtype={'cell_id':str});fixtures[cohort,component]=f
            assert np.array_equal(c.cell_id,f.cell_id) and np.array_equal(c.cell_type,f.cell_type)
            assert np.array_equal(~terminal,f.mask_shared_nonterminal.to_numpy())
            assert np.array_equal(f.endpoint.to_numpy(),c.cell_type.eq(component).to_numpy())
            assert np.array_equal(f.mask_starting.to_numpy(),c.cell_type.isin(config['starting_labels']).to_numpy())
            for column in [x for x in f if 'component proximity|seed=' in x]:
                method,seed=column.split('|seed=');seed=int(seed)
                if method.startswith('Diffusion'):
                    if config['probability_comparison']:
                        run=old/'runs'/cohort/'palantir'/('seed0_cutoff_checked' if cohort=='dapp1' and seed==0 else f'seed{seed}')
                    else:run=old/'runs'/cohort/'diffusion'/f'seed{seed}'
                    space=np.load(run/'outputs.npz',allow_pickle=False)['inference_space']
                else:
                    source=p/'analysis_replay/inputs/frozen'/cohort/'input.npz'
                    raw=np.load(source,allow_pickle=False);space=raw['latent' if method.startswith('PCA') else 'embedding']
                x=space.astype(float);ends=f.endpoint.to_numpy(bool)
                for index in np.linspace(0,len(f)-1,19,dtype=int):
                    d=np.sort(np.sqrt(np.square(x[ends]-x[index]).sum(1)))
                    value=-d[1:16].mean() if ends[index] else -d[:15].mean()
                    close(value,f[column].iloc[index],tol=1e-9);count['direct_distance_cells']+=1
            for method,label in [('palantir','Palantir, local variant'),('cellrank','CellRank 2, DPT kernel'),('fateid','FateID Python')]:
                if not config['probability_comparison']:continue
                for seed in [0,1,2]:
                    tag='seed0_cutoff_checked' if cohort=='dapp1' and method=='palantir' and seed==0 else f'seed{seed}_bounded' if cohort=='irradiation' and method=='fateid' else f'seed{seed}'
                    folder=old/'runs'/cohort/method/tag;receipt=json.loads((folder/'receipt.json').read_text())
                    if receipt['status']!='pass':assert f[f'{label}|seed={seed}'].isna().all();continue
                    raw=np.load(folder/'outputs.npz',allow_pickle=False)
                    value=raw['probability'][:,list(raw['component_names']).index(component)]
                    assert np.array_equal(value,f[f'{label}|seed={seed}'].to_numpy());count['component_raw_probability_columns']+=1
    associations=pd.read_csv(r/'components/associations.csv')
    for _,row in associations.iterrows():
        f=fixtures[row.cohort,row.component];m=f['mask_'+row.population].to_numpy(bool)
        assert m.sum()==row.n_cells and mask_sha(f.cell_id,m)==row.mask_sha256
        close(correlation(f[f'{row.method}|seed={int(row.seed)}'],f.program,m),row.spearman);count['component_associations']+=1
    trajectories=pd.read_csv(r/'components/trajectory_associations.csv');curves={}
    for _,row in trajectories.iterrows():
        f=fixtures[row.cohort,row.component]
        if row.cohort not in curves:curves[row.cohort]=np.load(old/'runs'/row.cohort/'slingshot/seed0_upstream_R/native_outputs.npz',allow_pickle=False)
        raw=curves[row.cohort];m=raw['evaluated'][:,int(row.lineage)]&f['mask_'+row.population].to_numpy(bool)
        assert m.sum()==row.n_cells and mask_sha(f.cell_id,m)==row.mask_sha256
        close(m.sum()/f['mask_'+row.population].sum(),row.coverage)
        x=raw['pseudotime'][:,int(row.lineage)] if row.method=='Slingshot pseudotime' else f[f'{row.method}|seed={int(row.seed)}']
        close(correlation(x,f.program,m),row.spearman);count['trajectory_associations']+=1
    dc=np.load(r/'components/irradiation_DC_expression.npz',allow_pickle=False)
    log=np.log1p(dc['counts'].astype(np.float32)/(dc['library_counts'].astype(np.float32)/np.float32(10000))[:,None])
    assert np.array_equal(log,dc['log1p']);z=(log-log.mean(0))/log.std(0,ddof=0)
    np.testing.assert_allclose(z.mean(1),dc['score'],atol=0,rtol=0)
    endpoint=pd.read_csv(r/'endpoint/evaluation.csv.gz',float_precision='round_trip');definition=np.load(r/'endpoint/definition_counts.npz',allow_pickle=False)
    strict=endpoint.original_endpoint.to_numpy()&(definition['counts']>0).all(1)
    assert np.array_equal(strict,endpoint.strict_endpoint) and strict.sum()==56
    split=pd.DataFrame(definition['split_log1p']);split=((split-split.mean())/split.std(ddof=0)).mean(1)
    np.testing.assert_allclose(split,endpoint.program_marker_split,atol=1e-14,rtol=0)
    for _,row in pd.read_csv(r/'endpoint/associations.csv').iterrows():
        m=endpoint.primary_mask.to_numpy(bool);x=endpoint[f'{row.definition}|{row.method}|seed={int(row.seed)}']
        assert m.sum()==row.n_cells and mask_sha(endpoint.cell_id,m)==row.mask_sha256
        close(correlation(x,endpoint['program_'+row.program],m),row.spearman);count['endpoint_associations']+=1
    original_input=np.load(p/'analysis_replay/inputs/frozen/dapp1/input.npz',allow_pickle=False)
    for definition_name in ['original','strict']:
        ends=endpoint[definition_name+'_endpoint'].to_numpy(bool)
        for method,seed in [('PCA30 proximity',-1),('UMAP proximity',-1)]+[('Diffusion proximity',s) for s in [0,1,2]]:
            if method.startswith('Diffusion'):
                tag='seed0_cutoff_checked' if seed==0 else f'seed{seed}'
                space=np.load(old/'runs/dapp1/palantir'/tag/'outputs.npz')['inference_space']
            else:space=original_input['latent' if method.startswith('PCA') else 'embedding']
            space=space.astype(float)
            for i in np.linspace(0,len(endpoint)-1,19,dtype=int):
                d=np.sort(np.sqrt(np.square(space[ends]-space[i]).sum(1)))
                value=-d[1:16].mean() if ends[i] else -d[:15].mean()
                close(value,endpoint[f'{definition_name}|{method}|seed={seed}'].iloc[i],tol=1e-9)
                count['direct_distance_cells']+=1
        for seed in [0,1,2]:
            tag='seed0_cutoff_checked' if seed==0 else f'seed{seed}'
            path=(old/'runs/dapp1/palantir'/tag/'outputs.npz') if definition_name=='original' else r/'endpoint'/f'seed{seed}/outputs.npz'
            z=np.load(path);np.testing.assert_array_equal(z['probability'][:,list(z['component_names']).index('erythroid')],endpoint[f'{definition_name}|Palantir, local variant|seed={seed}'])
    for _,row in pd.read_csv(r/'partitions/summary.csv').iterrows():
        f=pd.read_csv(r/'partitions'/f'{row.cohort}_evaluation.csv.gz',float_precision='round_trip')
        part=f['partition_'+row.partition].astype(str);a,b=('wt','ko') if row.cohort=='dapp1' else ('CT','SD')
        table=pd.crosstab(part,f.condition);eligible=table.index[(table[a]>=20)&(table[b]>=20)]
        m=part.isin(eligible);weight=table.loc[eligible,[a,b]].sum(1);weight=weight/weight.sum()
        score=f[f'{row.program}|{row.scaling}'];frame=pd.DataFrame({'group':part,'condition':f.condition,'score':score})
        means=frame.groupby(['group','condition']).score.mean().unstack()
        close(np.dot(weight,means.loc[eligible,b]-means.loc[eligible,a]),row.fixed)
        close(score[f.condition.eq(b)].mean()-score[f.condition.eq(a)].mean(),row.raw_all)
        close(score[f.condition.eq(b)&m].mean()-score[f.condition.eq(a)&m].mean(),row.raw_eligible)
        close(m.mean(),row.coverage);close(m[f.condition.eq(a)].mean(),row.coverage_A);close(m[f.condition.eq(b)].mean(),row.coverage_B)
        assert len(eligible)==row.n_eligible and mask_sha(f.cell_id,m)==row.eligible_mask_sha256
        count['partition_summaries']+=1
    # Materialize immutable diagnostic arrays once; NpzFile indexing otherwise
    # decompresses the same array at every simulated transition.
    with np.load(r/'training/environment_definition.npz',allow_pickle=False) as saved:environment=dict(saved)
    mapped=environment['mapped_grid_ids'];loc={int(g):i for i,g in enumerate(mapped)};boundary=set(environment['boundary'])-set(environment['starts']);N=int(environment['N'])
    steps_direction=np.asarray([(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)])
    frozen=cells['ucb'];target=frozen.program_megakaryocyte.to_numpy()
    embedding=np.load(p/'analysis_replay/inputs/matched/LiVAE/ucb/input.npz',allow_pickle=False)['embedding']
    for seed in [0,1,2]:
        folder=r/'training'/f'seed{seed}'
        with np.load(folder/'trajectories.npz',allow_pickle=False) as saved:raw=dict(saved)
        probs=np.load(folder/'policy_probabilities.npz',allow_pickle=False)['probabilities'];table=pd.read_csv(folder/'checkpoints.csv')
        ep=pd.read_csv(folder/'episodes.csv.gz');steps=pd.read_csv(folder/'steps.csv.gz')
        np.testing.assert_allclose(raw['returns'],ep['return'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(steps.groupby('episode').reward.sum().to_numpy(),raw['returns'],atol=1e-12,rtol=1e-12)
        d,index=NearestNeighbors(n_neighbors=15).fit(raw['grid']).kneighbors(embedding)
        sigma=np.maximum(d.std(1),np.finfo(float).eps);lw=-d*d/(2*sigma[:,None]**2);w=np.exp(lw-lw.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        for k,number in enumerate(raw['checkpoint_episodes']):
            row=table.loc[table.episode.eq(number)].iloc[0];v=raw['grid_values'][k]
            cell_value=np.sum(w*v[index],axis=1)
            for label,mask in [('all',np.ones(len(frozen),bool)),('nonterminal',~frozen.cell_type.eq('megakaryocyte').to_numpy()),('starting',frozen.cell_type.eq('primitive').to_numpy())]:close(correlation(cell_value,target,mask),row['rho_'+label])
            cdf=np.cumsum(probs[k].astype(float),axis=1);cdf/=cdf[:,-1,None]
            returns=[];squared_prediction=[];squared_TD=[]
            for j,start in enumerate(raw['evaluation_start_grids']):
                for rep in range(4):
                    current=int(start);visited={current};discount=1.;total=discounted=0.;time_step=0;done=trunc=False
                    while not done and not trunc:
                        i=loc[current];action=min(int(np.searchsorted(cdf[i],raw['evaluation_uniforms'][j,rep,time_step],side='right')),7)
                        move=steps_direction[action];nxt=current+int(move[0])+int(move[1])*N;reward=float(environment['rewards'][i,action]);time_step+=1
                        if reward==-1:reward=-2.;done=True;nxt=current
                        elif nxt in boundary:reward+=float(environment['pseudotime'][loc[nxt]]-environment['pseudotime'][i]);done=True
                        elif nxt in visited:reward=-1.;done=True
                        else:
                            visited.add(nxt)
                            if time_step>50:trunc=True
                        squared_TD.append((reward+.9*float(v[loc[nxt]])*(not done)-float(v[i]))**2)
                        total+=reward;discounted+=discount*reward;discount*=.9;current=nxt
                    comparison=discounted+(discount*float(v[loc[current]]) if trunc and not done else 0.)
                    returns.append(total);squared_prediction.append((float(v[loc[int(start)]])-comparison)**2);count['evaluation_episodes']+=1
            close(np.mean(returns),row.common_start_return_mean,tol=1e-9)
            close(np.sqrt(np.mean(squared_prediction)),row.common_start_prediction_RMSE,tol=1e-9)
            close(np.sqrt(np.mean(squared_TD)),row.common_start_TD_RMSE,tol=1e-9);count['training_checkpoints']+=1
    training_summary=pd.read_csv(r/'training/summary.csv')
    term=pd.read_csv(r/'training/termination_summary.csv')
    for _,row in training_summary.iterrows():
        folder=r/'training'/f'seed{int(row.seed)}';ep=pd.read_csv(folder/'episodes.csv.gz');steps=pd.read_csv(folder/'steps.csv.gz');cp=pd.read_csv(folder/'checkpoints.csv')
        last=ep.loc[ep.episode.between(row.episodes-499,row.episodes)];visited=steps.loc[steps.episode.le(row.episodes)];c=cp.loc[cp.episode.eq(row.episodes)].iloc[0]
        close(last['return'].mean(),row.last500_return);close(last.steps.mean(),row.last500_steps)
        assert len(set(visited.grid)|set(visited.next_grid))==row.visited_grids
        assert visited.grid.nunique()==row.visited_decision_grids and row.mapped_grids==1774
        for key,target in [('rho_all','rho_all'),('rho_nonterminal','rho_nonterminal'),('rho_starting','rho_starting'),('common_start_return','common_start_return_mean'),('prediction_RMSE','common_start_prediction_RMSE'),('TD_RMSE','common_start_TD_RMSE')]:close(row[key],c[target])
        group=term.loc[term.seed.eq(row.seed)&term.episodes.eq(row.episodes)]
        assert group.n_episodes.sum()==500
        for _,value in group.iterrows():assert last.termination_reason.eq(value.termination_reason).sum()==value.n_episodes
    extended=r/'slingshot_extended_dapp1';native=np.load(old/'runs/dapp1/slingshot/seed0_upstream_R/native_outputs.npz')
    shape=tuple(map(int,(extended/'shape.txt').read_text().split(',')))
    assert np.array_equal(np.asarray((extended/'cell_ids.txt').read_text().splitlines()),native['cell_id'])
    for name,key in [('pseudotime','pseudotime'),('weights','curve_weights')]:
        np.testing.assert_array_equal(np.fromfile(extended/(name+'.f64')).reshape(shape,order='F'),native[key])
    stopping=pd.read_csv(extended/'stopping.csv').iloc[0]
    assert stopping.iterations==5 and stopping.maxit==20 and stopping.converged
    count.update(training_budget_summaries=len(training_summary),Slingshot_extended_native_outputs_exact=True)
    result={'status':'pass',**count,'rank_implementation':'NumPy Pearson correlation of independently calculated pandas average ranks',
            'training_evaluation':'Pure NumPy rollout from native exported policies, saved environment and fixed random uniforms; source learning updates are not repeated.',
            'method_fitting_repeated':False,'script_sha256':sha(__file__)}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
