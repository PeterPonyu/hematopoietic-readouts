#!/usr/bin/env python3
"""Evaluate paired own-model/PCA downstream outputs without inventing gains."""
from __future__ import annotations
import argparse,datetime,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata,spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import adjusted_rand_score
PROGRAM={'erythroid':'erythroid','myeloid':'neutrophil','megakaryocyte':'megakaryocyte'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def rho(a,b):return float(spearmanr(a,b).statistic) if np.std(a)>0 and np.std(b)>0 else None

def analyze(D,N,dest,C):
 ARMS=["PCA","LiVAE"];dest.mkdir(parents=True,exist_ok=True);rows=[];stab=[];state=[];branches=[];receipts=[];stats=[];finitechecks=[]
 for cohort in C:
  for arm in ARMS:
   folder=D/arm/cohort;cells=pd.read_csv(folder/'cells.csv.gz');pack=np.load(folder/'input.npz',allow_pickle=False);protocol=json.loads((folder/'protocol.json').read_text());input_hash=sha(folder/'input.npz')
   assert np.array_equal(cells.cell_id.astype(str),pack['cell_id'])
   runs={};foci={}
   for f in sorted(folder.glob('*/receipt.json')):
    r=json.loads(f.read_text());assert r['input_sha256']==input_hash and r['output_sha256']==sha(f.parent/'outputs.npz') and r['n_cells']==len(cells)
    rec=dict(r);rec['path']=str(f);rec['sha256']=sha(f);receipts.append(rec)
    out=np.load(f.parent/'outputs.npz',allow_pickle=False)
    if r['method']=='scRL':runs[(r['seed'],r['lineage'],r['reward'])]=dict(out)
    else:foci[r['seed']]=dict(out)
   expected=3+9*len(protocol['ends']);assert len(runs)+len(foci)==expected,(cohort,arm,len(runs),len(foci),expected)
   for lineage,labels in protocol['ends'].items():
    end=cells.cell_type.isin(labels).to_numpy();start=cells.cell_type.isin(protocol['starts']).to_numpy();target=cells['program_'+PROGRAM[lineage]].to_numpy(float)
    assert end.sum()>=15 and start.any()
    x=pack['embedding'].astype(float);k=15;model=NearestNeighbors(n_neighbors=k+1).fit(x[end]);dist,_=model.kneighbors(x)
    geometry=-dist[:,:k].mean(1);geometry[end]=-dist[end,1:k+1].mean(1)
    ii=np.linspace(0,len(x)-1,19,dtype=int);direct=np.sort(cdist(x[ii],x[end]),axis=1);expected=np.array([-ds[1:k+1].mean() if end[i] else -ds[:k].mean() for i,ds in zip(ii,direct)])
    np.testing.assert_allclose(expected,geometry[ii],rtol=1e-10,atol=1e-10)
    masks={'all':np.ones(len(cells),bool),'nonterminal':~end,'starting':start}
    values={};percentiles={}
    for seed in [0,1,2]:
     for reward in ['decision','neutral','contribution']:
      out=runs[(seed,lineage,reward)];v=out['value'];pt=out['pseudotime'];assert np.isfinite(v).all() and len(v)==len(cells)
      per=(rankdata(v)-.5)/len(v);values[(seed,reward)]=v
      if reward=='neutral':percentiles[seed]=per
      for pop,mask in masks.items():rows.append({'cohort':cohort,'arm':arm,'lineage':lineage,'program':PROGRAM[lineage],'population':pop,'seed':seed,'reward':reward,'n_cells':int(mask.sum()),'n_endpoint':int(end.sum()),'value_program_rho':rho(v[mask],target[mask]),'geometry_program_rho':rho(geometry[mask],target[mask]),'increment_over_geometry':(rho(v[mask],target[mask])-rho(geometry[mask],target[mask])) if rho(v[mask],target[mask]) is not None and rho(geometry[mask],target[mask]) is not None else None})
      for cond in protocol['conditions']:
       for pop,mask in [('all',cells.condition.eq(cond).to_numpy()),('starting',cells.condition.eq(cond).to_numpy()&start)]:
        if mask.any():state.append({'cohort':cohort,'arm':arm,'lineage':lineage,'reward':reward,'seed':seed,'condition':cond,'population':pop,'n_cells':int(mask.sum()),'value_percentile_median':float(np.median(per[mask]))})
     for sa,sb in itertools.combinations([0,1,2],2):
      if seed==0:
       for reward in ['decision','neutral','contribution']:stab.append({'cohort':cohort,'arm':arm,'method':'scRL','lineage':lineage,'reward':reward,'seed_a':sa,'seed_b':sb,'metric':'value_Spearman','value':rho(runs[(sa,lineage,reward)]['value'],runs[(sb,lineage,reward)]['value'])})
    per=np.vstack([percentiles[s] for s in [0,1,2]])
    per_geom=(rankdata(geometry)-.5)/len(geometry)
    np.savez_compressed(dest/f'{cohort}_{arm}_{lineage}_readouts.npz',cell_id=pack['cell_id'],embedding=x,program=target,geometry_percentile=per_geom,value_percentile_median=np.median(per,axis=0),value_percentile_range=per.max(0)-per.min(0),starting=start,endpoint=end)
   for s in [0,1,2]:
    f=foci[s];raw=f['raw_weights'];norm=f['normalized_weights'];assert len(raw)==len(cells) and (raw>=0).all() and np.isfinite(raw).all() and np.allclose(norm.sum(1),1)
    programs=[col for col in cells if col.startswith('program_') and col!='program_annotation'];assign=f['assignment'];nbranch=norm.shape[1]
    stats.append({'cohort':cohort,'arm':arm,'seed':s,'branches':nbranch,'median_entropy':float(np.median(f['entropy'])),'min_raw_row_sum':float(raw.sum(1).min()),'max_raw_row_sum':float(raw.sum(1).max())})
    for b in range(nbranch):
     for cond in protocol['conditions']:
      mask=cells.condition.eq(cond).to_numpy();branches.append({'cohort':cohort,'arm':arm,'seed':s,'branch':b,'condition':cond,'n_cells':int(mask.sum()),'mean_normalized_weight':float(norm[mask,b].mean())})
   for a,b in itertools.combinations([0,1,2],2):stab.append({'cohort':cohort,'arm':arm,'method':'scFocus','lineage':'all','reward':'unsupervised','seed_a':a,'seed_b':b,'metric':'assignment_ARI','value':float(adjusted_rand_score(foci[a]['assignment'],foci[b]['assignment']))})
   finitechecks.append({'cohort':cohort,'arm':arm,'n_cells':len(cells),'runs':len(runs)+len(foci),'all_finite':True,'cell_axis_equal':True,'input_sha256':input_hash,'direct_geometry_checks':19*len(protocol['ends'])})
 data=pd.DataFrame(rows);data.to_csv(dest/'associations_all_rewards.csv',index=False);pd.DataFrame(stab).to_csv(dest/'seed_stability.csv',index=False);pd.DataFrame(state).to_csv(dest/'value_by_condition.csv',index=False);pd.DataFrame(branches).to_csv(dest/'focus_condition_weights.csv',index=False);pd.DataFrame(stats).to_csv(dest/'focus_summary.csv',index=False)
 neutral=data[data.reward=='neutral'];sums=[]
 for keys,g in neutral.groupby(['cohort','arm','lineage','population'],sort=False):
  sums.append(dict(zip(['cohort','arm','lineage','population'],keys),n_cells=int(g.n_cells.iloc[0]),value_min=float(g.value_program_rho.min()),value_median=float(g.value_program_rho.median()),value_max=float(g.value_program_rho.max()),geometry_rho=float(g.geometry_program_rho.iloc[0]),increment_median=float(g.increment_over_geometry.median())))
 summary=pd.DataFrame(sums);summary.to_csv(dest/'neutral_summary.csv',index=False)
 paired=[]
 for keys,g in summary.groupby(['cohort','lineage','population'],sort=False):
  g=g.set_index('arm');paired.append(dict(zip(['cohort','lineage','population'],keys),LiVAE_value_rho=float(g.loc['LiVAE','value_median']),PCA_value_rho=float(g.loc['PCA','value_median']),LiVAE_minus_PCA=float(g.loc['LiVAE','value_median']-g.loc['PCA','value_median']),LiVAE_increment_over_geometry=float(g.loc['LiVAE','increment_median']),PCA_increment_over_geometry=float(g.loc['PCA','increment_median'])))
 paired=pd.DataFrame(paired);paired.to_csv(dest/'paired_representation_effects.csv',index=False)
 save(dest/'receipt.json',{'status':'completed','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cohorts':C,'arms':ARMS,'complete_downstream_runs':len(receipts),'cell_checks':finitechecks,'tasks':sum(len(json.loads((D/'PCA'/c/'protocol.json').read_text())['ends']) for c in C),'population_comparisons':['all','nonterminal','starting'],'association_rows':len(data),'seed_stability_rows':len(stab),'driver_sha256':sha(__file__),'upstream_source_identity':[{'cohort':c,'receipt_sha256':sha(N/c/'receipt.json')} for c in C],'downstream_receipts':receipts,'interpretation':'same-data descriptive relationships, not fate accuracy; no biological replicates or unmeasured controls inferred','validation_is_external':False,'LiVAE_better_count_by_population':{p:int(((paired.population==p)&(paired.LiVAE_minus_PCA>0)).sum()) for p in ['all','nonterminal','starting']},'LiVAE_increment_over_geometry_positive_by_population':{p:int(((paired.population==p)&(paired.LiVAE_increment_over_geometry>0)).sum()) for p in ['all','nonterminal','starting']},'output_files':{p.name:sha(p) for p in sorted(dest.glob('*.csv'))}})
 print(paired.to_string(index=False))
