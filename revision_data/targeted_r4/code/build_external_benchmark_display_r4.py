#!/usr/bin/env python3
"""Compact, separate displays for fate-score and lineage-ordering associations."""
from pathlib import Path
import argparse,json,hashlib,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/hematopoietic-benchmark-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
OUT=Path(__file__).resolve().parents[2]/'results/external_method_benchmark_20261021'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

ORDER=['irradiation','ucb','dapp1','sleep']
LABEL={'irradiation':'Irrad.','ucb':'TPO','dapp1':'Dapp1','sleep':'Sleep'}
METHODS=['scRL value','PCA30 proximity','UMAP proximity','Diffusion proximity','Palantir','CellRank 2','FateID Python']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',default='analysis_v1')
    parser.add_argument('--folder',type=Path);parser.add_argument('--registration',type=Path)
    parser.add_argument('--output-folder',type=Path);args=parser.parse_args()
    folder=args.folder or OUT/args.analysis
    verification=json.loads((folder/'verification.json').read_text());assert verification['status']=='pass'
    summary=pd.read_csv(folder/'summary.csv');trajectory=pd.read_csv(folder/'trajectory_associations.csv')
    coverage=pd.read_csv(folder/'lineage_coverage.csv').fillna('')
    primary=summary.loc[summary.population.eq('shared_nonterminal')]
    traj=trajectory.loc[trajectory.population.eq('shared_nonterminal')]
    registry=json.loads((args.registration or OUT/'registration.json').read_text())
    dest=args.output_folder or folder/'figures';dest.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':7.5,
                         'ytick.labelsize':7.5,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(6.95,5.7))
    ax=fig.add_axes([.26,.65,.69,.235])
    tasks=[('irradiation','erythroid'),('irradiation','myeloid'),('dapp1','erythroid'),('dapp1','myeloid')]
    matrix=np.full((len(tasks),len(METHODS)),np.nan);display=[]
    missing_labels={}
    for i,(cohort,task) in enumerate(tasks):
        for j,method in enumerate(METHODS):
            row=primary.loc[primary.cohort.eq(cohort)&primary.task.eq(task)&primary.method.eq(method)]
            assert len(row)==1;row=row.iloc[0];matrix[i,j]=row['median']
            missing_labels[i,j]='invalid' if 'failed_fit' in row.statuses else 'NA'
            display.append({'panel':'A','cohort':cohort,'task':task,'method':method,'n_cells':int(row.n_cells),
                            'median':row['median'],'minimum':row.minimum,'maximum':row.maximum,
                            'all_defined':row.all_defined})
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e7e7e7')
    image=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(len(METHODS)),['scRL\nvalue','PCA30','UMAP','Diffusion','Palantir\nlocal variant','CellRank 2\nDPT kernel','FateID\nPython'])
    ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=4)
    ax.set_yticks(range(len(tasks)),[LABEL[c]+' / '+t for c,t in tasks])
    ax.set_xticks(np.arange(-.5,len(METHODS)),minor=True);ax.set_yticks(np.arange(-.5,len(tasks)),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.3);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(len(tasks)):
        for j in range(len(METHODS)):
            value=matrix[i,j]
            ax.text(j,i,f'{value:+.3f}' if np.isfinite(value) else missing_labels[i,j],ha='center',va='center',
                    fontsize=7.2 if np.isfinite(value) else 6.8,color='white' if abs(value)>.65 else '#222222')
    fig.text(.025,.975,'A',fontweight='bold',fontsize=11,va='top')
    fig.text(.26,.975,'Fate scores and value / geometry references',fontsize=8.7,va='top')
    fig.text(.26,.602,'Shared nonterminal cells: irradiation 21,468; Dapp1 8,298',fontsize=7.2,va='center')
    bar_ax=fig.add_axes([.76,.563,.19,.012]);bar=fig.colorbar(image,cax=bar_ax,orientation='horizontal',ticks=[-1,0,1])
    bar.ax.tick_params(labelsize=6.5,length=2,pad=1);bar.outline.set_visible(False)
    fig.text(.747,.568,'Signed ρ',fontsize=7,ha='right',va='center')

    failed_cohorts=coverage.loc[coverage.fit_status.eq('fail'),'cohort'].tolist()
    axb=fig.add_axes([.415,.13,.495,.31 if failed_cohorts else .34])
    row_keys=traj.loc[traj.method.eq('Slingshot pseudotime'),['cohort','task','lineage','end_cluster','n_cells']].copy()
    row_keys['order']=row_keys.cohort.map({c:i for i,c in enumerate(ORDER)})
    row_keys=row_keys.sort_values(['order','task','lineage']).reset_index(drop=True)
    labels=[];shapes={'Slingshot pseudotime':('o','#176b93'),'PCA30 proximity':('^','#ba793d'),
                      'UMAP proximity':('s','#4e8465'),'Diffusion proximity':('D','#85618d')}
    offsets={'Slingshot pseudotime':-.23,'PCA30 proximity':-.115,'UMAP proximity':0.,'Diffusion proximity':.115}
    component_maps={}
    for cohort in ORDER:
        path=next(iter(sorted(folder.glob(f'{cohort}_*_evaluation.csv.gz'))))
        cells=pd.read_csv(path,dtype={'cell_id':str,'leiden':str}).set_index('cell_id')
        mapping={}
        for name,cell in registry['cohorts'][cohort]['terminal_representatives'].items():
            mapping.setdefault(str(cells.loc[cell,'leiden']),[]).append(name)
        component_maps[cohort]=mapping
    transform=blended_transform_factory(axb.transAxes,axb.transData)
    for y,key in row_keys.iterrows():
        group=traj.loc[traj.cohort.eq(key.cohort)&traj.task.eq(key.task)&traj.lineage.eq(key.lineage)]
        focal=group.loc[group.method.eq('scRL value'),'spearman'].to_numpy()
        focal=focal[np.isfinite(focal)]
        if len(focal):
            axb.plot([focal.min(),focal.max()],[y+.23,y+.23],color='#333333',lw=1,zorder=2)
            axb.scatter(np.median(focal),y+.23,marker='x',s=20,lw=.9,color='#333333',zorder=4)
        for method,(marker,color) in shapes.items():
            selected=group.loc[group.method.eq(method),'spearman']
            assert len(selected)==1
            if np.isfinite(selected.iloc[0]):axb.scatter(selected.iloc[0],y+offsets[method],s=19,marker=marker,color=color,lw=.3,zorder=3)
        cov=coverage.loc[coverage.cohort.eq(key.cohort)&coverage.lineage.eq(key.lineage)]
        assert len(cov)==1
        axb.text(1.055,y,f'{100*cov.fraction_nonterminal_covered.iloc[0]:.0f}%',transform=transform,
                 fontsize=7,va='center',ha='center')
        end_label=','.join(component_maps[key.cohort].get(str(key.end_cluster),[str(key.end_cluster)]))
        labels.append(f'{LABEL[key.cohort]} / {end_label}   n={int(key.n_cells):,}')
        for method,subgroup in group.groupby('method',sort=False):
            defined=subgroup.loc[subgroup.status.eq('defined'),'spearman']
            display.append({'panel':'B','cohort':key.cohort,'task':key.task,'lineage':int(key.lineage),
                            'end_cluster':key.end_cluster,'method':method,'n_cells':int(key.n_cells),
                            'median':float(defined.median()),'minimum':float(defined.min()),'maximum':float(defined.max()),
                            'all_defined':bool(len(defined)==len(subgroup)),
                            'coverage_fraction':float(cov.fraction_nonterminal_covered.iloc[0])})
    axb.axvline(0,color='#aaaaaa',lw=.6,zorder=0)
    axb.set(xlim=(-1.04,1.04),ylim=(len(row_keys)-.5,-.5),xlabel='Signed Spearman ρ on the same assigned cells')
    axb.set_yticks(range(len(labels)),labels);axb.set_xticks([-1,-.5,0,.5,1])
    axb.spines[['right','top']].set_visible(False);axb.tick_params(axis='y',length=0,pad=4)
    axb.text(1.055,1.05,'Coverage',transform=axb.transAxes,fontsize=6.7,ha='center')
    fig.text(.025,.515,'B',fontweight='bold',fontsize=11,va='top')
    fig.text(.26,.515,'Slingshot ordering on assigned nonterminal cells',fontsize=8.7,va='top')
    if failed_cohorts:
        fig.text(.26,.484,', '.join(LABEL[c] for c in failed_cohorts)+': curve output unavailable within the resource window',
                 fontsize=6.9,color='#7b3a30',va='top')
    handles=[Line2D([],[],color='#333333',marker='x',lw=1,label='scRL median / range')]
    handles += [Line2D([],[],color=color,marker=marker,ls='',markersize=4,label=label)
                for method,(marker,color),label in [(k,v,{'Slingshot pseudotime':'Slingshot',
                  'PCA30 proximity':'PCA30','UMAP proximity':'UMAP','Diffusion proximity':'Diffusion'}[k]) for k,v in shapes.items()]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.61,.015),ncol=5,frameon=False,
               fontsize=6.9,handlelength=1,handletextpad=.35,columnspacing=.8,borderaxespad=0)
    for extension in ['png','pdf','svg']:
        fig.savefig(dest/f'external_method_comparison.{extension}',dpi=300)
    plt.close(fig)
    pd.DataFrame(display).to_csv(dest/'display_values.csv',index=False)
    write_json(dest/'display_receipt.json',{'status':'generated; visual review pending','source_verification_sha256':sha(folder/'verification.json'),
        'builder_sha256':sha(__file__),'n_probability_cells':28,'n_matched_lineage_rows':len(row_keys),
        'format':'Arial; explicit axes and label positions; distinct output families',
        'outputs':{p.name:sha(p) for p in dest.iterdir() if p.is_file()}})
    print(json.dumps({'figure_folder':str(dest),'matched_lineage_rows':len(row_keys)}))


if __name__=='__main__':main()
