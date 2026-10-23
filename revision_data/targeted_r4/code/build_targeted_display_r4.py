#!/usr/bin/env python3
"""Deterministic figures S4--S7 and tables S9--S12 from checked saved evidence."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,re
os.environ.setdefault('MPLCONFIGDIR','/tmp/hematopoietic-r4-display')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
LABEL={'dapp1':'Dapp1','irradiation':'Irrad.','ucb':'TPO','sleep':'Sleep'}
COMP={'erythroid':'Ery','neutrophil':'Neut','monocyte':'Mono','Ery':'Ery','Gra':'Gra','Mo':'Mo','DC':'DC','megakaryocyte':'Mk'}
METHODS=['PCA30 component proximity','UMAP component proximity','Diffusion component proximity','Palantir, local variant','CellRank 2, DPT kernel','FateID Python']
DISPLAY=['PCA30','UMAP','Diffusion','Palantir\nlocal variant','CellRank 2\nDPT kernel','FateID\nPython']
COLORS=['#23638c','#b15d3b','#62866b'];shapes=['o','s','^']
LEDGER=[]

def record(figure,table):
    for row in table.to_dict('records'):LEDGER.append({'figure':figure,**row})
def save(fig,dest,name):
    for ext in ['png','pdf','svg']:fig.savefig(dest/(name+'.'+ext),dpi=300)
    plt.close(fig)
def style(ax):
    ax.spines[['right','top']].set_visible(False);ax.tick_params(width=.6,length=3,pad=3)
def label(fig,letter,x,y,title):
    fig.text(x,y,letter,weight='bold',size=11,va='top');fig.text(x+.043,y,title,size=8.5,va='top')
def heat(ax,data,yticks,xticks):
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e5e5')
    im=ax.imshow(np.ma.masked_invalid(data),cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    ax.set_yticks(range(len(yticks)),yticks);ax.set_xticks(range(len(xticks)),xticks);ax.xaxis.tick_top()
    ax.tick_params(length=0,pad=5);ax.set_xticks(np.arange(-.5,len(xticks)),minor=True);ax.set_yticks(np.arange(-.5,len(yticks)),minor=True)
    ax.grid(which='minor',color='white',lw=1.2);ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    for i,row in enumerate(data):
        for j,x in enumerate(row):ax.text(j,i,f'{x:+.3f}' if np.isfinite(x) else 'invalid',ha='center',va='center',size=7,color='white' if abs(x)>.65 else '#222')
    return im

def component(r,d):
    table=pd.read_csv(r/'components/summary.csv');s=table.loc[table.population.eq('shared_nonterminal')]
    tasks=[('irradiation',x) for x in ['Ery','Gra','Mo','DC']]+[('dapp1',x) for x in ['erythroid','neutrophil','monocyte']]
    fig=plt.figure(figsize=(6.95,5.9));ax=fig.add_axes([.245,.535,.70,.335]);matrix=np.full((7,6),np.nan)
    for i,(c,e) in enumerate(tasks):
        for j,m in enumerate(METHODS):
            row=s.loc[s.cohort.eq(c)&s.component.eq(e)&s.method.eq(m)];assert len(row)==1;matrix[i,j]=row.iloc[0]['median']
    im=heat(ax,matrix,[LABEL[c]+' / '+COMP[e] for c,e in tasks],DISPLAY)
    label(fig,'A',.035,.975,'Component-matched program associations; common nonterminal cells')
    fig.text(.245,.492,'Irradiation n=21,468; Dapp1 n=8,298. Each proximity uses its own component.',size=7)
    ca=fig.add_axes([.765,.46,.18,.01]);fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[-1,0,1]).outline.set_visible(False);ca.tick_params(size=2,labelsize=6.5,pad=1)
    fig.text(.75,.465,'Signed ρ',size=7,ha='right',va='center')
    t=pd.read_csv(r/'components/trajectory_associations.csv');t=t.loc[t.population.eq('shared_nonterminal')]
    keys=t.loc[t.method.eq('Slingshot pseudotime')].sort_values(['cohort','component']).reset_index(drop=True)
    ax=fig.add_axes([.405,.11,.49,.27]);methods=['Slingshot pseudotime']+METHODS[:3]
    col=['#176b93','#ba793d','#4e8465','#85618d'];markers=['o','^','s','D'];offsets=np.linspace(-.22,.22,4)
    names=[]
    for y,row in keys.iterrows():
        group=t.loc[t.cohort.eq(row.cohort)&t.component.eq(row.component)&t.lineage.eq(row.lineage)]
        for m,color,marker,off in zip(methods,col,markers,offsets):
            q=group.loc[group.method.eq(m)];assert len(q)==1
            ax.scatter(q.spearman.iloc[0],y+off,s=20,color=color,marker=marker,zorder=3)
        names.append(f'{LABEL[row.cohort]} / {COMP[row.component]}  n={row.n_cells:,}')
        ax.text(1.05,y,f'{row.coverage:.1%}',transform=ax.get_yaxis_transform(),size=7,va='center',ha='left')
    ax.axvline(0,color='#aaa',lw=.6);ax.set(xlim=(-1.02,1.02),ylim=(len(keys)-.55,-.6),xlabel='Signed Spearman ρ on identical assigned cells')
    ax.set_yticks(range(len(keys)),names);ax.set_xticks([-1,-.5,0,.5,1]);style(ax);ax.tick_params(axis='y',length=0)
    label(fig,'B',.035,.425,'Matched Slingshot lineages; program and endpoint change together')
    ax.text(1.05,1.08,'Coverage',transform=ax.transAxes,size=6.8,ha='left')
    fig.legend([Line2D([],[],marker=m,color=c,ls='',markersize=4) for m,c in zip(markers,col)],['Slingshot','PCA30','UMAP','Diffusion'],ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.60,.015),fontsize=7,handlelength=1,columnspacing=1.1)
    record('S5A',s.loc[s.cohort.isin(['dapp1','irradiation'])]);record('S5B',t);save(fig,d,'component_matched')

def training(r,d):
    fig=plt.figure(figsize=(6.95,6.5));gs=fig.add_gridspec(3,2,left=.09,right=.975,bottom=.085,top=.925,wspace=.29,hspace=.40)
    titles=['Training return (250-episode mean)','All-cell value–program association','Common-start return','Common-start Critic prediction RMSE','Common-start TD RMSE','Cumulative mapped-grid visitation']
    axes=[fig.add_subplot(gs[i//2,i%2]) for i in range(6)]
    for seed in [0,1,2]:
        e=pd.read_csv(r/f'training/seed{seed}/episodes.csv.gz');t=pd.read_csv(r/f'training/seed{seed}/checkpoints.csv');steps=pd.read_csv(r/f'training/seed{seed}/steps.csv.gz')
        axes[0].plot(e.episode,e['return'].rolling(250,min_periods=250).mean(),lw=1,color=COLORS[seed])
        for k,key in [(1,'rho_all'),(2,'common_start_return_mean'),(3,'common_start_prediction_RMSE'),(4,'common_start_TD_RMSE')]:axes[k].plot(t.episode,t[key],lw=1,marker=shapes[seed],ms=2.5,color=COLORS[seed])
        grid=set();values=[]
        for ep,g in steps.groupby('episode',sort=True):
            grid.update(g.grid.astype(int));grid.update(g.next_grid.astype(int))
            if ep%500==0:values.append((ep,len(grid)/1774))
        axes[5].plot(*np.array(values).T,lw=1,marker=shapes[seed],ms=2.5,color=COLORS[seed]);record('S6',t.assign(seed=seed))
    for k,(ax,title) in enumerate(zip(axes,titles)):
        ax.set_title(chr(65+k)+'   '+title,loc='left',fontsize=8.4,pad=8);style(ax);ax.axvline(5000,color='#999',ls=':',lw=.8);ax.set(xlim=(0,10000),xticks=[0,5000,10000]);ax.ticklabel_format(axis='x',style='plain');ax.set_xlabel('Training episodes',labelpad=2)
        if k in [0,1,2]:ax.axhline(0,color='#bbb',lw=.5,zorder=0)
    axes[1].set(ylim=(-1,1),yticks=[-1,-.5,0,.5,1]);axes[5].set(ylim=(0,1),yticks=[0,.5,1]);axes[5].set_ylabel('Fraction of 1,774 mapped grids',labelpad=3)
    fig.legend([Line2D([],[],color=c,marker=m,ms=4,lw=1) for c,m in zip(COLORS,shapes)],['Seed 0','Seed 1','Seed 2'],loc='upper center',bbox_to_anchor=(.5,.998),ncol=3,frameon=False,fontsize=8)
    save(fig,d,'tpo_training_diagnostics')

def sensitivities(r,d):
    p=pd.read_csv(r/'partitions/summary.csv');fig=plt.figure(figsize=(6.95,5.65));axes=[fig.add_axes([.09+k*.305,.60,.245,.285]) for k in range(3)]
    partitions=['historical','leiden_0.5','leiden_1.5'];print(p.partition.unique())
    if not set(partitions)<=set(p.partition):partitions=['historical']+[s for s in p.partition.unique() if s!='historical']
    for i,(c,g,title) in enumerate([('dapp1','neutrophil','Dapp1 granulocytic'),('dapp1','cycling','Dapp1 cycling'),('sleep','neutrophil','Sleep granulocytic')]):
        ax=axes[i];q=p.loc[p.cohort.eq(c)&p.program.eq(g)]
        for j,scaling in enumerate(['cohort_reference','condition_A_reference']):
            rows=q.loc[q.scaling.eq(scaling)].set_index('partition').loc[partitions];yy=np.arange(3)+(j-.5)*.2;color=['#23638c','#b15d3b'][j]
            ax.scatter(rows.fixed,yy,color=color,marker=['o','s'][j],s=20,zorder=3)
            ax.scatter(rows.raw_all,yy,color=color,marker='|',s=70,lw=.9,zorder=3)
            for y,(_,row) in zip(yy,rows.iterrows()):ax.plot([row.raw_all,row.fixed],[y,y],color=color,lw=.7)
        ax.axvline(0,color='#aaa',lw=.7);ax.set_yticks(range(3),['Frozen','Leiden 0.5','Leiden 1.5'] if i==0 else ['','','']);ax.set_ylim(2.45,-.65);style(ax);ax.tick_params(axis='y',length=0)
        ax.set_xlabel('Program contrast');ax.set_title(chr(65+i)+'  '+title,loc='left',size=8.1,pad=9)
    fig.legend([Line2D([],[],marker='o',color='#23638c',ls=''),Line2D([],[],marker='s',color='#b15d3b',ls=''),Line2D([],[],marker='|',color='#444',ls='')],['Cohort reference, fixed','WT/CT reference, fixed','Unadjusted all cells'],loc='upper center',bbox_to_anchor=(.53,.545),ncol=3,frameon=False,fontsize=7,handlelength=1,columnspacing=.9)
    t=pd.read_csv(r/'endpoint/summary.csv');ax=fig.add_axes([.245,.115,.70,.28]);methods=['PCA30 proximity','UMAP proximity','Diffusion proximity','Palantir, local variant'];colors=['#ba793d','#4e8465','#85618d','#176b93'];markers=['^','s','D','o']
    keys=[('original','original'),('strict','original'),('original','marker_split'),('strict','marker_split')]
    for y,(definition,program) in enumerate(keys):
        for j,(m,color,marker) in enumerate(zip(methods,colors,markers)):
            q=t.loc[t.definition.eq(definition)&t.program.eq(program)&t.method.eq(m)];assert len(q)==1;row=q.iloc[0];yy=y+(j-1.5)*.13
            ax.plot([row.minimum,row.maximum],[yy,yy],color=color,lw=1);ax.scatter(row['median'],yy,color=color,marker=marker,s=20,zorder=3)
    ax.axvline(0,color='#aaa',lw=.6);ax.set(ylim=(3.5,-.5),xlim=(-.12,.25),xlabel='Signed Spearman ρ; identical 8,298 nonterminal cells');ax.set_xticks([-.1,0,.1,.2]);ax.set_yticks(range(4),['87 / full program','56 / full program','87 / marker split','56 / marker split']);style(ax);ax.tick_params(axis='y',length=0)
    label(fig,'D',.035,.435,'Dapp1 erythroid endpoint size / evaluation-program sensitivity')
    fig.legend([Line2D([],[],marker=m,color=c,ls='',ms=4) for m,c in zip(markers,colors)],['PCA30','UMAP','Diffusion','Palantir, local variant'],ncol=4,loc='lower center',bbox_to_anchor=(.59,.015),frameon=False,fontsize=7,handlelength=1,columnspacing=1.1)
    record('S7ABC',p);record('S7D',t);save(fig,d,'partition_endpoint_sensitivity')

def esc(s):
    s=str(s).replace('\\','/').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    return re.sub(r'[a-f0-9]{40,64}',lambda m:r'\allowbreak{}'.join(m[0][i:i+8] for i in range(0,len(m[0]),8)),s)
def fmt(x):return f'{x:.3f}' if np.isfinite(x) else 'invalid'
def value(row):
    if not np.isfinite(row['median']):return 'invalid (0/3)'
    if int(row.n_runs)==1:return fmt(row['median'])
    return r'\shortstack{'+fmt(row['median'])+r'\\{\tiny('+fmt(row.minimum)+', '+fmt(row.maximum)+')}}'
def block(title,headers,rows,widths,note=''):
    assert abs(sum(widths)-.98)<.04
    usable=.975-4*(len(widths)-1)/450
    widths=[round(w/sum(widths)*usable,5) for w in widths]
    lines=[r'\Needspace{.18\textheight}',r'\noindent\textbf{'+title+r'} '+note+r'\par',r'{\scriptsize\setlength{\tabcolsep}{2pt}\renewcommand{\arraystretch}{1.18}',r'\begin{longtable}{@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\textwidth}' for w in widths)+'@{}}',r'\toprule']
    head=' & '.join(headers)+r' \\'
    lines +=[head,r'\midrule\endfirsthead',r'\toprule',head,r'\midrule\endhead']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows];lines +=[r'\bottomrule\end{longtable}}','']
    return '\n'.join(lines)

def tables(r,p,out):
    # Current S9 starts from the historical table, retaining every numerical entry.
    text=(p/'revision_data/external_benchmark/supplement_table.tex').read_text().replace('Palantir','Palantir, local variant').replace('CellRank 2','CellRank 2, DPT kernel')
    text=text.replace('Myeloid tasks retain their broad terminal unions but are evaluated against the granulocytic program.','Myeloid tasks retain their historical broad terminal unions and granulocytic evaluation; component-matched results are supplied separately in Table~S10.')
    fail=pd.read_csv(r/'implementation/palantir_failure_diagnostics.csv');rows=[]
    for _,x in fail.iterrows():rows.append([int(x.seed),f'{x.failed_cells:,} ({x.failed_fraction:.1%})'.replace('%',r'\%'),f'{x.zero_mass_cells:,}',f'{x.above_unit_mass_cells:,}',f'[{x.score_minimum:.3g}, {x.score_maximum:.6g}]',f'[{x.row_mass_minimum:.3g}, {x.row_mass_maximum:.6g}]'])
    text+=block('(e) Palantir, local variant: irradiation native-output diagnostics.',['Seed','Failed cells','Zero mass','Mass $>1$','Score extrema','Row-mass extrema'],rows,[.06,.18,.14,.14,.20,.24],r'Checks require finite nonnegative scores and row mass in $[0.96-10^{-6},1+10^{-6}]$. Failed-cell fractions use all 41,252 cells; no normalization repairs these outputs.')
    rows=[[int(x.seed),f'{x.unreachable_waypoints}/{x.waypoints}',f'{x.condition_number:.3g}',f'{x.smallest_singular_value:.3g}'] for _,x in fail.iterrows()]
    text+=block('Reachability and conditioning for the same three runs.',['Seed','Unreachable / total waypoints','Condition number','Smallest singular value'],rows,[.08,.34,.25,.29],r'Values copy the recorded diagnostic snapshot. Independent native-array replay verifies failed scores, reachability and the near-singular diagnosis; extreme condition numbers and smallest singular values vary with numerical roundoff. These diagnostics do not identify a proven cause or validate terminal representatives.')
    identity=pd.read_csv(r/'implementation/implementation_identity.csv');rows=[[esc(x.method),esc(x.upstream),esc(x.change_type)+'. '+esc(x.changes)+' '+esc(x.conformance)] for _,x in identity.iterrows()]
    text+=block('(f) Pinned implementation identity and change classification.',['Configured method','Pinned source','Changes and conformance'],rows,[.18,.25,.55],r'Exact sources, release hashes and patch files accompany the code. Source identity and internal numerical conformance do not establish agreement with another algorithm or biological validity.')
    rows=[['Dapp1','10; old count unavailable','20; stops at 5','Yes, threshold 0.001; native outputs identical'],['TPO','10; count unavailable','Not repeated','Finite output; stopping status unavailable'],['Sleep','10; count unavailable','Not repeated','Finite output; stopping status unavailable'],['Irradiation','10; resource window exceeded','Not repeated','No native curve output; no convergence inference']]
    text+=block('(g) Native R Slingshot stopping status.',['Cohort','Original cap / status','Focused longer cap','Interpretation'],rows,[.12,.25,.25,.36],r'The Dapp1 exit logger records the native inner curve-fitting loop without changing its update. Final relative total-distance change was below 0.001 at iteration 5; curve weights and pseudotime match the original maxit 10 output exactly. This does not recover old stopping counts for other cohorts.')
    text+=r'\clearpage'+'\n'+r'\subsection*{Supplementary Table S10. Component-matched external outputs and proximity}'+'\n'
    s=pd.read_csv(r/'components/summary.csv');s=s.loc[s.population.eq('shared_nonterminal')];tasks=[('irradiation',x) for x in ['Ery','Gra','Mo','DC']]+[('dapp1',x) for x in ['erythroid','neutrophil','monocyte']]
    rows=[]
    for c,e in tasks:
        group=s.loc[s.cohort.eq(c)&s.component.eq(e)].set_index('method');rows.append([LABEL[c]+' / '+COMP[e]]+[value(group.loc[m]) for m in METHODS])
    text+=block('(a) Matched probability-component associations.',['Cohort / component','PCA30','UMAP','Diffusion','Palantir, local variant','CellRank 2, DPT kernel','FateID Python'],rows,[.16]+[.136]*6,r'Each output is matched to its own program and its own full-membership endpoint reference. N=21,468 irradiation and 8,298 Dapp1 common nonterminal cells. Entries give medians (minimum, maximum) across all three stochastic seeds; geometry without stochastic fitting has one value. No broad-target scRL fit is relabelled as component-specific. The new irradiation dendritic program uses Flt3, Clec9a, Siglech, Tcf4 and Itgax, all detected on the retained gene axis; the count audit accompanies the code.')
    t=pd.read_csv(r/'components/trajectory_associations.csv');t=t.loc[t.population.eq('shared_nonterminal')];keys=t.loc[t.method.eq('Slingshot pseudotime')].sort_values(['cohort','component']);rows=[]
    for _,x in keys.iterrows():
        group=t.loc[t.cohort.eq(x.cohort)&t.component.eq(x.component)&t.lineage.eq(x.lineage)].set_index('method');rows.append([f'{LABEL[x.cohort]} / {COMP[x.component]}',f'{x.n_cells:,}',f'{x.coverage:.1%}'.replace('%',r'\%')]+[fmt(group.loc[m].spearman) for m in ['Slingshot pseudotime']+METHODS[:3]])
    text+=block('(b) Matched lineage pseudotime and component-specific references.',['Cohort / component','Assigned n','Coverage','Slingshot','PCA30','UMAP','Diffusion'],rows,[.205,.115,.12,.135,.135,.135,.135],r'All four associations for a row use identical native assigned nonterminal cells. Coverage divides by the full common nonterminal population. Monocyte lineages now use the monocyte program, whereas Table~S9c preserves the historical granulocytic evaluation. Overlapping sets need not sum to 100\%.')
    text+=r'\clearpage'+'\n'+r'\subsection*{Supplementary Table S11. Longer-budget TPO Contribution diagnostics}'+'\n'
    t=pd.read_csv(r/'training/summary.csv');rows=[]
    for _,x in t.iterrows():rows.append([int(x.seed),f'{x.episodes:,}',fmt(x.last500_return),fmt(x.rho_all),fmt(x.rho_nonterminal),fmt(x.rho_starting),fmt(x.common_start_return),fmt(x.prediction_RMSE),fmt(x.TD_RMSE)])
    text+=block('(a) Training and common-start evaluation at nested budgets.',['Seed','Episodes','Last 500 return',r'$\rho$ all',r'$\rho$ nonterm.',r'$\rho$ start','Common return','Prediction RMSE','TD RMSE'],rows,[.055,.10,.12,.10,.12,.10,.12,.135,.12],r'Each seed is one native 10,000-episode run; its 5,000-episode prefix exactly reproduces the historical returns and projected values. All-checkpoint histories are in Fig.~S6. Common evaluations use 64 identical start grids and 4 fixed-uniform rollouts per grid at each checkpoint; their randomness is separate from training. Prediction targets are discounted realized return plus native continuation bootstrap at truncation; TD error uses the native one-step target. These errors are conditional on policy and this environment, not biological calibration or proof of adequate learning.')
    term=pd.read_csv(r/'training/termination_summary.csv');rows=[]
    for _,x in t.iterrows():
        group=term.loc[term.seed.eq(x.seed)&term.episodes.eq(x.episodes)].set_index('termination_reason');counts=[int(group.loc[k].n_episodes) if k in group.index else 0 for k in ['boundary','revisit','time_limit','invalid_move']]
        rows.append([int(x.seed),f'{x.episodes:,}',f'{x.visited_grids}/{x.mapped_grids}',f'{x.last500_steps:.3f}']+counts)
    text+=block('(b) Visitation and last-500-episode termination.',['Seed','Episodes','Cumulative visited / mapped grids','Mean steps','Boundary','Revisit','Time limit','Invalid move'],rows,[.06,.10,.24,.14,.11,.11,.11,.11],r'Counts in each row sum to 500. Visitation is the union of pre-move and post-move grids reached up to that budget; it does not establish adequate coverage or exploration. The native time limit truncates after 51 moves; invalid moves and revisits terminate with their native penalties.')
    text+=r'\clearpage'+'\n'+r'\subsection*{Supplementary Table S12. Endpoint, partition and scaling sensitivity}'+'\n'
    t=pd.read_csv(r/'endpoint/summary.csv');rows=[]
    for definition,program in [('original','original'),('strict','original'),('original','marker_split'),('strict','marker_split')]:
        group=t.loc[t.definition.eq(definition)&t.program.eq(program)].set_index('method');rows.append([('87 original' if definition=='original' else '56 strict'),('Full' if program=='original' else 'Marker split')]+[value(group.loc[m]) for m in ['PCA30 proximity','UMAP proximity','Diffusion proximity','Palantir, local variant']])
    text+=block('(a) Dapp1 erythroid endpoint and evaluation-program sensitivity.',['Endpoint','Program','PCA30','UMAP','Diffusion','Palantir, local variant'],rows,[.15,.13,.15,.15,.18,.22],r'The stricter endpoint requires positive integer Klf1 and Gata1 UMI counts within the original 87 cells, retaining 56 (19 WT, 37 KO). Its PCA-centroid representative changes; root and other component representatives, diffusion spaces, seeds and Palantir settings remain fixed. All three new fits pass native score checks. The same original 8,298-cell nonterminal evaluation population is used throughout; the 31 removed endpoint cells are not added to evaluation. Marker-split scoring uses Alas2, Gypa and Tfrc, removing both definition markers. These are RNA-internal sensitivity controls, not independent annotation or validation.')
    t=pd.read_csv(r/'partitions/summary.csv');rows=[]
    for _,x in t.iterrows():
        program='Gran.' if x.program=='neutrophil' else 'Cycling';partition={'historical':'Frozen'}.get(x.partition,x.partition.replace('fresh_','Leiden '));scale='Cohort' if x.scaling=='cohort_reference' else 'WT/CT'
        rows.append([LABEL[x.cohort]+' / '+program,esc(partition),scale,f'{x.n_eligible}/{x.n_groups}',f'{100*x.coverage:.2f}',f'{100*x.coverage_A:.2f} / {100*x.coverage_B:.2f}',fmt(x.raw_all),fmt(x.raw_eligible),fmt(x.fixed)])
    text+=block('(b) Fresh coarse/fine partitions and alternative gene-scaling reference.',['Cohort / program','Partition','Scaling','Eligible / all groups',r'Coverage (\%)',r'A / B coverage (\%)','Raw all','Raw eligible','Fixed'],rows,[.15,.125,.085,.115,.095,.13,.095,.095,.095],r'Fresh Leiden resolutions 0.5 and 1.5 use a 15-neighbour PCA30 graph, seed 0, undirected igraph and two iterations. Eligibility requires 20 cells per condition per group; common weights use pooled eligible counts. A/B denotes WT/KO or CT/SD. Scores use cohort means/population SD or the WT/CT reference; zero SD is replaced by 1. This checked set retains the three headline directions while their magnitudes vary. It is not an exhaustive partition or scaling robustness claim.')
    assert '\r' not in text, 'Unexpected carriage-return in TeX display'
    out.write_text(text)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--packet',type=Path,required=True);ap.add_argument('--article-root',type=Path,required=True);ap.add_argument('--output-folder',type=Path,required=True);args=ap.parse_args()
    r=args.packet;p=args.article_root;d=args.output_folder;d.mkdir(parents=True,exist_ok=False)
    assert json.loads((r/'replay_verification.json').read_text())['status']=='pass'
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7.2,'axes.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    component(r,d);training(r,d);sensitivities(r,d);tables(r,p,d/'supplement_table.tex')
    pd.DataFrame(LEDGER).to_csv(d/'display_values.csv',index=False)
    (d/'display_receipt.json').write_text(json.dumps({'status':'generated; visual check pending','builder_sha256':sha(Path(__file__)),'replay_sha256':sha(r/'replay_verification.json'),'table_groups':['S9','S10','S11','S12'],'outputs':{f.name:sha(f) for f in sorted(d.iterdir()) if f.is_file()}},indent=2)+'\n')
if __name__=='__main__':main()
