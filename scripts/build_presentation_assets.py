#!/usr/bin/env python3
"""Compose the current scientific sequence from immutable saved numerical outputs.

The legacy builder preserves canonical source bindings and plot artists.
Current layouts consume CSV cells, with a separate value/mask record ledger.
No expression scoring, fitting or association calculation occurs in this file.
"""
from pathlib import Path
import csv, hashlib, json, os
os.environ.setdefault('MPLCONFIGDIR','/tmp/scientific-reports-matplotlib')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import build_figures as f
import build_legacy_presentation_assets as old
import publication_style as s

P=Path(__file__).resolve().parents[1]; E=P/'evidence'; F=P/'figures'
R4='revision_data/targeted_r4/'; R3='revision_data/external_benchmark/'
MAIN=['article_figure1_composition','article_figure2_irradiation','article_figure3_tpo',
      'article_figure4_controls','article_figure5_components','article_figure6_interfaces','article_figure7_training']
SUPPLEMENT=['article_supplementary_figure1_contrasts','article_supplementary_figure2_rewards',
 'article_supplementary_figure3_optimization','article_supplementary_figure4_historical',
 'article_supplementary_figure5_lineages','article_supplementary_figure6_training',
 'article_supplementary_figure7_partitions','article_supplementary_figure8_marrow',
 'article_supplementary_figure9_focus']
SOURCES={}; VALUES=[]; RECORDS=[]; TABLES={}; QA=[]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def data(rel):
    if rel not in TABLES:
        TABLES[rel]=pd.read_csv(P/rel, float_precision='round_trip')
        SOURCES[rel]=sha(P/rel)
    return TABLES[rel]
def cell(rel, index, column, figure, panel, **context):
    row=data(rel).loc[index]; result=float(row[column])
    VALUES.append(dict(display_mark=f'{figure}:{panel}:row{index+1}:{column}',source_file=rel,
        source_row_1based=int(index)+1,source_column=column,transform='identity',value=result))
    record={key:'' for key in ['figure','panel','source_file','source_row_1based','source_column',
        'cohort','component','method','implementation','definition','program','population',
        'mask_sha256','seed','episode','statistic','n_cells','coverage','value']}
    for key in record:
        if key in row:record[key]=row[key]
    record.update(figure=figure,panel=panel,source_file=rel,source_row_1based=int(index)+1,
        source_column=column,statistic=column,value=result)
    if 'eligible_mask_sha256' in row:record['mask_sha256']=row.eligible_mask_sha256
    if not record['component']:
        record['component']=row.get('task', row.get('program',''))
    if not record['implementation']:
        record['implementation']='Palantir 1.4.2 unpatched' if str(record['method']).startswith('Palantir') else record['method']
    if rel.endswith('summary.csv') and 'components/' in rel and not record['mask_sha256']:
        a=data(R4+'components/associations.csv')
        m=a.cohort.eq(row.cohort)&a.component.eq(row.component)&a.population.eq(row.population)
        hashes=a.loc[m,'mask_sha256'].unique();assert len(hashes)==1
        record['mask_sha256']=hashes[0]
    elif rel==R3+'tables/summary.csv' and not record['mask_sha256']:
        a=data(R3+'tables/associations.csv')
        hashes=a.loc[a.cohort.eq(row.cohort)&a.task.eq(row.task)&a.population.eq(row.population),'mask_sha256'].unique()
        assert len(hashes)==1;record['mask_sha256']=hashes[0]
    elif rel==R4+'endpoint/summary.csv':
        a=data(R4+'endpoint/associations.csv')
        hashes=a.loc[a.definition.eq(row.definition)&a.program.eq(row.program),'mask_sha256'].unique()
        assert len(hashes)==1;record['mask_sha256']=hashes[0];record['population']='shared_nonterminal';record['component']='erythroid';record['cohort']='dapp1'
    if '/training/seed' in rel:
        seed=int(rel.split('/training/seed')[1].split('/')[0])
        record.update(cohort='ucb',component='megakaryocyte',method='scRL Contribution',implementation='native retained r4',seed=seed,definition='frozen endpoint',program='megakaryocytic')
        if str(column).startswith('rho_'):
            pop=column.removeprefix('rho_');a=data('revision_data/structural_r5/value_scale.csv')
            mask=a.loc[a.seed.eq(seed)&a.episode.eq(row.episode)&a.population.eq(pop)].iloc[0]
            record.update(population=pop,mask_sha256=mask.mask_sha256,n_cells=mask.n_cells)
        else:record['population']='native episodes' if rel.endswith('episodes.csv.gz') else 'common-start rollouts'
    if rel.startswith('revision_data/structural_r5/training_'):
        record.update(cohort='ucb',component='megakaryocyte',method='scRL Contribution',implementation='native retained r4',definition='frozen endpoint',program='megakaryocytic',population='native training episodes' if 'return' in rel else 'mapped training grids')
    record.update(context); RECORDS.append(record)
    return result

def pick(rel, **selector):
    d=data(rel)
    for key,value in selector.items():d=d.loc[d[key].eq(value)]
    assert len(d)==1,(rel,selector,len(d))
    return d.index[0]

def vector(rel,column,figure,panel,**selector):
    d=data(rel)
    for k,v in selector.items():d=d.loc[d[k].eq(v)]
    return [cell(rel,i,column,figure,panel) for i in d.index]

def emit(fig,name):s.export(fig,name,F,QA,crop_vertical=True)
def position(ax,fig,x,y,w,h):ax.set_position([x/s.WIDTH,y/fig.get_figheight(),w/s.WIDTH,h/fig.get_figheight()])
def remove_letters(fig):
    for ax in fig.axes:
        for obj in list(ax.texts):
            if getattr(obj,'_panel_letter',False):obj.remove()
def clear_figtext(fig):
    for obj in list(fig.texts):obj.remove()
    for obj in list(fig.legends):obj.remove()

def contrast_legend(fig,y):
    fig.legend(handles=[Line2D([],[],color=s.GRAY,marker='o',mfc='white',ls='',label='Raw eligible'),
        Line2D([],[],color=s.BLUE,marker='o',ls='',label='Fixed'),
        Line2D([],[],color=s.GRAY,marker='x',ls='',label='Raw all')],
        loc='center',bbox_to_anchor=(.54,y/fig.get_figheight()),frameon=False,ncol=3,
        columnspacing=1.5,handlelength=1,handletextpad=.4)

def partition_panel(fig,box,letter,cohort,program,heading,show_y=True,annotate_coverage=False):
    ax=s.axes(fig,*box); rel=R4+'partitions/summary.csv'
    for y,part in enumerate(['historical','fresh_0.5','fresh_1.5']):
        for delta,scale,color,marker in [(-.13,'cohort_reference',s.BLUE,'o'),(.13,'condition_A_reference',s.PURPLE,'s')]:
            i=pick(rel,cohort=cohort,program=program,partition=part,scaling=scale)
            raw,fixed,allraw=[cell(rel,i,c,'1' if annotate_coverage else 'S7',letter) for c in ['raw_eligible','fixed','raw_all']]
            s.raw_fixed(ax,y+delta,raw,fixed,allraw,color,marker)
    ax.axvline(0,color='#999999',ls='--',lw=.7);ax.grid(axis='x',color='#E7E7E7',lw=.6)
    ax.set_ylim(2.55,-.55);ax.set_yticks(range(3),['Reference','Leiden 0.5','Leiden 1.5'] if show_y else ['']*3)
    ax.set_xlim((-.47,.08) if cohort=='dapp1' and program=='neutrophil' else (-.18,.11) if cohort=='dapp1' else (0,.16))
    ax.set_xticks([-.4,-.2,0] if cohort=='dapp1' and program=='neutrophil' else [-.1,0,.1] if cohort=='dapp1' else [0,.05,.10,.15])
    ax.set_xlabel('Program difference');s.panel(ax,letter,heading)
    return ax

def composition(plates):
    fig=plates[MAIN[0]];axes=fig.axes[:4];marker=fig.axes[-1]
    # The two fraction illustrations move to S1. Keep their canonical CSV values.
    for ax in [axes[0],axes[1],fig.axes[4]]:fig.delaxes(ax)
    clear_figtext(fig);remove_letters(fig);fig.set_size_inches(s.WIDTH,6.4)
    for ax,box,letter,title in [(axes[2],(1.10,4.95,2.17,1.0),'a','Dapp1: KO − WT'),
                                (axes[3],(4.05,4.95,2.17,1.0),'b','Sleep: SD − CT')]:
        position(ax,fig,*box);s.panel(ax,letter,title)
    position(marker,fig,1.95,3.50,4.15,.65);s.panel(marker,'c','Marker omission reveals program dependence',offset=-83)
    # Retain the marker legend, but give it its own row.
    fig.legend(handles=[Line2D([],[],color=s.GRAY,marker='D',ls='',label='Full program'),
        Line2D([],[],color=s.BLUE,marker='o',ls='',label='Omit one marker'),
        Line2D([],[],color=s.ORANGE,marker='o',ls='',label='Sleep: omit Mpo')],
        loc='center',bbox_to_anchor=(.55,2.85/6.4),ncol=3,frameon=False,
        handlelength=.8,columnspacing=1,handletextpad=.4)
    for box,letter,c,p,title,sy in [((1.10,.93,1.56,1.12),'d','dapp1','neutrophil','Dapp1 gran.',True),
        ((2.98,.93,1.51,1.12),'e','dapp1','cycling','Dapp1 cycling',False),
        ((4.81,.93,1.40,1.12),'f','sleep','neutrophil','Sleep gran.',False)]:
        partition_panel(fig,box,letter,c,p,title,sy,True)
    fig.text(.53,2.55/6.4,'Partition/scaling: circles, cohort reference; squares, WT/CT reference',ha='center')
    contrast_legend(fig,.35)
    fig.text(.53,.04/6.4,'Fine Dapp1 coverage: 95.25% overall; WT 93.15%, KO 98.52%',ha='center')
    emit(fig,MAIN[0])

def fractions_supplement(plates):
    fig=plates[SUPPLEMENT[0]];fig.set_size_inches(s.WIDTH,5.05);remove_letters(fig)
    for ax,letter,heading in zip(fig.axes,'cde',['Dapp1 granulocytic','Dapp1 cycling','Sleep granulocytic']):
        pos=ax.get_position();position(ax,fig,pos.x0*s.WIDTH,.45,pos.width*s.WIDTH,2.64);s.panel(ax,letter,heading)
    for x,cohort,letter,heading in [(1.1,'dapp1','a','Dapp1 granulocytic cluster 6'),(4.05,'sleep','b','Sleep granulocytic labels')]:
        ax=s.axes(fig,x,3.88,2.1,.65)
        rows=[r for r in f.LEDGER if r['figure']==2 and r['panel']==('a' if cohort=='dapp1' else 'b')]
        vals=[r['value'] for r in rows];assert len(vals)==2
        ax.bar([0,1],vals,color=['#A8B5BE',s.ORANGE if cohort=='dapp1' else s.GREEN],width=.54)
        ax.set_xticks([0,1],['WT','KO'] if cohort=='dapp1' else ['CT','SD']);ax.set_ylabel('Captured (%)' if cohort=='dapp1' else '')
        ax.set_ylim(0,15 if cohort=='dapp1' else 55);ax.set_yticks([0,10] if cohort=='dapp1' else [0,25,50])
        for k,v in enumerate(vals):ax.text(k,v+(.6 if cohort=='dapp1' else 1.4),f'{v:.2f}%',ha='center')
        s.panel(ax,letter,heading)
    emit(fig,SUPPLEMENT[0])

def frozen(plates):
    fig=plates[MAIN[3]];marrow=fig.axes[:2];evalaxes=fig.axes[2:5]
    # Rebuild the small technical-control plate directly from its bound source.
    sup=plt.figure(figsize=(s.WIDTH,3.55));rel='source_data/results/application_reanalysis_20260921/marrow/reward_timing.csv'
    for j,lineage in enumerate(['erythroid','myeloid']):
        ax=s.axes(sup,1.02+j*2.87,.65,2.12,2.05)
        for x,(reward,color) in enumerate([('decision',s.BLUE),('contribution',s.ORANGE),('neutral',s.GREEN)]):
            vals=vector(rel,'top_decile_mean_pseudotime','S8','ab'[j],lineage=lineage,reward=reward)
            for k,v in enumerate(vals):ax.scatter(x+(k-1)*.09,v,marker=s.SEEDS[k][1],color=color,s=25)
        ax.set_xticks(range(3),['Decision','Contribution','No-time']);ax.set_ylim(0,.85);ax.set_yticks([0,.25,.5,.75])
        if j==0:ax.set_ylabel('High-value pseudotime')
        s.panel(ax,'ab'[j],f'Marrow {lineage}')
    sup.legend(handles=[Line2D([],[],color=s.GRAY,marker=m,ls='',label=f'Seed {i}') for i,(_,m) in enumerate(s.SEEDS)],loc='lower center',bbox_to_anchor=(.54,.015),frameon=False,ncol=3);emit(sup,SUPPLEMENT[7])
    for ax in marrow+[fig.axes[-1]]:fig.delaxes(ax)
    clear_figtext(fig);remove_letters(fig);fig.set_size_inches(s.WIDTH,3.7)
    for j,ax in enumerate(evalaxes):
        position(ax,fig,.95+j*1.84,.90,1.48,1.30);s.panel(ax,'bcd'[j],['All cells','Nonterminal','Selected start'][j])
    ax=old.diagram_axis(fig,'a','Shared information in program-labelled tasks',[.045,2.85/3.7,.93,.48/3.7],label_x=.95/s.WIDTH-22/(72*s.WIDTH))
    for x,kind,label in [(.10,'program','Program'),(.36,'endpoints','Endpoints'),(.64,'balance','Geometry / scRL'),(.91,'correlation','Association')]:old.glyph_label(ax,x,kind,label,size=20)
    for a,b in [(.16,.29),(.43,.57),(.71,.84)]:old.arrow(ax,(a,.70),(b,.70))
    fig.legend(handles=[Line2D([],[],color=s.BLUE,marker='o',label='scRL median / range'),
        Line2D([],[],color=s.GREEN,marker='^',ls='',label='PCA30 proximity'),
        Line2D([],[],color=s.ORANGE,marker='s',ls='',label='UMAP proximity')],
        loc='lower center',bbox_to_anchor=(.54,.005),frameon=False,ncol=3,handlelength=1.1,columnspacing=1)
    emit(fig,MAIN[3])

def components():
    fig=plt.figure(figsize=(s.WIDTH,6.65));rel=R4+'components/summary.csv'
    rows=[('irradiation','Ery','IR Ery'),('irradiation','Gra','IR Gran.'),
        ('irradiation','Mo','IR Mono.'),('irradiation','DC','IR DC'),
        ('dapp1','erythroid','Dapp1 Ery'),('dapp1','neutrophil','Dapp1 Neut.'),('dapp1','monocyte','Dapp1 Mono.')]
    methods=['PCA30 component proximity','UMAP component proximity','Diffusion component proximity','Palantir, local variant','CellRank 2, DPT kernel','FateID Python']
    values=[]
    for c,comp,_ in rows:
        vals=[]
        for method in methods:
            i=pick(rel,cohort=c,component=comp,method=method,population='shared_nonterminal')
            vals.append(cell(rel,i,'median','5','a',implementation='Palantir 1.4.2 unpatched' if method.startswith('Palantir') else method))
            for col in ['minimum','maximum']:cell(rel,i,col,'5','a')
        values.append(vals)
    ax=s.axes(fig,1.12,4.10,5.12,2.05);s.matrix(ax,values,[q[2] for q in rows],['PCA30','UMAP','Diffusion','Palantir\n1.4.2','CellRank\nDPT','FateID\nPython'])
    s.panel(ax,'a','Matched components and programs · median Spearman ρ')
    # Two fixed masks; only the program changes.
    ax=s.axes(fig,1.12,2.68,5.10,.55);relold=R3+'tables/trajectory_associations.csv';relnew=R4+'components/trajectory_associations.csv'
    for y,c in enumerate(['dapp1','sleep']):
        i=pick(relnew,cohort=c,component='monocyte',method='Slingshot pseudotime',population='shared_nonterminal')
        row=data(relnew).loc[i]
        j=pick(relold,cohort=c,lineage=row.lineage,method='Slingshot pseudotime',population='shared_nonterminal')
        assert data(relold).loc[j,'mask_sha256']==row.mask_sha256
        a=cell(relold,j,'spearman','5','b');b=cell(relnew,i,'spearman','5','b')
        ax.plot([a,b],[y,y],color=s.GRAY,lw=1.0)
        ax.scatter(a,y,facecolor='white',edgecolor=s.BLUE,s=36);ax.scatter(b,y,color=s.BLUE,s=36)
        for v,dy in [(a,9),(b,9)]:ax.annotate(f'{v:.3f}',(v,y),xytext=(0,dy),textcoords='offset points',ha='center')
    ax.set_yticks([0,1],['Dapp1','Sleep']);ax.set_ylim(1.45,-.45);ax.set_xlim(-.25,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('Spearman ρ')
    ax.axvline(0,color=s.GRAY,lw=.7,ls='--');s.panel(ax,'b','Same monocyte lineage; open: granulocytic, filled: monocyte')
    ax=s.axes(fig,1.12,.70,5.10,1.05);rel=R4+'endpoint/associations.csv';summ=R4+'endpoint/summary.csv'
    combos=[('original','original'),('strict','original'),('original','marker_split'),('strict','marker_split')]
    for y,(definition,program) in enumerate(combos):
        for method,off in [('PCA30 proximity',-.23),('UMAP proximity',-.09),('Diffusion proximity',.06),('Palantir, local variant',.23)]:
            key=method.split()[0].rstrip(',').replace('PCA30','PCA');color,mark=s.METHODS[key]
            i=pick(summ,definition=definition,program=program,method=method)
            lo,hi,median=[cell(summ,i,col,'5','c') for col in ['minimum','maximum','median']]
            ax.plot([lo,hi],[y+off]*2,color=color,lw=1.0)
            if key=='Palantir':
                group=data(rel).loc[lambda d:d.definition.eq(definition)&d.program.eq(program)&d.method.eq(method)]
                for j,row in group.iterrows():
                    v=cell(rel,j,'spearman','5','c');seed=int(row.seed)
                    ax.scatter(v,y+off+(seed-1)*.035,facecolor='white',edgecolor=color,marker=s.SEEDS[seed][1],s=24,zorder=3)
                ax.scatter(median,y+off,color='black',marker='|',s=65,zorder=4)
            else:ax.scatter(median,y+off,color=color,marker=mark,s=26,zorder=3)
    ax.set_yticks(range(4),['87 / full','56 / full','87 / residual','56 / residual']);ax.set_ylim(3.6,-.6)
    ax.set_xlim(-.15,.26);ax.set_xticks([-.1,0,.1,.2]);ax.set_xlabel('Spearman ρ');ax.axvline(0,color=s.GRAY,lw=.7,ls='--')
    s.panel(ax,'c','Endpoint cells / evaluation program · fixed n = 8,298')
    # This figure uses full-width rows so ordinary type remains 10 pt.
    reference_handles=[Line2D([],[],color=c,marker=m,ls='',label=label) for label,(c,m) in
                       [('PCA30',s.METHODS['PCA']),('UMAP',s.METHODS['UMAP']),('Diffusion',s.METHODS['Diffusion'])]]
    seed_handles=[Line2D([],[],color='black',marker=m,mfc='white',ls='',label=f'Palantir {i}')
                  for i,(_,m) in enumerate(s.SEEDS)]
    fig.legend(handles=reference_handles+seed_handles,loc='lower center',bbox_to_anchor=(.54,-.005),
               frameon=False,ncol=6,columnspacing=.7,handletextpad=.3,handlelength=.7)
    emit(fig,MAIN[4])

def pipeline(plates):
    fig=plates['article_figure5_interfaces'];fig.set_size_inches(s.WIDTH,6.22);clear_figtext(fig);remove_letters(fig)
    for j,ax in enumerate(fig.axes[:3]):
        position(ax,fig,1.33+j*1.78,2.97,1.27,1.45);s.panel(ax,'bcd'[j],['All cells','Nonterminal','Selected start'][j])
    position(fig.axes[-1],fig,.22,5.00,5.97,.90)
    fig.text(.53,6.02/6.22,'Sensitivity to complete pipelines',ha='center',fontsize=s.TITLE)
    item=fig.text(.16,6.02/6.22,'a',fontsize=s.PANEL);item._panel_letter=True
    rel='source_data/results/recovered_integration_20261001/native_analysis/seed_stability.csv'
    ax=s.axes(fig,1.02,.75,5.0,1.25)
    for y,(c,lineage) in enumerate(f.TASKS):
        for arm,off in [('PCA',-.20),('LiVAE',.20)]:
            vals=vector(rel,'value','6','e',cohort=c,lineage=lineage,arm=arm,method='scRL',reward='neutral')
            ax.plot([min(vals),max(vals)],[y+off]*2,color=f.ARM_COLORS[arm],lw=.9)
            for k,v in enumerate(vals):ax.scatter(v,y+off+(k-1)*.09,color=f.ARM_COLORS[arm],marker=s.SEEDS[k][1],s=25,zorder=3)
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_yticks(range(6),f.TASK_LABELS);ax.set_ylim(5.6,-.6)
    ax.set_xlabel('Between-seed value Spearman ρ');s.panel(ax,'e','No-time scRL seed agreement',offset=0)
    fig.legend(handles=[Line2D([],[],color=s.BLUE,marker='s',label='PCA interface'),Line2D([],[],color=s.PURPLE,marker='o',label='LiVAE interface'),Line2D([],[],color=s.GRAY,marker='D',mfc='white',ls='',label='Proximity')],loc='center',bbox_to_anchor=(.54,2.43/6.22),frameon=False,ncol=3,columnspacing=.9,handlelength=.8,handletextpad=.35)
    fig.legend(handles=[Line2D([],[],color=s.GRAY,marker=m,ls='',label='Pair '+pair) for m,pair in zip(['o','s','^'],['0–1','0–2','1–2'])],loc='lower center',bbox_to_anchor=(.54,.005),frameon=False,ncol=3)
    emit(fig,MAIN[5])
    fig=plates['article_figure6_stability'];fig.delaxes(fig.axes[2]);clear_figtext(fig);remove_letters(fig);fig.set_size_inches(s.WIDTH,3.25)
    for j,ax in enumerate(fig.axes):
        position(ax,fig,1.02+j*2.80,.93,2.12,1.45)
        if ax.get_legend():ax.get_legend().remove()
        s.panel(ax,'ab'[j],['scFocus assignment ARI','scFocus weight entropy'][j])
    fig.legend(handles=[Line2D([],[],color=f.ARM_COLORS[a],marker='o',ls='',label=a) for a in f.ARMS],loc='center',bbox_to_anchor=(.54,.38/3.25),ncol=2,frameon=False)
    fig.legend(handles=[Line2D([],[],color=s.GRAY,marker=m,ls='',label='Pair '+['0–1','0–2','1–2'][i]+' / seed '+str(i)) for i,(_,m) in enumerate(s.SEEDS)],loc='lower center',bbox_to_anchor=(.54,.005),ncol=3,frameon=False)
    emit(fig,SUPPLEMENT[8])

def training():
    cps=[data(R4+f'training/seed{seed}/checkpoints.csv') for seed in range(3)]
    fig=plt.figure(figsize=(s.WIDTH,5.05))
    for j,(col,ylabel,title) in enumerate([('rho_all','Spearman ρ','All cells (n = 39,911)'),('rho_nonterminal','Spearman ρ','Nonterminal (n = 22,803)'),('common_start_return_mean','Mean return','Identical common starts')]):
        x,y=(.78,3.10) if j==0 else (3.98,3.10) if j==1 else (.78,.77)
        ax=s.axes(fig,x,y,2.03,1.35)
        vals=[vector(R4+f'training/seed{seed}/checkpoints.csv',col,'7','abc'[j]) for seed in range(3)]
        s.trajectory(ax,cps[0].episode,vals,ylabel,col.startswith('rho'));s.panel(ax,'abc'[j],title)
        if j==2:ax.set_yticks([0,5,10])
    ax=s.axes(fig,4.08,.77,2.05,1.35);rel=R4+'training/summary.csv';matrix=[]
    labels=[]
    for seed in range(3):
        for budget in [5000,10000]:
            i=pick(rel,seed=seed,episodes=budget)
            matrix.append([cell(rel,i,col,'7','d',population=pop) for col,pop in [('rho_all','all'),('rho_nonterminal','nonterminal'),('rho_starting','starting')]])
            labels.append(f'{seed} / {budget//1000}k')
    s.matrix(ax,matrix,labels,['All','Nonterm.','Start']);s.panel(ax,'d','Seed / budget · Spearman ρ',offset=-29)
    s.seed_legend(fig);emit(fig,MAIN[6])


def lineage_plot(fig,box,letter,historical=False,show_letter=True):
    ax=s.axes(fig,*box);rel=R3+'tables/trajectory_associations.csv' if historical else R4+'components/trajectory_associations.csv'
    order=[('ucb','megakaryocyte',1,'TPO Mk'),('dapp1','erythroid',8,'Dapp1 Ery'),('dapp1','neutrophil',6,'Dapp1 Neut.'),('dapp1','monocyte',5,'Dapp1 Mono.'),('sleep','monocyte',1,'Sleep Mono.'),('sleep','neutrophil',1,'Sleep Neut.')]
    # Resolve lineage indices by component. Sleep has two component evaluations
    # of the same one broad-target lineage, and both are retained.
    labels=[];counts=[];coverage=[]
    for y,(c,comp,_,label) in enumerate(order):
        matched=data(R4+'components/trajectory_associations.csv')
        sel=matched.loc[matched.cohort.eq(c)&matched.component.eq(comp)&matched.method.eq('Slingshot pseudotime')&matched.population.eq('shared_nonterminal')]
        assert len(sel)==1
        matchedrow=sel.iloc[0];lineage=matchedrow.lineage
        labels.append(label);counts.append(matchedrow.n_cells);coverage.append(matchedrow.coverage)
        if historical:
            names=['Slingshot pseudotime','PCA30 proximity','UMAP proximity','Diffusion proximity','scRL value']
            chosen=data(rel).loc[lambda d:d.cohort.eq(c)&d.lineage.eq(lineage)&d.population.eq('shared_nonterminal')]
        else:
            names=['Slingshot pseudotime','PCA30 component proximity','UMAP component proximity','Diffusion component proximity']
            chosen=data(rel).loc[lambda d:d.cohort.eq(c)&d.component.eq(comp)&d.population.eq('shared_nonterminal')]
        for k,method in enumerate(names):
            rows=chosen.loc[chosen.method.eq(method)];assert len(rows)>0,(c,comp,method)
            vals=[cell(rel,i,'spearman','S4' if historical else 'S5',letter,coverage=matchedrow.coverage) for i in rows.index]
            key=method.split()[0].rstrip(',').replace('PCA30','PCA');color,mark=s.METHODS[key]
            yy=y+(k-(len(names)-1)/2)*.13
            ax.plot([min(vals),max(vals)],[yy]*2,color=color,lw=.8)
            ax.scatter(np.median(vals),yy,color=color,marker=mark,s=27,zorder=3)
    n=len(labels);ax.set_ylim(n-.45,-.65);ax.set_yticks([]);s.correlation_axis(ax)
    title_y=ax.get_position().y1+.36/fig.get_figheight()
    fig.text(.52,title_y,'Slingshot and references · assigned nonterminal cells',ha='center',fontsize=s.TITLE)
    if show_letter:
        item=fig.text(.08/s.WIDTH,title_y,letter,fontsize=s.PANEL);item._panel_letter=True
    fig.canvas.draw();s.count_columns(fig,ax,labels,counts,coverage,x_name=.08,x_n=1.84,x_coverage=2.57)
    handles=[Line2D([],[],color=c,marker=m,ls='',label=key) for key,(c,m) in s.METHODS.items() if key in (['PCA','UMAP','Diffusion','Slingshot','scRL'] if historical else ['PCA','UMAP','Diffusion','Slingshot'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.003),ncol=len(handles),frameon=False,columnspacing=1,handlelength=1,handletextpad=.4)


def external_supplements():
    fig=plt.figure(figsize=(s.WIDTH,4.65));rel=R3+'tables/summary.csv'
    tasks=[('irradiation','erythroid','IR Ery'),('irradiation','myeloid','IR Myeloid'),('dapp1','erythroid','Dapp1 Ery'),('dapp1','myeloid','Dapp1 Myeloid')]
    methods=['scRL value','PCA30 proximity','UMAP proximity','Diffusion proximity','Palantir','CellRank 2','FateID Python'];matrix=[]
    for c,task,_ in tasks:
        matrix.append([cell(rel,pick(rel,cohort=c,task=task,method=m,population='shared_nonterminal'),'median','S4','a') for m in methods])
    ax=s.axes(fig,1.12,3.15,5.10,.95);s.matrix(ax,matrix,[t[2] for t in tasks],['scRL','PCA30','UMAP','Diffusion','Palantir\n1.4.2','CellRank\nDPT','FateID\nPython'])
    s.panel(ax,'a','Broad-target program associations',offset=-75)
    lineage_plot(fig,(2.85,.77,3.32,1.45),'b',historical=True);emit(fig,SUPPLEMENT[3])
    fig=plt.figure(figsize=(s.WIDTH,3.3));lineage_plot(fig,(2.85,.77,3.32,1.60),'a',historical=False,show_letter=False);emit(fig,SUPPLEMENT[4])


def training_supplement():
    fig=plt.figure(figsize=(s.WIDTH,8.5))
    for k,(column,title,ylabel) in enumerate([('native_return','Training return','250-episode mean'),('common_start_prediction_RMSE','Critic prediction error','RMSE'),('common_start_TD_RMSE','One-step TD error','RMSE'),('visitation','Cumulative mapped-grid coverage','Fraction visited'),('SD','Nonterminal projected-value scale','Population SD')]):
        x=.83+(k%2)*3.12;y=6.35-(k//2)*2.50
        ax=s.axes(fig,x,y,2.13,1.50)
        if k in [0,3]:
            rel='revision_data/structural_r5/'+('training_return_curve.csv.gz' if k==0 else 'training_grid_coverage.csv')
            d=data(rel)
            for seed,(color,marker) in enumerate(s.SEEDS):
                selected=d.loc[d.seed.eq(seed)]
                vals=vector(rel,'rolling250_return' if k==0 else 'fraction_visited','S6','a' if k==0 else 'd',seed=seed)
                ax.plot(selected.episode,vals,color=color,lw=.9,**({} if k==0 else dict(marker=marker,ms=3)))
            ax.axvline(5000,color=s.GRAY,ls=':',lw=.8);ax.set_xlim(0,10000);ax.set_xticks([0,5000,10000],['0','5,000','10,000']);ax.set_xlabel('Training episode');ax.set_ylabel(ylabel)
            if k==3:ax.set_ylim(0,1)
        else:
            rel='revision_data/structural_r5/value_scale.csv' if k==4 else None
            series=[vector(rel,'SD','S6','e',seed=seed,population='nonterminal') if k==4 else vector(R4+f'training/seed{seed}/checkpoints.csv',column,'S6','abcde'[k]) for seed in range(3)]
            s.trajectory(ax,np.arange(0,10001,500),series,ylabel)
        s.panel(ax,'abcde'[k],title)
    ax=s.axes(fig,3.95,1.35,2.13,1.50);rel=R4+'training/termination_summary.csv'
    reasons=['boundary','revisit','time_limit','invalid_move'];colors=[s.BLUE,s.ORANGE,s.GREEN,'#AAAAAA']
    bottom=np.zeros(6)
    for reason,color in zip(reasons,colors):
        vals=[cell(rel,pick(rel,seed=seed,episodes=ep,termination_reason=reason),'fraction','S6','f') for seed in range(3) for ep in [5000,10000]]
        ax.bar(range(6),vals,bottom=bottom,color=color,width=.7,label=reason.replace('_',' '));bottom+=vals
    np.testing.assert_allclose(bottom,1)
    ax.set_xticks(range(6),['0/5','0/10','1/5','1/10','2/5','2/10'],rotation=45,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Last-500 fraction');ax.set_xlabel('Seed / budget (k episodes)');s.panel(ax,'f','Episode termination')
    fig.legend(handles=[Line2D([],[],color=c,lw=5,label=r.replace('_',' ')) for c,r in zip(colors,reasons)],loc='lower center',bbox_to_anchor=(.54,-.002),ncol=4,frameon=False,columnspacing=.8,handlelength=.8,handletextpad=.4)
    s.seed_legend(fig,.055);emit(fig,SUPPLEMENT[5])


def partitions_supplement():
    fig=plt.figure(figsize=(s.WIDTH,4.45))
    for box,letter,c,p,title,show_y in [((1.06,2.25,1.52,1.25),'a','dapp1','neutrophil','Dapp1 gran.',True),
        ((2.91,2.25,1.52,1.25),'b','dapp1','cycling','Dapp1 cycling',False),
        ((4.76,2.25,1.45,1.25),'c','sleep','neutrophil','Sleep gran.',False)]:
        partition_panel(fig,box,letter,c,p,title,show_y)
    contrast_legend(fig,1.66)
    fig.text(.53,1.40/4.45,'Circles: cohort scaling; squares: WT/CT scaling. Coverage is per condition.',ha='center')
    rel=R4+'partitions/summary.csv'
    for cohort,label,shift in [('dapp1','Dapp1',0),('sleep','Sleep',3.25)]:
        fig.text((1.58+shift)/s.WIDTH,1.15/4.45,label+' eligible populations',ha='center')
        for x,heading,ha in [(.10,'Partition','left'),(1.35,'n','right'),(2.22,'WT/CT','right'),(3.03,'KO/SD','right')]:
            fig.text((x+shift)/s.WIDTH,.91/4.45,heading,ha=ha)
        for y,part,partition in [(.68,'historical','Reference'),(.45,'fresh_0.5','Leiden 0.5'),(.22,'fresh_1.5','Leiden 1.5')]:
            i=pick(rel,cohort=cohort,program='neutrophil',partition=part,scaling='cohort_reference')
            n,A,B=[cell(rel,i,col,'S7','metadata') for col in ['n_covered','coverage_A','coverage_B']]
            for x,value,ha in [(.10,partition,'left'),(1.35,f'{int(n):,}','right'),(2.22,f'{A:.2%}','right'),(3.03,f'{B:.2%}','right')]:
                fig.text((x+shift)/s.WIDTH,y/4.45,value,ha=ha)
    emit(fig,SUPPLEMENT[6])


def compact_existing(plates):
    fig=plates[MAIN[1]];fig.set_size_inches(s.WIDTH,5.60)
    for ax,box in zip(fig.axes,[(.889,4.00,2.191,1.08),(4.032,4.00,2.191,1.08),(.889,2.20,2.191,1.08),(4.032,2.20,2.191,1.08),(1.0795,.44,5.1435,1.00)]):
        position(ax,fig,*box)
    # Expose the sparse day-5 subtype counts at the expression panel itself.
    ax=fig.axes[2]
    rel='supplement_tables/s23_irradiation_subtype_programs.csv'
    counts=[]
    for label in ['HSC','MPP2/3','MPP4']:
        index=pick(rel,day=5,inherited_cell_type=label)
        counts.append(int(cell(rel,index,'n_cells','2','c',cohort='irradiation')))
    assert counts==[4,54,8]
    annotation=next(item for item in ax.texts if item.get_text()=='Day 5: 66 HSPC')
    annotation.set_text(f'Day 5: {counts[0]} HSC\n{counts[1]} MPP2/3; {counts[2]} MPP4')
    annotation.set_position((11,.59))
    annotation.set_verticalalignment('top')
    emit(fig,MAIN[1])
    fig=plates[MAIN[2]];old_height=fig.get_figheight();positions=[(x,x.get_position().bounds) for x in fig.axes]
    text_positions=[(x,x.get_position()) for x in fig.texts]
    fig.set_size_inches(s.WIDTH,6.08)
    for index,(ax,box) in enumerate(positions):
        x,y,w,h=box
        if index<3:position(ax,fig,x*s.WIDTH,3.46,w*s.WIDTH,1.10)
        elif index<6:position(ax,fig,x*s.WIDTH,.42,w*s.WIDTH,2.32)
        else:position(ax,fig,x*s.WIDTH,5.31,w*s.WIDTH,.50)
    for obj,(x,y) in text_positions:obj.set_position((x,(y*old_height-.36)/6.08))
    emit(fig,MAIN[2])
    fig=plates[SUPPLEMENT[1]];fig.set_size_inches(s.WIDTH,3.30)
    for ax in fig.axes:
        box=ax.get_position();position(ax,fig,box.x0*s.WIDTH,.98,box.width*s.WIDTH,1.75)
    handles=[handle for legend in fig.legends for handle in legend.legend_handles]
    labels=[item.get_text().replace('Revised LiVAE','LiVAE')
            for legend in fig.legends for item in legend.get_texts()]
    for legend in list(fig.legends):legend.remove()
    fig.legend(handles,labels,loc='center',bbox_to_anchor=(.54,.35/3.30),
               frameon=False,ncol=6,handlelength=.7,columnspacing=.8,handletextpad=.35)
    emit(fig,SUPPLEMENT[1])
    fig=plates[SUPPLEMENT[2]];fig.set_size_inches(s.WIDTH,8.20)
    for i,ax in enumerate(fig.axes):
        box=ax.get_position();position(ax,fig,box.x0*s.WIDTH,6.64-(i//2)*1.85,box.width*s.WIDTH,1.10)
    emit(fig,SUPPLEMENT[2])


def main():
    captured={};original_save=f.save;original_setup=f.setup
    def capture(fig,name):captured[name]=fig;original_save(fig,name)
    def setup():result=original_setup();s.configure();return result
    f.save=capture;f.setup=setup
    old.main()
    f.save=original_save;f.setup=original_setup;s.configure()
    SOURCES.update(old.SOURCES);VALUES.extend(old.VALUES)
    legacy=json.loads((E/'presentation_assets.json').read_text())
    # Recompose the native layouts while retaining canonical numerical bindings.
    composition(captured);fractions_supplement(captured);frozen(captured);components();pipeline(captured);training()
    external_supplements();training_supplement();partitions_supplement()
    compact_existing(captured)
    for name in ['scripts/build_figures.py','scripts/build_legacy_presentation_assets.py','scripts/publication_style.py',
                 'evidence/figure_values.csv']:
        SOURCES[name]=sha(P/name)
    with (E/'presentation_display_values.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(VALUES[0]));w.writeheader();w.writerows(VALUES)
    pd.DataFrame(RECORDS).to_csv(P/'revision_data/structural_r5/plot_records.csv',index=False)
    mapping=dict(scope='Current seven-main-figure sequence; original canonical numerical IDs remain frozen',row_ledger='evidence/figure_values.csv',additional_display_ledger='evidence/presentation_display_values.csv',record_ledger='revision_data/structural_r5/plot_records.csv',
        figures=[dict(article_figure=i,filename_stem=n,pdf='figures/'+n+'.pdf',pdf_sha256=sha(F/(n+'.pdf'))) for i,n in enumerate(MAIN,1)],
        supplementary_figures=[dict(figure='S'+str(i),filename_stem=n,pdf='figures/'+n+'.pdf',pdf_sha256=sha(F/(n+'.pdf'))) for i,n in enumerate(SUPPLEMENT,1)])
    (E/'figure_number_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    record=dict(schema_version=1,script_sha256=sha(__file__),policy='Presentation of saved outputs only; scale diagnostic is calculated separately.',main_figures=MAIN,supplementary_figures=SUPPLEMENT,canonical_ledger_values_checked=legacy['canonical_ledger_values_checked'],additional_display_values=len(VALUES),sources=SOURCES,qa=QA,
        outputs={str((F/(n+'.'+ext)).relative_to(P)):sha(F/(n+'.'+ext)) for n in MAIN+SUPPLEMENT for ext in ['pdf','png','svg']})
    for rel in ['evidence/presentation_display_values.csv','evidence/figure_number_mapping.json','revision_data/structural_r5/plot_records.csv']:record['outputs'][rel]=sha(P/rel)
    (E/'presentation_assets.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'pass','main_figures':len(MAIN),'supplementary_figures':len(SUPPLEMENT),'saved_value_records':len(VALUES),'identified_plot_records':len(RECORDS)}))

if __name__=='__main__':main()
