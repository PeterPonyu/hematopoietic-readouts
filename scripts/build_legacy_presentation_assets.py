#!/usr/bin/env python3
"""Compose publication displays without changing saved scientific calculations.

Canonical component figures and numerical ledger IDs remain stable. New panels
copy existing marker-sensitivity and annotation CSV cells; their display ledger
is separate. No expression scores, correlations, fits or tests are calculated.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/tmp/scientific-reports-matplotlib')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Arc
from matplotlib.transforms import Affine2D

import build_figures as f

P = Path(__file__).resolve().parents[1]
E, F, T = P/'evidence', P/'figures', P/'supplement_tables'
MAIN = ['article_figure1_composition', 'article_figure2_irradiation',
        'article_figure3_tpo', 'article_figure4_controls',
        'article_figure5_interfaces', 'article_figure6_stability']
SUPPLEMENT = ['article_supplementary_figure1_contrasts',
              'article_supplementary_figure2_rewards',
              'article_supplementary_figure3_optimization']
SOURCES, VALUES = {}, []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    path = T/(name+'.csv')
    SOURCES[str(path.relative_to(P))] = sha(path)
    with path.open() as stream:
        return [(i, row) for i, row in enumerate(csv.DictReader(stream), 1)]


def value(name, index, row, column, mark):
    result = float(row[column])
    VALUES.append({'display_mark': mark, 'source_file': 'supplement_tables/'+name+'.csv',
                   'source_row_1based': index, 'source_column': column,
                   'transform': 'identity', 'value': result})
    return result


def reset_labels(axes):
    for ax in axes:
        for obj in list(ax.texts):
            if getattr(obj, '_panel_letter', False):
                obj.remove()


def diagram_axis(fig, letter, heading, box, label_x):
    """Reserve a physical header above an original explanatory vector panel."""
    ax = fig.add_axes(box)
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    header = box[1] + box[3] + 3 / (72 * fig.get_figheight())
    fig.text(.53, header, heading, ha='center', va='bottom', fontsize=10.5)
    obj = fig.text(label_x, header, letter, ha='left', va='bottom', fontsize=13)
    obj._panel_letter = True
    return ax


def icon(ax, x, y, kind, color=f.BLUE, size=22):
    """Original vector glyphs in physical points, with no enclosing flow boxes."""
    box = ax.get_position(); fig = ax.figure
    t = (Affine2D().scale(size/(72*fig.get_figwidth()*box.width),
                          size/(72*fig.get_figheight()*box.height))
         .translate(x,y) + ax.transAxes)
    def line(xs,ys,c=color,**kw):
        ax.plot(xs,ys,transform=t,color=c,lw=1.1,clip_on=False,**kw)
    def dot(x,y,r=.11,c=color,fill='white'):
        ax.add_patch(Circle((x,y),r,transform=t,facecolor=fill,
                            edgecolor=c,lw=1.1,clip_on=False))
    if kind == 'cells':
        for u,v,c in [(-.30,.13,color),(.10,.28,color),(.32,-.15,f.ORANGE)]:
            dot(u,v,.23,c);dot(u+.025,v,.065,c,c)
    elif kind == 'balance':
        line([0,0],[-.43,.35]);line([-.48,.48],[.25,.25])
        line([-.22,.22],[-.43,-.43]);dot(0,.30,.055)
        for x0 in [-.36,.36]:
            line([x0-.14,x0,x0+.14],[.25,-.15,.25])
            line([x0-.17,x0+.17],[-.16,-.16]);dot(x0,-.06,.075)
    elif kind in ['marker','program']:
        line([-.45,-.15,.15,.45],[.16,-.16,.16,-.16])
        for u,v in [(-.45,.16),(-.15,-.16),(.15,.16),(.45,-.16)]:dot(u,v,.085)
        if kind=='marker':
            line([.0,.3],[-.04,.36],c=f.ORANGE)
            line([.0,.3],[.36,-.04],c=f.ORANGE)
    elif kind == 'standardize':
        line([-.48,.48],[0,0],c='#777777');line([0,0],[-.45,.45],c='#777777',ls='--')
        for u,v in [(-.32,.20),(-.16,-.18),(0,.30),(.16,-.30),(.32,.12)]:dot(u,v,.075)
    elif kind == 'cluster':
        points=[(-.36,.20),(-.16,.36),(-.12,.03),(.21,-.24),(.40,-.04),(.13,.08)]
        for a,b in [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(2,5)]:
            line([points[a][0],points[b][0]],[points[a][1],points[b][1]],c='#9DA5AC')
        for i,(u,v) in enumerate(points):dot(u,v,.10,color if i<3 else f.ORANGE)
    elif kind == 'lock':
        ax.add_patch(Arc((0,.13),.50,.57,theta1=0,theta2=180,transform=t,
                         edgecolor=color,lw=1.2,clip_on=False))
        ax.add_patch(FancyBboxPatch((-.37,-.38),.74,.57,
                     boxstyle='round,pad=0,rounding_size=.08',transform=t,
                     facecolor='white',edgecolor=color,lw=1.1,clip_on=False))
        dot(0,-.06,.065);line([0,0],[-.10,-.24])
    elif kind == 'calendar':
        ax.add_patch(Rectangle((-.43,-.41),.86,.76,transform=t,
                     facecolor='white',edgecolor=color,lw=1.1,clip_on=False))
        line([-.43,.43],[.12,.12]);line([-.23,-.23],[.25,.47]);line([.23,.23],[.25,.47])
        for u,v in [(-.24,-.08),(0,-.08),(.24,-.08),(-.24,-.28),(0,-.28),(.24,-.28)]:dot(u,v,.045,color,color)
    elif kind in ['endpoints','umap']:
        for u,v in [(-.40,-.28),(-.18,-.17),(-.12,.12),(.12,.05),(.29,.26),(.42,.13)]:dot(u,v,.065)
        if kind == 'endpoints':dot(.34,.20,.23,f.ORANGE)
    elif kind == 'proximity':
        dot(.24,.10,.26,f.ORANGE);dot(.24,.10,.06,f.ORANGE,f.ORANGE)
        for u,v in [(-.42,-.29),(-.33,.25),(-.12,-.12)]:
            dot(u,v,.065);line([u,.24],[v,.10],c='#9DA5AC',ls=':')
    elif kind == 'critic':
        layers=[(-.42,[-.25,.25]),(0,[-.40,0,.40]),(.42,[-.20,.20])]
        for (u,ys),(v,zs) in zip(layers,layers[1:]):
            for yy in ys:
                for zz in zs:line([u,v],[yy,zz],c='#A9B5BE')
        for u,ys in layers:
            for v in ys:dot(u,v,.08)
    elif kind == 'correlation':
        line([-.40,-.40,.45],[.43,-.40,-.40],c='#777777')
        line([-.32,.38],[-.27,.32])
        for u,v in [(-.29,-.18),(-.12,-.21),(.06,.11),(.22,.06),(.35,.36)]:dot(u,v,.055)
    elif kind == 'pca':
        line([-.42,.42],[-.35,.35]);line([-.34,.34],[.38,-.38],c='#9DA5AC')
        for u,v in [(-.25,-.08),(-.12,-.22),(.09,.18),(.26,.14)]:dot(u,v,.06)
    elif kind == 'latent':
        for r in [.18,.39]:
            ax.add_patch(Arc((0,0),2*r,1.2*r,theta1=0,theta2=360,
                         transform=t,edgecolor=color,lw=1.1,clip_on=False))
        line([-.38,0,.38],[.05,-.38,.05]);line([-.30,0,.30],[.20,.39,.20])
        for u,v in [(-.28,.0),(.14,.14),(.04,-.25)]:dot(u,v,.065)
    elif kind == 'branches':
        line([-.38,-.08,.32],[-.35,.0,.32]);line([-.08,.37],[0,-.09]);line([-.08,.13],[0,-.38])
        for u,v in [(-.38,-.35),(.32,.32),(.37,-.09),(.13,-.38)]:dot(u,v,.065)
    else:raise ValueError(kind)


def arrow(ax, start, end, dashed=False):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle':'-|>', 'color':'#777777', 'lw':1.0,
                            'linestyle':'--' if dashed else '-',
                            'shrinkA':2, 'shrinkB':2})


def glyph_label(ax, x, kind, text, y=.70, label_y=.13, color=f.BLUE, size=22):
    icon(ax,x,y,kind,color,size)
    ax.text(x,label_y,text,ha='center',va='center',fontsize=f.MIN_FONT,linespacing=1.05)


def composition_logic(fig):
    ax=diagram_axis(fig,'a','Composition and marker controls',[.045,5.75/6.55,.93,.56/6.55],
                    label_x=.17-22/(72*f.WIDTH))
    for x,kind,text in [(.14,'cells','Observed mixture'),(.50,'balance','Common weights'),(.85,'marker','Marker omission')]:
        glyph_label(ax,x,kind,text)
    arrow(ax,(.24,.70),(.40,.70));arrow(ax,(.60,.70),(.75,.70))


def annotation_logic(fig):
    height=fig.get_figheight()
    ax=diagram_axis(fig,'a','Pooled labels and day-specific profiles',[.055,5.67/height,.91,.50/height],
                    label_x=.115-22/(72*f.WIDTH))
    for x,kind,text in [(.10,'standardize','Pooled z scores'),(.36,'cluster','Cluster maximum'),
                        (.63,'lock','Fix labels'),(.90,'calendar','Day profiles')]:
        glyph_label(ax,x,kind,text,size=20)
    for start,end in [(.17,.28),(.44,.55),(.71,.82)]:arrow(ax,(start,.70),(end,.70))


def evaluation_logic(fig):
    ax=diagram_axis(fig,'a','Shared information in program-labelled tasks',[.045,4.93/5.90,.93,.57/5.90],
                    label_x=.15-22/(72*f.WIDTH))
    items=[(.10,'program','Program'),(.36,'endpoints','Endpoints'),
           (.64,'balance','Geometry / scRL'),(.91,'correlation','Association')]
    for x,kind,text in items:glyph_label(ax,x,kind,text,size=20)
    for start,end in [(.16,.29),(.43,.57),(.71,.84)]:arrow(ax,(start,.70),(end,.70))
    ax.plot([.07,.025,.025,1.01,1.01],[.70,.70,-.15,-.15,.70],color='#9DA5AC',ls='--',lw=1.0,clip_on=False)
    arrow(ax,(1.01,.70),(.95,.70),dashed=True)


def interface_logic(fig):
    height=4.25
    ax=diagram_axis(fig,'a','Matched two-dimensional interfaces',[.035,2.62/height,.94,1.22/height],
                    label_x=.21-22/(72*f.WIDTH))
    glyph_label(ax,.075,'cells','Same cells\nand labels',y=.57,label_y=.23,size=20)
    for y,kind,text in [(.85,'pca','Full-axis PCA30'),(.34,'latent','Selected-axis LiVAE10')]:
        glyph_label(ax,.345,kind,text,y=y,label_y=y-.23,size=19)
        glyph_label(ax,.62,'umap','UMAP2',y=y,label_y=y-.23,size=19)
        arrow(ax,(.14,.60 if y>.5 else .47),(.26,y))
        arrow(ax,(.43,y),(.54,y))
        ax.plot([.69,.75,.75],[y,y,.59],color='#777777',lw=1.0)
    for y,kind,text in [(.85,'critic','scRL: 2D state/grid'),(.34,'branches','scFocus: 2D input')]:
        glyph_label(ax,.90,kind,text,y=y,label_y=y-.23,size=19)
        arrow(ax,(.75,.59),(.83,y))



def composition(captured, export):
    f.fig2()
    fig = captured.pop('figure2_composition')
    fig.set_size_inches(f.WIDTH, 6.55)
    axes = fig.axes[:4]
    reset_labels(axes)
    for ax, box, letter in zip(axes, [[.17,4.10/6.55,.34,1.05/6.55], [.64,4.10/6.55,.34,1.05/6.55],
                              [.17,2.37/6.55,.34,.90/6.55], [.64,2.37/6.55,.34,.90/6.55]], 'bcde'):
        ax.set_position(box)
        f.title(ax, letter, ax.get_title())
    for ax in axes[:2]:
        for obj in ax.texts:
            if getattr(obj,'_panel_letter',False):obj.set_position((-22,20))
    composition_logic(fig)
    for legend in list(fig.legends):
        legend.remove()
    raw = Line2D([], [], marker='o', markerfacecolor='white', color=f.GRAY, ls='', label='Unadjusted')
    fixed = Line2D([], [], marker='o', color=f.BLUE, ls='', label='Fixed composition')
    fig.legend(handles=[raw, fixed], loc='center', ncol=2, frameon=False,
               bbox_to_anchor=(.55,1.80/6.55))
    ax = fig.add_axes([.31,.72/6.55,.66,.72/6.55])
    groups = [('dapp1','neutrophil','Dapp1 granulocytic'),
              ('dapp1','cycling','Dapp1 cycling'),
              ('sleep','neutrophil','Sleep granulocytic')]
    rows = read('s20_marker_sensitivity')
    for y, (cohort, program, _) in enumerate(groups):
        omissions = []
        for i, row in rows:
            if row['cohort'] != cohort or row['program'] != program:
                continue
            if row['analysis_type'] == 'full_program':
                x = value('s20_marker_sensitivity', i, row, 'composition_standardized_delta',
                          f'Figure1f:{cohort}:{program}:full')
                ax.scatter(x, y, marker='D', s=35, facecolors='white', edgecolors='black', zorder=4)
            elif row['analysis_type'] == 'leave_one_marker_out':
                omissions.append((i, row))
        if len(omissions) != 6:
            raise ValueError(('Expected six saved omissions', cohort, program, len(omissions)))
        for offset, (i, row) in zip(np.linspace(-.22,.22,6), omissions):
            x = value('s20_marker_sensitivity', i, row, 'composition_standardized_delta',
                      f'Figure1f:{cohort}:{program}:omit:{row["marker"]}')
            special = cohort == 'sleep' and row['marker'].lower() == 'mpo'
            ax.scatter(x, y+offset, color=f.ORANGE if special else f.BLUE,
                       s=23, edgecolor='white', linewidth=.4, zorder=3)
    ax.axvline(0, color=f.GRAY, ls='--', lw=.7)
    ax.set_xlim(-.27,.10); ax.set_xticks([-.2,-.1,0,.1])
    ax.set_yticks(range(3), [g[2] for g in groups]); ax.set_ylim(2.55,-.55)
    ax.set_xlabel('Fixed-composition difference'); f.clean(ax, 'x')
    f.title(ax, 'f', 'Single-marker omission');ax.set_title('Single-marker omission',pad=8)
    fig.legend(handles=[Line2D([],[],marker='D',markerfacecolor='white',color='black',ls='',label='Full program'),
                        Line2D([],[],marker='o',color=f.BLUE,ls='',label='Omit one marker'),
                        Line2D([],[],marker='o',color=f.ORANGE,ls='',label='Sleep: omit Mpo')],
               loc='lower center', bbox_to_anchor=(.54,-.004), ncol=3, frameon=False,
               columnspacing=1.0, handletextpad=.45)
    export(fig, MAIN[0])


def tpo(captured, export):
    f.fig3()
    fig = captured.pop('figure3_tpo')
    height=6.45
    fig.set_size_inches(f.WIDTH, height)
    axes = fig.axes[:4]
    reset_labels(axes)
    for obj in list(fig.texts):
        obj.remove()
    for legend in list(fig.legends):
        legend.remove()
    boxes = [[.115,3.65/height,.205,1.15/height], [.435,3.65/height,.205,1.15/height],
             [.755,3.65/height,.205,1.15/height], [.115,.49/height,.205,2.32/height]]
    titles = ['Label fractions', 'Megakaryocyte program',
              'Component genes', 'No-time value']
    for i,(ax,box,title) in enumerate(zip(axes, boxes, titles)):
        ax.set_position(box); f.title(ax, 'bcde'[i], title)
        ax.set_xticks([0,7,14])
        if i<3:
            ax.set_title(title,pad=38)
    # Keep the visible gene-expression ticks explicit. Autoscaling also creates
    # off-frame tick labels that Matplotlib does not draw but Text.findobj sees.
    axes[2].set_yticks([0,1,2,3])
    genes, gene_names = axes[2].get_legend_handles_labels()
    axes[2].get_legend().remove()
    axes[2].legend(genes, gene_names, loc='lower center', bbox_to_anchor=(.5,1.03),
                   ncol=2, frameon=False, handlelength=.8, handletextpad=.35,
                   columnspacing=.7,
                   borderaxespad=0, labelspacing=.2)
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, ['Prim.','Mk','Ery','Other'],
                   loc='lower center', bbox_to_anchor=(.5,1.03), ncol=2,
                   frameon=False, columnspacing=.7, handletextpad=.35,
                   handlelength=1.0, borderaxespad=0, labelspacing=.2)
    handles,names=axes[1].get_legend_handles_labels()
    axes[1].legend(handles,names,loc='lower center',bbox_to_anchor=(.5,1.03),
                   ncol=1,frameon=False,handlelength=1.0,handletextpad=.4,
                   borderaxespad=0,labelspacing=.2)
    # These two panels previously appeared below the group-contrast supplement.
    ax = fig.add_axes([.435,.49/height,.205,2.32/height])
    rows = sorted(read('s15_tpo_cluster_composition'), key=lambda item:int(item[1]['cluster']))
    counts=[]
    for i,row in rows:
        counts.append(value('s15_tpo_cluster_composition',i,row,'n_D0',f'Figure3f:cluster:{row["cluster"]}'))
    colors=[f.ORANGE if int(row['cluster'])==5 else f.BLUE for _,row in rows]
    ax.barh(range(len(rows)),counts,color=colors,height=.68)
    ax.set_yticks(range(len(rows)),[row['cluster'] for _,row in rows])
    ax.set_ylim(len(rows)-.4,-.6); ax.set_xlim(0,4200)
    ax.set_xticks([0,3000],['0','3,000'])
    for y,n in enumerate(counts):
        if int(rows[y][1]['cluster']) == 5:
            ax.scatter(n,y,s=20,color=f.ORANGE,edgecolors='none',zorder=3)
        if n>0:ax.text(n+65,y,f'{int(n):,}',va='center',fontsize=f.MIN_FONT)
    ax.set_xlabel('D0 cells'); ax.set_ylabel('Cluster',labelpad=2)
    f.clean(ax,'x');f.title(ax,'f','D0 clusters')
    ax = fig.add_axes([.755,.49/height,.205,2.32/height])
    annotation=read('s17_annotation_diagnostics');data=[]
    scopes=['pooled','D0','D4','D7','D11','D14']
    for scope in scopes:
        chosen=[(i,r) for i,r in annotation if r['cohort']=='ucb' and r['scope']==scope]
        if len(chosen)!=1:raise ValueError(('TPO scope',scope,len(chosen)))
        i,row=chosen[0]
        data.append([value('s17_annotation_diagnostics',i,row,'program_'+program,
                           f'Figure3g:{scope}:{program}') for program in ['primitive','neutrophil','dendritic']])
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        'score_with_black_text',plt.get_cmap('RdBu_r')(np.linspace(.20,.80,256)))
    matrix=np.asarray(data)
    ax.pcolormesh(range(3),range(6),matrix,cmap=cmap,vmin=-1.1,vmax=1.1,
                  shading='nearest',edgecolors='none',rasterized=False)
    ax.set_xlim(-.5,2.5);ax.set_ylim(5.5,-.5)
    ax.set_yticks(range(6),['Pooled','D0','D4','D7','D11','D14'])
    ax.set_xticks(range(3),['Prim.','Gran.','Dendr.'])
    for y in range(6):
        for x in range(3):ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',fontsize=f.MIN_FONT)
    f.title(ax,'g','Cluster 5 scores')
    for index,axis in enumerate(fig.axes):
        for obj in axis.texts:
            if getattr(obj,'_panel_letter',False):obj.set_position((-22,38 if index<3 else 13))
    annotation_logic(fig)
    export(fig,MAIN[2])


def controls(captured, export):
    f.fig4(); fig = captured.pop('figure4_controls')
    fig.set_size_inches(f.WIDTH, 5.90)
    for obj in list(fig.texts): obj.remove()
    for legend in list(fig.legends): legend.remove()
    reset_labels(fig.axes)
    boxes = [[.15,3.10/5.90,.37,.80/5.90], [.61,3.10/5.90,.37,.80/5.90],
             [.15,.82/5.90,.235,1.35/5.90], [.445,.82/5.90,.235,1.35/5.90], [.745,.82/5.90,.235,1.35/5.90]]
    for ax, box, letter in zip(fig.axes, boxes, 'bcdef'):
        ax.set_position(box); f.title(ax, letter, ax.get_title())
    for ax in fig.axes[:2]: ax.set_yticks([0,.25,.5,.75])
    for ax,title in zip(fig.axes[:2],['Marrow erythroid','Marrow myeloid']): ax.set_title(title)
    fig.legend(handles=[Line2D([],[],color=f.GRAY,marker=m,ls='',label=f'Seed {i}')
                        for i,m in enumerate(['o','s','^'])],
               loc='center',ncol=3,frameon=False,bbox_to_anchor=(.60,4.45/5.90),
               columnspacing=1.4,handletextpad=.4)
    fig.legend(handles=[Line2D([],[],color=f.BLUE,marker='o',label='scRL median / seed range'),
                        Line2D([],[],ls='',marker='s',color=f.ORANGE,label='UMAP proximity'),
                        Line2D([],[],ls='',marker='^',color=f.GREEN,label='PCA30 proximity')],
               loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.55,0),
               columnspacing=1.0,handlelength=1.4)
    evaluation_logic(fig)
    export(fig, MAIN[3])


def interfaces(export):
    fig=plt.figure(figsize=(f.WIDTH,4.25))
    for i,pop in enumerate(['all','nonterminal','starting']):
        ax=fig.add_axes([.21+i*.28,.72/4.25,.20,1.41/4.25])
        f.correlation_panel(ax,5,'abc'[i],pop,matched=True,show_y=i==0,details=False)
        f.title(ax,'bcd'[i],['All cells','Nonterminal','Selected start'][i])
    interface_logic(fig)
    fig.legend(handles=[Line2D([],[],color=f.BLUE,marker='s',label='PCA'),
                        Line2D([],[],color=f.ARM_COLORS['LiVAE'],marker='o',label='LiVAE'),
                        Line2D([],[],color=f.GRAY,marker='D',markerfacecolor='white',ls='',label='Interface proximity')],
               loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.54,-.002),
               columnspacing=1.0,handlelength=1.2,handletextpad=.4)
    export(fig,MAIN[4])


def irradiation(captured,export):
    f.fig7();fig=captured.pop('figure7_irradiation')
    fig.set_size_inches(f.WIDTH,6.10)
    axes=fig.axes[:4];reset_labels(axes)
    for ax in fig.axes[4:]:fig.delaxes(ax)
    for ax,box,letter in zip(axes,[[.14,4.37/6.10,.345,1.08/6.10],[.635,4.37/6.10,.345,1.08/6.10],
                                  [.14,2.38/6.10,.345,1.08/6.10],[.635,2.38/6.10,.345,1.08/6.10]],'abcd'):
        ax.set_position(box);f.title(ax,letter,ax.get_title())
    axes[0].set_yticks([0,20,40,60,80]);axes[2].set_yticks([0,.5,1.0])
    for obj in axes[2].texts:
        if obj.get_text()=='Day 5: 66 HSPC':obj.set_position((12,.27))
    ax=fig.add_axes([.17,.44/6.10,.81,1.10/6.10])
    days=[0,2,5,8,11,14,21,30]
    rows=[('captured n','Captured',f.GRAY),('HSPC n','HSPC',f.BLUE),('Ery n','Ery',f.ORANGE)]
    counts=[]
    for suffix,label,color in rows:
        counts.append([next(q['value'] for q in f.LEDGER if q['figure']==7 and q['mark']==f'd{day}:{suffix}') for day in days])
    maximum=max(max(row) for row in counts)
    for y,((_,label,color),values) in enumerate(zip(rows,counts)):
        ax.scatter(range(8),[y]*8,s=np.asarray(values)/maximum*100,color=color,alpha=.85,edgecolors='none')
        for x,n in enumerate(values):ax.annotate(f'{int(n):,}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',va='bottom',fontsize=f.MIN_FONT)
    ax.set_xlim(-.55,7.55);ax.set_ylim(2.40,-.60)
    ax.set_yticks(range(3),[q[1] for q in rows]);ax.set_xticks(range(8),[str(x) for x in days])
    ax.set_xlabel('Sampling day (categorical columns)',labelpad=2)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.tick_params(length=0,pad=3)
    f.title(ax,'e','Retained counts · area scales with cell number')
    export(fig,MAIN[1])


def stability(captured,export):
    f.fig6();fig=captured.pop('figure6_stability');fig.set_size_inches(f.WIDTH,5.45)
    axes=fig.axes[:3];reset_labels(axes)
    for ax,box,letter in zip(axes,[[.16,3.45/5.45,.345,1.20/5.45],
                                  [.60,3.45/5.45,.345,1.20/5.45],
                                  [.16,.93/5.45,.79,1.47/5.45]],'abc'):
        ax.set_position(box);f.title(ax,letter,ax.get_title())
    axes[1].get_legend().set_bbox_to_anchor((.5,-.40))
    for legend in list(fig.legends):legend.remove()
    fig.legend(handles=[Line2D([],[],color=f.ARM_COLORS[x],marker='o',ls='',label='LiVAE' if x=='LiVAE' else x) for x in f.ARMS],
               loc='center',ncol=2,frameon=False,bbox_to_anchor=(.55,.36/5.45))
    fig.legend(handles=[Line2D([],[],color=f.GRAY,marker=m,ls='',label=f'Pair {pair}') for m,pair in zip(['o','s','^'],['0–1','0–2','1–2'])],
               loc='center',ncol=3,frameon=False,bbox_to_anchor=(.55,.10/5.45),columnspacing=1.1,handletextpad=.45)
    export(fig,MAIN[5])


def supplements(export):
    import build_interpretation_assets as q
    captured={}
    class Captured(Exception):pass
    def capture(fig,name):
        captured[name]=fig
        raise Captured()
    original=q.save;q.save=capture
    try:
        for function,name in [(q.supplementary_figure,'supplementary_figure1_diagnostics'),
                              (q.reward_comparison,'supplementary_figure2_rewards')]:
            try:function()
            except Captured:pass
            fig=captured.pop(name)
            for obj in list(fig.texts):obj.remove()
            if name=='supplementary_figure1_diagnostics':
                for ax in fig.axes[3:]:fig.delaxes(ax)
                fig.set_size_inches(f.WIDTH,3.72)
                for ax,letter in zip(fig.axes,'abc'):
                    old=ax.get_position();ax.set_position([old.x0,.48/3.72,old.width,2.71/3.72]);f.title(ax,letter,ax.get_title())
                export(fig,SUPPLEMENT[0])
            else:
                fig.set_size_inches(f.WIDTH,4.15)
                for ax,letter in zip(fig.axes,'abc'):
                    old=ax.get_position();ax.set_position([old.x0,.95/4.15,old.width,2.65/4.15]);f.title(ax,letter,ax.get_title())
                for legend,y in zip(fig.legends,[.32/4.15,.055/4.15]):legend.set_bbox_to_anchor((.565,y))
                export(fig,SUPPLEMENT[1])
    finally:q.save=original
    # The supplemental renderer's existing exact source-cell ledger remains
    # authoritative; drawing it again never reaches its table-writing code.
    SOURCES.update(q.SOURCES)
    for name in ['scripts/build_interpretation_assets.py','evidence/interpretation_display_values.csv']:
        path=P/name
        if path.exists():SOURCES[name]=sha(path)



def captions(text):
    result=[];offset=0
    while True:
        start=text.find(r'\caption{',offset)
        if start<0:return result
        body=start+len(r'\caption{');depth=1;end=body
        while depth:
            if text[end]=='{':depth+=1
            elif text[end]=='}':depth-=1
            end+=1
        result.append(text[body:end-1]);offset=end


def optimization(export):
    """Display retained epoch records; do not refit or smooth an objective."""
    fig=plt.figure(figsize=(f.WIDTH,7.55))
    cohorts=[('irradiation','Irradiation'),('ucb','TPO'),('dapp1','Dapp1'),('sleep','Sleep')]
    colors=[f.BLUE,f.ORANGE,'#76559A','#168578']
    columns=['training_loss','validation_loss','validation_KL','validation_unweighted_geometry']
    for row,(cohort,label) in enumerate(cohorts):
        relative='revision_data/review_support/'+cohort+'_training_history.csv'
        path=P/relative;SOURCES[relative]=sha(path)
        with path.open() as stream:records=list(csv.DictReader(stream))
        data={key:[] for key in ['epoch']+columns}
        for index,record in enumerate(records,1):
            for column in data:
                result=float(record[column]);data[column].append(result)
                VALUES.append({'display_mark':f'S3:{cohort}:epoch{index}:{column}',
                    'source_file':relative,'source_row_1based':index,'source_column':column,
                    'transform':'identity','value':result})
        bottom=.80-row*.225
        left=fig.add_axes([.14,bottom,.32,.135])
        right=fig.add_axes([.64,bottom,.32,.135])
        for column,color in zip(columns[:2],colors[:2]):
            left.plot(data['epoch'],data[column],color=color,lw=1.25)
        for column,color in zip(columns[2:],colors[2:]):
            right.plot(data['epoch'][-40:],data[column][-40:],color=color,lw=1.25)
        left.set_xlim(1,200);left.set_xticks([1,100,200]);left.set_ylabel('Objective',labelpad=2)
        left.set_yticks({'irradiation':[1700,2700,3700],'ucb':[1400,2000,2600],
                         'dapp1':[3500,5000,6500],'sleep':[1000,1400,1800]}[cohort])
        right.set_xlim(161,200);right.set_xticks([170,185,200]);right.set_ylabel('Component',labelpad=2)
        right.set_yticks([5,15])
        for ax in [left,right]:
            ax.set_xlabel('Epoch',labelpad=1);f.clean(ax)
        f.title(left,'abcdefgh'[2*row],label)
        f.title(right,'abcdefgh'[2*row+1],'Late components')
        for ax in [left,right]:
            for text in ax.texts:
                if getattr(text,'_panel_letter',False):text.set_position((-40,13))
    fig.legend(handles=[Line2D([],[],color=color,lw=1.3,label=label) for color,label in
               zip(colors,['Training','Validation','KL','Geometry (unweighted)'])],
               loc='lower center',bbox_to_anchor=(.54,.005),frameon=False,ncol=2,
               columnspacing=1.2,handlelength=1.3,labelspacing=.25)
    export(fig,SUPPLEMENT[2])


def main():
    captured={};original_export=f.save
    def export(fig,name):
        original_export(fig,name)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        texts=[obj for obj in fig.findobj(matplotlib.text.Text) if obj.get_visible() and obj.get_text()]
        # Annotation bounds include the arrow; compare glyph bounds only.
        overlaps=[]
        for i,item in enumerate(texts):
            a=matplotlib.text.Text.get_window_extent(item,renderer)
            for other in texts[i+1:]:
                b=matplotlib.text.Text.get_window_extent(other,renderer)
                if min(a.x1,b.x1)-max(a.x0,b.x0)>.5 and min(a.y1,b.y1)-max(a.y0,b.y0)>.5:
                    overlaps.append([item.get_text(),other.get_text()])
        f.QA[-1]['overlapping_text']=overlaps
        if overlaps:raise ValueError(('Overlapping publication text',name,overlaps))
    f.save=lambda fig,name:captured.__setitem__(name,fig)
    with tempfile.TemporaryDirectory(prefix='hematopoietic-presentation-') as temp:
        f.EVIDENCE=Path(temp);f.OUT=F;f.setup()
        f.fig1()
        plt.close(captured.pop('figure1_design'))
        composition(captured,export);irradiation(captured,export);tpo(captured,export);controls(captured,export);interfaces(export);stability(captured,export);supplements(export);optimization(export)
    reference={}
    with (E/'figure_values.csv').open() as stream:
        for row in csv.DictReader(stream):
            reference[(row['figure'],row['panel'],row['mark'])]=row
    for row in f.LEDGER:
        key=(str(row['figure']),row['panel'],row['mark'])
        if key not in reference or not np.isclose(float(reference[key]['value']),row['value'],rtol=0,atol=1e-12):
            raise ValueError(('Canonical display value changed',key))
    for path in [P/'scripts/build_figures.py',E/'figure_values.csv']:
        SOURCES[str(path.relative_to(P))]=sha(path)
    ledger=E/'presentation_display_values.csv'
    with ledger.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(VALUES[0]));writer.writeheader();writer.writerows(VALUES)
    mapping={'scope':'Publication numbering mapped to unchanged canonical numerical identifiers',
             'row_ledger':'evidence/figure_values.csv',
             'additional_display_ledger':'evidence/presentation_display_values.csv',
             'panel_mapping':{'1':{'a':'original composition schematic', 'b':'2a', 'c':'2b',
                                  'd':'2c', 'e':'2d', 'f':'marker sensitivity'},
                              '2':{'a':'7a', 'b':'7b', 'c':'7c', 'd':'7d', 'e':'7ab/7ac/7b retained counts'},
                              '3':{'a':'original annotation schematic', 'b':'3a', 'c':'3b',
                                   'd':'3c', 'e':'3d', 'f':'D0 composition', 'g':'annotation scores'},
                              '4':{'a':'original shared-information schematic', 'b':'4a', 'c':'4b',
                                   'd':'4c', 'e':'4d', 'f':'4e'},
                              '5':{'a':'original paired-interface schematic', 'b':'5a', 'c':'5b', 'd':'5c'},
                              '6':{'a':'6a', 'b':'6b', 'c':'6c'}},
             'figures':[{'article_figure':i,'filename_stem':name,'renderer_internal_figure_id':internal,
                         'figure_values_csv_selector':{'figure':internal},'pdf':'figures/'+name+'.pdf',
                         'pdf_sha256':sha(F/(name+'.pdf'))}
                        for i,(name,internal) in enumerate(zip(MAIN,[2,7,3,4,5,6]),1)],
             'supplementary_figures':[{'figure':'S'+str(i),'filename_stem':name,'pdf':'figures/'+name+'.pdf',
                                      'pdf_sha256':sha(F/(name+'.pdf'))} for i,name in enumerate(SUPPLEMENT,1)]}
    (E/'figure_number_mapping.json').write_text(json.dumps(mapping,indent=2)+'\n')
    record={'schema_version':1,'script_sha256':sha(Path(__file__)),
            'policy':'Saved-value display and vector composition only; no new scientific estimates.',
            'main_figures':MAIN,'supplementary_figures':SUPPLEMENT,
            'canonical_ledger_values_checked':len(f.LEDGER),'additional_display_values':len(VALUES),
            'sources':SOURCES,'qa':f.QA,
            'outputs':{str(p.relative_to(P)):sha(p) for name in MAIN+SUPPLEMENT
                       for ext in ['pdf','png','svg'] for p in [F/(name+'.'+ext)]}}
    record['outputs'][str(ledger.relative_to(P))]=sha(ledger)
    for name in ['figure_number_mapping.json']:
        record['outputs']['evidence/'+name]=sha(E/name)
    (E/'presentation_assets.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'main_figures':len(MAIN),'supplementary_figures':len(SUPPLEMENT),
                      'canonical_values_checked':len(f.LEDGER),'additional_display_values':len(VALUES)}))


if __name__=='__main__':
    main()
