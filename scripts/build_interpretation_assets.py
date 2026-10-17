#!/usr/bin/env python3
"""Display saved cluster contrasts, annotation competition and task diagnostics.

Reads only the already generated supplementary CSVs. No expression scoring,
correlation, model fitting, relabelling or inferential analysis is performed.
Every displayed number is linked to a source row and column in a separate ledger.
"""
from pathlib import Path
import csv
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/scientific-reports-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
T = P/'supplement_tables'
E = P/'evidence'
F = P/'figures'
SOURCES, LEDGER, QA = {}, [], []
WIDTH = 6.35
MIN_FONT = 9.5
BLUE, ORANGE, GRAY, INK = '#17689A', '#C6602A', '#657480', '#203440'
TASKS = [('ucb', 'megakaryocyte', 'TPO Mk'), ('dapp1', 'erythroid', 'Dapp1 Ery'),
         ('dapp1', 'myeloid', 'Dapp1 My'), ('sleep', 'myeloid', 'Sleep My')]


def load(name):
    path = T/(name+'.csv')
    SOURCES[str(path.relative_to(P))] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = pd.read_csv(path, keep_default_na=False)
    data.index = range(1, len(data)+1)
    return data


def value(name, data, column, mark, transform='identity'):
    values = data[column].astype(float)
    if transform == 'identity':
        assert len(values) == 1, (name, mark, len(values))
        result = values.iloc[0]
    elif transform == 'min':
        result = values.min()
    elif transform == 'max':
        result = values.max()
    elif transform == 'median':
        result = values.median()
    else:
        raise ValueError(transform)
    LEDGER.append({'display_mark': mark, 'source_file': 'supplement_tables/'+name+'.csv',
                   'source_rows_1based_excluding_header': ';'.join(map(str, data.index)),
                   'source_column': column, 'transform': transform, 'value': float(result)})
    return float(result)


def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=15, fontsize=10.5, fontweight='bold')
    ax.annotate(letter.lower(), (0, 1), xycoords='axes fraction', xytext=(-16, 15),
                textcoords='offset points', ha='right', va='bottom', fontsize=12.5,
                fontweight='bold', annotation_clip=False)



def save(fig, name):
    text_objects=[obj for obj in fig.findobj(matplotlib.text.Text) if obj.get_visible() and obj.get_text()]
    for obj in text_objects:
        obj.set_fontsize(max(MIN_FONT,obj.get_fontsize()))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    clipped=[]
    for item in text_objects:
        box=item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1):
            clipped.append(item.get_text())
    QA.append({'figure':name,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),
               'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,
               'min_text_pt':min(obj.get_fontsize() for obj in text_objects),
               'font_sizes_pt':sorted(set(obj.get_fontsize() for obj in text_objects)),
               'out_of_canvas_text':clipped})
    assert not clipped,(name,clipped)
    F.mkdir(exist_ok=True)
    fig.savefig(F/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(F/(name+'.png'),dpi=600)
    plt.close(fig)


def supplementary_figure():
    group=load('s14_group_contrasts')
    composition=load('s15_tpo_cluster_composition')
    annotation=load('s17_annotation_diagnostics')
    fig=plt.figure(figsize=(WIDTH,8.85))
    gs=fig.add_gridspec(2,6,left=.10,right=.915,top=.90,bottom=.13,
                         height_ratios=[1.18,1],hspace=.28,wspace=1.35)
    for col,(cohort,program,title) in enumerate([
        ('dapp1','program_neutrophil','Dapp1 granulocytic'),
        ('dapp1','program_cycling','Dapp1 cycling'),
        ('sleep','program_neutrophil','Sleep granulocytic')]):
        ax=fig.add_subplot(gs[0,2*col:2*col+2])
        selected=group[(group.cohort==cohort)&(group.program==program)].sort_values('cluster')
        assert len(selected)==(19 if cohort=='sleep' else 11)
        for y,(i,row) in enumerate(selected.iterrows()):
            d=selected.loc[[i]]
            x=value('s14_group_contrasts',d,'mean_difference_b_minus_a',f'S1{col}:cluster{row.cluster}:contrast')
            w=value('s14_group_contrasts',d,'pooled_weight',f'S1{col}:cluster{row.cluster}:weight')
            ax.axhspan(y-.43,y+.43,color='#F3F6F8' if y%2==0 else 'white',zorder=0)
            ax.plot([0,x],[y,y],color='#CCD6DD',lw=1,zorder=1)
            ax.scatter(x,y,s=18+200*w,color=BLUE if x<0 else ORANGE,edgecolor='white',linewidth=.4,zorder=3)
        ax.set_yticks(range(len(selected)),selected.cluster.astype(str))
        ax.set_ylim(len(selected)-.45,-.55);ax.axvline(0,color=GRAY,ls='--',lw=.7)
        ax.set_xlabel('Program difference');ax.set_ylabel('Cluster',labelpad=2)
        ax.grid(axis='x',color='#E6EBEF',lw=.6);ax.set_axisbelow(True)
        limits=[(-.58,.03),(-.25,.38),(-.055,.16)][col]
        ticks=[[-.5,-.25,0],[-.2,0,.2],[-.05,0,.1]][col]
        ticklabels=[["−0.50","−0.25","0"],["−0.2","0","0.2"],["−0.05","0","0.10"]][col]
        ax.set_xlim(*limits);ax.set_xticks(ticks,ticklabels)
        panel(ax,'ABC'[col],title.replace(' ','\n',1))
        ax.text(0,1.012,'SD − CT' if cohort=='sleep' else 'KO − WT',transform=ax.transAxes,fontsize=MIN_FONT,color=GRAY)
    ax=fig.add_subplot(gs[1,:3]);composition=composition.sort_values('cluster');counts=[]
    for i,row in composition.iterrows():
        counts.append(value('s15_tpo_cluster_composition',composition.loc[[i]],'n_D0',f'S1D:cluster{row.cluster}:n'))
    colors=[ORANGE if int(x)==5 else BLUE for x in composition.cluster]
    ax.barh(range(len(composition)),counts,color=colors,height=.68)
    ax.set_yticks(range(len(composition)),composition.cluster.astype(str))
    ax.set_ylim(len(composition)-.4,-.6);ax.set_xlim(0,3450);ax.set_xticks([0,1000,2000,3000],['0','1,000','2,000','3,000'])
    for y,(cluster,n) in enumerate(zip(composition.cluster,counts)):
        if n>0:ax.text(n+45,y,f'{int(n):,}',va='center',fontsize=MIN_FONT,color=ORANGE if cluster==5 else INK)
    ax.set_xlabel('Retained cells at D0');ax.set_ylabel('Cluster',labelpad=2)
    ax.grid(axis='x',color='#E6EBEF',lw=.6);ax.set_axisbelow(True)
    panel(ax,'D','TPO: full D0 composition')
    ax=fig.add_subplot(gs[1,3:]);scopes=['pooled','D0','D4','D7','D11','D14'];programs=['primitive','neutrophil','dendritic'];data=[];labels=[]
    for scope in scopes:
        selected=annotation[(annotation.cohort=='ucb')&(annotation.scope==scope)]
        assert len(selected)==1
        data.append([value('s17_annotation_diagnostics',selected,'program_'+program,f'S1E:{scope}:{program}') for program in programs])
        labels.append('Pooled' if scope=='pooled' else scope)
    matrix=np.asarray(data);im=ax.imshow(matrix,cmap='RdBu_r',vmin=-1.1,vmax=1.1,aspect='auto')
    ax.set_yticks(range(6),labels)
    ax.set_xticks(range(3),['Primitive','Granulocytic','Dendritic'],rotation=35,ha='right')
    for y in range(6):
        for x in range(3):
            ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',fontsize=10,color='white' if abs(matrix[y,x])>.68 else INK)
    panel(ax,'E','Cluster 5: program scores')
    colorbar=fig.colorbar(im,ax=ax,shrink=.9,pad=.03,fraction=.055,ticks=[-1,0,1])
    colorbar.set_label('Mean program score',fontsize=MIN_FONT)
    colorbar.ax.tick_params(labelsize=MIN_FONT)
    fig.text(.10,.985,'Group contrasts and annotation diagnostics',color=GRAY,fontsize=MIN_FONT,va='top')
    fig.text(.10,.044,'a–c: all eligible clusters; area = pooled weight. d: all 16 D0 clusters.',fontsize=MIN_FONT,color=GRAY)
    fig.text(.10,.016,'Orange = cluster 5 (D0 n = 17). e: competing scores; full scores in S15–S17.',fontsize=MIN_FONT,color=GRAY)
    save(fig,'supplementary_figure1_diagnostics')


def task_table():
    neutral = load('s06_paired_neutral_geometry')
    stability = load('s10_seed_stability')
    denominators = load('s13_task_denominators')
    records, tex_rows, seed_notes = [], [], []
    pop_labels = {'all': 'All', 'nonterminal': 'Nonterminal', 'starting': 'Selected start'}
    for cohort, lineage, label in TASKS:
        den = denominators[(denominators.cohort == cohort) & (denominators.lineage == lineage)]
        n_end = value('s13_task_denominators', den, 'n_endpoint', f'S18:{label}:endpoint_n')
        note = ('Cluster 5 is granulocytic-max at D0 and dendritic-max at D14; D0 n=17' if cohort == 'ucb' else
                'Endpoint cluster 8: WT mast-max, pooled erythroid-max; all labels program-derived' if lineage == 'erythroid' else
                'Endpoints selected by frozen program maxima; no independent fate validation')
        ranges = {}
        for arm in ['PCA', 'LiVAE']:
            selected = stability[(stability.cohort == cohort) & (stability.lineage == lineage) &
                                 (stability.arm == arm) & (stability.method == 'scRL') & (stability.reward == 'neutral')]
            assert len(selected) == 3
            ranges[arm] = [value('s10_seed_stability', selected, 'value', f'S18:{label}:{arm}:seed_{fn}', fn)
                           for fn in ['min', 'max']]
        for pop in ['all', 'nonterminal', 'starting']:
            shown = {}
            for arm in ['PCA', 'LiVAE']:
                selected = neutral[(neutral.cohort == cohort) & (neutral.lineage == lineage) &
                                   (neutral.arm == arm) & (neutral.population == pop)]
                metrics = {key: value('s06_paired_neutral_geometry', selected, key, f'S18:{label}:{pop}:{arm}:{key}')
                           for key in ['n_cells', 'value_median', 'geometry_rho', 'increment_median']}
                shown[arm] = metrics
                records.append({'cohort': cohort, 'task': lineage, 'population': pop, 'arm': arm,
                                'n_endpoint': int(n_end), **metrics,
                                'between_seed_rho_min_all_cells': ranges[arm][0],
                                'between_seed_rho_max_all_cells': ranges[arm][1], 'annotation_note': note})
            assert shown['PCA']['n_cells'] == shown['LiVAE']['n_cells']
            display = [label if pop == 'all' else '', pop_labels[pop], f"{int(shown['PCA']['n_cells']):,}"]
            display += [f"{shown[arm]['value_median']:.3f}" for arm in ['PCA', 'LiVAE']]
            display += [f"{shown[arm]['increment_median']:+.3f}" for arm in ['PCA', 'LiVAE']]
            tex_rows.append(' & '.join(display)+r' \\')
        tex_rows.append(r'\addlinespace[.4em]')
        seed_notes.append(r'\noindent '+label+
                        f': endpoint $n={int(n_end):,}$; all-cell between-seed value correlations: PCA '+
                        f'{ranges["PCA"][0]:.3f}--{ranges["PCA"][1]:.3f}, LiVAE '+
                        f'{ranges["LiVAE"][0]:.3f}--{ranges["LiVAE"][1]:.3f}.'+r'\par')
    with (T/'s18_task_evidence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    text = r'''\begin{table}[htbp]
\centering\small
\caption{Task evidence viewed jointly under the matched two-dimensional protocol. All associations use no-time reward. $\rho$ is the three-seed median value--program correlation; $\Delta$ is that median minus endpoint proximity in the same interface. Seed ranges below each task describe all-cell between-run value correlations, not the three population-specific associations or biological intervals.}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrrr@{}}
\toprule
Task & Population & $n$ & $\rho_{\rm PCA}$ & $\rho_{\rm LiVAE}$ & $\Delta_{\rm PCA}$ & $\Delta_{\rm LiVAE}$ \\
\midrule
'''+ '\n'.join(tex_rows)+r'''
\bottomrule
\end{tabular*}
\par\smallskip\footnotesize
'''+ '\n'.join(seed_notes)+r'''
\par\smallskip
The task labels identify frozen program-defined endpoint sets. Dapp1 Ery has 87 endpoint cells (33 WT and 54 KO); its WT mast score exceeds its erythroid score. TPO's day-specific start is the intersection of frozen cluster 5 with that day's captured cells, with only 17 cells at D0; the pooled primitive-max label is granulocytic-max at D0 and dendritic-max at D14. No task has independent observed-fate validation. A gain in association, a gain over proximity, and seed consistency are distinct diagnostics. The companion CSV includes all same-interface proximity correlations and annotation notes. Sources: Tables S6, S10, S13 and S17.
\end{table}
'''
    (T/'s18_task_evidence.tex').write_text(text)


def reward_comparison():
    saved = load('s08_all_reward_associations')
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.65))
    fig.subplots_adjust(left=.15, right=.98, top=.83, bottom=.32, wspace=.32)
    records, tex_rows = [], []
    colors = {'PCA': BLUE, 'LiVAE': '#8856A7'}
    rewards = [('decision', 'Decision'), ('neutral', 'No-time'), ('contribution', 'Contribution')]
    for k, (reward, reward_label) in enumerate(rewards):
        ax = axes[k]
        for y, (cohort, lineage, label) in enumerate(TASKS):
            if y % 2 == 0:
                ax.axhspan(y-.44, y+.44, color='#F3F6F8', zorder=0)
            for arm, offset in [('PCA', -.19), ('LiVAE', .19)]:
                selected = saved[(saved.cohort == cohort) & (saved.lineage == lineage) &
                                 (saved.arm == arm) & (saved.reward == reward) & (saved.population == 'all')].sort_values('seed')
                assert len(selected) == 3 and selected.geometry_program_rho.nunique() == 1
                correlations = [value('s08_all_reward_associations', selected.loc[[i]], 'value_program_rho',
                                      f'S2:{label}:{arm}:{reward}:seed{int(row.seed)}') for i, row in selected.iterrows()]
                geometry = value('s08_all_reward_associations', selected.iloc[[0]], 'geometry_program_rho',
                                 f'S2:{label}:{arm}:{reward}:proximity')
                ax.plot([min(correlations), max(correlations)], [y+offset]*2, color=colors[arm], alpha=.5, lw=.9)
                for seed, (rho, marker) in enumerate(zip(correlations, ['o', 's', '^'])):
                    ax.scatter(rho, y+offset+(seed-1)*.085, s=24, marker=marker,
                               color=colors[arm], edgecolor='white', lw=.35, zorder=4)
                ax.scatter(geometry, y+offset+.12, s=19, marker='D', facecolor='white',
                           edgecolor=colors[arm], lw=.85, zorder=5)
                med = value('s08_all_reward_associations', selected, 'value_program_rho',
                            f'S19:{label}:{arm}:{reward}:median', 'median')
                increment = value('s08_all_reward_associations', selected, 'increment_over_geometry',
                                  f'S19:{label}:{arm}:{reward}:increment_median', 'median')
                record = {'cohort': cohort, 'task': lineage, 'arm': arm, 'reward': reward, 'population': 'all',
                          'n_cells': int(selected.n_cells.iloc[0]), 'n_endpoint': int(selected.n_endpoint.iloc[0]),
                          'rho_seed0': correlations[0], 'rho_seed1': correlations[1], 'rho_seed2': correlations[2],
                          'rho_median': med, 'rho_min': min(correlations), 'rho_max': max(correlations),
                          'geometry_rho': geometry, 'increment_median': increment}
                records.append(record)
                tex_rows.append(' & '.join([label, reward_label, arm, *[f'{x:.3f}' for x in correlations],
                                            f'{geometry:.3f}'])+r' \\')
            tex_rows.append(r'\addlinespace[.25em]')
        ax.set_yticks(range(4), [x[2] for x in TASKS] if k == 0 else ['']*4)
        ax.set_ylim(3.55, -.55); ax.set_xlim(-1.03, 1.03); ax.set_xticks([-1, 0, 1])
        ax.axvline(0, color=GRAY, ls='--', lw=.7); ax.set_xlabel('Spearman ρ', fontsize=MIN_FONT)
        ax.grid(axis='x', color='#E6EBEF', lw=.6); ax.set_axisbelow(True)
        panel(ax, 'ABC'[k], reward_label)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=colors[a], marker='o', ls='', label='Revised LiVAE' if a == 'LiVAE' else a)
                        for a in ['PCA', 'LiVAE']], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.55, .175), fontsize=MIN_FONT)
    fig.text(.15, .985, 'Matched interfaces · every retained reward · all cells', fontsize=MIN_FONT, color=GRAY, va='top')
    fig.text(.55,.132,'Circle / square / triangle: seeds 0 / 1 / 2. Open diamond: proximity.',ha='center',fontsize=MIN_FONT,color=GRAY)
    fig.text(.55,.086,'Segments span three seeds; they are not biological intervals.',ha='center',fontsize=MIN_FONT,color=GRAY)
    fig.text(.55,.035,'TPO LiVAE Contribution associations change sign across seeds.',ha='center',fontsize=MIN_FONT,color=GRAY)
    save(fig,'supplementary_figure2_rewards')
    with (T/'s19_reward_evidence.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (T/'s19_reward_evidence.tex').write_text(r'''\begin{table}[htbp]
\centering\small
\caption{All-cell matched-interface associations for every retained task and reward. Each seed column is one saved value--program Spearman correlation; proximity uses the same interface and is reward-independent. These are same-data associations, not observed-fate accuracy.}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lllrrrr@{}}
\toprule
Task & Reward & Interface & Seed 0 & Seed 1 & Seed 2 & Proximity \\
\midrule
'''+ '\n'.join(tex_rows)+r'''
\bottomrule
\end{tabular*}
\par\smallskip\footnotesize
Every four-task, three-reward and two-interface combination is shown. The three seeds share cells, programs, endpoint sets and fixed upstream fits. No-time removes temporal reward terms but retains the task environment. The TPO Contribution sign disagreement in the revised LiVAE interface is conditional on this task and training configuration; it is not evidence of a general failure of the method. Dapp1 Ery uses the annotation-ambiguous 87-cell endpoint. Companion CSVs provide medians, ranges, denominators and increments over proximity; the complete population-specific saved correlations remain in Table S8. Source: Table S8.
\end{table}
''')


def main():
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': MIN_FONT, 'axes.labelsize': MIN_FONT, 'xtick.labelsize': MIN_FONT, 'ytick.labelsize': MIN_FONT, 'legend.fontsize': MIN_FONT, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6, 'pdf.fonttype': 42,
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    E.mkdir(exist_ok=True)
    supplementary_figure(); task_table(); reward_comparison()
    with (E/'interpretation_display_values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LEDGER[0])); writer.writeheader(); writer.writerows(LEDGER)
    outputs = ['figures/'+name+'.'+ext for name in ['supplementary_figure1_diagnostics', 'supplementary_figure2_rewards'] for ext in ['pdf', 'png']]
    outputs += ['supplement_tables/'+name+'.'+ext for name in ['s18_task_evidence', 's19_reward_evidence'] for ext in ['csv', 'tex']]
    outputs += ['evidence/interpretation_display_values.csv']
    manifest = {'scope': 'Display assembly from saved supplementary CSVs; no new scientific fits or scores',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sources': SOURCES,
                'outputs': {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in outputs},
                
                'ledger_values': len(LEDGER), 'figure_checks': QA, 'minimum_native_text_pt':MIN_FONT, 'intended_print_width_inches':WIDTH}
    (E/'interpretation_assets.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': 'pass', 'display_values': len(LEDGER), 'task_evidence_rows': 24}))


if __name__ == '__main__':
    main()
