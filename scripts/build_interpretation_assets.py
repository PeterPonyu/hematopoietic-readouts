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
MIN_FONT = 10
BLUE, ORANGE, GRAY, INK = '#17689A', '#C6602A', '#707070', '#000000'
TASKS = [('irradiation', 'erythroid', 'IR Ery'), ('irradiation', 'myeloid', 'IR My'),
         ('ucb', 'megakaryocyte', 'TPO Mk'), ('dapp1', 'erythroid', 'Dapp1 Ery'),
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
    ax.set_title(title, loc='center', pad=10, fontsize=10.5, fontweight='normal')
    label = ax.figure.text(0, 0, letter.lower(), ha='right', va='top',
                           fontsize=13, fontweight='bold', color=INK,
                           gid='panel-letter-'+letter.lower())
    label._panel_axis = ax



def save(fig, name):
    text_objects=[obj for obj in fig.findobj(matplotlib.text.Text) if obj.get_visible() and obj.get_text()]
    for obj in text_objects:
        obj.set_fontsize(max(MIN_FONT,obj.get_fontsize()))
        obj.set_color(INK)
        obj.set_fontweight('bold' if (obj.get_gid() or '').startswith('panel-letter-') else 'normal')
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for obj in text_objects:
        if (obj.get_gid() or '').startswith('panel-letter-'):
            ax = obj._panel_axis
            title_box = ax.title.get_window_extent(renderer)
            obj.set_position((ax.get_position().x0 - 12 / 72 / fig.get_figwidth(),
                              title_box.y1 / fig.bbox.height))
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();width,height=fig.canvas.get_width_height()
    clipped=[]
    for item in text_objects:
        box=item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1):
            clipped.append(item.get_text())
    overlaps=[]
    for i,item in enumerate(text_objects):
        a=item.get_window_extent(renderer)
        for other in text_objects[i+1:]:
            b=other.get_window_extent(renderer)
            if min(a.x1,b.x1)-max(a.x0,b.x0) > .5 and min(a.y1,b.y1)-max(a.y0,b.y0) > .5:
                overlaps.append([item.get_text(),other.get_text()])
    QA.append({'figure':name,'width_inches':fig.get_figwidth(),'height_inches':fig.get_figheight(),
               'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,
               'min_text_pt':min(obj.get_fontsize() for obj in text_objects),
               'font_sizes_pt':sorted(set(obj.get_fontsize() for obj in text_objects)),
               'text_colors':sorted(set(obj.get_color() for obj in text_objects)),
               'bold_text':[obj.get_text() for obj in text_objects if obj.get_fontweight() == 'bold'],
               'overlapping_text':overlaps,
               'out_of_canvas_text':clipped})
    assert not clipped,(name,clipped)
    assert not overlaps,(name,overlaps)
    F.mkdir(exist_ok=True)
    fig.savefig(F/(name+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(F/(name+'.png'),dpi=600)
    fig.savefig(F/(name+'.svg'),metadata={'Date':None})
    plt.close(fig)


def supplementary_figure():
    group=load('s14_group_contrasts')
    composition=load('s15_tpo_cluster_composition')
    annotation=load('s17_annotation_diagnostics')
    fig=plt.figure(figsize=(WIDTH,7.6))
    top = fig.add_gridspec(1,3,left=.085,right=.975,top=.935,bottom=.535,wspace=.24)
    bottom = fig.add_gridspec(1,2,left=.085,right=.88,top=.43,bottom=.105,
                              width_ratios=[.98,1],wspace=.36)
    for col,(cohort,program,title) in enumerate([
        ('dapp1','program_neutrophil','Dapp1 granulocytic'),
        ('dapp1','program_cycling','Dapp1 cycling'),
        ('sleep','program_neutrophil','Sleep granulocytic')]):
        ax=fig.add_subplot(top[0,col])
        selected=group[(group.cohort==cohort)&(group.program==program)].sort_values('cluster')
        assert len(selected)==(19 if cohort=='sleep' else 11)
        for y,(i,row) in enumerate(selected.iterrows()):
            d=selected.loc[[i]]
            x=value('s14_group_contrasts',d,'mean_difference_b_minus_a',f'S1{col}:cluster{row.cluster}:contrast')
            w=value('s14_group_contrasts',d,'pooled_weight',f'S1{col}:cluster{row.cluster}:weight')
            ax.axhspan(y-.43,y+.43,color='#F5F5F5' if y%2==0 else 'white',zorder=0)
            ax.plot([0,x],[y,y],color='#CCCCCC',lw=1,zorder=1)
            ax.scatter(x,y,s=18+200*w,color=BLUE if x<0 else ORANGE,edgecolor='white',linewidth=.4,zorder=3)
        ax.set_yticks(range(len(selected)),selected.cluster.astype(str))
        ax.set_ylim(len(selected)-.45,-.55);ax.axvline(0,color=GRAY,ls='--',lw=.7)
        ax.set_xlabel('Program difference');ax.set_ylabel('Cluster' if col == 0 else '',labelpad=2)
        ax.grid(axis='x',color='#E8E8E8',lw=.6);ax.set_axisbelow(True)
        limits=[(-.58,.03),(-.25,.38),(-.055,.16)][col]
        ticks=[[-.5,-.25,0],[-.2,0,.2],[-.05,0,.1]][col]
        ticklabels=[["−0.50","−0.25","0"],["−0.2","0","0.2"],["−0.05","0","0.10"]][col]
        ax.set_xlim(*limits);ax.set_xticks(ticks,ticklabels)
        panel(ax,'ABC'[col],title+'\n'+('SD − CT' if cohort=='sleep' else 'KO − WT'))
    ax=fig.add_subplot(bottom[0,0]);composition=composition.sort_values('cluster');counts=[]
    for i,row in composition.iterrows():
        counts.append(value('s15_tpo_cluster_composition',composition.loc[[i]],'n_D0',f'S1D:cluster{row.cluster}:n'))
    colors=[ORANGE if int(x)==5 else BLUE for x in composition.cluster]
    ax.barh(range(len(composition)),counts,color=colors,height=.68)
    ax.set_yticks(range(len(composition)),composition.cluster.astype(str))
    ax.set_ylim(len(composition)-.4,-.6);ax.set_xlim(0,3450);ax.set_xticks([0,1000,2000,3000],['0','1,000','2,000','3,000'])
    for y,(cluster,n) in enumerate(zip(composition.cluster,counts)):
        if int(cluster) == 5:
            ax.scatter(n,y,s=16,color=ORANGE,edgecolors='none',zorder=3,clip_on=False)
        if n>0:ax.text(n+(70 if int(cluster) == 5 else 45),y,f'{int(n):,}',va='center',fontsize=MIN_FONT,color=INK)
    ax.set_xlabel('Retained cells at D0');ax.set_ylabel('Cluster',labelpad=2)
    ax.grid(axis='x',color='#E8E8E8',lw=.6);ax.set_axisbelow(True)
    panel(ax,'D','TPO: full D0 composition')
    ax=fig.add_subplot(bottom[0,1]);scopes=['pooled','D0','D4','D7','D11','D14'];programs=['primitive','neutrophil','dendritic'];data=[];labels=[]
    for scope in scopes:
        selected=annotation[(annotation.cohort=='ucb')&(annotation.scope==scope)]
        assert len(selected)==1
        data.append([value('s17_annotation_diagnostics',selected,'program_'+program,f'S1E:{scope}:{program}') for program in programs])
        labels.append('Pooled' if scope=='pooled' else scope)
    score_colors = matplotlib.colors.LinearSegmentedColormap.from_list(
        'score_with_black_text', plt.get_cmap('RdBu_r')(np.linspace(.20,.80,256)))
    matrix=np.asarray(data)
    im=ax.pcolormesh(range(3),range(6),matrix,cmap=score_colors,vmin=-1.1,vmax=1.1,
                     shading='nearest',edgecolors='none',rasterized=False)
    ax.set_xlim(-.5,2.5);ax.set_ylim(5.5,-.5)
    ax.set_yticks(range(6),labels)
    ax.set_xticks(range(3),['Primitive','Granulocytic','Dendritic'],rotation=35,ha='right',rotation_mode='anchor')
    for y in range(6):
        for x in range(3):
            ax.text(x,y,f'{matrix[y,x]:.2f}',ha='center',va='center',fontsize=MIN_FONT,color=INK)
    panel(ax,'E','Cluster 5: program scores')
    colorbar_ax=fig.add_axes([.90,.105,.018,.325])
    colorbar=fig.colorbar(im,cax=colorbar_ax,ticks=[-1,0,1])
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor('face')
    colorbar.set_label('Mean program score',fontsize=MIN_FONT)
    colorbar.ax.tick_params(labelsize=MIN_FONT)
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
        note = ('HSC, MPP2/3 and MPP4 starting labels; day 5 starting n=66; mixed public and author-supplied sources' if cohort == 'irradiation' else
                'Cluster 5 is granulocytic-max at D0 and dendritic-max at D14; D0 n=17' if cohort == 'ucb' else
                'Endpoint cluster 8: WT mast-max, pooled erythroid-max; all labels program-derived' if cohort == 'dapp1' and lineage == 'erythroid' else
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
IR denotes irradiation and uses frozen HSC, MPP2/3 and MPP4 starting labels; its day-5 starting subset contains 66 cells (Table S22). Other task labels identify frozen program-defined endpoint sets. Dapp1 Ery has 87 endpoint cells (33 WT and 54 KO); its WT mast score exceeds its erythroid score. TPO's day-specific start is the intersection of frozen cluster 5 with that day's captured cells, with only 17 cells at D0; the pooled primitive-max label is granulocytic-max at D0 and dendritic-max at D14. No task has independent observed-fate validation. A gain in association, a gain over proximity, and seed consistency are distinct diagnostics. The companion CSV includes all same-interface proximity correlations and annotation notes. Sources: Tables S6, S10, S13, S17 and S22.
\end{table}
'''
    (T/'s18_task_evidence.tex').write_text(text)


def reward_comparison():
    saved = load('s08_all_reward_associations')
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 4.8))
    fig.subplots_adjust(left=.155, right=.98, top=.90, bottom=.26, wspace=.22)
    records, tex_rows = [], []
    colors = {'PCA': BLUE, 'LiVAE': '#8856A7'}
    rewards = [('decision', 'Decision'), ('neutral', 'No-time'), ('contribution', 'Contribution')]
    for k, (reward, reward_label) in enumerate(rewards):
        ax = axes[k]
        for y, (cohort, lineage, label) in enumerate(TASKS):
            if y % 2 == 0:
                ax.axhspan(y-.44, y+.44, color='#F5F5F5', zorder=0)
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
        ax.set_yticks(range(len(TASKS)), [x[2] for x in TASKS] if k == 0 else ['']*len(TASKS))
        ax.set_ylim(len(TASKS)-.45, -.55); ax.set_xlim(-1.03, 1.03); ax.set_xticks([-1, 0, 1])
        ax.axvline(0, color=GRAY, ls='--', lw=.7); ax.set_xlabel('Spearman ρ', fontsize=MIN_FONT)
        ax.grid(axis='x', color='#E8E8E8', lw=.6); ax.set_axisbelow(True)
        panel(ax, 'ABC'[k], reward_label)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=colors[a], marker='o', ls='', label='Revised LiVAE' if a == 'LiVAE' else a)
                        for a in ['PCA', 'LiVAE']], loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(.565, .075), fontsize=MIN_FONT, handletextpad=.5,
               borderaxespad=0, columnspacing=1.5)
    fig.legend(handles=[Line2D([], [], color=INK, marker=marker, ls='', markersize=4.5,
                               markerfacecolor='white' if marker == 'D' else INK, label=label)
                        for marker,label in [('o','Seed 0'),('s','Seed 1'),('^','Seed 2'),('D','Proximity')]],
               loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(.565, .005),
               fontsize=MIN_FONT, handletextpad=.4, columnspacing=1, borderaxespad=0)
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
Every six-task, three-reward and two-interface combination is shown. IR denotes irradiation. The three seeds share cells, programs, endpoint sets and fixed upstream fits. No-time removes temporal reward terms but retains the task environment. The TPO Contribution sign disagreement in the revised LiVAE interface is conditional on this task and training configuration; it is not evidence of a general failure of the method. Dapp1 Ery uses the annotation-ambiguous 87-cell endpoint. Companion CSVs provide medians, ranges, denominators and increments over proximity; the complete population-specific saved correlations remain in Table S8. Source: Table S8.
\end{table}
''')



def flowing_evidence_table(text):
    """Let long joint-evidence tables use page space in numerical order."""
    lines=text.splitlines()
    caption=next(line for line in lines if line.startswith(r'\caption{'))
    start=next(i for i,line in enumerate(lines) if line.startswith(r'\begin{tabular*}'))
    mid=next(i for i,line in enumerate(lines[start:],start) if line==r'\midrule')
    end=next(i for i,line in enumerate(lines) if line==r'\end{tabular*}')
    columns=lines[start].replace(r'\begin{tabular*}{\textwidth}',r'\begin{longtable}')
    header=lines[start+1:mid+1]
    trailing=[line.replace(r'\footnotesize',r'\small') for line in lines[end+1:] if line!=r'\end{table}']
    return '\n'.join([r'\begingroup',r'\small',r'\setlength{\LTpre}{6pt}',r'\setlength{\LTpost}{6pt}',
        columns,caption+r'\\',*header,r'\endfirsthead',
        r'\multicolumn{7}{l}{\small Table \thetable\ continued}\\',*header,r'\endhead',
        *lines[mid+1:end],r'\end{longtable}',*trailing,r'\endgroup',''])

def main():
    font_manager.findfont('Arial', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': MIN_FONT, 'axes.labelsize': MIN_FONT, 'xtick.labelsize': MIN_FONT, 'ytick.labelsize': MIN_FONT, 'legend.fontsize': MIN_FONT, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none', 'svg.hashsalt': 'hematopoietic-readouts',
                         'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})
    E.mkdir(exist_ok=True)
    supplementary_figure(); task_table(); reward_comparison()
    for name in ['s18_task_evidence', 's19_reward_evidence']:
        file=T/(name+'.tex')
        file.write_text(flowing_evidence_table(file.read_text()))
    with (E/'interpretation_display_values.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LEDGER[0])); writer.writeheader(); writer.writerows(LEDGER)
    outputs = ['figures/'+name+'.'+ext for name in ['supplementary_figure1_diagnostics', 'supplementary_figure2_rewards'] for ext in ['pdf', 'png', 'svg']]
    outputs += ['supplement_tables/'+name+'.'+ext for name in ['s18_task_evidence', 's19_reward_evidence'] for ext in ['csv', 'tex']]
    outputs += ['evidence/interpretation_display_values.csv']
    manifest = {'scope': 'Display assembly from saved supplementary CSVs; no new scientific fits or scores',
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'sources': SOURCES,
                'outputs': {name: hashlib.sha256((P/name).read_bytes()).hexdigest() for name in outputs},
                
                'ledger_values': len(LEDGER), 'figure_checks': QA, 'minimum_native_text_pt':MIN_FONT, 'intended_print_width_inches':WIDTH}
    (E/'interpretation_assets.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'status': 'pass', 'display_values': len(LEDGER), 'task_evidence_rows': 36}))


if __name__ == '__main__':
    main()
