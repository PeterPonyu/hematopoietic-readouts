#!/usr/bin/env python3
"""Format existing saved results; never fit models or recalculate scientific estimands.

Run from any directory with the repository tree intact. Uses Python standard library.
Full-precision CSVs are filtered copies or lossless reshapes of saved CSV/JSON fields.
"""
from pathlib import Path
import csv
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT
PACKAGED = PROJECT / 'source_data'
OUT = PROJECT / 'supplement_tables'
FROZEN = ROOT / 'results/application_reanalysis_20260921'
NATIVE = ROOT / 'results/recovered_integration_20261001'
COHORTS = ('ucb', 'dapp1', 'sleep')
COHORT_LABEL = {'ucb': 'TPO', 'dapp1': 'Dapp1', 'sleep': 'Sleep', 'marrow': 'Marrow'}
TASKS = {('ucb','megakaryocyte'): 'TPO-Mk', ('dapp1','erythroid'): 'Dapp1-E', ('dapp1','myeloid'): 'Dapp1-M', ('sleep','myeloid'): 'Sleep-M'}
SOURCES = {}


def resolve(path):
    relative = path.relative_to(PACKAGED) if path.is_relative_to(PACKAGED) else path.relative_to(ROOT)
    packaged = PACKAGED / relative
    return packaged if packaged.exists() else ROOT / relative


def tracked(path):
    relative = path.relative_to(PACKAGED) if path.is_relative_to(PACKAGED) else path.relative_to(ROOT)
    actual = resolve(path)
    SOURCES[str(relative)] = hashlib.sha256(actual.read_bytes()).hexdigest()
    return actual


def read_csv(path, scoped=False):
    rows = list(csv.DictReader(tracked(path).open()))
    return [r for r in rows if r['cohort'] in COHORTS] if scoped else rows


def read_json(path):
    relative = str(Path(path).relative_to(ROOT))
    metadata = PROJECT / 'aggregate_metadata.json'
    SOURCES['aggregate_metadata.json'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return json.loads(metadata.read_text())['records'][relative]


def csv_out(name, rows):
    assert rows, name
    with (OUT / (name+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def esc(value):
    return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}.get(c,c) for c in str(value))


def number(value, digits=3):
    return f'{float(value):.{digits}f}'


def cell(value):
    return f'{int(value):,}'


def program(value):
    return value.removeprefix('program_').replace('_',' ').replace('neutrophil','granulocytic').replace('cycling','cell cycle')


def task(row):
    return TASKS[row['cohort'], row['lineage']]


def pop(row):
    return {'all':'All', 'nonterminal':'Nonterminal', 'starting':'Starting'}[row['population']]


def table(name, caption, headers, widths, rows):
    assert len(headers)==len(widths)
    cols='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\linewidth}' for w in widths)+'@{}'
    header=' & '.join(headers)+r' \\'
    tex = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{2.5pt}', r'\renewcommand{\arraystretch}{1.13}', r'\begin{longtable}{'+cols+'}', r'\caption{'+caption+r'}\label{tab:'+name+r'}\\', r'\toprule', header, r'\midrule', r'\endfirsthead', r'\multicolumn{'+str(len(headers))+r'}{l}{\small\tablename\ \thetable\ continued}\\',r'\toprule',header,r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(len(headers))+r'}{r}{\small Continued on next page}\\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
    for row in rows:
        assert len(row)==len(headers), name
        tex.append(' & '.join(esc(v) for v in row)+r' \\')
    tex += [r'\end{longtable}',r'\endgroup','']
    (OUT/(name+'.tex')).write_text('\n'.join(tex))


def main():
    OUT.mkdir(exist_ok=True)
    protocols={c:read_json(FROZEN/c/'protocol.json') for c in COHORTS}
    rows=[]
    for c in COHORTS:
        order=['D0','D4','D7','D11','D14'] if c=='ucb' else ['wt','ko'] if c=='dapp1' else ['CT','SD']
        for condition in order:
            p=protocols[c]
            rows.append({'cohort':c,'accession':p['accession'],'condition':condition,'n_cells':p['conditions'][condition],'cohort_input_cells':p['n_input'],'cohort_retained_genes':p['n_genes']})
    assert sum(r['n_cells'] for r in rows)==79262
    csv_out('s01_cohort_counts', rows)
    table('s01_cohort_counts','Retained cells by public cohort and observed condition. Input cells and retained genes are cohort totals repeated to identify the denominator; input cells precede the additional cell filters described in Methods. These counts do not enumerate independent animals or donors.', ['Cohort','Accession','Condition','Cells','Input cells','Genes'], [.12,.18,.12,.15,.15,.15], [[COHORT_LABEL[r['cohort']],r['accession'],r['condition'].upper(),cell(r['n_cells']),cell(r['cohort_input_cells']),cell(r['cohort_retained_genes'])] for r in rows])

    rows=[{'cohort':c,**r} for c in ('dapp1','sleep') for r in read_csv(FROZEN/c/'composition_standardized_effects.csv')]
    assert len(rows)==30
    csv_out('s02_program_contrasts',rows)
    table('s02_program_contrasts','Complete saved program contrasts. Raw is the all-cell condition mean difference; fixed uses the pooled mixture of eligible Leiden clusters with at least 20 cells in each condition. Directions are KO minus WT and SD minus CT. Coverage is the fraction of all retained cohort cells in eligible clusters. Scores are cohort-specific and descriptive.', ['Cohort','Program','Raw','Fixed','Groups','Coverage'], [.12,.27,.12,.12,.12,.12], [[COHORT_LABEL[r['cohort']],program(r['program']),number(r['unadjusted_delta']),number(r['composition_standardized_delta']),r['matched_groups'],number(r['fraction_cells_covered'])] for r in rows])

    rows=read_csv(FROZEN/'ucb/condition_program_means.csv')
    csv_out('s03_tpo_program_means',rows)
    byday={r['condition']:r for r in rows}
    table('s03_tpo_program_means','Complete TPO all-cell program means at actual culture days. Each entry is the saved mean of the cohort-standardized gene program; values are not independently standardized within day.', ['Program','D0','D4','D7','D11','D14'], [.27,.13,.13,.13,.13,.13], [[program(p)]+[number(byday[d][p]) for d in ('D0','D4','D7','D11','D14')] for p in rows[0] if p!='condition'])

    rows=[{'cohort':c,'program':p,'genes':';'.join(genes),'n_genes':len(genes)} for c in COHORTS for p,genes in protocols[c]['program_genes'].items()]
    assert len(rows)==45
    csv_out('s04_detected_program_genes',rows)
    table('s04_detected_program_genes','Exact detected fixed-program genes used in each cohort. Matching to the predefined lists was case-insensitive; symbols below preserve the saved matrix naming. Missing genes were omitted, with no replacement selected from the observed contrasts. Granulocytic denotes the saved neutrophil program.', ['Cohort','Program','Detected genes'], [.12,.22,.59], [[COHORT_LABEL[r['cohort']],program(r['program']),r['genes'].replace(';',', ')] for r in rows])

    rows=read_csv(FROZEN/'geometry_reference/summary.csv',True)
    assert len(rows)==12
    csv_out('s05_frozen_geometry',rows)
    table('s05_frozen_geometry','All retained task-by-subset comparisons in the frozen PCA30-state protocol. Entries are Spearman correlations with the target program. scRL reports the three-seed median and range under no-time reward; UMAP and PCA are negative mean distances to 15 nearest endpoint cells in the respective spaces. TPO-Mk is megakaryocyte; E is erythroid; M is myeloid (granulocytic expression reference).', ['Task','Subset','Cells','scRL median [min, max]','UMAP','PCA30'], [.15,.17,.12,.27,.11,.11], [[task(r),pop(r),cell(r['n_cells']),f"{number(r['scrl_median'])} [{number(r['scrl_min'])}, {number(r['scrl_max'])}]",number(r['umap']),number(r['pca'])] for r in rows])

    rows=read_csv(NATIVE/'native_analysis/neutral_summary.csv',True)
    assert len(rows)==24
    csv_out('s06_paired_neutral_geometry',rows)
    table('s06_paired_neutral_geometry','Matched-interface no-time reward results, with every retained task, arm and subset. Both arms use UMAP2 for state and action coordinates; geometry is measured in that same arm. Correlations are with the fixed program. The scRL interval is the minimum--maximum across three seeds, not a confidence interval.', ['Task / arm','Subset','Cells','scRL median [min, max]','Geometry','Increment'], [.20,.15,.11,.25,.11,.11], [[task(r)+' / '+r['arm'],pop(r),cell(r['n_cells']),f"{number(r['value_median'])} [{number(r['value_min'])}, {number(r['value_max'])}]",number(r['geometry_rho']),number(r['increment_median'])] for r in rows])

    rows=read_csv(NATIVE/'native_analysis/paired_representation_effects.csv',True)
    assert len(rows)==12
    csv_out('s07_paired_differences',rows)
    table('s07_paired_differences','Saved paired representation differences. Value correlations are three-seed medians under no-time reward. Difference is LiVAE minus PCA; each increment subtracts that arm\'s own geometry correlation. All differences retain their signs.', ['Task','Subset','LiVAE','PCA','Difference','LiVAE increment','PCA increment'], [.15,.16,.11,.11,.12,.13,.13], [[task(r),pop(r),number(r['LiVAE_value_rho']),number(r['PCA_value_rho']),number(r['LiVAE_minus_PCA']),number(r['LiVAE_increment_over_geometry']),number(r['PCA_increment_over_geometry'])] for r in rows])

    rows=read_csv(NATIVE/'native_analysis/associations_all_rewards.csv',True)
    assert len(rows)==216
    csv_out('s08_all_reward_associations',rows)
    pivot={}
    for r in rows:
        key=(r['cohort'],r['arm'],r['lineage'],r['population'],r['reward'])
        pivot.setdefault(key,{})[r['seed']]=r
    assert len(pivot)==72 and all(set(v)=={'0','1','2'} for v in pivot.values())
    display=[]
    for key,rs in pivot.items():
        r=rs['0']; assert all(x['geometry_program_rho']==r['geometry_program_rho'] for x in rs.values())
        display.append([task(r)+' / '+r['arm'],pop(r),{'decision':'Decision','neutral':'No time','contribution':'Contribution'}[r['reward']]]+[number(rs[str(s)]['value_program_rho']) for s in range(3)]+[number(r['geometry_program_rho'])])
    table('s08_all_reward_associations','Complete per-seed program correlations under all three rewards in the matched-interface experiment. The 216 saved correlations are reshaped into 72 rows without averaging. Geometry is fixed within an arm/task/subset. The companion CSV additionally retains cell counts, endpoint counts and each saved value-minus-geometry increment.', ['Task / arm','Subset','Reward','Seed 0','Seed 1','Seed 2','Geometry'], [.20,.15,.18,.09,.09,.09,.11],display)

    rows=read_csv(NATIVE/'native_analysis/focus_summary.csv',True)
    assert len(rows)==18
    for r in rows:
        rec=read_json(NATIVE/'paired_downstream'/r['arm']/r['cohort']/('scfocus_seed'+r['seed'])/'receipt.json')
        r['member_episodes']=';'.join(map(str,rec['actual_episodes']))
    csv_out('s09_focus_summary',rows)
    table('s09_focus_summary','Complete scFocus run summaries for the matched interfaces. Entropy is the median cell entropy after row normalization, divided by the log merged branch count. Raw sums show that original weights are not row-normalized probabilities. Episodes lists actual counts for the three ensemble members.', ['Cohort / arm','Seed','Branches','Entropy','Raw row sums [min, max]','Episodes'], [.22,.08,.10,.12,.25,.17], [[COHORT_LABEL[r['cohort']]+' / '+r['arm'],r['seed'],r['branches'],number(r['median_entropy']),f"[{number(r['min_raw_row_sum'])}, {number(r['max_raw_row_sum'])}]",r['member_episodes'].replace(';',', ')] for r in rows])

    rows=read_csv(NATIVE/'native_analysis/seed_stability.csv',True)
    csv_out('s10_seed_stability',rows)
    pivot={}
    for r in rows:
        key=(r['cohort'],r['arm'],r['method'],r['lineage'],r['reward'],r['metric'])
        pivot.setdefault(key,{})[(r['seed_a'],r['seed_b'])]=r
    display=[]
    for key,rs in pivot.items():
        r=next(iter(rs.values())); focus=r['method'].lower()=='scfocus'
        identity=COHORT_LABEL[r['cohort']] if focus else task(r)
        display.append([identity+' / '+r['arm'],'scFocus' if focus else 'scRL',r['reward'] or '--','ARI' if focus else 'Spearman']+[number(rs[p]['value']) for p in [('0','1'),('0','2'),('1','2')]])
    table('s10_seed_stability','Complete pairwise seed stability in the matched-interface experiment. scFocus uses adjusted Rand index (ARI) between hard assignments; scRL uses Spearman correlation of cell values. Neutral denotes no-time reward. Pairwise comparisons share runs and are not independent replicates.', ['Task / arm','Method','Reward','Metric','0 vs 1','0 vs 2','1 vs 2'], [.21,.11,.17,.12,.10,.10,.10],display)

    rows=read_csv(FROZEN/'marrow/reward_timing.csv')
    assert len(rows)==18
    csv_out('s11_marrow_reward_timing',rows)
    table('s11_marrow_reward_timing','All 18 marrow reward-control runs. Top-decile timing is the mean grid-derived pseudotime of cells in the top value decile. Correlation is the saved value--pseudotime Spearman coefficient. Neutral denotes no-time reward. Sorted MEP and GMP labels define erythroid and myeloid endpoints in this single-donor technical control.', ['Lineage','Reward','Seed','Top-decile timing','Value--pseudotime correlation'], [.17,.20,.10,.22,.24], [[r['lineage'],r['reward'],r['seed'],number(r['top_decile_mean_pseudotime']),number(r['q_pseudotime_spearman'])] for r in rows])

    rows=[]
    for c in COHORTS:
        r=read_json(NATIVE/'native_representation'/c/'receipt.json')
        rows.append({'cohort':c,**{k:r[k] for k in ['training_cells','optimization_validation_cells','actual_epochs','best_epoch','initial_validation_loss','best_validation_loss','relative_optimization_loss_reduction','mean_15NN_overlap_with_frozen_PCA','median_15NN_overlap_with_frozen_PCA']}})
    csv_out('s12_representation_training',rows)
    table('s12_representation_training','Saved upstream LiVAE optimization diagnostics. All fits used seed 0 and reached 200 epochs. Losses describe the declared weighted objective; relative reduction and neighbour overlap are saved diagnostics, not independent biological validation. The companion CSV includes exact train/validation counts and median neighbour overlaps.', ['Cohort','Best epoch','Initial loss','Best loss','Reduction','Mean 15NN overlap'], [.15,.14,.18,.18,.14,.15], [[COHORT_LABEL[r['cohort']],r['best_epoch'],number(r['initial_validation_loss'],2),number(r['best_validation_loss'],2),number(r['relative_optimization_loss_reduction']),number(r['mean_15NN_overlap_with_frozen_PCA'])] for r in rows])

    associations=read_csv(FROZEN/'geometry_reference/associations.csv',True)
    rows=[]
    for c,lineage in TASKS:
        selected=[r for r in associations if r['cohort']==c and r['lineage']==lineage]
        p=protocols[c]
        bypop={r['population']:r for r in selected}
        rows.append({'cohort':c,'lineage':lineage,'root':p['root'],'start_labels':';'.join(p['starts']),'endpoint_labels':';'.join(p['ends'][lineage]),'n_all':bypop['all']['n_cells'],'n_nonterminal':bypop['nonterminal']['n_cells'],'n_starting':bypop['starting']['n_cells'],'n_endpoint':selected[0]['n_endpoint'],'program':selected[0]['program']})
    csv_out('s13_task_denominators',rows)
    table('s13_task_denominators','Frozen task labels and subset denominators, shared by the matched-interface arms. All application roots and starting labels are primitive. Erythroid endpoints are sparse in Dapp1; sleep has no added erythroid task. Denominators come from the saved geometry-association tables.', ['Task','Endpoint labels','All','Nonterminal','Starting','Endpoint'], [.15,.25,.13,.14,.12,.12], [[task(r),r['endpoint_labels'].replace(';',', '),cell(r['n_all']),cell(r['n_nonterminal']),cell(r['n_starting']),cell(r['n_endpoint'])] for r in rows])

    # Field-selected aggregate inventory; no historical receipt files required.
    metadata = PROJECT / 'aggregate_metadata.json'
    SOURCES['aggregate_metadata.json'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    rows = json.loads(metadata.read_text())['matched_run_inventory']
    assert len(rows) == 90
    assert sum(r['method'] == 'scRL' for r in rows) == 72
    csv_out('matched_run_inventory',rows)
    (OUT/'source_checksums.json').write_text(json.dumps({'scope':list(COHORTS),'application_cells':79262,'matched_downstream_runs':90,'operation':'filter, reshape and format existing saved CSV/JSON; no scientific analysis recomputed','source_sha256':SOURCES},indent=2)+'\n')
    print(json.dumps({'tables':13,'matched_runs':len(rows),'source_files':len(SOURCES),'output_directory':str(OUT)},indent=2))


if __name__=='__main__':
    main()
