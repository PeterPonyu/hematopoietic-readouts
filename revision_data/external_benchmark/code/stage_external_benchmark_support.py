#!/usr/bin/env python3
"""Stage reviewed benchmark evidence as a closed manuscript-support packet.

This does not edit manuscript prose or the historical scientific tables. The
caller integrates the new TeX inputs only after reviewing the staged evidence.
"""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/external_method_benchmark_20261021'
PROJECT=ROOT/'paper/scientific-reports-hematopoietic-readouts'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tex(value):
    return str(value).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')


def number(value):return f'{value:.3f}' if np.isfinite(value) else 'undefined'


def interval(row):
    if not row.n_defined:return 'invalid (0/3)' if 'failed_fit' in row.statuses else 'undefined'
    if row.n_runs==1:return '$'+number(row['median'])+'$'
    return r'\shortstack{$'+number(row['median'])+r'$\\{\tiny$('+number(row.minimum)+', '+number(row.maximum)+r')$}}'


def table(columns,rows,widths):
    header=' & '.join(columns)+r' \\'
    output=[r'{\scriptsize\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.18}',
            r'\begin{longtable}{@{}'+''.join('p{'+w+r'\textwidth}' for w in widths)+r'@{}}',
            r'\toprule',header,r'\midrule\endfirsthead',r'\toprule',header,r'\midrule\endhead']
    output += [' & '.join(row)+r' \\' for row in rows]
    output += [r'\bottomrule\end{longtable}}']
    return '\n'.join(output)+'\n'


def build_table(folder,packet):
    summary=pd.read_csv(folder/'summary.csv');primary=summary.loc[summary.population.eq('shared_nonterminal')]
    applicability=pd.read_csv(folder/'applicability.csv')
    coverage=pd.read_csv(folder/'lineage_coverage.csv').fillna('')
    trajectory=pd.read_csv(folder/'trajectory_associations.csv')
    inventory=pd.read_csv(folder/'run_inventory.csv').fillna('')
    cohort_names={'irradiation':'Irradiation','ucb':'TPO culture','dapp1':'Dapp1','sleep':'Sleep loss'}
    order=['irradiation','ucb','dapp1','sleep']
    methods=['scRL value','PCA30 proximity','UMAP proximity','Diffusion proximity','Palantir','CellRank 2','FateID Python']
    tasks=[('irradiation','erythroid'),('irradiation','myeloid'),('dapp1','erythroid'),('dapp1','myeloid')]
    output=[r'\clearpage',r'\subsection*{Supplementary Table S9. External configured inference comparisons}',
            'All results use the frozen cohorts, labels and expression programs. These are same-data associations rather than measured future-fate accuracy. The primary population excludes every supplied terminal group: 21,468 irradiation and 8,298 Dapp1 cells. Myeloid tasks retain their broad terminal unions but are evaluated against the granulocytic program. Dapp1 erythroid remains an annotation-sensitive operational endpoint. Complete all-cell, starting-population and per-seed results accompany these displays.\n',
            r'\noindent\textbf{(a) Signed program associations in the common nonterminal population.} Entries are medians, with seed minima and maxima in parentheses; these ranges are not confidence intervals. Geometry uses the negative mean distance to 15 nearest full-target cells. Diffusion proximity uses min--max-scaled multiscale coordinates. Invalid Palantir irradiation scores are excluded from correlations and remain visible.\par']
    rows=[]
    for method in methods:
        row=[tex(method)]
        for cohort,task in tasks:
            selected=primary.loc[primary.cohort.eq(cohort)&primary.task.eq(task)&primary.method.eq(method)]
            assert len(selected)==1;row.append(interval(selected.iloc[0]))
        rows.append(row)
    output.append(table(['Score / reference','Irradiation erythroid','Irradiation myeloid','Dapp1 erythroid','Dapp1 myeloid'],rows,['.235','.17','.17','.17','.17']))
    output.append(r'\noindent\textbf{(b) Applicability and completed primary fits.} Counts are numerically valid fits / registered fits. A dash denotes a single-broad-target probability task without an informative competing broad fate; it does not indicate method failure. Slingshot was attempted on all four cohorts. Failed or resource-limited fits remain in the denominator.\par')
    rows=[]
    for cohort in order:
        subset=applicability.loc[applicability.cohort.eq(cohort)].set_index('method')
        row=[cohort_names[cohort],f'{int(subset.n_cells.iloc[0]):,}']
        for method in ['palantir','cellrank','fateid','slingshot']:
            r=subset.loc[method];row.append(f'{int(r.valid_runs)}/{int(r.planned_runs)}' if r.applicable else '---')
        rows.append(row)
    output.append(table(['Cohort','Cells','Palantir','CellRank 2','FateID Python','Slingshot'],rows,['.22','.13','.13','.14','.16','.12']))
    output.append(r'\noindent\textbf{(c) Slingshot lineage ordering and coverage.} Only lineages ending at a supplied terminal cluster are associated with a configured task here. Erythroid lineages use the erythroid program; all myeloid lineages, including monocyte and dendritic ends, retain the granulocytic program. These are lineage-specific masks, not new component-matched focal fits. Every inferred lineage, including unmatched ends, remains in the companion coverage table. Coverage divides assigned nonterminal cells by the entire nonterminal population of that cohort. Scores and references for each lineage use exactly its assigned-cell mask.\par')
    rows=[]
    selected=trajectory.loc[trajectory.population.eq('shared_nonterminal')&trajectory.method.eq('Slingshot pseudotime')].copy()
    selected['order']=selected.cohort.map({c:i for i,c in enumerate(order)});selected=selected.sort_values(['order','task','lineage'])
    registry=json.loads((OUT/'registration.json').read_text())
    for _,r in selected.iterrows():
        cov=coverage.loc[coverage.cohort.eq(r.cohort)&coverage.lineage.eq(r.lineage)].iloc[0]
        support=next(folder.glob(f'{r.cohort}_*_evaluation.csv.gz'))
        cells=pd.read_csv(support,dtype={'cell_id':str,'leiden':str}).set_index('cell_id')
        names=[name for name,cell in registry['cohorts'][r.cohort]['terminal_representatives'].items()
               if str(cells.loc[cell,'leiden'])==str(r.end_cluster)]
        assert names
        rows.append([cohort_names[r.cohort],tex(', '.join(names)),str(int(r.lineage)+1),f'{int(r.n_cells):,}',
                     f'{100*cov.fraction_nonterminal_covered:.1f}'+r'\%',number(r.spearman)])
    output.append(table(['Cohort','End component','Lineage','Assigned cells','Coverage',r'Signed $\rho$'],rows,['.205','.22','.10','.15','.12','.10']))
    output.append(r'\Needspace{.4\textheight}')
    output.append(r'\noindent\textbf{(d) Runtime and process memory of primary fits.} Runtime is median (minimum, maximum) over completed registered seeds, or the single deterministic fit. Peak RSS is the largest process peak across those fits, including the R child for Slingshot. Shared mappings can be counted in process RSS. Jobs ran concurrently; these records are descriptive and do not establish a hardware-controlled speed ranking. Technical pilot failures and resource restarts remain separately recorded.\par')
    rows=[]
    for cohort in order:
        for method,label in [('palantir','Palantir'),('cellrank','CellRank 2'),('fateid','FateID Python'),('slingshot','Slingshot')]:
            sub=inventory.loc[inventory.cohort.eq(cohort)&inventory.method.eq(method)&inventory.role.eq('primary_attempt')]
            if not len(sub):continue
            passed=sub.loc[sub.status.eq('pass')]
            used=passed if len(passed) else sub.loc[sub.status.eq('fail')]
            if not len(used):continue
            seconds=pd.to_numeric(used.elapsed_seconds,errors='coerce').dropna()
            rss=pd.to_numeric(used.peak_rss_MiB,errors='coerce').dropna()
            resource=used.error.astype(str).str.contains('Resource limit',case=False).any()
            if len(seconds):duration=f'{seconds.median():.1f}' if len(used)==1 else f'{seconds.median():.1f} ({seconds.min():.1f}, {seconds.max():.1f})'
            else:duration=r'$>3600$ (wall limit)' if resource else 'not retained'
            status='valid' if len(passed) else ('resource limit' if resource else 'invalid score')
            rows.append([cohort_names[cohort],label,status,duration,f'{rss.max()/1024:.2f}' if len(rss) else 'not retained'])
    output.append(table(['Cohort','Pipeline','Output status','Elapsed seconds','Peak RSS (GiB)'],rows,['.19','.19','.16','.26','.10']))
    output.append('Palantir receives one fixed PCA-centroid representative per component; CellRank and FateID receive full terminal membership. Slingshot uses the frozen Leiden root and terminal clusters. Terminal information therefore differs between configured pipelines. FateID is the retained Python variant; equivalence to the original R algorithm is not certified. The Slingshot Python pilot failed with invalid spline inputs; the reported results use upstream R Slingshot. The three Palantir irradiation runs failed native probability checks under the frozen endpoint configuration; their scores were not normalized or used for ranking.\n')
    (packet/'supplement_table.tex').write_text('\n\n'.join(output)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',default='analysis_v1');args=parser.parse_args()
    folder=OUT/args.analysis
    assert json.loads((folder/'verification.json').read_text())['status']=='pass'
    assert json.loads((folder/'replay_verification.json').read_text())['status']=='pass'
    review=json.loads((folder/'scientific_review.json').read_text())
    assert review['integration_decision']=='include_with_limits'
    assert json.loads((folder/'figures/visual_review.json').read_text())['status']=='pass'
    packet=PROJECT/'revision_data/external_benchmark';packet.mkdir(parents=True,exist_ok=False)
    files=[]
    def copy(source,relative):
        target=packet/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert digest(source)==digest(target);files.append(target)
    for path in sorted(folder.glob('*.csv*')):copy(path,Path('tables')/path.name)
    for name in ['verification.json','replay_verification.json']:copy(folder/name,Path('tables')/name)
    for path in sorted(OUT.glob('registration*.json')):copy(path,Path('protocol')/path.name)
    for source,name in [('pilot_review.json','pilot_validation.json'),('fateid_membership_equivalence.json','fateid_membership_equivalence.json'),
                        ('fateid_bounded_equivalence.json','fateid_bounded_equivalence.json'),('fateid_median_kernel_equivalence.json','fateid_median_kernel_equivalence.json'),
                        ('fateid_expression_input_bindings.json','fateid_expression_input_bindings.json')]:
        copy(OUT/source,Path('protocol')/name)
    copy(OUT/'palantir_irradiation_diagnostic/review.json',Path('protocol/palantir_irradiation_diagnostic.json'))
    for cohort in ['ucb','sleep']:
        for name in ['outputs.npz','receipt.json']:copy(OUT/cohort/'diffusion/seed0'/name,Path('runs')/cohort/'diffusion/seed0'/name)
    inventory=pd.read_csv(folder/'run_inventory.csv')
    allowed=['receipt.json','outputs.npz','driver_snapshot.py','traceback.txt','lineages.json',
             'fit.rds','native_outputs.npz','native_export_receipt.json','sessionInfo.txt','package_version.txt',
             'lineage_graph_diagnostic.csv']
    for run in inventory.run:
        relative=Path(run);assert not relative.is_absolute() and '..' not in relative.parts
        source=OUT/relative
        for name in allowed:
            if (source/name).is_file():copy(source/name,Path('runs')/relative/name)
        cohort,method,seed_name=relative.parts
        log_name=f'{cohort}_{method}_{seed_name}.log'
        log=OUT/'logs'/log_name
        if not log.exists():log=OUT/f'{method}_{cohort}_{seed_name}.log'
        if log.exists():copy(log,Path('logs')/log_name)
    for path in sorted((OUT/'manual_sources').rglob('*')):
        if path.is_file():copy(path,Path('manual_sources')/path.relative_to(OUT/'manual_sources'))
    for name in ['hematopoietic_external.py','run_external_batch.py','analyze_hematopoietic_external.py',
                 'replay_external_associations.py','build_external_benchmark_display.py','hematopoietic_slingshot.R',
                 'export_slingshot_native.py','export_slingshot_native.R','build_external_diffusion_control.py',
                 'stage_external_benchmark_support.py']:
        copy(ROOT/'scripts/benchmarks'/name,Path('code')/name)
    for name in ['benchmark_overlay_requirements.txt','slingshot_requirements.txt','r_local_packages.csv','r_slingshot_example_check.txt','observed_task_versions.json']:
        copy(OUT/'environment'/name,Path('environment')/name)
    for extension in ['pdf','png','svg']:copy(folder/'figures'/f'external_method_comparison.{extension}',Path('figure')/f'external_method_comparison.{extension}')
    for name in ['display_values.csv','display_receipt.json','visual_review.json']:
        copy(folder/'figures'/name,Path('figure')/('visual_validation.json' if name=='visual_review.json' else name))
    copy(OUT/'PROTOCOL_AND_REPLAY.txt',Path('PROTOCOL_AND_REPLAY.txt'))
    build_table(folder,packet);files.append(packet/'supplement_table.tex')
    methods=r'''\subsection*{External configured inference comparisons}
Four external method families broadened the geometric controls: locally refined
Palantir~\cite{setty2019palantir}, CellRank~2~\cite{weiler2024cellrank2}, a retained
Python variant of FateID~\cite{herman2018fateid}, and upstream R
Slingshot~\cite{street2018slingshot}. They reused the frozen cells, programs,
PCA30 coordinates and Leiden partition. The scRL comparison reuses the frozen
no-time-reward fits, rather than the separately fitted paired interfaces. The
task registration fixed roots,
terminal groups, seeds and evaluation populations before inspecting comparisons.
Technical amendments and every attempted fit are supplied separately. These are
configured-pipeline comparisons, not equal-information algorithm ablations.

The root cell was the cell nearest its root-label centroid in PCA30, with cell-ID
tie breaking. One analogous representative was fixed per supplied terminal
component. Palantir received those representatives; CellRank and FateID received
all cells in the component groups. Slingshot received the clusters containing
the root and terminal representatives. The terminal information therefore differed
by pipeline. Irradiation used four components (Ery, Gra, Mo and DC), and Dapp1
used erythroid, neutrophil and monocyte components. Broad-myeloid scores summed
the corresponding components and retained the granulocytic evaluation program.
TPO and sleep each had one configured broad target; its normalized probability
would be uninformative without independently justified competing broad fates.
Their trajectory and geometry comparisons were retained, without inventing
alternative endpoints.

Palantir used the existing locally refined Python implementation, with source
fingerprints supplied. Diffusion maps requested ten components with 30 neighbours
and anisotropy parameter zero. The automatic multiscale space was min--max scaled
per component for inference. The fit requested 1,200 waypoints and at most 25
pseudotime-refinement iterations, using the fixed root directly. The native
result constructor sets branch scores below 0.01 to zero without normalizing
their row sums; these scores were preserved. Valid outputs had finite
nonnegative scores, row mass no greater than one and a deficit no greater than
0.01 times the component count, within numerical tolerance. All three configured
irradiation fits failed these checks. Directed waypoint reachability and the
fundamental-system conditioning were inspected; the failed scores were not
normalized, filled or used for association rankings.

CellRank used a 30-neighbour PCA30 graph and root-derived diffusion pseudotime
with ten diffusion components~\cite{haghverdi2016dpt}. Its PseudotimeKernel applied
the soft directional threshold. Supplied component memberships were set directly
as terminal states in GPCCA, followed by direct absorption-probability solving.
Every supplied terminal member was checked, and score rows summed to one within
tolerance. Experimental sampling day and condition were absent from direction
construction; DPT is an expression-derived prior rather than an independent
temporal outcome.

The retained FateID Python port used saved log-normalized expression for the
2,000 selected genes. An explicit cell-distance matrix, one minus Pearson
correlation, avoided a gene/cell-axis defect in the port's default distance call.
The settings were 20 candidate cells, 50 training cells per component, 500 trees,
adaptive selection and confidence 0.75, with two forest threads. Distances were
stored as float32 after double-precision centering and normalization. Resource
refinements cached list membership, skipped identity reindex copies and computed
the same finite medians in 256-column blocks using four threads. They reproduced
the entire native Dapp1 seed-zero probability output exactly; full irradiation
median comparisons also matched exactly. The distance cache remained unchanged.
Every cell had to be processed; unprocessed uniform defaults were inadmissible.
Its internal binomial vote test selected trajectory updates and was not treated
as a biological significance test. The port converts mean leaf probabilities to rounded votes, differing from the
original R hard-tree votes. Equivalence to the R algorithm is not certified.

The Slingshot Python port passed a small branching example but failed the full
Dapp1 fit with invalid spline inputs. The registered fallback used upstream R
Slingshot with the same PCA30, frozen Leiden partition and root/terminal clusters,
ten maximum curve iterations, 150 approximation points and remaining upstream
defaults. Native doubles were exported directly from each saved R fit. A cell
contributed to a lineage only when its pseudotime was finite and its native
curve weight was positive; no missing scores were filled and weights were not
renormalized. Each lineage ending at a supplied terminal cluster was evaluated
separately against its task's program, with the identical cell mask applied to
scRL and geometry references. Every inferred lineage, including unmatched ends,
is retained in the coverage table. Displayed lineage numbers are one-based;
machine-readable array indices are zero-based. Curve weights and pseudotime
were not treated as fate probabilities.

The primary population excluded the union of all supplied terminal groups,
leaving 21,468 irradiation and 8,298 Dapp1 cells for probability comparisons.
This common mask differs from the task-endpoint-only exclusion in the frozen
six-task reference table. Assigned lineage sets can overlap, so their coverage
percentages need not sum to one. All-cell and frozen-starting-set associations
were secondary. Signed Spearman
correlations used average tie ranks within each evaluation population. Signs and
target definitions were not chosen from performance. Stochastic fits retained
seeds 0, 1 and 2; Slingshot used one fit on fixed inputs. Medians and ranges
describe algorithmic variation, not biological uncertainty. Undefined scores
remain explicit. Each PCA and UMAP reference was retained separately, and a
diffusion-space reference used the same 15-nearest-full-target proximity rule in
the min--max-scaled multiscale coordinates. Diffusion controls used three seeds
on probability cohorts and seed zero on TPO and sleep. Differences from each
reference subtract the corresponding score and reference medians; they are not
the median of paired seed differences. An independent replay checks average-rank
correlations, component aggregation, cell masks, coverage and direct distances
without refitting. These RNA-internal controls do not supply measured future
fates, independent annotations, calibration or biological replication.
'''
    (packet/'methods.tex').write_text(methods+'\n');files.append(packet/'methods.tex')
    readme='''External configured inference comparison\n\nExtract Supplementary Code 1 and every Supplementary Data archive into one parent directory.\nFrom the extracted project root, replay the new associations without model fitting:\n\npython revision_data/external_benchmark/code/replay_external_associations.py --folder revision_data/external_benchmark/tables --raw-root revision_data/external_benchmark/runs --registration revision_data/external_benchmark/protocol/registration.json\n\nNumPy and pandas are required. Figure regeneration additionally requires matplotlib and Arial:\n\npython revision_data/external_benchmark/code/build_external_benchmark_display.py --folder revision_data/external_benchmark/tables --registration revision_data/external_benchmark/protocol/registration.json --output-folder /tmp/external-benchmark-figure\n\nThe fitting sources document the original execution layout and isolated environments; saved-artifact replay does not repeat raw preparation or model fitting. Complete raw task outputs, failed native Palantir outputs, all attempt receipts and Slingshot R fits are retained. The initial Palantir validation failure predates automatic driver snapshots; its algorithm-source hashes and output are retained. Technical refinement check runs are separate from primary seed replication. No biological p-values or independent future-fate accuracy are inferred.\n'''
    (packet/'README.txt').write_text(readme);files.append(packet/'README.txt')
    record={'schema_version':1,'operation':'byte-identical scientific copies plus derived display table',
            'source_verification_sha256':digest(folder/'verification.json'),
            'source_replay_sha256':digest(folder/'replay_verification.json'),
            'files':[{'path':str(p.relative_to(PROJECT)),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(files)]}
    (packet/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'staged','files':len(files),'bytes':sum(x['bytes'] for x in record['files'])}))


if __name__=='__main__':main()
