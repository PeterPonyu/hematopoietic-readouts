#!/usr/bin/env python3
"""Display extra frozen-protocol summaries with precise, numbered destinations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output-root',type=Path)
    args=ap.parse_args()
    root=args.root.resolve(); dest=(args.output_root or root).resolve()
    prefix='source_data/results/application_reanalysis_20260921/'
    sources={}; rows=[]
    def read(name):
        path=root/(prefix+name)
        sources[prefix+name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
        with path.open(newline='') as f: return list(csv.DictReader(f))
    def add(cohort,quantity,setting,value,unit,source,n=''):
        rows.append(dict(cohort=cohort,quantity=quantity,setting=setting,value=value,unit=unit,n_cells=n,source=prefix+source))
    name='dapp1_spliced/composition_standardized_effects.csv'
    r=next(r for r in read(name) if r['program']=='program_neutrophil')
    for label,field in [('Spliced raw KO-WT','unadjusted_delta'),('Spliced fixed KO-WT','composition_standardized_delta')]:
        add('Dapp1','Granulocytic program',label,float(r[field]),'cohort-standardized program score',name)
    name='ucb/state_values_by_condition.csv'; data=read(name)
    for pop,label in [('all','All cells'),('starting_population','Cluster 5')]:
        for day in ['D0','D4','D7','D11','D14']:
            selected=[r for r in data if r['reward']=='neutral' and r['lineage']=='megakaryocyte' and r['population']==pop and r['condition']==day]
            assert len(selected)==3 and len({r['n_cells'] for r in selected})==1
            add('TPO','No-time value percentile',day+'; '+label,median(float(r['median_value_percentile']) for r in selected),'median of three seed-specific condition medians',name,selected[0]['n_cells'])
    name='dapp1/seed_stability.csv'
    for r in read(name):
        if r['method']=='scRL' and r['reward']=='neutral' and r['metric']=='cell_value_spearman' and r['lineage'] in ['erythroid','myeloid']:
            add('Dapp1','No-time seed agreement',r['lineage'].capitalize()+'; '+r['seed_a']+'-'+r['seed_b'],float(r['value']),'all-cell value Spearman correlation',name)
    name='ucb/condition_gene_means.csv'; data=read(name)
    for day,gene in [('D11','PF4'),('D11','PPBP'),('D0','ITGA2B'),('D11','ITGA2B')]:
        r=next(r for r in data if r['condition']==day)
        add('TPO',gene+' mean log expression',day,float(r[gene]),'mean log1p(counts per 10000)',name)
    assert len(rows)==22
    folder=dest/'supplement_tables'; folder.mkdir(parents=True,exist_ok=True)
    with (folder/'s21_additional_frozen_readouts.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    tex=[r'\small',r'\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{@{}lp{4.0cm}p{4.1cm}rr@{}}',
        r'\caption{Additional readouts under the frozen application protocol. The spliced-count granulocytic score is a separate expression-layer sensitivity. TPO value percentiles are the median across three seed-specific condition medians under no-time reward; cluster 5 denotes its intersection with each day. Seed agreement uses frozen PCA30-state/UMAP-action scRL values, rather than the separate matched two-dimensional experiment in Table S10. The four marker means use all TPO cells in the specified day. Full-precision CSV values identify their exact saved source files.}\label{tab:s21}\\',
        r'\toprule Cohort & Readout & Setting & Value & Cells\\\midrule\endfirsthead',
        r'\multicolumn{5}{l}{\textbf{Supplementary Table S21 continued}}\\',
        r'\toprule Cohort & Readout & Setting & Value & Cells\\\midrule\endhead',r'\bottomrule\endfoot']
    for r in rows:
        tex.append(f"{r['cohort']} & {r['quantity']} & {r['setting']} & {r['value']:.3f} & {r['n_cells']}"+r'\\')
    tex += [r'\end{longtable}',r'\normalsize','']
    (folder/'s21_additional_frozen_readouts.tex').write_text('\n'.join(tex))
    target=dest/'evidence/frozen_readout_table_sources.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps({'operation':'filter and format saved aggregate summaries; median of saved seed summaries only','rows':len(rows),'sources':sources},indent=2)+'\n')
    print(json.dumps({'rows':len(rows),'all_sources_bound':True}))


if __name__=='__main__': main()
