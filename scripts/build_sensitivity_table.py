#!/usr/bin/env python3
"""Format S20 and its numerical claim summary from supplied marker sensitivity.

This display utility does not reconstruct counts. Use calculate_program_sensitivity.py
against the extracted scientific supplements first to recompute the inputs.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--input-dir', type=Path)
    ap.add_argument('--output-root', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    source = (args.input_dir or root/'sensitivity_data').resolve()
    dest = (args.output_root or root).resolve()
    file = source/'marker_sensitivity_full_precision.csv'
    with file.open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 39
    for row in rows:
        assert row['n_eligible_groups'] == ('11' if row['cohort'] == 'dapp1' else '19')
        assert float(row['fraction_cells_covered']) == 1
    out = dest/'supplement_tables'
    out.mkdir(parents=True,exist_ok=True)
    (out/'s20_marker_sensitivity.csv').write_bytes(file.read_bytes())
    tex = [r'\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{longtable}{@{}llp{5.4cm}rr@{}}',
        r'\caption{Marker sensitivity on the same observed cells and frozen groups. Each full program contains six genes. Single-marker scores and omission scores use the original cohort-wise gene standardization. Raw and fixed contrasts are KO minus WT for Dapp1 and SD minus CT for sleep. Fixed contrasts use the original pooled weights over all 11 Dapp1 or 19 sleep groups; all cells are covered. These alternatives do not provide independent biological validation or sensitivity to another partition. Full-precision values and exact score genes accompany this table as CSV.}\label{tab:s20}\\',
        r'\toprule Cohort & Program & Score definition & Raw & Fixed\\\midrule\endfirsthead',
        r'\multicolumn{5}{l}{\textbf{Supplementary Table S20 continued}}\\',
        r'\toprule Cohort & Program & Score definition & Raw & Fixed\\\midrule\endhead',
        r'\bottomrule\endfoot']
    last = None
    for r in rows:
        key = r['cohort'],r['program']
        if last is not None and key != last:
            tex.append(r'\addlinespace[4pt]')
        label = {'full_program':'Full six-gene program','single_marker':'Single: '+r['marker'],
                 'leave_one_marker_out':'Omit '+r['marker']}[r['analysis_type']]
        cohort={'dapp1':'Dapp1','sleep':'Sleep'}[r['cohort']]
        program={'neutrophil':'Granulocytic','cycling':'Cycling'}[r['program']]
        tex.append(f"{cohort} & {program} & {label} & {float(r['unadjusted_delta']):+.3f} & {float(r['composition_standardized_delta']):+.3f}"+r'\\')
        last=key
    tex += [r'\end{longtable}',r'\normalsize','']
    (out/'s20_marker_sensitivity.tex').write_text('\n'.join(tex))
    summary={}
    for cohort,program in [('dapp1','neutrophil'),('dapp1','cycling'),('sleep','neutrophil')]:
        group=[r for r in rows if r['cohort']==cohort and r['program']==program]
        omitted=[r for r in group if r['analysis_type']=='leave_one_marker_out']
        fixed=[float(r['composition_standardized_delta']) for r in omitted]
        raw=[float(r['unadjusted_delta']) for r in omitted]
        summary[cohort+'_'+program]={'n_omissions':len(omitted),'fixed_min':min(fixed),'fixed_max':max(fixed),
            'raw_min':min(raw),'raw_max':max(raw),'all_fixed_negative':all(x<0 for x in fixed),
            'all_fixed_positive':all(x>0 for x in fixed),'all_raw_negative':all(x<0 for x in raw)}
    mpo=next(r for r in rows if r['cohort']=='sleep' and r['analysis_type']=='leave_one_marker_out' and r['marker']=='Mpo')
    summary['sleep_neutrophil']['omit_Mpo_fixed']=float(mpo['composition_standardized_delta'])
    weights=json.loads((source/'marker_sensitivity_manifest.json').read_text())['weight_sign_results']
    summary['weight_sign_results']=weights
    summary['source']={'path':'sensitivity_data/'+file.name,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'rows':len(rows)}
    target=dest/'evidence/program_sensitivity_summary.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'outputs':['supplement_tables/s20_marker_sensitivity.csv','supplement_tables/s20_marker_sensitivity.tex','evidence/program_sensitivity_summary.json']}))


if __name__=='__main__':
    main()
