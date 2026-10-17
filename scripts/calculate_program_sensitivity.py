#!/usr/bin/env python3
"""Marker sensitivity from retained integer counts; no refitting or biological replication.

Uses retained integer marker counts and library totals over ALL retained genes.
Groups and eligible pooled composition weights remain fixed. The full score,
each constituent gene score and each single-gene-omission score are diagnostic
alternatives on the same observed libraries, not independent confirmations.

Example:
  python calculate_program_sensitivity.py --root /path/to/scientific-reports-hematopoietic-readouts --output-dir /tmp/marker-check

The two existing tolerance policies are read without alteration. Saved-score
summary replay uses saved_summary; fresh float32 expression reduction uses
expression_float32. These checks must not be conflated.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(name, actual, expected, policy):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, name
    result = {'name': name, 'n_values': int(actual.size),
              'rtol': policy['rtol'], 'atol': policy['atol'],
              'max_abs_error': float(np.max(np.abs(actual-expected))),
              'passed': bool(np.allclose(actual, expected, **policy))}
    if not result['passed']:
        raise AssertionError(json.dumps(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out = (args.output_dir or root/'sensitivity_data').resolve()
    out.mkdir(parents=True, exist_ok=True)
    tolerance_file = root/'analysis_replay/tolerances.json'
    tolerance = json.loads(tolerance_file.read_text())
    summary_policy = {k: tolerance['saved_summary'][k] for k in ['rtol','atol']}
    expression_policy = {k: tolerance['expression_float32'][k] for k in ['rtol','atol']}
    reference_path = root/'supplement_tables/s02_program_contrasts.csv'
    reference = pd.read_csv(reference_path)
    source_files = {tolerance_file, reference_path}
    checks, rows, group_rows, saved_reference_rows, sign_results = [], [], [], [], []

    for cohort, program in [('dapp1','neutrophil'), ('dapp1','cycling'), ('sleep','neutrophil')]:
        folder = root/'analysis_replay/inputs/frozen'/cohort
        input_paths = {folder/n for n in ['expression_support.npz','expression_support.json','cells.csv.gz','protocol.json','marker_log1p.csv.gz']}
        source_files.update(input_paths)
        cells = pd.read_csv(folder/'cells.csv.gz')
        protocol = json.loads((folder/'protocol.json').read_text())
        support = np.load(folder/'expression_support.npz', allow_pickle=False)
        assert np.array_equal(cells.cell_id.to_numpy(), support['cell_id'])
        assert np.array_equal(support['full_retained_genes'][support['marker_gene_indices']], support['marker_genes'])
        assert (support['library_counts_all_retained_genes'] > 0).all()
        assert (support['marker_counts'].sum(axis=1) <= support['library_counts_all_retained_genes']).all()
        observed = pd.read_csv(folder/'marker_log1p.csv.gz', index_col=0)
        assert np.array_equal(observed.index.to_numpy(), support['cell_id'])
        assert np.array_equal(observed.columns.to_numpy(), support['marker_genes'])
        divisor = support['library_counts_all_retained_genes'].astype(np.float32)/np.float32(10000)
        expression = pd.DataFrame(np.log1p(support['marker_counts'].astype(np.float32)/divisor[:,None]), columns=support['marker_genes'])
        checks.append(verify(cohort+'/'+program+': reconstructed marker log1p', expression, observed, expression_policy))
        genes = protocol['program_genes'][program]
        assert len(genes) == 6 and len(set(genes)) == 6
        selected = expression[genes]
        z = (selected-selected.mean())/selected.std(ddof=0).replace(0,1)
        full_score = z.mean(axis=1)
        checks.append(verify(cohort+'/'+program+': full per-cell score', full_score, cells['program_'+program], expression_policy))

        condition_a, condition_b = ('wt','ko') if cohort == 'dapp1' else ('CT','SD')
        group = cells.leiden.astype(str)
        counts = pd.crosstab(group, cells.condition)
        eligible = counts.index[(counts[condition_a]>=20)&(counts[condition_b]>=20)]
        weights = counts.loc[eligible, [condition_a,condition_b]].sum(axis=1).astype(float)
        weights /= weights.sum()
        covered = float(counts.loc[eligible].to_numpy().sum()/len(cells))
        assert len(eligible) == (11 if cohort=='dapp1' else 19)
        assert covered == 1.0
        assert np.isclose(weights.sum(), 1, rtol=0, atol=1e-14)
        for g in eligible:
            group_rows.append({'cohort':cohort, 'program':program, 'cluster':g,
                               'condition_a':condition_a,'condition_b':condition_b,
                               'n_a':int(counts.loc[g,condition_a]),'n_b':int(counts.loc[g,condition_b]),
                               'pooled_weight':float(weights.loc[g])})

        def summarize(score):
            frame = pd.DataFrame({'score':np.asarray(score,dtype=float),'condition':cells.condition,'group':group})
            means = frame.groupby(['group','condition']).score.mean().unstack()
            condition_means = frame.groupby('condition').score.mean()
            differences = means.loc[eligible,condition_b]-means.loc[eligible,condition_a]
            return float(condition_means.loc[condition_b]-condition_means.loc[condition_a]), float((differences*weights).sum()), differences

        saved_raw, saved_fixed, saved_differences = summarize(cells['program_'+program])
        target = reference[(reference.cohort==cohort)&(reference.program=='program_'+program)].iloc[0]
        checks.append(verify(cohort+'/'+program+': SAVED per-cell score summary against S2',
                             [saved_raw,saved_fixed], [target.unadjusted_delta,target.composition_standardized_delta], summary_policy))
        saved_reference_rows.append({'cohort':cohort,'program':program,
                                     'saved_per_cell_raw_recomputed':saved_raw,'existing_s02_raw':float(target.unadjusted_delta),
                                     'saved_per_cell_fixed_recomputed':saved_fixed,'existing_s02_fixed':float(target.composition_standardized_delta),
                                     'raw_absolute_error':abs(saved_raw-target.unadjusted_delta),
                                     'fixed_absolute_error':abs(saved_fixed-target.composition_standardized_delta)})
        full_raw, full_fixed, _ = summarize(full_score)
        checks.append(verify(cohort+'/'+program+': RECONSTRUCTED float32 score summary against S2',
                             [full_raw,full_fixed],[target.unadjusted_delta,target.composition_standardized_delta],expression_policy))
        sign_results.append({'cohort':cohort,'program':program,'n_negative_saved_group_contrasts':int((saved_differences<0).sum()),
                             'n_positive_saved_group_contrasts':int((saved_differences>0).sum()),
                             'min_saved_group_contrast':float(saved_differences.min()),'max_saved_group_contrast':float(saved_differences.max()),
                             'direction_invariant_under_any_nonnegative_common_weights_on_these_groups':bool((saved_differences<0).all() or (saved_differences>0).all())})

        alternatives = [('full_program','',genes,full_score)]
        alternatives += [('single_marker',g,[g],z[g]) for g in genes]
        alternatives += [('leave_one_marker_out',g,[h for h in genes if h!=g],z[[h for h in genes if h!=g]].mean(axis=1)) for g in genes]
        for kind, marker, score_genes, score in alternatives:
            raw,fixed,_ = summarize(score)
            rows.append({'cohort':cohort,'program':program,'contrast':condition_b+'-'+condition_a,
                         'analysis_type':kind,'marker':marker,'n_score_genes':len(score_genes),'score_genes':';'.join(score_genes),
                         'n_condition_a':int((cells.condition==condition_a).sum()),'n_condition_b':int((cells.condition==condition_b).sum()),
                         'n_eligible_groups':len(eligible),'fraction_cells_covered':covered,
                         'unadjusted_delta':raw,'composition_standardized_delta':fixed,
                         'origin':'retained_count_marker_sensitivity_20261017','normalization_denominator':'all_retained_genes',
                         'score_reduction':'float32_normalization_log1p_gene_zscore_and_program_mean;float64_contrast'})

    assert len(rows) == 39
    paths = []
    for name, values in [('marker_sensitivity_full_precision.csv',rows),('marker_sensitivity_group_weights.csv',group_rows),
                         ('marker_sensitivity_existing_reference_checks.csv',saved_reference_rows)]:
        path=out/name
        pd.DataFrame(values).to_csv(path,index=False,float_format='%.17g')
        paths.append(path)
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'analysis_origin':'Diagnostic marker sensitivity from retained counts; distinct from original saved-score summaries',
                'script':{'name':Path(__file__).name,'sha256':sha(Path(__file__))},
                'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
                'inputs':{str(p.relative_to(root)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(source_files)},
                'outputs':{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths},
                'tolerance_policy_version':tolerance['policy_version'],'checks':checks,'all_checks_passed':all(c['passed'] for c in checks),
                'full_program_versus_saved_reference':'Fresh count reductions agree under expression_float32; saved per-cell score replay agrees separately under saved_summary. Bitwise identity of fresh expression is not asserted.',
                'weight_sign_results':sign_results,
                'boundaries':{'refitting':False,'reannotation':False,'reclustering':False,'new_observations':False,
                              'independent_biological_replication':False,'partition_robustness_established':False,
                              'general_marker_set_robustness_established':False,'causal_composition_decomposition':False},
                'interpretation':'Marker omission is sensitivity on the same cells and frozen groups. It is not a held-out test because endpoint and cluster definitions can reuse these markers. Convex-weight sign invariance concerns only the same 11 Dapp1 clusters; it does not establish robustness to another partition.'}
    (out/'marker_sensitivity_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'all_checks_passed':manifest['all_checks_passed'],
                      'max_saved_summary_error':max(c['max_abs_error'] for c in checks if 'SAVED per-cell' in c['name']),
                      'max_fresh_expression_summary_error':max(c['max_abs_error'] for c in checks if 'RECONSTRUCTED float32 score summary' in c['name'])},indent=2))


if __name__=='__main__':
    main()
