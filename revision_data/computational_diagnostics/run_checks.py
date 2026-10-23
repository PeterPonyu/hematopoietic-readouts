"""Post-hoc diagnostics using immutable, staged saved outputs; never fits models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'inputs'
OUT = HERE / 'results'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name, **kwargs):
    return pd.read_csv(INPUT / name, float_precision='round_trip', **kwargs)


def limits(values):
    assert np.isfinite(values).all()
    low, high = np.quantile(values, [.025, .975], method='linear')
    half = np.quantile(values[:len(values)//2], [.025, .975], method='linear')
    return dict(lower=float(low), upper=float(high),
                resampling_SD=float(np.std(values, ddof=1)),
                positive_resample_fraction=float(np.mean(values > 0)),
                half_vs_full_max_quantile_difference=float(np.max(np.abs(half-[low, high]))))


def composition(protocol, saved):
    reference = read('marker_sensitivity.csv', keep_default_na=False)
    original = read('program_contrasts.csv')
    rows = []
    for ci, (cohort, programs) in enumerate(protocol['composition']['programs'].items()):
        cells = read(cohort+'_cells.csv.gz')
        expression = read(cohort+'_marker_log1p.csv.gz', index_col=0).astype(np.float32)
        assert np.array_equal(expression.index.astype(str), cells.cell_id)
        definitions = []
        source_protocol = json.loads((INPUT/(cohort+'_protocol.json')).read_text())
        for program in programs:
            genes = source_protocol['program_genes'][program]
            assert len(genes) == 6
            selected = expression[genes]
            z = (selected-selected.mean())/selected.std(ddof=0).replace(0, 1)
            definitions.append((program, 'full_program', '', cells['program_'+program].to_numpy()))
            definitions += [(program, 'single_marker', g, z[g].to_numpy()) for g in genes]
            definitions += [(program, 'leave_one_marker_out', g,
                             z[[h for h in genes if h != g]].mean(axis=1).to_numpy()) for g in genes]
        matrix = np.column_stack([d[3] for d in definitions]).astype(float)
        condition_a, condition_b = ('wt', 'ko') if cohort == 'dapp1' else ('CT', 'SD')
        group = cells.leiden.astype(str)
        counts = pd.crosstab(group, cells.condition)
        eligible = counts.index[(counts[condition_a] >= 20) & (counts[condition_b] >= 20)]
        assert len(eligible) == (11 if cohort == 'dapp1' else 19)
        assert counts.loc[eligible].to_numpy().sum() == len(cells)
        weights = counts.loc[eligible].sum(axis=1)/len(cells)
        rng = np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'], 100+ci]))
        shape = (protocol['n_resamples'], len(definitions))
        raw, fixed = np.zeros(shape), np.zeros(shape)
        point_raw, point_fixed = np.zeros(len(definitions)), np.zeros(len(definitions))
        strata = []
        for label in eligible:
            for condition, sign in [(condition_a, -1), (condition_b, 1)]:
                mask = group.eq(label) & cells.condition.eq(condition)
                x = matrix[mask]
                raw_weight = sign*len(x)/int(cells.condition.eq(condition).sum())
                fixed_weight = sign*float(weights.loc[label])
                point_raw += raw_weight*x.mean(axis=0)
                point_fixed += fixed_weight*x.mean(axis=0)
                strata.append(dict(cohort=cohort, cluster=label, condition=condition,
                                   n_cells=len(x), original_pooled_weight=float(weights.loc[label])))
                for start in range(0, len(raw), 16):
                    stop = min(start+16, len(raw))
                    indices = rng.integers(0, len(x), size=(stop-start, len(x)))
                    mean = x[indices].mean(axis=1)
                    raw[start:stop] += raw_weight*mean
                    fixed[start:stop] += fixed_weight*mean
        pd.DataFrame(strata).to_csv(OUT/(cohort+'_strata.csv'), index=False)
        for j, (program, kind, marker, _) in enumerate(definitions):
            expected = reference.loc[reference.cohort.eq(cohort) & reference.program.eq(program)
                & reference.analysis_type.eq(kind) & reference.marker.eq(marker)]
            assert len(expected) == 1
            expected = expected.iloc[0]
            if kind == 'full_program':
                baseline = original.loc[original.cohort.eq(cohort)
                    & original.program.eq('program_'+program)].iloc[0]
            else:
                baseline = expected
            tolerance = 1e-10 if kind == 'full_program' else 2e-6
            for stat, observed, distribution, column in [
                ('raw', point_raw[j], raw[:, j], 'unadjusted_delta'),
                ('fixed', point_fixed[j], fixed[:, j], 'composition_standardized_delta')]:
                point = float(baseline[column])
                assert abs(observed-point) <= tolerance, (cohort, program, kind, marker, stat, observed, point)
                rows.append(dict(cohort=cohort, program=program, score_definition=kind,
                    marker=marker, statistic=stat, point_copied_from_existing_table=point,
                    unresampled_conformance_error=float(abs(observed-point)),
                    **limits(distribution)))
        saved[cohort+'_composition_raw'] = raw
        saved[cohort+'_composition_fixed'] = fixed
        print('Composition diagnostics complete:', cohort, flush=True)
    result = pd.DataFrame(rows)
    assert len(result) == 78
    result.to_csv(OUT/'composition_resampling.csv', index=False, float_format='%.17g')


def correlations(matrix):
    """Each replicate recalculates average-tie ranks over its sampled rows."""
    ranks = rankdata(matrix, axis=1, method='average')
    ranks -= ranks.mean(axis=1, keepdims=True)
    target, scores = ranks[:, :, :1], ranks[:, :, 1:]
    denominator = np.sqrt((target**2).sum(axis=1)*(scores**2).sum(axis=1))
    assert (denominator > 0).all()
    return (target*scores).sum(axis=1)/denominator


def starting_correlations(protocol, saved):
    summary, associations = read('geometry_summary.csv'), read('geometry_associations.csv')
    rows = []
    for ci, (cohort, lineage) in enumerate([('ucb', 'megakaryocyte'), ('sleep', 'myeloid')]):
        cells = read(cohort+'_cells.csv.gz')
        scores = read(f'{cohort}_{lineage}_cells.csv.gz')
        assert np.array_equal(cells.cell_id, scores.cell_id)
        assert np.array_equal(scores.starting, cells.cell_type.eq('primitive'))
        assert np.array_equal(scores.program, cells['program_'+('megakaryocyte' if cohort == 'ucb' else 'neutrophil')])
        mask = scores.starting.to_numpy()
        selected = scores.loc[mask, ['program', 'scRL seed 0', 'scRL seed 1', 'scRL seed 2',
                                     'UMAP distance', 'PCA30 distance']].to_numpy()
        observed = correlations(selected[None, :, :])[0]
        baseline = summary.loc[summary.cohort.eq(cohort) & summary.lineage.eq(lineage)
                               & summary.population.eq('starting')].iloc[0]
        per_seed = associations.loc[associations.cohort.eq(cohort)
            & associations.lineage.eq(lineage) & associations.population.eq('starting')].set_index('method')
        expected_seed = np.array([per_seed.loc[f'scRL seed {seed}', 'spearman'] for seed in range(3)])
        np.testing.assert_allclose(observed[:3], expected_seed, atol=1e-10, rtol=0)
        np.testing.assert_allclose(observed[3:], [baseline.umap, baseline.pca], atol=1e-10, rtol=0)
        rng = np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'], 200+ci]))
        condition = cells.loc[mask, 'condition'].to_numpy()
        strata = [np.flatnonzero(condition == c) for c in np.unique(condition)]
        values = np.empty((protocol['n_resamples'], 5))
        for start in range(0, len(values), 32):
            stop = min(start+32, len(values))
            indices = np.empty((stop-start, len(selected)), dtype=int)
            for members in strata:
                indices[:, members] = rng.choice(members, size=(stop-start, len(members)), replace=True)
            values[start:stop] = correlations(selected[indices])
        for ref, k, original_ref in [('UMAP', 3, float(baseline.umap)), ('PCA30', 4, float(baseline.pca))]:
            for seed in range(4):
                name = f'seed {seed}' if seed < 3 else 'three-seed median'
                model = values[:, seed] if seed < 3 else np.median(values[:, :3], axis=1)
                original_model = expected_seed[seed] if seed < 3 else float(baseline.scrl_median)
                delta = model-values[:, k]
                rows.append(dict(cohort=cohort, lineage=lineage, population='original starting cells',
                    n_cells=int(mask.sum()), aggregation=name, reference=ref,
                    model_point_copied=float(original_model), reference_point_copied=original_ref,
                    point_increment=float(original_model-original_ref), **limits(delta)))
        saved[cohort+'_starting_correlations'] = values
        print('Paired starting-cell correlation diagnostics complete:', cohort, flush=True)
    pd.DataFrame(rows).to_csv(OUT/'starting_correlation_resampling.csv', index=False, float_format='%.17g')


def tpo_masks():
    cells = read('ucb_cells.csv.gz')
    scores = read('ucb_megakaryocyte_cells.csv.gz')
    with np.load(INPUT/'ucb_expression_support.npz', allow_pickle=False) as support:
        assert np.array_equal(support['cell_id'], cells.cell_id)
        column = np.flatnonzero(support['marker_genes'] == 'CD34')
        assert len(column) == 1
        detected = support['marker_counts'][:, column[0]] > 0
    assert np.array_equal(cells.cell_id, scores.cell_id)
    d0 = cells.condition.eq('D0').to_numpy()
    cluster = cells.leiden.astype(str)
    masks = {'all D0': d0, 'D0 cluster 5': d0 & cluster.eq('5'),
             'D0 clusters 0 and 1': d0 & cluster.isin(['0', '1']),
             'D0 CD34 count > 0': d0 & detected,
             'original cluster 5, all days': cluster.eq('5')}
    assert int(np.sum(masks['D0 cluster 5'])) == 17
    assert np.array_equal(masks['original cluster 5, all days'], scores.starting)
    rows = []
    for name, mask in masks.items():
        mask = np.asarray(mask)
        target = scores.loc[mask, 'program'].to_numpy()
        assert mask.sum() >= 3
        for seed in range(3):
            value = scores.loc[mask, f'scRL seed {seed}'].to_numpy()
            rho = float(spearmanr(value, target).statistic) if np.ptp(value) > 0 and np.ptp(target) > 0 else np.nan
            low, high = np.quantile(value, [.25, .75], method='linear')
            rows.append(dict(mask=name, n_cells=int(mask.sum()), seed=seed,
                SD=float(value.std(ddof=0)), IQR=float(high-low), q25=float(low), q75=float(high),
                mean=float(value.mean()), spearman_megakaryocytic=rho,
                mask_sha256=hashlib.sha256('\n'.join(cells.cell_id[mask]).encode()).hexdigest(),
                correlation_status='defined' if np.isfinite(rho) else 'undefined constant input'))
    pd.DataFrame(rows).to_csv(OUT/'TPO_mask_diagnostics.csv', index=False, float_format='%.17g')
    pd.DataFrame({'cell_id': cells.cell_id, **masks}).to_csv(OUT/'TPO_masks.csv.gz', index=False,
            compression={'method':'gzip', 'mtime':0})


def main():
    started = time.time()
    protocol = json.loads((HERE/'protocol.json').read_text())
    assert protocol['status'] == 'fixed_before_execution'
    for name, expected in protocol['inputs'].items():
        assert sha(INPUT/name) == expected['sha256'], name
    OUT.mkdir(exist_ok=False)
    samples = {}
    composition(protocol, samples)
    starting_correlations(protocol, samples)
    tpo_masks()
    np.savez_compressed(OUT/'resampling_distributions.npz', **samples)
    record = dict(status='pass', finished_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.time()-started, protocol_sha256=sha(HERE/'protocol.json'),
        script_sha256=sha(__file__), input_files_verified=len(protocol['inputs']),
        n_resamples=protocol['n_resamples'], composition_statistic_rows=78,
        paired_correlation_increment_rows=16, TPO_mask_seed_rows=15,
        fitting=False, original_points_replaced=False, biological_inference=False,
        packages={name: __import__(name).__version__ for name in ['numpy', 'pandas', 'scipy']},
        outputs={f.name:dict(sha256=sha(f), bytes=f.stat().st_size) for f in sorted(OUT.iterdir()) if f.is_file()})
    (HERE/'execution.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
