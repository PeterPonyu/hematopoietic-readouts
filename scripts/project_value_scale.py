#!/usr/bin/env python3
"""Project saved grid values and audit their scale, without fitting a model.

The projection is the frozen 15-neighbour Gaussian projection used in r4.
Population SD uses ddof=0; quartiles use NumPy's linear interpolation. A check
also recalculates all 189 saved population-specific Spearman correlations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculate(p):
    r = p / 'revision_data/targeted_r4'
    cells = pd.read_csv(p/'analysis_replay/inputs/frozen/ucb/cells.csv.gz',
                        dtype={'cell_id': str}, float_precision='round_trip')
    embedding = np.load(p/'analysis_replay/inputs/matched/LiVAE/ucb/input.npz')['embedding']
    masks = {'all': np.ones(len(cells), bool),
             'nonterminal': ~cells.cell_type.eq('megakaryocyte').to_numpy(),
             'starting': cells.cell_type.eq('primitive').to_numpy()}
    target = cells.program_megakaryocyte.to_numpy()
    rows, sources = [], {}
    for seed in range(3):
        source = r/'training'/f'seed{seed}'/'trajectories.npz'
        sources[str(source.relative_to(p))] = sha(source)
        saved = dict(np.load(source, allow_pickle=False))
        np.testing.assert_array_equal(saved['cell_id'], cells.cell_id)
        distances, neighbours = NearestNeighbors(n_neighbors=15).fit(saved['grid']).kneighbors(embedding)
        sigma = np.maximum(distances.std(axis=1), np.finfo(float).eps)
        logw = -distances**2/(2*sigma[:, None]**2)
        weights = np.exp(logw - logw.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        checkpoint = pd.read_csv(source.with_name('checkpoints.csv'), float_precision='round_trip')
        for episode, grid_values in zip(saved['checkpoint_episodes'], saved['grid_values']):
            projected = np.sum(weights*grid_values[neighbours], axis=1)
            cp = checkpoint.loc[checkpoint.episode.eq(episode)].iloc[0]
            for population, mask in masks.items():
                values = projected[mask]
                ranks = pd.Series(values).rank().to_numpy()
                program_ranks = pd.Series(target[mask]).rank().to_numpy()
                rho = float(np.corrcoef(ranks, program_ranks)[0, 1])
                np.testing.assert_allclose(rho, cp['rho_'+population], atol=1e-10, rtol=0)
                q25, q75 = np.quantile(values, [.25, .75], method='linear')
                mask_hash = hashlib.sha256('\n'.join(cells.cell_id[mask]).encode()).hexdigest()
                rows.append(dict(cohort='ucb', component='megakaryocyte',
                    method='scRL Contribution', implementation='native r4 retained checkpoint',
                    endpoint='frozen megakaryocyte', program='full megakaryocytic',
                    seed=seed, episode=int(episode), population=population,
                    n_cells=int(mask.sum()), mask_sha256=mask_hash,
                    mean=float(values.mean()), SD=float(values.std(ddof=0)),
                    q25=float(q25), q75=float(q75), IQR=float(q75-q25),
                    minimum=float(values.min()), maximum=float(values.max()),
                    range=float(np.ptp(values)), spearman=rho,
                    grid_source=str(source.relative_to(p)), grid_source_sha256=sha(source)))
    for name in ['analysis_replay/inputs/frozen/ucb/cells.csv.gz',
                 'analysis_replay/inputs/matched/LiVAE/ucb/input.npz']:
        sources[name] = sha(p/name)
    return pd.DataFrame(rows), sources


def training_displays(p):
    """Prepare display-only rolling means and cumulative counts separately."""
    r = p/'revision_data/targeted_r4/training'
    summary = pd.read_csv(r/'summary.csv', float_precision='round_trip')
    returns, coverage = [], []
    for seed in range(3):
        episodes = pd.read_csv(r/f'seed{seed}/episodes.csv.gz', float_precision='round_trip')
        rolling = episodes['return'].rolling(250, min_periods=250).mean()
        for episode, raw, smooth in zip(episodes.episode, episodes['return'], rolling):
            returns.append(dict(seed=seed, episode=int(episode), native_return=raw,
                                rolling250_return=smooth))
        steps = pd.read_csv(r/f'seed{seed}/steps.csv.gz')
        visited = set()
        for budget in range(0,10001,500):
            selected = steps.loc[steps.episode.gt(budget-500)&steps.episode.le(budget)]
            visited.update(selected.grid); visited.update(selected.next_grid)
            coverage.append(dict(seed=seed, episode=budget, visited_grids=len(visited),
                                 mapped_grids=1774, fraction_visited=len(visited)/1774))
            if budget in [5000,10000]:
                row = summary.loc[summary.seed.eq(seed)&summary.episodes.eq(budget)].iloc[0]
                assert len(visited) == row.visited_grids
                np.testing.assert_allclose(episodes.loc[episodes.episode.between(budget-499,budget),'return'].mean(),
                                           row.last500_return, atol=1e-12, rtol=0)
    return pd.DataFrame(returns), pd.DataFrame(coverage)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--article-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args(); p = args.article_root.resolve()
    target = p/'revision_data/structural_r5/value_scale.csv'
    data, sources = calculate(p)
    curves, visited = training_displays(p)
    for rel in ['revision_data/targeted_r4/training/summary.csv'] + [
            f'revision_data/targeted_r4/training/seed{seed}/{name}' for seed in range(3)
            for name in ['episodes.csv.gz','steps.csv.gz','checkpoints.csv']]:
        sources[rel] = sha(p/rel)
    assert len(data) == 189
    if args.check:
        expected = pd.read_csv(target, float_precision='round_trip')
        assert list(expected) == list(data)
        for column in data:
            if data[column].dtype.kind in 'f':
                np.testing.assert_allclose(data[column], expected[column], atol=1e-12, rtol=1e-12)
            else:
                np.testing.assert_array_equal(data[column], expected[column])
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(target, index=False)
        data.loc[data.episode.isin([5000,10000])].to_csv(target.with_name('value_scale_key_checkpoints.csv'), index=False)
    for name, frame in [('training_return_curve.csv.gz', curves), ('training_grid_coverage.csv', visited)]:
        path = target.with_name(name)
        if args.check:
            expected = pd.read_csv(path, float_precision='round_trip')
            pd.testing.assert_frame_equal(frame, expected, check_exact=False, rtol=1e-12, atol=1e-12)
        else:
            frame.to_csv(path, index=False, compression={'method':'gzip','mtime':0} if path.suffix=='.gz' else None)
    record = dict(status='pass', operation='check' if args.check else 'calculate',
        population_checkpoint_records=len(data), saved_correlations_checked=len(data),
        key_checkpoint_records=int(data.episode.isin([5000,10000]).sum()),
        rolling_return_records=len(curves), cumulative_grid_records=len(visited),
        model_fitting_repeated=False, sources=sources, script_sha256=sha(__file__),
        SD_definition='population standard deviation, ddof=0',
        quartile_definition='NumPy quantile with linear interpolation')
    output = args.output or target.with_name('value_scale_verification.json')
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
