#!/usr/bin/env python3
"""Registered external inference comparisons on the retained four cohorts.

New outputs are isolated from the historical scientific release. Expression
programs are read from saved tables, never redefined to favour a new method.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hematopoietic-benchmark-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/hematopoietic-benchmark-numba')
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import minmax_scale

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/external_method_benchmark_20261021'
INPUT = ROOT/'results/application_reanalysis_20260921'
PAPER = ROOT/'paper/scientific-reports-hematopoietic-readouts'
COHORTS = ['dapp1', 'irradiation', 'ucb', 'sleep']
FATEID = ROOT/'data/remote-dl4080-20260912/files/scRL_LAB/FateID.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def load(cohort):
    folder = INPUT/cohort
    cells = pd.read_csv(folder/'cells.csv.gz', index_col=0)
    cells.index = cells.index.astype(str)
    with np.load(folder/'input.npz', allow_pickle=False) as data:
        pack = {k: data[k] for k in data.files}
    assert np.array_equal(pack['cell_id'], cells.index)
    protocol = json.loads((folder/'protocol.json').read_text())
    return cells, pack, protocol


def representative(cells, x, labels):
    indices = np.flatnonzero(cells.cell_type.isin(labels).to_numpy())
    assert len(indices)
    # Nearest-to-centroid representative, fixed in saved PCA30 before fitting.
    distance = np.square(x[indices].astype(float)-x[indices].mean(axis=0)).sum(axis=1)
    order = np.lexsort((cells.index.to_numpy()[indices], distance))
    return str(cells.index[indices[order[0]]])


def register():
    target = OUT/'registration.json'
    if target.exists():
        raise FileExistsError('Registration already exists; do not change it after observing fits.')
    result = {'schema_version': 1, 'registered_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Retrospective configured-pipeline comparison; expression labels and programs are shared, not measured future fates.',
              'user_authorized_execution': 'Install dependencies in isolated environments, use refined Python modules, run Dapp1 Palantir pilot then four-method benchmark, review before manuscript integration.',
              'primary_population': 'Exclude the union of every supplied terminal population.',
              'secondary_populations': ['all', 'retained starting set'],
              'seeds': [0, 1, 2], 'slingshot_seeds': [0],
              'parameters': {'palantir': {'n_components': 10, 'knn': 30, 'num_waypoints': 1200,
                                           'max_iterations': 25, 'use_early_cell_as_start': True,
                                           'scale_components': True, 'n_jobs': 2},
                             'cellrank': {'knn': 30, 'diffusion_components': 10,
                                          'kernel': 'DPT PseudotimeKernel, soft direction threshold',
                                          'terminal_states': 'all cells in supplied component labels',
                                          'solver': 'direct', 'no_experimental_day_or_condition_input': True},
                             'fateid': {'implementation': 'retained author Python port',
                                        'minnr': 20, 'minnrh': 50, 'nbtree': 500,
                                        'confidence': .75, 'adapt': True, 'n_jobs': 2,
                                        'distance': '1 - Pearson correlation between saved selected-gene expression rows',
                                        'explicit_z_reason': 'The retained port default x.T.corr() uses a gene axis; explicit cell-distance input avoids that orientation defect.'},
                             'slingshot': {'implementation': 'pyslingshot 0.2.0',
                                          'input': 'saved PCA30 and frozen Leiden partition',
                                          'iterations': 10, 'coverage': 'positive lineage curve weight; no missing-value filling'}},
              'cohorts': {}}
    for cohort in COHORTS:
        cells, pack, protocol = load(cohort)
        component_labels = list(dict.fromkeys(v for group in protocol['ends'].values() for v in group))
        components = {label: representative(cells, pack['latent'], [label]) for label in component_labels}
        root = representative(cells, pack['latent'], [protocol['root']])
        result['cohorts'][cohort] = {
            'n_cells': len(cells), 'root_cell': root, 'root_label': protocol['root'],
            'starting_labels': protocol['starts'], 'tasks': protocol['ends'],
            'terminal_representatives': components,
            'terminal_label_counts': {label: int((cells.cell_type==label).sum()) for label in components},
            'probability_comparison': len(protocol['ends'])>1,
            'single_fate_gate': None if len(protocol['ends'])>1 else 'No informative broad-target probability comparison without separately validated competing fates.',
            'cell_axis_sha256': hashlib.sha256('\n'.join(cells.index).encode()).hexdigest(),
            'input_sha256': sha(INPUT/cohort/'input.npz'),
            'cells_sha256': sha(INPUT/cohort/'cells.csv.gz'),
            'protocol_sha256': sha(INPUT/cohort/'protocol.json'),
            'annotation': 'inherited historical labels' if cohort=='irradiation' else 'frozen expression-program labels',
            'terminal_input_difference': 'Palantir receives one PCA-centroid representative per component; CellRank and FateID receive full component membership. These are configured pipelines, not an equal-terminal-constraint algorithm ablation.'}
    write_json(target, result)
    print(json.dumps({'registration': str(target), 'cohorts': len(result['cohorts'])}))


def package_sources(package):
    path = Path(package.__file__).resolve().parent
    return {str(p): sha(p) for p in sorted(path.rglob('*.py'))}


def palantir_fit(cells, pack, config, seed, dest):
    import palantir
    latent = pd.DataFrame(pack['latent'].astype(float), index=cells.index)
    diffusion = palantir.utils.run_diffusion_maps(latent, n_components=10, knn=30, alpha=0, seed=seed)
    multiscale = palantir.utils.determine_multiscale_space(diffusion)
    assert np.array_equal(multiscale.index, cells.index)
    terminal_ids = list(config['terminal_representatives'].values())
    pr = palantir.core.run_palantir(multiscale, config['root_cell'], terminal_states=terminal_ids,
                                   knn=30, num_waypoints=1200, n_jobs=2, scale_components=True,
                                   use_early_cell_as_start=True, max_iterations=25, seed=seed)
    components = list(config['terminal_representatives'])
    columns = config['terminal_representatives']
    probabilities = pr.branch_probs.loc[cells.index, [columns[k] for k in components]].to_numpy()
    np.savez_compressed(dest/'outputs.npz', cell_id=cells.index.to_numpy(dtype=str),
                        component_names=np.asarray(components,dtype=str), probability=probabilities,
                        pseudotime=pr.pseudotime.reindex(cells.index).to_numpy(),
                        entropy=pr.entropy.reindex(cells.index).to_numpy(),
                        diffusion_space=multiscale.to_numpy(),
                        inference_space=minmax_scale(multiscale.to_numpy()),
                        diffusion_eigenvalues=np.asarray(diffusion['EigenValues']),
                        evaluated=np.ones(len(cells),dtype=bool))
    assert np.isfinite(probabilities).all()
    assert probabilities.min() >= -1e-8
    # The retained PResults constructor zeroes scores <0.01 without row
    # renormalization. Preserve those native scores and validate their mass
    # deficit, rather than silently change the algorithm's output.
    row_sum=probabilities.sum(axis=1)
    assert np.all(row_sum<=1+1e-6) and np.all(row_sum>=1-.01*len(components)-1e-6)
    return {'package_version': importlib.metadata.version('palantir'),
            'source_files': package_sources(palantir), 'multiscale_dimensions': multiscale.shape[1],
            'terminal_ids': terminal_ids, 'native_probability_cutoff': .01,
            'row_sum_range': [float(row_sum.min()),float(row_sum.max())],
            'output_renormalized': False,
            'implementation': 'existing locally refined Palantir package; source hashes retained'}


def cellrank_fit(cells, pack, config, seed, dest):
    import anndata as ad
    import scanpy as sc
    import cellrank as cr
    from scipy.sparse.csgraph import connected_components
    data = ad.AnnData(sparse.csr_matrix((len(cells),1)), obs=cells[['leiden']].copy())
    data.obsm['X_pca'] = pack['latent'].copy()
    sc.pp.neighbors(data, n_neighbors=30, use_rep='X_pca', random_state=seed)
    n_components = connected_components(data.obsp['connectivities'], directed=False, return_labels=False)
    sc.tl.diffmap(data, n_comps=10, random_state=seed)
    data.uns['iroot'] = cells.index.get_loc(config['root_cell'])
    sc.tl.dpt(data, n_dcs=10)
    if not np.isfinite(data.obs['dpt_pseudotime']).all():
        raise ValueError('DPT has unreachable cells; do not replace infinite pseudotimes.')
    kernel = cr.kernels.PseudotimeKernel(data, time_key='dpt_pseudotime')
    kernel.compute_transition_matrix(threshold_scheme='soft', n_jobs=1, show_progress_bar=False)
    estimator = cr.estimators.GPCCA(kernel)
    components = list(config['terminal_representatives'])
    terminal = {label: cells.index[cells.cell_type==label].tolist() for label in components}
    estimator.set_terminal_states(terminal)
    actual_terminals = estimator.terminal_states
    for label, cell_ids in terminal.items():
        assert set(actual_terminals.index[actual_terminals==label]) == set(cell_ids)
    estimator.compute_fate_probabilities(solver='direct', use_petsc=False, n_jobs=1,
                                         show_progress_bar=False)
    probabilities = np.asarray(estimator.fate_probabilities[:,components])
    for j, label in enumerate(components):
        np.testing.assert_allclose(probabilities[cells.cell_type.eq(label),j],1,atol=1e-8,rtol=0)
    np.savez_compressed(dest/'outputs.npz', cell_id=cells.index.to_numpy(dtype=str),
                        component_names=np.asarray(components,dtype=str), probability=probabilities,
                        pseudotime=data.obs['dpt_pseudotime'].to_numpy(),
                        evaluated=np.ones(len(cells),dtype=bool))
    assert np.isfinite(probabilities).all() and probabilities.min() >= -1e-7
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
    return {'package_version': importlib.metadata.version('cellrank'),
            'source_files': package_sources(cr), 'graph_components': int(n_components),
            'terminal_state_counts': {label:len(x) for label,x in terminal.items()},
            'full_terminal_membership_verified': True,
            'kernel_parameters': str(kernel.params)}


def selected_expression(cohort, cells):
    import anndata as ad
    data = ad.read_h5ad(INPUT/cohort/'prepared.h5ad', backed='r')
    try:
        assert np.array_equal(data.obs_names.astype(str), cells.index)
        hvg = data.var.highly_variable.to_numpy(dtype=bool)
        expression = data[:,hvg].X
        if sparse.issparse(expression):
            expression = expression.toarray()
        return np.asarray(expression,dtype=np.float32), data.var_names[hvg].astype(str).to_numpy()
    finally:
        data.file.close()


def correlation_cache(cohort, expression):
    path = OUT/cohort/'cell_correlation_distance.npy'
    if path.exists():
        return np.load(path, mmap_mode='r')
    # Float32 stored distances with float64 centering/norms; block multiplication
    # prevents multiple dense 41k-by-41k copies. Reused across all forest seeds.
    centered = expression-expression.mean(axis=1,keepdims=True,dtype=np.float64)
    norm = np.linalg.norm(centered,axis=1,keepdims=True)
    if (norm==0).any():
        raise ValueError('A selected-gene expression row has zero variance.')
    normalized = np.asarray(centered/norm,dtype=np.float32)
    temp = path.with_suffix('.building.npy')
    matrix = np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(len(expression),len(expression)))
    for start in range(0,len(expression),1024):
        stop = min(start+1024,len(expression))
        matrix[start:stop] = np.maximum(0,1-normalized[start:stop] @ normalized.T)
    np.fill_diagonal(matrix,0)
    matrix.flush(); del matrix
    temp.replace(path)
    return np.load(path,mmap_mode='r')


def fateid_fit(cohort, cells, pack, config, seed, dest):
    source = ROOT/'scripts/benchmarks/FateID_membership_cache.py' if os.environ.get('HEMATO_FATEID_MEMBERSHIP_CACHE')=='1' else FATEID
    if os.environ.get('HEMATO_FATEID_BLOCK_MEDIAN')=='1':source=ROOT/'scripts/benchmarks/FateID_blocked_median.py'
    if os.environ.get('HEMATO_FATEID_PARALLEL_MEDIAN')=='1':source=ROOT/'scripts/benchmarks/FateID_parallel_median.py'
    spec = importlib.util.spec_from_file_location('_retained_fateid', source)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    expression, genes = selected_expression(cohort,cells)
    distance = correlation_cache(cohort,expression)
    x = pd.DataFrame(expression.T,index=genes,columns=cells.index)
    z = pd.DataFrame(distance,index=cells.index,columns=cells.index,copy=False)
    components = list(config['terminal_representatives'])
    result = module.fate_bias(x,cells.cell_type,components,z=z,minnr=20,minnrh=50,
                             adapt=True,confidence=.75,nbtree=500,seed=seed,n_jobs=2,
                             verbose=True)
    probabilities = result['probs'].loc[cells.index,['t'+k for k in components]].to_numpy(dtype=float)
    evaluated = cells.index.isin(result['trall'])
    np.savez_compressed(dest/'outputs.npz',cell_id=cells.index.to_numpy(dtype=str),
                        component_names=np.asarray(components,dtype=str),probability=probabilities,
                        evaluated=evaluated,genes=np.asarray(genes,dtype=str))
    assert evaluated.all(), 'Unprocessed uniform defaults cannot count as fitted probabilities.'
    assert np.isfinite(probabilities).all() and probabilities.min()>=0
    np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-8,rtol=0)
    return {'implementation':'retained refined FateID Python port; no R-equivalence certification',
            'source_files':{str(FATEID):sha(FATEID),str(source):sha(source)},
            'membership_cache_only':source.name=='FateID_membership_cache.py',
            'blocked_finite_median':source.name in ['FateID_blocked_median.py','FateID_parallel_median.py'],
            'median_threads':4 if source.name=='FateID_parallel_median.py' else 1,
            'scikit_learn':importlib.metadata.version('scikit-learn'),
            'gene_axis_sha256':hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
            'correlation_distance_path':str(OUT/cohort/'cell_correlation_distance.npy'),
            'forest_iterations':len(result['rfl']),'processed_cells':int(evaluated.sum()),
            'rf_leaf_probability_votes':'Retained Python-port semantics; differs from R randomForest hard-tree votes.'}


def slingshot_fit(cells, pack, config, seed, dest):
    if os.environ.get('HEMATO_SLINGSHOT_R')=='1':
        return slingshot_r_fit(cells,pack,config,seed,dest)
    from pyslingshot import Slingshot
    import pyslingshot
    cluster = pd.Categorical(cells.leiden.astype(str))
    root_index = cells.index.get_loc(config['root_cell'])
    root_cluster = int(cluster.codes[root_index])
    endpoints = sorted({int(cluster.codes[cells.index.get_loc(cell)])
                        for cell in config['terminal_representatives'].values()})
    # The ndarray API requires one-hot cluster membership, not integer codes.
    membership = np.eye(len(cluster.categories), dtype=int)[cluster.codes]
    model = Slingshot(pack['latent'].astype(float), membership,
                      start_node=root_cluster, end_nodes=endpoints)
    model.fit(num_epochs=10)
    # Output extraction is verified against the installed Python-port API.
    times = np.column_stack([curve.pseudotimes_interp for curve in model.curves])
    weights = np.asarray(model.cell_weights)
    if weights.shape != times.shape:
        raise ValueError(('Slingshot output axis mismatch',weights.shape,times.shape))
    assert np.isfinite(weights).all() and (weights>=0).all()
    assert np.all((np.isfinite(times)&(weights>0)).sum(axis=0)>1)
    lineages = [list(map(int,lineage)) for lineage in model.lineages]
    np.savez_compressed(dest/'outputs.npz',cell_id=cells.index.to_numpy(dtype=str),
                        pseudotime=times,curve_weights=weights,
                        cluster_categories=np.asarray(cluster.categories,dtype=str),
                        evaluated=np.isfinite(times)&(weights>0))
    write_json(dest/'lineages.json',{'lineages':lineages,'root_cluster':root_cluster,
                                    'terminal_clusters':endpoints,'cluster_categories':list(cluster.categories)})
    return {'package_version':importlib.metadata.version('pyslingshot'),
            'source_files':package_sources(pyslingshot),'lineages':lineages,
            'implementation':'Python Slingshot port; no R-equivalence certification',
            'n_lineages':len(lineages)}


def slingshot_r_fit(cells,pack,config,seed,dest):
    script=ROOT/'scripts/benchmarks/hematopoietic_slingshot.R'
    library=ROOT/'.rlib/hematopoietic-benchmark-20261021'
    clusters=pd.Categorical(cells.leiden.astype(str))
    root=str(cells.loc[config['root_cell'],'leiden'])
    endpoints=sorted({str(cells.loc[k,'leiden']) for k in config['terminal_representatives'].values()})
    pd.DataFrame(pack['latent'],index=cells.index).to_csv(dest/'pca.csv.gz',float_format='%.17g')
    pd.DataFrame({'leiden':cells.leiden.astype(str)},index=cells.index).to_csv(dest/'clusters.csv')
    # /usr/bin/time records the child R process, which Python's own ru_maxrss
    # cannot measure. The raw fit and sessionInfo are retained for inspection.
    command=['/usr/bin/time','-f','%M','-o',str(dest/'r_peak_rss_kib.txt'),
             'Rscript',str(script),str(dest),root,','.join(endpoints),str(library)]
    r_environment=os.environ.copy()
    # igraph was linked against the existing Conda libxml2; scope its loader
    # search path to this R child rather than modify shell or shared libraries.
    r_environment['LD_LIBRARY_PATH']='/home/zeyufu/miniconda3/lib'+(':'+r_environment['LD_LIBRARY_PATH'] if r_environment.get('LD_LIBRARY_PATH') else '')
    subprocess.run(command,check=True,env=r_environment)
    times=pd.read_csv(dest/'pseudotime.csv.gz',index_col=0,float_precision='round_trip')
    weights=pd.read_csv(dest/'curve_weights.csv.gz',index_col=0,float_precision='round_trip')
    assert np.array_equal(times.index,cells.index) and np.array_equal(weights.index,cells.index)
    assert times.shape==weights.shape and np.isfinite(weights).all().all()
    paths=[line.split(',') for line in (dest/'lineage_paths.txt').read_text().splitlines()]
    codes={str(k):i for i,k in enumerate(clusters.categories)}
    lineages=[[codes[k] for k in path] for path in paths]
    mask=np.isfinite(times.to_numpy())&(weights.to_numpy()>0)
    assert np.all(mask.sum(axis=0)>1)
    np.savez_compressed(dest/'outputs.npz',cell_id=cells.index.to_numpy(dtype=str),
                        pseudotime=times.to_numpy(),curve_weights=weights.to_numpy(),
                        cluster_categories=np.asarray(clusters.categories,dtype=str),evaluated=mask)
    write_json(dest/'lineages.json',{'lineages':lineages,'root_cluster':codes[root],
                                    'terminal_clusters':[codes[k] for k in endpoints],
                                    'cluster_categories':list(clusters.categories)})
    return {'package_version':(dest/'package_version.txt').read_text().strip(),
            'implementation':'upstream R Slingshot; fallback after full Python-port pilot failure',
            'source_files':{str(script):sha(script),
                            str(library/'slingshot/DESCRIPTION'):sha(library/'slingshot/DESCRIPTION')},
            'n_lineages':len(lineages),'lineages':lineages,
            'curve_parameters':{'maxit':10,'approx_points':150,'other_settings':'upstream defaults'},
            'r_peak_rss_MiB':float((dest/'r_peak_rss_kib.txt').read_text().strip())/1024,
            'output_na_filling':False,'lineage_weight_normalization':False}


def fit(args):
    registration = json.loads((OUT/'registration.json').read_text())
    config = registration['cohorts'][args.cohort]
    if args.method in ['palantir','cellrank','fateid'] and not config['probability_comparison']:
        raise ValueError('Single-target probability comparison is not applicable; do not fit a constant broad-fate score.')
    cells, pack, protocol = load(args.cohort)
    assert sha(INPUT/args.cohort/'input.npz')==config['input_sha256']
    assert sha(INPUT/args.cohort/'cells.csv.gz')==config['cells_sha256']
    assert os.environ.get('PYTHONHASHSEED')=='0', 'Retained Python-port set iteration requires a fixed process hash seed.'
    dest = OUT/args.cohort/args.method/(f'seed{args.seed}'+('_'+args.tag if args.tag else ''))
    dest.mkdir(parents=True,exist_ok=False)
    start = time.time()
    receipt = {'status':'running','cohort':args.cohort,'method':args.method,'seed':args.seed,
               'started_utc':datetime.now(timezone.utc).isoformat(),'python':sys.executable,
               'driver_sha256':sha(__file__),'registration_sha256':sha(OUT/'registration.json'),
               'input_sha256':config['input_sha256'],'cell_axis_sha256':config['cell_axis_sha256'],
               'n_cells':len(cells),'scope':registration['scope']}
    write_json(dest/'receipt.json',receipt)
    shutil.copy2(__file__,dest/'driver_snapshot.py')
    try:
        if args.method=='palantir':details=palantir_fit(cells,pack,config,args.seed,dest)
        elif args.method=='cellrank':details=cellrank_fit(cells,pack,config,args.seed,dest)
        elif args.method=='fateid':details=fateid_fit(args.cohort,cells,pack,config,args.seed,dest)
        else:details=slingshot_fit(cells,pack,config,args.seed,dest)
        receipt.update(status='pass',**details)
        receipt['outputs']={f.name:sha(f) for f in dest.iterdir() if f.is_file() and f.name!='receipt.json'}
    except BaseException as error:
        receipt.update(status='fail',error=f'{type(error).__name__}: {error}')
        (dest/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        receipt.update(elapsed_seconds=time.time()-start,
                       peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        write_json(dest/'receipt.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','cohort','method','seed','elapsed_seconds','peak_rss_MiB']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('register')
    run=sub.add_parser('fit');run.add_argument('cohort',choices=COHORTS)
    run.add_argument('method',choices=['palantir','cellrank','fateid','slingshot'])
    run.add_argument('--seed',type=int,choices=[0,1,2],default=0)
    run.add_argument('--tag',default='')
    args=parser.parse_args()
    register() if args.command=='register' else fit(args)


if __name__=='__main__':main()
