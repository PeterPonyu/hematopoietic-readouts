#!/usr/bin/env python3
"""Run a frozen batch without replacing existing run directories or hiding failures."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/external_method_benchmark_20261021'


def run(job):
    method=job['method']; cohort=job['cohort']; seed=job.get('seed',0)
    envname='hematopoietic-slingshot-20261021' if method=='slingshot' else 'hematopoietic-benchmark-20261021'
    python=ROOT/'.venv'/envname/'bin/python'
    command=[str(python),'-u',str(ROOT/'scripts/benchmarks/hematopoietic_external.py'),
             'fit',cohort,method,'--seed',str(seed)]
    if job.get('tag'): command+=['--tag',job['tag']]
    label=f"{cohort}_{method}_seed{seed}"+('_'+job['tag'] if job.get('tag') else '')
    path=OUT/'logs'/f'{label}.log';path.parent.mkdir(exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    env=os.environ.copy()
    env.update(PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
               NUMBA_CACHE_DIR='/tmp/hematopoietic-benchmark-numba',
               MPLCONFIGDIR='/tmp/hematopoietic-benchmark-mpl',
               OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2')
    start=time.time()
    with path.open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                              start_new_session=True)
        try:
            code=proc.wait(timeout=job.get('timeout_seconds',3600));status='pass' if code==0 else 'fail'
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            code=proc.returncode;status='timeout'
    result={**job,'status':status,'exit_code':code,'elapsed_seconds':time.time()-start,
            'log':str(path),'command':command}
    (OUT/'logs'/f'{label}.process.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text())
    target=args.manifest.with_suffix('.completion.json')
    if target.exists():raise FileExistsError(target)
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,job) for job in manifest['jobs']]
        for future in as_completed(futures):results.append(future.result())
    target.write_text(json.dumps({'completed_utc':datetime.now(timezone.utc).isoformat(),
                                 'jobs':results},indent=2)+'\n')


if __name__=='__main__':main()
