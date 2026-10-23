#!/usr/bin/env python3
"""Fast integrity check of the allowed release files and successful replay bindings."""
import argparse,hashlib,json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(article):
    root=article/'analysis_replay';manifest=json.loads((root/'release_manifest.json').read_text());paths=[]
    for row in manifest['files']:
        rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
        f=article/rel;assert f.is_file() and f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],str(rel);paths.append(str(rel))
    assert len(paths)==len(set(paths))
    allowed=set(paths)|{'analysis_replay/release_manifest.json'}
    actual={str(f.relative_to(article)) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
    assert actual=={p for p in allowed if p.startswith('analysis_replay/')},'unmanifested or missing replay file'
    receipt=json.loads((root/'evidence/isolated_replay_receipt.json').read_text());assert receipt['status']=='passed'
    assert receipt['support_manifest_sha256']==sha(root/'support_manifest.json')
    assert receipt['policy_sha256']==sha(root/'tolerances.json')
    for name,digest in receipt['code_sha256'].items():assert sha(root/name)==digest,name
    figure=json.loads((root/'evidence/figure_replay_receipt.json').read_text());assert figure['status']=='passed'
    assert figure['source_replay_receipt_sha256']==sha(root/'evidence/isolated_replay_receipt.json')
    assert figure['renderer_sha256']==sha(root/'figure_calculations.py') and figure['wrapper_sha256']==sha(root/'render_figures.py')
    assert figure['expected_ledger_sha256']==sha(root/'expected_figure_values.csv')
    applications=['irradiation','ucb','dapp1','sleep']
    assert receipt['application_cohorts']==applications
    assert receipt['cohorts']==applications+['marrow','dapp1_spliced']
    assert receipt['tasks']==6 and receipt['matched_downstream_runs']==132
    assert receipt['matched_method_counts']=={'scRL':108,'scfocus':24}
    expected={str(f.relative_to(root/'expected')) for f in (root/'expected').rglob('*.csv')}
    retained={'results/recovered_integration_20261001/native_representation/irradiation/training_history.csv'}
    assert len(expected)==50 and set(receipt['all_reference_tables_compared'])==expected
    assert len(receipt['summary_tables_compared'])==49 and set(receipt['summary_tables_compared'])==expected-retained
    assert set(receipt['retained_training_history_tables_compared'])==retained
    assert [row['cohort'] for row in receipt['checkpoints']]==applications
    assert figure['figure_count']==7
    return {'status':'passed','files':len(paths),'bytes':sum(r['bytes'] for r in manifest['files']),'scientific_tables':49,'retained_training_history_tables':1,'checkpoint_cohorts':4,'application_tasks':6,'matched_downstream_runs':132,'figures':7,'ledger_rows':figure['ledger_rows'],'scope':'integrity of retained successful saved-artifact replay; analysis not recomputed by this quick check; no training or raw preparation replay'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--article-root',type=Path,required=True);a=p.parse_args();print(json.dumps(verify(a.article_root.resolve()),indent=2))
if __name__=='__main__':main()
