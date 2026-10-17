#!/usr/bin/env python3
"""Render the six article figures from regenerated replay tables, never originals."""
import argparse,json,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from replay_common import sha256,save_json

def render(out):
    import pandas as pd
    receipt=json.loads((out/'replay_receipt.json').read_text())
    assert receipt['status']=='passed' and set(['ucb','dapp1','sleep','marrow']).issubset(receipt['cohorts'])
    # Every scientific summary must still match the successful replay receipt.
    for row in receipt['output_files']:
        assert sha256(out/row['path'])==row['sha256'],row['path']
    import figure_calculations as f
    f.PROJECT=out;f.OUT=out/'figures';f.EVIDENCE=out/'evidence';f.CACHE.clear();f.LEDGER.clear();f.SOURCES.clear();f.QA.clear()
    def strict_table(path):
        name=str(path)
        if name not in f.CACHE:
            source=out/'source_data'/path
            frame=pd.read_csv(source);frame.index=range(1,len(frame)+1);f.CACHE[name]=frame;f.SOURCES[name]={'sha256':sha256(source),'rows':len(frame),'columns':list(frame.columns)}
        return f.CACHE[name]
    f.table=strict_table
    font=f.setup()
    for draw in [f.fig1,f.fig2,f.fig3,f.fig4,f.fig5,f.fig6]:draw()
    ledger=pd.DataFrame(f.LEDGER);ledger.to_csv(out/'evidence/figure_values.csv',index=False)
    expected=pd.read_csv(HERE/'expected_figure_values.csv');assert list(ledger.columns)==list(expected.columns)
    import numpy as np
    for c in ['figure','panel','mark','metric','unit','source_file','source_rows_1based_excluding_header','source_columns','display_transform']:
        assert ledger[c].fillna('').astype(str).tolist()==expected[c].fillna('').astype(str).tolist(),c
    np.testing.assert_allclose(ledger.value,expected.value,rtol=1e-10,atol=1e-10)
    return {'status':'passed','scope':'six figures rendered only from regenerated scientific CSVs; every plotted ledger value matched frozen reference','source_replay_receipt_sha256':sha256(out/'replay_receipt.json'),'renderer_sha256':sha256(HERE/'figure_calculations.py'),'wrapper_sha256':sha256(__file__),'expected_ledger_sha256':sha256(HERE/'expected_figure_values.csv'),'font':font,'figure_count':6,'ledger_rows':len(ledger),'maximum_ledger_value_error':float(np.max(np.abs(ledger.value-expected.value))),'sources':f.SOURCES,'figures':[{'path':str(p.relative_to(out)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted((out/'figures').glob('figure*'))]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--replay-output',type=Path,required=True);a=p.parse_args();out=a.replay_output.resolve()
    if not out.is_dir():raise FileNotFoundError(out)
    receipt_path=out/'figure_replay_receipt.json'
    # Invalidate any earlier pass before imports, input checks or rendering.
    save_json(receipt_path,{'status':'in_progress','scope':'figure replay attempt'})
    try:
        result=render(out)
    except BaseException as exc:
        save_json(receipt_path,{'status':'failed','failure':{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}})
        raise
    save_json(receipt_path,result)
if __name__=='__main__':main()
