#!/usr/bin/env python3
"""Derive nested-budget summaries from retained native episode/step/checkpoint logs."""
from pathlib import Path
import argparse,json
import pandas as pd

def derive(folder):
    rows=[];termination=[]
    for seed in [0,1,2]:
        f=folder/f'seed{seed}';ep=pd.read_csv(f/'episodes.csv.gz');steps=pd.read_csv(f/'steps.csv.gz');check=pd.read_csv(f/'checkpoints.csv');receipt=json.loads((f/'receipt.json').read_text())
        assert receipt['status']=='pass' and all(receipt['historical_prefix_exact_checks'].values())
        for n in [5000,10000]:
            last=ep.loc[ep.episode.between(n-499,n)];visited=steps.loc[steps.episode.le(n)];c=check.loc[check.episode.eq(n)].iloc[0]
            rows.append({'seed':seed,'episodes':n,'last500_return':last['return'].mean(),
                         **{k:c[k] for k in ['rho_all','rho_nonterminal','rho_starting']},
                         'common_start_return':c.common_start_return_mean,'prediction_RMSE':c.common_start_prediction_RMSE,'TD_RMSE':c.common_start_TD_RMSE,
                         'visited_grids':len(set(visited.grid)|set(visited.next_grid)),'visited_decision_grids':visited.grid.nunique(),'mapped_grids':1774,
                         'last500_steps':last.steps.mean(),'historical_5000_prefix_exact':True})
            for name in ['boundary','revisit','time_limit','invalid_move']:
                number=int(last.termination_reason.eq(name).sum());termination.append({'seed':seed,'episodes':n,'window':'last500','termination_reason':name,'n_episodes':number,'fraction':number/500})
    return pd.DataFrame(rows),pd.DataFrame(termination)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--training',type=Path,required=True);args=ap.parse_args()
    a,b=derive(args.training);a.to_csv(args.training/'summary.csv',index=False);b.to_csv(args.training/'termination_summary.csv',index=False)
if __name__=='__main__':main()
