#!/usr/bin/env python3
"""Read-only instrumentation of native scRL training and common-start evaluation."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, importlib, json, os, pickle, random, sys, time
os.environ.setdefault('MPLCONFIGDIR','/tmp/hematopoietic-r4-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/hematopoietic-r4-numba')
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/targeted_review_20261021_r4'
FOLDER=ROOT/'results/recovered_integration_20261001/paired_downstream/LiVAE/ucb'
sys.path.insert(0,str(ROOT/'scripts/applications'))
from common import seed_all,sha256,save_json
from run_models import project_values


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();seed=args.seed
    reg=json.loads((OUT/'registration.json').read_text());assert seed in reg['TPO_training']['seeds']
    dest=OUT/'training'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=False)
    started=time.time();source_record=json.loads((FOLDER/f'scRL_seed{seed}_megakaryocyte_contribution/receipt.json').read_text())
    assert all(sha256(ROOT/name)==digest for name,digest in source_record['source_sha256'].items())
    sys.path.insert(0,str(ROOT/'code/scRL'));import scRL
    for name in ['GridCore','EnvironmentCore','utils']:
        importlib.import_module('scRL.'+name).get_dist=lambda a,b: np.linalg.norm(a-b,axis=1) if a.ndim==1 else cdist(a,b)
    importlib.import_module('scRL.ActorCriticCore').device=torch.device('cpu')
    pack=np.load(FOLDER/'input.npz',allow_pickle=False);cells=pd.read_csv(FOLDER/'cells.csv.gz',index_col=0)
    assert np.array_equal(pack['cell_id'],cells.index.astype(str))
    with (FOLDER/'grid.pkl').open('rb') as f:gres=pickle.load(f)
    seed_all(seed);scRL.d_rewards(gres,starts=['primitive'],ends=['megakaryocyte'],beta=1,mode='Contribution')
    trainer=scRL.trainer('ActorCritic',gres,reward_type='d',reward_mode='Contribution',starts_prob=False,
                         X_latent=pack['latent'],hidden_dim=128,num_episodes=10000,gamma=.9,max_step=50,KNN=10)
    env=trainer.env;agent=trainer.agent;states=np.mean(env.state_space,axis=1)
    tensor=torch.tensor(states,dtype=torch.float32)
    grid=gres.grids['grids'][env.mapped_grids]
    loc={int(g):i for i,g in enumerate(env.mapped_grids)}
    starts=np.asarray(sorted(env.starts_grids),dtype=np.int64)
    selected=starts[np.unique(np.linspace(0,len(starts)-1,min(64,len(starts)),dtype=int))]
    uniforms=np.random.default_rng(20261021).random((len(selected),4,51))
    target=cells.program_megakaryocyte.to_numpy();mask_start=cells.cell_type.eq('primitive').to_numpy()
    mask_nonterminal=~cells.cell_type.eq('megakaryocyte').to_numpy()
    checkpoints=[];grid_values=[];episode_rows=[];step_rows=[];evaluation_rows=[]
    original_step=env.step;original_update=agent.update
    episode=0;episode_offset=0;episode_reasons=[]

    def evaluate(number):
        # Evaluation uses its own environment and NumPy Generator draws. The
        # training streams are restored even if a future diagnostic uses RNG.
        py_state=random.getstate();np_state=np.random.get_state();torch_state=torch.random.get_rng_state()
        try:
            with torch.no_grad():
                values=agent.critic(tensor).cpu().numpy().ravel().copy()
                probs=agent.actor(tensor).cpu().numpy()
            projected=project_values(pack['embedding'],grid,values)
            row={'episode':number}
            for label,mask in [('all',np.ones(len(cells),bool)),('nonterminal',mask_nonterminal),('starting',mask_start)]:
                row['rho_'+label]=float(spearmanr(projected[mask],target[mask]).statistic)
            evaluation=copy.deepcopy(env)
            # Drop the instance logging closure so this separate environment
            # calls its native class method, bound to its own state.
            if 'step' in vars(evaluation):delattr(evaluation,'step')
            returns=[];errors=[];td=[]
            for j,g in enumerate(selected):
                for rep in range(4):
                    evaluation.state_idx=int(g);evaluation.state=states[loc[int(g)]].copy()
                    evaluation.time_step=0;evaluation.reset_=True;evaluation.trajectory=[int(g)]
                    total=discounted=0.;discount=1.;done=trunc=False;steps=0
                    while not done and not trunc:
                        before=int(evaluation.state_idx);i=loc[before]
                        cdf=np.cumsum(probs[i].astype(float));cdf/=cdf[-1]
                        action=min(int(np.searchsorted(cdf,uniforms[j,rep,steps],side='right')),7)
                        _,reward,done,trunc=evaluation.step(action);nxt=int(evaluation.state_idx)
                        delta=reward+.9*float(values[loc[nxt]])*(not done)-float(values[i]);td.append(delta*delta)
                        total+=reward;discounted+=discount*reward;discount*=.9;steps+=1
                    comparison=discounted+(discount*float(values[loc[nxt]]) if trunc and not done else 0.)
                    error=float(values[loc[int(g)]])-comparison;errors.append(error*error);returns.append(total)
                    evaluation_rows.append({'episode':number,'start_grid':int(g),'replicate':rep,'steps':steps,
                        'return':total,'discounted_return':discounted,'truncated':bool(trunc),'terminated':bool(done),
                        'critic_start':float(values[loc[int(g)]]),'continuation_target':comparison,'prediction_error':error})
            row.update(common_start_return_mean=float(np.mean(returns)),common_start_prediction_RMSE=float(np.sqrt(np.mean(errors))),
                       common_start_TD_RMSE=float(np.sqrt(np.mean(td))),n_evaluation_episodes=len(returns))
            checkpoints.append(row);grid_values.append(values)
            torch.save({'actor':agent.actor.state_dict(),'critic':agent.critic.state_dict(),
                        'actor_optimizer':agent.actor_optimizer.state_dict(),'critic_optimizer':agent.critic_optimizer.state_dict(),
                        'episode':number,'python_rng':py_state,'numpy_rng':np_state,'torch_rng':torch_state},dest/f'checkpoint_{number:05}.pt')
        finally:
            random.setstate(py_state);np.random.set_state(np_state);torch.random.set_rng_state(torch_state)

    def logged_step(action):
        before=int(env.state_idx);reward_entry=float(env.rewards_df.iloc[loc[before],action])
        result=original_step(action);_,reward,done,trunc=result;nxt=int(env.state_idx)
        reason=('invalid_move' if reward_entry==-1 else 'boundary' if done and nxt in set(env.mapped_boundary)-set(env.starts_grids)
                else 'revisit' if done else 'time_limit' if trunc else 'continue')
        step_rows.append((episode+1,before,action,float(reward),nxt,bool(done),bool(trunc),reason))
        return result

    def logged_update(transition):
        nonlocal episode,episode_offset
        with torch.no_grad():
            s=torch.tensor(np.asarray(transition['states']),dtype=torch.float32)
            ns=torch.tensor(np.asarray(transition['next_states']),dtype=torch.float32)
            reward=torch.tensor(transition['rewards'],dtype=torch.float32).view(-1,1)
            done=torch.tensor(transition['dones'],dtype=torch.float32).view(-1,1)
            error=reward+.9*agent.critic(ns)*(1-done)-agent.critic(s)
            mse=float(torch.mean(error.square()))
        original_update(transition);episode+=1
        recent=step_rows[episode_offset:];episode_offset=len(step_rows)
        episode_rows.append({'episode':episode,'return':sum(r[3] for r in recent),'steps':len(recent),
                             'pre_update_TD_MSE':mse,'termination_reason':recent[-1][-1],
                             'start_grid':recent[0][1],'unique_grids':len(set(r[1] for r in recent))})
        if episode%500==0:evaluate(episode)

    env.step=logged_step;agent.update=logged_update
    evaluate(0);returns,smooth=trainer.train()
    assert len(returns)==10000 and len(episode_rows)==10000
    np.savez_compressed(dest/'trajectories.npz',returns=np.asarray(returns),smoothed_values=np.asarray(smooth),
                        checkpoint_episodes=np.arange(0,10001,500),grid_values=np.stack(grid_values),grid=grid,
                        mapped_grid_ids=np.asarray(env.mapped_grids),state_means=states,cell_id=pack['cell_id'],
                        evaluation_start_grids=selected,evaluation_uniforms=uniforms)
    np.testing.assert_allclose(returns,[r['return'] for r in episode_rows],atol=1e-12,rtol=1e-12)
    pd.DataFrame(episode_rows).to_csv(dest/'episodes.csv.gz',index=False)
    pd.DataFrame(step_rows,columns=['episode','grid','action','reward','next_grid','done','truncated','reason']).to_csv(dest/'steps.csv.gz',index=False)
    pd.DataFrame(checkpoints).to_csv(dest/'checkpoints.csv',index=False)
    pd.DataFrame(evaluation_rows).to_csv(dest/'common_start_evaluations.csv.gz',index=False)
    old=np.load(FOLDER/f'scRL_seed{seed}_megakaryocyte_contribution/outputs.npz',allow_pickle=False)
    idx=10;new5=grid_values[idx];projected=project_values(pack['embedding'],grid,new5)
    prefix_checks={'returns':np.array_equal(old['returns'],np.asarray(returns[:5000])),
                   'initial_grid_value':np.array_equal(old['initial_grid_value'],grid_values[0]),
                   'grid_value_5000':np.array_equal(old['grid_value'],new5),
                   'cell_value_5000':np.array_equal(old['value'],projected),
                   'smooth_values_prefix':np.array_equal(old['smoothed_values'],np.asarray(smooth[:sum(r['steps'] for r in episode_rows[:5000])]))}
    save_json(dest/'receipt.json',{'status':'pass','seed':seed,'arm':'LiVAE','episodes':10000,
        'elapsed_seconds':time.time()-started,'completed_utc':datetime.now(timezone.utc).isoformat(),
        'source_identity_matches_retained_receipt':True,'historical_prefix_exact_checks':prefix_checks,
        'input_sha256':sha256(FOLDER/'input.npz'),'grid_sha256':sha256(FOLDER/'grid.pkl'),
        'source_files_sha256':source_record['source_sha256'],'driver_sha256':sha256(__file__),
        'evaluation_policy':'fixed common starting grids and random uniforms; continuation bootstrap only at time-limit truncation',
        'criterion':'Diagnostic observations; episode count, return or finite output do not certify convergence.',
        'outputs':{f.name:sha256(f) for f in dest.iterdir() if f.is_file()}})
    print(json.dumps({'seed':seed,'prefix_checks':prefix_checks,'elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
