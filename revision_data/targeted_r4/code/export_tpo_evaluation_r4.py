#!/usr/bin/env python3
"""Export native policy probabilities and environment arrays for NumPy replay."""
from pathlib import Path
import hashlib,importlib,json,pickle,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/targeted_review_20261021_r4/training'
sys.path.insert(0,str(ROOT/'code/scRL'))
from scRL.ActorCriticCore import PolicyNet
import scRL
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2)
folder=ROOT/'results/recovered_integration_20261001/paired_downstream/LiVAE/ucb'
with (folder/'grid.pkl').open('rb') as f:gres=pickle.load(f)
scRL.d_rewards(gres,starts=['primitive'],ends=['megakaryocyte'],beta=1,mode='Contribution')
mapped=np.asarray(gres.grids['mapped_grids'])
np.savez_compressed(OUT/'environment_definition.npz',mapped_grid_ids=mapped,
    rewards=gres.qlearning['d_Contribution_rewards'].loc[mapped].to_numpy(dtype=float),
    pseudotime=gres.grids['pseudotime'].loc[mapped].to_numpy(),
    boundary=np.asarray(gres.grids['mapped_boundary']),starts=np.asarray(gres.grids['starts_cluster_grids']),N=np.asarray(gres.grids['n']))
record={'status':'pass','native_policy_exports':[],'environment_sha256':sha(OUT/'environment_definition.npz'),
        'source_grid_sha256':sha(folder/'grid.pkl'),'script_sha256':sha(__file__),'no_training_or_parameter_update':True}
for seed in [0,1,2]:
    dest=OUT/f'seed{seed}';receipt=json.loads((dest/'receipt.json').read_text());data=np.load(dest/'trajectories.npz')
    assert receipt['status']=='pass';tensor=torch.tensor(data['state_means'],dtype=torch.float32);probs=[]
    model=PolicyNet(tensor.shape[1],128,8)
    for episode in data['checkpoint_episodes']:
        path=dest/f'checkpoint_{int(episode):05}.pt';assert sha(path)==receipt['outputs'][path.name]
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['actor'])
        with torch.no_grad():probs.append(model(tensor).numpy())
    target=dest/'policy_probabilities.npz';np.savez_compressed(target,probabilities=np.stack(probs))
    record['native_policy_exports'].append({'seed':seed,'policy_output_sha256':sha(target),'source_receipt_sha256':sha(dest/'receipt.json')})
(OUT/'evaluation_export_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
