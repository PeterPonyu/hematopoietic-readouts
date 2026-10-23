#!/usr/bin/env python3
"""Stage the registered follow-up evidence; preserve the earlier benchmark packet."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/scientific-reports-hematopoietic-readouts'
R=ROOT/'results/targeted_review_20261021_r4'
T=P/'revision_data/targeted_r4'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest():
    files=[p for p in sorted(T.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
    record={'schema_version':1,'operation':'Byte-identical primary scientific evidence plus derived displays; earlier benchmark unchanged',
      'registration_sha256':sha(T/'registration.json'),'replay_sha256':sha(T/'replay_verification.json'),
      'files':[{'path':str(p.relative_to(P)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]}
    (T/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(p.stat().st_size for p in files)}))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-only',action='store_true');args=ap.parse_args()
    if args.manifest_only:manifest();return
    assert json.loads((R/'replay_verification.json').read_text())['status']=='pass'
    T.mkdir(exist_ok=False)
    for name in ['registration.json','replay_verification.json']:
        shutil.copy2(R/name,T/name)
    for folder in ['components','partitions','endpoint','training','implementation','slingshot_extended_dapp1']:
        shutil.copytree(R/folder,T/folder,ignore=shutil.ignore_patterns('__pycache__'))
    code=T/'code';code.mkdir()
    for name in ['component_and_partition_r4.py','endpoint_sensitivity_r4.py','tpo_training_diagnostics_r4.py',
                 'export_tpo_evaluation_r4.py','slingshot_stopping_r4.R','replay_targeted_r4.py','stage_targeted_r4.py']:
        shutil.copy2(ROOT/'scripts/benchmarks'/name,code/name)
    native=code/'native_scRL';native.mkdir()
    for p in (ROOT/'code/scRL/scRL').glob('*.py'):shutil.copy2(p,native/p.name)
    (T/'README.txt').write_text('''TARGETED FOLLOW-UP: 21 OCTOBER 2026, R4

This review-informed follow-up uses the earlier four-cohort data and benchmark.
The registration was recorded before its new outcomes, with the earlier results
already known. It adds no independent biological sample or future-fate outcome.
The historical broad-myeloid comparisons remain in ../external_benchmark.

From the extracted common article root, run:
  python revision_data/targeted_r4/code/replay_targeted_r4.py \\
    --packet revision_data/targeted_r4 --article-root . --output /tmp/r4-replay.json

NumPy, pandas, SciPy and scikit-learn are required; the supplied environment
inventories record the executed versions. Replay uses the supplied arrays and
native policy probabilities. It recalculates all association, partition summary
and common-start rollout diagnostics; it does not repeat method fitting, fresh
Leiden fitting or learning updates. Write verification outputs separately.

components/: all component-matched populations, scores, own-endpoint proximity,
  full marker counts for the irradiation dendritic program, and lineage masks.
endpoint/: original87 vs Klf1-and-Gata1-positive56, with three new Palantir fits;
  the original8298-cell evaluation mask is fixed. Alas2/Gypa/Tfrc evaluation omits
  the two definition markers but still reuses observed RNA. Configuration has an
  unused legacy terminal_label_counts field; actual supplied representative and
  n_strict=56 govern the fit. No annotation independence is claimed.
partitions/: frozen and fresh Leiden0.5/1.5 groupings, both cohort and WT/CT gene
  references, all/eligible contrasts and per-condition coverage.
training/: three native10000-episode TPO Contribution runs, checkpoint0:500:10000,
  episode and step records, policy/critic checkpoints and common-start rollouts.
  Native updates preserve the exact5000-episode historical prefix in all seeds.
  Each evaluation uses64 common starting grids and4 identical-uniform rollouts.
  Prediction errors include continuation bootstrap only at native truncation.
  They assess this environment, not calibration or biological accuracy.
implementation/: pinned source identity, exact patch, GPL notice and failure data.
  Palantir1.4.2 Python source is12/12 byte-identical to its upstream wheel.
  The local variant name describes the wrapper/configuration, not a source patch.
  FateID Python is not certified equivalent to the original R hard-vote algorithm.
slingshot_extended_dapp1/: native R fit under maxit20 stops at5, threshold0.001;
  pseudotime and weights exactly match the retained maxit10 result. Old TPO/sleep
  stopping counts remain unavailable; irradiation remains resource-limited.

code/: executed analysis drivers with repository-context source paths, the native
  scRL source snapshot, portable saved-output replay, and deterministic display
  generation. Repeating training requires reconstructing the original driver
  context; the replay command above works directly in this supplied edition.
figure/, tables/, methods.tex and supplement_table.tex: current scientific displays.
manifest.json: closed file paths, sizes and SHA-256 hashes (self-hash excluded).

Technical attempts are not extra biological or stochastic replicates. Before the
primary results, an evaluation deepcopy closure error stopped before training;
three full logging runs then failed only a return-summation check (maximum8.89e-15
roundoff), corrected to1e-12 without changing native updates. A float32 distance
check was replaced with float64 distance verification before associations. The
initial R tracing attempt did not fit a model. Their logs/outputs remain in the
local evidence workspace and their disposition is recorded in validation.json.
No run was selected for a favorable correlation or return.
''')
    validation={'status':'pass','saved_output_replay':json.loads((R/'replay_verification.json').read_text()),
      'primary_scientific_followups':{'training_runs':3,'training_episodes_per_run':10000,'strict_endpoint_Palantir_fits':3,'extended_Slingshot_fits':1},
      'technical_attempts':[{'attempt':'evaluation closure','stage':'before training','disposition':'correctly bind clone native method'},
                           {'attempt':'return summation','stage':'three full diagnostic runs','maximum_discrepancy':8.89e-15,'disposition':'native updates unchanged; validator tolerance1e-12; all primary seeds rerun'},
                           {'attempt':'component float32 distance check','stage':'before associations','disposition':'float64 primary distance calculation and independent direct checks'},
                           {'attempt':'R method tracing','stage':'before fit','disposition':'native inner-method exit logger only'}],
      'limitations':['No independent fate outcomes or new biological samples','No alternative TPO root fit','No upstream encoder refitting','TPO/sleep old Slingshot stopping counts unavailable']}
    (T/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    manifest()
if __name__=='__main__':main()
