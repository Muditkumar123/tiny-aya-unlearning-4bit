"""Execute the frozen MANSU comparison sequentially, retaining every stage log."""
from __future__ import annotations
from datetime import datetime,timezone
import argparse,json,os
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from run_mansu import checkpoint_identity,file_sha256
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,default=ROOT/'runs/mansu_20261115')
parser.add_argument('--config',type=Path,default=ROOT/'configs/mansu.yaml')
parser.add_argument('--protocol',type=Path,help='optional written run plan; its hash is recorded')
parser.add_argument('--reuse-discovery',action='store_true')
args=parser.parse_args()
RUN=args.run.resolve()
REPORT=RUN/'execution_stages.json'
if REPORT.exists():raise ValueError('execution receipt already exists; use a new run')
state={'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':file_sha256(args.protocol) if args.protocol else None,'config_sha256':file_sha256(args.config),
       'inputs_receipt_sha256':file_sha256(RUN/'inputs/inputs_receipt.json'),'stages':[]}
def save():REPORT.write_text(json.dumps(state,indent=2)+'\n')
def run(name,args):
    row={'name':name,'started_at_utc':datetime.now(timezone.utc).isoformat(),'command':args,'status':'running'}
    script=ROOT/args[0];row['script_sha256']=file_sha256(script)
    state['stages'].append(row);save()
    with (RUN/(name+'.log')).open('x') as stream:
        child=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        row['pid']=child.pid;save();code=child.wait()
    row.update(status='complete' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat());save()
    if code:raise RuntimeError(f'{name} failed with exit {code}; see retained log')
try:
    smoke=json.loads((RUN/'discovery_smoke/attribution.json').read_text())
    assert len(smoke['sample_ids'])==1 and smoke['ig_steps']==1
    if args.reuse_discovery:
        discovery=json.loads((RUN/'discovery/attribution.json').read_text())
        assert discovery['pairs_sha256']==file_sha256(RUN/'inputs/discovery.json')
        assert discovery['model_weights_sha256']==checkpoint_identity(ROOT/'HF/models/c0v2')
        state['reused_discovery_sha256']=file_sha256(RUN/'discovery/attribution.json');save()
    else:
        run('discovery',['compression/discover_mansu.py','--checkpoint','HF/models/c0v2','--pairs',str(RUN/'inputs/discovery.json'),'--out',str(RUN/'discovery')])
    run('training',['training/run_mansu.py','--config',str(args.config),'--checkpoint','HF/models/c0v2','--circuit',str(RUN/'discovery/attribution.json'),'--forget',str(RUN/'inputs/forget.json'),'--retain',str(RUN/'inputs/retain.json'),'--validation',str(RUN/'inputs/validation.json'),'--out',str(RUN/'training'),'--reference-device','cuda:0'])
    candidates={'c0':ROOT/'HF/models/c0v2','npo':ROOT/'HF/models/c2','mansu':RUN/'training/checkpoint'}
    for method,path in candidates.items():
        run('nf4_'+method,['compression/build_mansu_nf4.py',str(path),str(RUN/('nf4_'+method)),'--expect-source-sha',*checkpoint_identity(path).values()])
    for method,path in candidates.items():
        for precision,checkpoint in [('bf16',path),('nf4',RUN/('nf4_'+method))]:
            label=method+'_'+precision
            run('eval_'+label,['evaluation/run_mansu_comparison.py','--checkpoint',str(checkpoint),'--tokenizer','HF/models/c0v2','--inputs',str(RUN/'inputs'),'--label',label,'--precision',precision,'--out',str(RUN/'evaluation')])
    run('post_attribution',['compression/discover_mansu.py','--checkpoint',str(candidates['mansu']),'--pairs',str(RUN/'inputs/discovery.json'),'--out',str(RUN/'post_attribution')])
    state['status']='complete'
except Exception as exc:
    state['status']='failed';state['failure']=str(exc);raise
finally:
    state['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
