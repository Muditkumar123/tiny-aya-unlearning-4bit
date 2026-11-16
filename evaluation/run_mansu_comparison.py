"""Evaluate a pinned local comparison checkpoint with the existing AuditKIT scorer."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
import subprocess
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'shared'),str(ROOT/'evaluation')]
from run_mansu import checkpoint_identity,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--label',required=True)
    p.add_argument('--precision',choices=['bf16','nf4'],required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if (a.out/a.label).exists():p.error('use a fresh label output directory')
    weights=checkpoint_identity(a.checkpoint)
    data=json.loads((a.inputs/'evaluation.json').read_text())
    frozen=json.loads((a.inputs/'inputs_receipt.json').read_text())
    if file_sha256(a.inputs/'evaluation.json')!=frozen['input_sha256']['evaluation.json']:
        raise ValueError('evaluation inputs changed after preparation')
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);tok.padding_side='left'
    if tok.convert_ids_to_tokens([6,3])!=['<|END_OF_TURN_TOKEN|>','<EOS_TOKEN>']:raise ValueError('stop-token mismatch')
    torch.manual_seed(0)
    kwargs={'dtype':torch.bfloat16,'local_files_only':True}
    if a.precision=='nf4':kwargs['device_map']={'':a.device}
    gpu_before=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,**kwargs)
    if a.precision=='bf16':model.to(a.device)
    model.eval();model.config.use_cache=True
    import bitsandbytes as bnb
    quantized={n:m for n,m in model.named_modules() if isinstance(m,bnb.nn.Linear4bit)}
    if a.precision=='nf4':
        if len(quantized)!=252:raise ValueError('expected252 NF4 linear modules')
        for n,m in quantized.items():
            if m.weight.quant_state.quant_type!='nf4' or m.weight.quant_state.blocksize!=64:raise ValueError('NF4 precision mismatch')
            if m.compute_dtype!=torch.bfloat16 or m.weight.quant_state.nested:raise ValueError('NF4 compute/double-quant settings differ')
    elif quantized:raise ValueError('bf16 candidate contains NF4 modules')
    from inmemory_model import InMemoryHFModel
    import tofu_eval
    ak=InMemoryHFModel(model,tok,name=f'mansu-comparison-{a.label}@{json.dumps(weights,sort_keys=True)}',
                       batch_size=a.batch_size,fp32_logits=True,eos_token_id=[6,3])
    tofu_eval.load_sets=lambda limit=None:{k:v[:limit] for k,v in data.items()}
    summary=tofu_eval.evaluate(ak,a.label,out=a.out)
    lengths={}
    from auditkit.model import Request
    for k,rows in data.items():
        n=[len(tok(ak._render_prompt(Request(prompt=r['prompt'])),add_special_tokens=False).input_ids) for r in rows]
        lengths[k]={'min':min(n),'max':max(n),'count':len(n)}
    report={'finished_at_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_weights_sha256':weights,
            'evaluation_sha256':file_sha256(a.inputs/'evaluation.json'),'inputs_receipt_sha256':file_sha256(a.inputs/'inputs_receipt.json'),
            'code_sha256':{str(q.relative_to(ROOT)):file_sha256(q) for q in [Path(__file__),ROOT/'evaluation/tofu_eval.py',ROOT/'shared/inmemory_model.py']},
            'precision':a.precision,'nf4_linears':len(quantized),'batch_size':a.batch_size,'padding_side':tok.padding_side,'gpu_before_load':gpu_before,
            'seed':0,'stop_ids':[6,3],'rendered_lengths':lengths,'weight_bytes':sum(x.stat().st_size for x in a.checkpoint.glob('*.safetensors')),
            'peak_gpu_memory_bytes':torch.cuda.max_memory_allocated(),'versions':{k:version(k) for k in ['torch','transformers','auditkit','bitsandbytes']}}
    (a.out/a.label/'execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary['sets'],indent=2),flush=True)

if __name__=='__main__':main()
