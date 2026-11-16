"""Measure CAD, activation shifts and actual NF4 weight changes for one completed run."""
from __future__ import annotations
import argparse,gc,json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'training'),str(ROOT/'evaluation')]
from run_mansu import encode_examples,checkpoint_identity,file_sha256
from mansu_diagnostics import cad,activation_shift,quantized_update_survival


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():p.error('diagnostic output must be new')
    before=json.loads((a.run/'discovery/attribution.json').read_text());after=json.loads((a.run/'post_attribution/attribution.json').read_text());layers=before['selected_mlp_layers']
    report={'cad':cad(before,after,layers),'layers':layers,'source_sha256':{str(x.relative_to(a.run)):file_sha256(x) for x in [a.run/'discovery/attribution.json',a.run/'post_attribution/attribution.json']}}
    from transformers import AutoModelForCausalLM,AutoTokenizer
    base=ROOT/'HF/models/c0v2';trained=a.run/'training/checkpoint'
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
    data=json.loads((a.run/'inputs/evaluation.json').read_text())['forget'][:10]
    batches=encode_examples(data,tok,512)
    def load(path,quant=False):
        kw={'dtype':torch.bfloat16,'local_files_only':True}
        if quant:kw['device_map']={'':'cuda:0'}
        m=AutoModelForCausalLM.from_pretrained(path,**kw)
        return (m if quant else m.to('cuda:0')).eval()
    b,t=load(base),load(trained)
    report['activation_shift']=activation_shift(b,t,batches,layers);report['activation_ids']=[r['id'] for r in data]
    del b,t;gc.collect();torch.cuda.empty_cache()
    b,t=load(a.run/'nf4_c0',True),load(a.run/'nf4_mansu',True)
    from safetensors import safe_open
    import bitsandbytes as bnb
    def tensors(directory):
        result={}
        for file in directory.glob('*.safetensors'):
            with safe_open(file,framework='pt',device='cpu') as f:
                for name in f.keys():
                    if any(name==f'model.layers.{layer}.mlp.{projection}.weight' for layer in layers for projection in ['gate_proj','up_proj','down_proj']):result[name]=f.get_tensor(name).float()
        return result
    bweights,tweights=tensors(base),tensors(trained)
    bn,tn=dict(b.named_parameters()),dict(t.named_parameters());per={}
    for name,original in bweights.items():
        qb=bnb.functional.dequantize_4bit(bn[name].data,quant_state=bn[name].quant_state).float().cpu()
        qt=bnb.functional.dequantize_4bit(tn[name].data,quant_state=tn[name].quant_state).float().cpu()
        per[name]=quantized_update_survival(qb,qt,tweights[name]-original)
    total=sum(x['updated_coordinates'] for x in per.values());survived=sum(x['changed_after_quantization'] for x in per.values())
    report['nf4_updates']={'per_matrix':per,'updated_coordinates':total,'changed_after_quantization':survived,'survival_fraction':survived/total,'semantic_forgetting_established':False}
    report['weights_sha256']={name:checkpoint_identity(path) for name,path in [('source',base),('trained',trained),('source_nf4',a.run/'nf4_c0'),('trained_nf4',a.run/'nf4_mansu')]}
    report['code_sha256']=file_sha256(__file__)
    with a.out.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({'cad':report['cad'],'activation_shift':{k:v for k,v in report['activation_shift'].items() if k!='per_layer'},'nf4_survival_fraction':survived/total},indent=2))
if __name__=='__main__':main()
