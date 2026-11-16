"""Export text-free comparison rows and paired uncertainty from a completed local run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'shared'),str(ROOT/'training')]
from cluster_ci import cluster_ci
from paired_ci import paired_ci
from run_mansu import file_sha256


def export(run,out):
    if out.exists():raise ValueError('numeric export directory must be new')
    methods=['c0','npo','mansu'];labels=[m+'_'+p for m in methods for p in ['bf16','nf4']]
    values={};sources={};levels={};contrasts={}
    for label in labels:
        d=run/'evaluation'/label
        values[label]={}
        for subset in ['forget','retain','mmlu_en','mmlu_hi']:
            path=d/(subset+'.jsonl');rows=[json.loads(line) for line in path.read_text().splitlines()]
            assert len({r['id'] for r in rows})==len(rows)
            fields=['id','author','rougeL_recall','lp_answer','truth_ratio'] if subset in ['forget','retain'] else ['id','subject','answer','lp','pred','correct']
            values[label][subset]=[{k:r[k] for k in fields} for r in rows]
            sources[str(path.relative_to(run))]=file_sha256(path)
        execution=json.loads((d/'execution.json').read_text())
        levels[label]={'weight_bytes':execution['weight_bytes'],'sets':{}}
        for subset,rows in values[label].items():
            metrics=['rougeL_recall','lp_answer','truth_ratio'] if subset in ['forget','retain'] else ['correct']
            levels[label]['sets'][subset]={'n':len(rows),**{m:float(np.mean([r[m] for r in rows])) for m in metrics}}
            if subset in ['forget','retain']:
                ratios=np.array([r['truth_ratio'] for r in rows])
                levels[label]['sets'][subset]['truth_ratio_tofu']=float(np.mean(np.minimum(ratios,1/ratios) if subset=='forget' else np.maximum(0,1-1/ratios)))
    for method in methods:
        left,right=method+'_bf16',method+'_nf4';contrasts[method]={}
        for subset in values[left]:
            a={r['id']:r for r in values[left][subset]};b={r['id']:r for r in values[right][subset]}
            assert a.keys()==b.keys()
            metrics=['rougeL_recall','lp_answer'] if subset in ['forget','retain'] else ['correct']
            result={}
            for metric in metrics:
                va={k:r[metric] for k,r in a.items()};vb={k:r[metric] for k,r in b.items()}
                if subset in ['forget','retain']:
                    result[metric]=cluster_ci({'before':va,'after':vb},{k:r['author'] for k,r in a.items()},lambda m:m['after']-m['before'])
                else:result[metric]=paired_ci(va,vb)
            contrasts[method][subset]=result
    out.mkdir(parents=True)
    for label in labels:
        d=out/label;d.mkdir()
        for subset,rows in values[label].items():(d/(subset+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        shutil.copyfile(run/'evaluation'/label/'execution.json',d/'execution.json')
    for source,name in [(run/'training/run.json','training.json'),(run/'discovery/attribution.json','attribution_before.json'),(run/'post_attribution/attribution.json','attribution_after.json'),(run/'inputs/inputs_receipt.json','inputs_receipt.json'),(run/'execution_stages.json','execution_stages.json')]:
        shutil.copyfile(source,out/name)
    if (run/'diagnostics.json').exists():shutil.copyfile(run/'diagnostics.json',out/'diagnostics.json')
    for name in ['source_identity_check.json','rendered_lengths_preflight.json']:
        if (run/name).exists():shutil.copyfile(run/name,out/name)
    report={'levels':levels,'nf4_minus_bf16':contrasts,'source_row_sha256':sources,'export_code_sha256':file_sha256(__file__),
            'interpretation':'Concrete checkpoints, not equal-budget methods; lexical and likelihood proxies do not establish erasure.',
            'uncertainty':{'confidence':.95,'draws':10000,'seed':0,'tofu_unit':'author','mmlu_unit':'question','multiplicity_correction':False}}
    (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# MANSU comparison: new NF4 experiment','','Raw ROUGE-L recall is word overlap. Log-probability is in nats/token. MMLU values are percentages. These rows use the new protocol and must not replace or be pooled with Study U.','','| Checkpoint | Forget ROUGE-L | Forget log-prob | Retain ROUGE-L | English MMLU | Hindi MMLU | Weights, GB |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label,d in levels.items():
        s=d['sets'];lines.append(f"| {label} | {s['forget']['rougeL_recall']:.3f} | {s['forget']['lp_answer']:.3f} | {s['retain']['rougeL_recall']:.3f} | {100*s['mmlu_en']['correct']:.1f}% | {100*s['mmlu_hi']['correct']:.1f}% | {d['weight_bytes']/1e9:.2f} |")
    lines+=['','| NF4 − bf16 | Forget ROUGE-L Δ [95% CI] | Forget log-prob Δ [95% CI] | Retain ROUGE-L Δ [95% CI] | English MMLU Δ, pp [95% CI] |','| --- | --- | --- | --- | --- |']
    def interval(x,scale=1):return f"{scale*x.get('point',x.get('delta')):+.3f} [{scale*x['ci_low']:+.3f}, {scale*x['ci_high']:+.3f}]"
    for method,c in contrasts.items():lines.append(f"| {method} | {interval(c['forget']['rougeL_recall'])} | {interval(c['forget']['lp_answer'])} | {interval(c['retain']['rougeL_recall'])} | {interval(c['mmlu_en']['correct'],100)} |")
    lines+=['','One seed. MANSU has at most30 singleton optimization steps. NPO has different historical training and selection. Final forget overlaps available unlearning inputs, excludes validation, and includes60 additional rows. C0 originally trained on all TOFU rows. Validation utility IDs are excluded from English/Hindi test sets. No metric alone proves factual erasure.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.run,a.out)
