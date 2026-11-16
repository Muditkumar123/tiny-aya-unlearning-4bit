"""Check and summarize saved MANSU validation aggregates; this does not rescore a model."""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def summarize(results):
    records=[]
    for name,label in [('mansu_20261115_attempt','Tensor-range'),('mansu_nf4_scaled_20261115_attempt','NF4-scaled')]:
        folder=results/name
        raw=(folder/'training.json').read_bytes();training=json.loads(raw)
        summary=json.loads((folder/'attempt_summary.json').read_text())
        assert hashlib.sha256(raw).hexdigest()==summary['source_training_report_sha256']
        assert training['status']=='failed' and training['phase']=='floor'
        t=training['training'];history=t['history'];cfg=t['config']
        assert len(history)==summary['training_steps_executed']==cfg['max_steps']==30
        checks=[r for r in history if 'validation' in r]
        eligible=[r for r in checks if t['baseline']['utility_accuracy']-r['validation']['utility_accuracy']<=cfg['max_utility_drop']+1e-12]
        selected=min(eligible,key=lambda r:r['validation']['forget_accuracy'])
        assert selected['step']==t['selected_step']==summary['selected_step']
        assert selected['validation']==t['selected_metrics']==summary['selected_before_floor']
        assert t['baseline']==summary['baseline'] and training['post_floor_validation']==summary['after_floor']
        drop=t['baseline']['utility_accuracy']-training['post_floor_validation']['utility_accuracy']
        assert drop>cfg['max_utility_drop']+1e-12
        assert math.isclose(drop*100,summary['utility_drop_percentage_points'],abs_tol=1e-9)
        assert all(math.isfinite(r[k]) for r in history for k in ['forget_nll','retain_kl','objective'])
        for metrics in [summary['baseline'],summary['selected_before_floor'],summary['after_floor']]:
            for field,n in [('forget_accuracy',40),('utility_accuracy',100)]:
                assert 0<=metrics[field]<=1 and math.isclose(metrics[field]*n,round(metrics[field]*n),abs_tol=1e-9)
        assert not any(summary[k] for k in ['checkpoint_exported','nf4_comparison_run','final_test_scoring_run'])
        records.append((label,summary))
    print('Saved validation aggregate consistency checks pass for both attempts; no model predictions rescored.')
    print('| Floor | Steps run / selected | Forget MCQ, before -> after | Utility, before -> after | Result |')
    print('| --- | --- | --- | --- | --- |')
    for label,s in records:
        before,after=s['selected_before_floor'],s['after_floor']
        print(f"| {label} | {s['training_steps_executed']} / {s['selected_step']} | {before['forget_accuracy']:.1%} -> {after['forget_accuracy']:.1%} | {before['utility_accuracy']:.1%} -> {after['utility_accuracy']:.1%} | Utility guard failed |")

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,default=ROOT/'results');summarize(parser.parse_args().results)
