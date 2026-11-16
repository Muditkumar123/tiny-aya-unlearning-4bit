"""Check pairing and removal of source text from new MANSU result exports."""
import contextlib,io,json
from pathlib import Path
import tempfile,unittest
from export_mansu_results import export


class ExportTests(unittest.TestCase):
    def make_run(self,root):
        for rel in ['training/run.json','discovery/attribution.json','post_attribution/attribution.json','inputs/inputs_receipt.json','execution_stages.json']:
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
        for method in ['c0','npo','mansu']:
            for precision in ['bf16','nf4']:
                d=root/'evaluation'/(method+'_'+precision);d.mkdir(parents=True)
                (d/'execution.json').write_text(json.dumps({'weight_bytes':100}))
                for subset in ['forget','retain','mmlu_en','mmlu_hi']:
                    rows=[]
                    for i in range(4):
                        r={'id':str(i),'prompt':'PRIVATE PROMPT','answer':'PRIVATE ANSWER'}
                        if subset in ['forget','retain']:
                            r.update(author=i//2,rougeL_recall=.2+(.1 if precision=='nf4' else 0),lp_answer=-1.,truth_ratio=.5)
                        else:r.update(subject='toy',answer='A',lp=[-1,-2,-3,-4],pred='A',correct=1)
                        rows.append(r)
                    (d/(subset+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_export_strips_text_and_pairs_intervals(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'run';self.make_run(root);out=Path(t)/'export'
            with contextlib.redirect_stdout(io.StringIO()):export(root,out)
            report=json.loads((out/'comparison.json').read_text())
            ci=report['nf4_minus_bf16']['mansu']['forget']['rougeL_recall']
            self.assertAlmostEqual(ci['point'],.1);self.assertAlmostEqual(ci['ci_low'],.1)
            self.assertEqual(ci['n_clusters'],2)
            for p in out.rglob('*.jsonl'):
                self.assertNotIn('PRIVATE',p.read_text())
            with self.assertRaises(ValueError):export(root,out)

    def test_mismatched_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)/'run';self.make_run(root)
            p=root/'evaluation/mansu_nf4/forget.jsonl';rows=p.read_text().splitlines()
            row=json.loads(rows[0]);row['id']='different';rows[0]=json.dumps(row);p.write_text('\n'.join(rows)+'\n')
            with self.assertRaises(AssertionError):export(root,Path(t)/'export')

if __name__=='__main__':unittest.main(verbosity=2)
