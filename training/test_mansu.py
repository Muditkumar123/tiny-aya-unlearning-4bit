"""CPU MANSU checks using synthetic inputs and tiny random models; no downloads."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mansu import MansuConfig, MansuTrainer, completion_nll, estimate_fisher, magnitude_floor, rank_mlp_layers, reference_kl
from run_mansu import checkpoint_identity, file_sha256, validate_split_ids
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'compression'), str(ROOT / 'evaluation')]
from discover_mansu import aggregate_edges
from mansu_diagnostics import activation_shift, cad, quantized_update_survival

torch.set_num_threads(1)
def tiny():
    torch.manual_seed(7)
    return LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)).eval()
def batch(ids=(2, 4, 5, 3)):
    x = torch.tensor([ids]); y = x.clone(); y[:, :2] = -100
    return {'input_ids': x, 'attention_mask': torch.ones_like(x), 'labels': y}

class MansuTests(unittest.TestCase):
    def test_configuration_and_shared_reference_are_rejected(self):
        for kw in ({'max_steps':1.5}, {'evaluate_every':True}, {'max_utility_drop':1.1}):
            with self.assertRaises(ValueError):MansuConfig(**kw)
        model = tiny()
        with self.assertRaises(ValueError):MansuTrainer(model, copy.copy(model), [0], MansuConfig())

    def test_per_example_fisher_avoids_gradient_cancellation(self):
        class Opposite(nn.Module):
            def __init__(self):
                super().__init__(); self.w = nn.Parameter(torch.tensor(0.0))
            def forward(self, input_ids, **kw):
                return SimpleNamespace(logits=torch.stack([self.w, -self.w]).expand(*input_ids.shape, 2))
        m = Opposite()
        examples = [{'input_ids': torch.tensor([[0, 0]]), 'labels': torch.tensor([[-100, a]])} for a in (0, 1)]
        self.assertEqual(estimate_fisher(m, {'w': m.w}, examples)['w'].item(), 1.0)
        with self.assertRaises(ValueError):
            estimate_fisher(m, {'w': m.w}, [{'input_ids': torch.zeros(2, 2, dtype=torch.long)}])

    def test_completion_mask_and_reference_kl_direction(self):
        labels = torch.tensor([[-100, -100, 1]])
        a = torch.randn(1, 3, 3, requires_grad=True); b = torch.randn(1, 3, 3, requires_grad=True)
        loss = reference_kl(a, b, labels); p = b[0, 1].softmax(-1)
        self.assertAlmostEqual(loss.item(), (p * (p.log() - a[0, 1].log_softmax(-1))).sum().item(), places=6)
        loss.backward(); self.assertIsNone(b.grad)
        self.assertEqual(reference_kl(a.detach(), a.detach(), labels).item(), 0)
        changed = a.detach().clone(); changed[:, 0] += 99
        self.assertEqual(completion_nll(a, labels).item(), completion_nll(changed, labels).item())
        with self.assertRaises(ValueError):completion_nll(a, torch.full_like(labels, -100))

    def test_floor_preserves_direction_zeros_and_protected_coordinates(self):
        original = torch.tensor([1., -2., .5, .75]); trained = original + torch.tensor([.001, -.001, 0, .001])
        mask = torch.tensor([True, True, True, False])
        result, info = magnitude_floor(original, trained, allowed=mask)
        self.assertAlmostEqual(float(result[0]-original[0]), info['floor'])
        self.assertAlmostEqual(float(result[1]-original[1]), -info['floor'])
        self.assertTrue(torch.equal(result[2:], original[2:]))
        zero, _ = magnitude_floor(original, trained, policy='zero', allowed=mask)
        self.assertTrue(torch.equal(zero, original))
        with self.assertRaises(ValueError):magnitude_floor(original, trained, policy='invalid')

    def test_mean_absolute_attribution_before_aggregation(self):
        signed, absolute = aggregate_edges([{'x':1., 'y':.5}, {'x':-1., 'y':.5}])
        self.assertEqual(signed['x'], 0)
        self.assertEqual(rank_mlp_layers(absolute, {'x':1, 'y':0}, 1), [1])
        with self.assertRaises(ValueError):rank_mlp_layers({'x':0}, {'x':0}, 1)
        with self.assertRaises(ValueError):aggregate_edges([{'x':1}, {'y':1}])

    def test_nf4_floor_uses_each_block_scale_and_keeps_masks(self):
        original=torch.cat([torch.full((64,),.01),torch.full((64,),.1)])
        trained=original+1e-5;trained[2]=original[2]
        mask=torch.ones(128,dtype=torch.bool);mask[1]=False
        result,info=magnitude_floor(original,trained,scale='nf4_block',allowed=mask)
        self.assertAlmostEqual(float(result[0]-original[0]),.01*.0796*.704,places=7)
        self.assertAlmostEqual(float(result[64]-original[64]),.1*.0796*.704,places=7)
        self.assertTrue(torch.equal(result[1:3],original[1:3]))
        self.assertEqual(info['scale'],'nf4_block')

    def test_optimizer_and_floor_leave_all_protected_weights_unchanged(self):
        m = tiny(); ref = copy.deepcopy(m); initial = {n:p.detach().clone() for n,p in m.named_parameters()}
        t = MansuTrainer(m, ref, [0], MansuConfig(learning_rate=.01, max_steps=2, kl_weight=.2, fisher_quantile=.5))
        retain = [batch((2,6,7,3)), batch((2,8,9,3))]
        t.prepare_fisher(retain); t.train([batch()], retain); t.apply_floor(); changed = 0
        for n,p in m.named_parameters():
            if n not in t.parameters:self.assertTrue(torch.equal(p, initial[n]), n)
            else:
                self.assertTrue(torch.equal(p[~t.masks[n]], initial[n][~t.masks[n]]), n)
                changed += int((p != initial[n]).sum())
        self.assertGreater(changed, 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in ref.parameters()))
        with self.assertRaises(ValueError):MansuTrainer(tiny(), tiny(), [8], MansuConfig())
        bad = copy.deepcopy(m)
        with torch.no_grad():next(bad.parameters()).add_(1)
        with self.assertRaises(ValueError):MansuTrainer(m, bad, [0], MansuConfig())

    def test_best_checkpoint_selection_and_early_stop(self):
        m = tiny(); t = MansuTrainer(m, copy.deepcopy(m), [0], MansuConfig(learning_rate=.01, max_steps=3,
            kl_weight=0, evaluate_every=1, early_stop_utility_drop=.05))
        t.prepare_fisher([batch()])
        scores = iter([{'forget_accuracy':.8,'utility_accuracy':.8}, {'forget_accuracy':.6,'utility_accuracy':.79},
                       {'forget_accuracy':.4,'utility_accuracy':.5}])
        def evaluate(model):
            self.assertFalse(model.training)
            return next(scores)
        report = t.train([batch()], [batch()], evaluate=evaluate)
        self.assertEqual(report['selected_step'], 1); self.assertEqual(len(report['history']), 2)

    def test_cad_rejects_different_inputs_missing_edges_and_zero_denominator(self):
        a = {'purpose':'forget_knowledge','algorithm':'EAP-IG-inputs','ig_steps':5,'metric':'fixed',
             'pairs_sha256':'same','sample_ids':['d0'],'tokenizer_sha256':{'tokenizer.json':'fixed'},'seed':0,
             'edge_destination_layer':{'x':0,'y':1},'edges_signed_mean':{'x':2.,'y':1.}}
        b = copy.deepcopy(a); b['edges_signed_mean']['x'] = 1.
        self.assertEqual(cad(a,b,[0])['cad'], .5)
        b['pairs_sha256'] = 'different'
        with self.assertRaises(ValueError):cad(a,b,[0])
        b = copy.deepcopy(a); b['edges_signed_mean'].pop('x')
        with self.assertRaises(ValueError):cad(a,b,[0])
        a['edges_signed_mean']['x'] = 0
        with self.assertRaises(ValueError):cad(a,a,[0])

    def test_exact_utility_threshold_does_not_stop_on_float_rounding(self):
        m=tiny();t=MansuTrainer(m,copy.deepcopy(m),[0],MansuConfig(max_steps=2,evaluate_every=1))
        t.prepare_fisher([batch()])
        scores=iter([{'forget_accuracy':.8,'utility_accuracy':.8},
                     {'forget_accuracy':.7,'utility_accuracy':.78},
                     {'forget_accuracy':.6,'utility_accuracy':.77}])
        report=t.train([batch()],[batch()],evaluate=lambda model:next(scores))
        self.assertEqual(len(report['history']),2)

    def test_range_floor_is_not_a_quantization_guarantee(self):
        a = torch.tensor([.1,.2]); b,_ = magnitude_floor(a,a+.001)
        result = quantized_update_survival(a.round(),b.round(),b-a)
        self.assertEqual(result['survival_fraction'], 0)
        self.assertFalse(result['semantic_forgetting_established'])

    def test_activation_shift_identity_and_hook_cleanup(self):
        m = tiny(); r = activation_shift(copy.deepcopy(m),m,[batch()],[0])
        self.assertEqual(r['as_c'],0); self.assertEqual(r['as_nc'],0)
        self.assertTrue(all(not l.mlp._forward_hooks for l in m.model.layers))
        with self.assertRaises(ValueError):activation_shift(copy.deepcopy(m),m,[batch()],[0,1])

    def test_validation_overlap_rejected(self):
        v = {'forget':[{'id':'fe','answer':'A'}],'utility':[{'id':'ue','answer':'B'}]}
        validate_split_ids([{'id':'ft'}],[{'id':'rt'}],v,{'sample_ids':['ft']})
        with self.assertRaises(ValueError):validate_split_ids([{'id':'fe'}],[{'id':'rt'}],v,{'sample_ids':['d']})
        v['forget'][0]['id'] = ''
        with self.assertRaises(ValueError):validate_split_ids([{'id':'ft'}],[{'id':'rt'}],v,{'sample_ids':['d']})

    def test_full_cpu_command_exports_reloadable_checkpoint(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); source = d/'source'; tiny().to(torch.bfloat16).save_pretrained(source)
            words = ['[UNK]','[PAD]','[BOS]','<EOS_TOKEN>','A','B','<|END_OF_TURN_TOKEN|>','C','D',
                     'user','assistant','foo','bar','baz','qux']
            raw = Tokenizer(WordLevel({w:i for i,w in enumerate(words)},unk_token='[UNK]')); raw.pre_tokenizer=WhitespaceSplit()
            tok = PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',pad_token='[PAD]',eos_token='<EOS_TOKEN>')
            tok.chat_template="{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
            tok.save_pretrained(source)
            for name,row in [('forget',{'id':'ft','prompt':'foo','answer':'A'}),('retain',{'id':'rt','prompt':'bar','answer':'B'})]:
                (d/f'{name}.json').write_text(json.dumps({'rows':[row]}))
            (d/'validation.json').write_text(json.dumps({'forget':[{'id':'fe','prompt':'baz','answer':'A'}],
                                                       'utility':[{'id':'ue','prompt':'qux','answer':'B'}]}))
            (d/'circuit.json').write_text(json.dumps({'purpose':'forget_knowledge','algorithm':'EAP-IG-inputs',
                'sample_ids':['ft'],'model_weights_sha256':checkpoint_identity(source),'tokenizer_sha256':{p.name:file_sha256(p) for p in source.iterdir() if p.name in {'tokenizer.json','tokenizer_config.json','chat_template.jinja'}},'edges_abs_mean':{'x':1.,'y':.1},
                'edge_destination_layer':{'x':0,'y':1}}))
            (d/'config.json').write_text(json.dumps({'seed':0,'circuit_size':1,'fisher_examples':1,'max_length':32,
                'training':{'learning_rate':.05,'max_steps':2,'kl_weight':.1,'evaluate_every':1,
                            'max_utility_drop':1.,'early_stop_utility_drop':1.}}))
            args=[sys.executable,'-B',str(ROOT/'training/run_mansu.py'),'--checkpoint',str(source)]
            for flag,name in [('circuit','circuit'),('forget','forget'),('retain','retain'),('validation','validation'),('config','config')]:
                args.extend(['--'+flag,str(d/(name+'.json'))])
            args.extend(['--out',str(d/'run'),'--device','cpu'])
            p=subprocess.run(args+['--check-inputs'],capture_output=True,text=True,timeout=60)
            self.assertEqual(p.returncode,0,p.stderr); self.assertFalse((d/'run').exists())
            p=subprocess.run(args,capture_output=True,text=True,timeout=120)
            self.assertEqual(p.returncode,0,p.stderr)
            report=json.loads((d/'run/run.json').read_text())
            self.assertEqual(report['status'],'trained_and_exported_not_quantization_validated')
            self.assertFalse(report['quantization_survival_verified'])
            loaded=LlamaForCausalLM.from_pretrained(d/'run/checkpoint',dtype=torch.bfloat16)
            self.assertEqual(loaded.config.num_hidden_layers,2)
            p=subprocess.run(args,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(p.returncode,0)

if __name__ == '__main__':unittest.main(verbosity=2)
