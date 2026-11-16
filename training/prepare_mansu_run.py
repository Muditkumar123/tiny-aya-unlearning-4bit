"""Prepare deterministic local MANSU inputs; source text stays in ignored runs/."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from transformers import AutoTokenizer
from run_mansu import checkpoint_identity, file_sha256


def prepare(source, checkpoint, out):
    if out.exists():
        raise ValueError('input output directory must be new')
    tok = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    names = ['tofu_author_split.json','tofu_forget10_perturbed.json','tofu_retain_perturbed.json',
             'global_mmlu_en500.json','global_mmlu_hi200.json']
    data = {n: json.loads((source/n).read_text())['rows'] for n in names}
    pert = {r['id']:r for r in data[names[1]]}
    forget = [{**r, **pert[r['id']]} for r in data[names[0]]]
    grouped = defaultdict(list)
    for row in forget: grouped[row['author']].append(row)
    train, validation, final = [], [], []
    for author in sorted(grouped):
        rows = grouped[author]
        assert len(rows)==20
        train.extend(rows[:15]); validation.extend(rows[15:17]); final.extend(rows[:15]+rows[17:])
    retain_groups = defaultdict(list)
    for row in data[names[2]]: retain_groups[row['author']].append(row)
    retain_train, retain_final = [], []
    for author in sorted(retain_groups):
        retain_train.extend(retain_groups[author][:15]); retain_final.extend(retain_groups[author][15:])
    rng = random.Random(0)
    rng.shuffle(train); rng.shuffle(retain_train)
    utility = list(data[names[3]]); rng.shuffle(utility)
    utility_validation = utility[:100]
    reserved = {r['id'] for r in utility_validation}
    def mcq(row, position):
        wrong = row['perturbed_answer'][:3]
        assert len(wrong)==3 and row['answer'] not in wrong and len(set(wrong))==3
        options = list(wrong); options.insert(position,row['answer'])
        prompt = 'Answer the following multiple-choice question with the letter of the correct option only.\n\n'+row['prompt']+'\n'+'\n'.join(f'{letter}. {text}' for letter,text in zip('ABCD',options))
        return {'id':row['id'],'prompt':prompt,'answer':'ABCD'[position]}
    def render(prompt):
        return tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    pairs=[]
    for row in train:
        clean, corrupt = mcq(row,0), mcq(row,1)
        a,b = render(clean['prompt']),render(corrupt['prompt'])
        if len(tok(a,add_special_tokens=False).input_ids) == len(tok(b,add_special_tokens=False).input_ids):
            pairs.append({'id':row['id'],'clean_prompt':a,'clean_answer':'A','corrupt_prompt':b,'corrupt_answer':'B'})
        if len(pairs)==50: break
    assert len(pairs)==50, 'fewer than 50 exact token-length-matched MCQ contrasts'
    val = {'forget':[mcq(r,i%4) for i,r in enumerate(validation)],'utility':utility_validation}
    sets = {'forget':final,'retain':retain_final,'mmlu_en':[r for r in data[names[3]] if r['id'] not in reserved],
            'mmlu_hi':[r for r in data[names[4]] if r['id'] not in reserved]}
    out.mkdir(parents=True)
    objects = {'forget.json':{'rows':train},'retain.json':{'rows':retain_train},'validation.json':val,
               'discovery.json':{'purpose':'forget_knowledge','pairs':pairs},'evaluation.json':sets}
    for name,obj in objects.items(): (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    receipt={'prepared_at_utc':datetime.now(timezone.utc).isoformat(),'seed':0,
             'source_sha256':{n:file_sha256(source/n) for n in names},'checkpoint_weights_sha256':checkpoint_identity(checkpoint),
             'input_sha256':{n:file_sha256(out/n) for n in objects},
             'ids':{'forget_train':[r['id'] for r in train],'retain_train':[r['id'] for r in retain_train],
                    'discovery':[r['id'] for r in pairs],'forget_validation':[r['id'] for r in val['forget']],
                    'utility_validation':[r['id'] for r in utility_validation],**{k+'_evaluation':[r['id'] for r in v] for k,v in sets.items()}},
             'actual_maximum_training_exposure':{'forget_ids_first_30':[r['id'] for r in train[:30]],'retain_ids_first_30':[r['id'] for r in retain_train[:30]]},
             'caveats':['C0 trained on all TOFU rows.','Final forget includes unlearning input rows; validation IDs excluded.',
                        'MCQ option swaps are a new attribution adaptation; no causal fact-circuit validation.',
                        'Historical NPO training and selection differ from this 30-step singleton run.']}
    (out/'inputs_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'counts':{k:len(v) for k,v in sets.items()},'training':[len(train),len(retain_train)],'discovery':len(pairs),'validation':[len(validation),len(utility_validation)]}))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-data',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();prepare(a.source_data,a.checkpoint,a.out)
