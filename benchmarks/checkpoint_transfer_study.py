"""Prospective screen, twin-donor causal intervention, and file-disjoint text confirmation."""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, forbid_donor_forward
from nexus_expanse.checkpoint_transport import CONFIG, METHODS, CorrectedLinear, ImportedSkill, SkillBank, load_skill

SOURCE='d48b2dfc2b15607c27f6d89672f690ebdce18265'
RULE_SEEDS=(211,212,213,214,215)
TEXT_SEEDS=(311,312,313,314,315)
TEXT_FILES={'python':['nexus_expanse/core.py','nexus_expanse/memory.py'],
            'fit':['README.md','docs/ARCHITECTURE_V1.md'],
            'val':['docs/GATE25_PREREG.md'],'test':['docs/GATE26_PREREG.md'],
            'retention':['nexus_expanse/arbitration.py','nexus_expanse/fast_checkpoint.py']}
OUT=ROOT/'evidence'
CLOCK=time.perf_counter


def digest(x):
    return hashlib.sha256(x).hexdigest()


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def atomic(path,data):
    path=Path(path)
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def corpus():
    seen=set()
    splits={}
    manifest={}
    for split,paths in TEXT_FILES.items():
        blocks=[]
        for path in paths:
            raw=git('show',SOURCE+':'+path)
            spans=[]
            for start in range(0,len(raw)-63,64):
                chunk=raw[start:start+64]
                grams={chunk[i:i+32] for i in range(33)}
                if grams & seen:
                    continue
                seen.update(grams)
                spans.append([start,digest(chunk)])
                blocks.append(list(chunk))
                if len(spans)==16:
                    break
            manifest[path]=dict(file_sha256=digest(raw),spans=spans,split=split)
        if len(blocks)<4:
            raise ValueError('insufficient disjoint windows for '+split)
        data=torch.tensor(blocks)
        splits[split]=(data[:,:-1],data[:,1:])
    return splits,manifest


def rules(seed):
    rng=random.Random(seed+17171)
    a,b=[],[]
    for _ in range(4):
        p=rng.sample(range(8),8)
        while True:
            q=rng.sample(range(8),8)
            if all(i!=j for i,j in zip(p,q)):
                break
        a+=p
        b+=q
    return [torch.tensor(x)+145 for x in (a,b)]


def rule_batch(seed, split, step=0):
    rng=torch.Generator().manual_seed(100000*seed+1000*split+step)
    n=32 if split==0 else 128
    k=torch.arange(n)%32
    nonce=torch.randint(0,16,(n,3),generator=rng)+180+split*16
    x=torch.cat((nonce,(40+k%8)[:,None],(80+k//8)[:,None],torch.full((n,1),125)),1)
    return x,k


def batch(domain,data,seed,step,table):
    if domain=='rule':
        x,k=rule_batch(seed,0,step)
        return x,table[k]
    x,y=data
    g=torch.Generator().manual_seed(1000*seed+step)
    ix=torch.randperm(len(x),generator=g)[:8]
    return x[ix],y[ix]


def setup_donor(seed):
    torch.manual_seed(seed)
    return DenseTokenDonorLM(vocab_size=256,dim=32,n_layers=2,n_heads=4,ffn_mult=2,max_seq_len=64)


def score_logits(logits,y,reference=None):
    logp=logits.double().log_softmax(-1)
    pred=logp.argmax(-1)
    score=dict(bpb=float(-logp.gather(-1,y.unsqueeze(-1)).mean()/math.log(2)),
               accuracy=float((pred==y).double().mean()))
    if reference is not None:
        ref=reference.double().log_softmax(-1)
        score.update(kl_bits=float((ref.exp()*(ref-logp)).sum(-1).mean()/math.log(2)),
                     agreement=float((pred==ref.argmax(-1)).double().mean()))
    return score


def task_logits(logits,y):
    return logits[:,-1] if y.ndim==1 else logits


def loss(logits,y):
    return F.cross_entropy(task_logits(logits,y).reshape(-1,256),y.reshape(-1))


def optim(module):
    return torch.optim.AdamW(module.parameters(),lr=.004,weight_decay=.01)


def update(model,opt,logits,y):
    opt.zero_grad(set_to_none=True)
    loss(logits,y).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()


def source_pack(path,model,ids):
    torch.save(dict(weights=model.state_dict(),byte_to_row=ids),path)


def produce(path,seed,domain,splits):
    """Separate process. Only PT packages are exposed to the static converter."""
    start=CLOCK()
    ids=torch.randperm(256,generator=torch.Generator().manual_seed(seed+16000))
    parent=setup_donor(seed)
    def output(model,x):
        return model(ids[x])[...,ids]
    opt=optim(parent)
    for step in range(32):
        x,y=batch('text',splits['python'],seed,step,None)
        update(parent,opt,output(parent,x),y)
    source_pack(path/'parent.pt',parent,ids)
    parent_seconds=CLOCK()-start
    old=NexusExpanseByteCore(**CONFIG)
    opt=optim(old)
    for step in range(32):
        x,y=batch('text',splits['python'],seed,step,None)
        update(old,opt,old.forward_neural(x,targets=y)[0],y)
    torch.save(old.state_dict(),path/'old.pt')
    common_seconds=CLOCK()-start
    receipt={'common_build_seconds':common_seconds,'parent_creation_seconds':parent_seconds,
             'receiver_creation_seconds':common_seconds-parent_seconds,'donors':{}}
    tables=rules(seed)
    for label,table in [('A',tables[0]),('B',tables[1])][:2 if domain=='rule' else 1]:
        begin=CLOCK()
        donor=copy.deepcopy(parent)
        donor.forward_calls=0
        opt=optim(donor)
        validation,test=eval_sets(domain,seed,splits,table)
        for step in range(1,321 if domain=='rule' else 65):
            donor.train()
            x,y=batch(domain,splits['fit'],seed,step,table)
            update(donor,opt,output(donor,x),y)
            if domain=='rule' and step%8==0:
                donor.eval()
                with torch.inference_mode():
                    x,y=validation
                    val=score_logits(task_logits(output(donor,x),y),y)
                    if val['accuracy']==1 and val['bpb']<=.06:
                        break
        donor.eval()
        saved={}
        with torch.inference_mode():
            for name,(x,y) in [('validation',validation),('test',test)]:
                saved[name]=task_logits(output(donor,x),y).clone()
        source_pack(path/(label+'.pt'),donor,ids)
        torch.save(saved,path/(label+'.reference'))
        receipt['donors'][label]=dict(training_steps=step,producer_forward_calls=donor.forward_calls,
            creation_and_reference_seconds=CLOCK()-begin,checkpoint_bytes=(path/(label+'.pt')).stat().st_size,
            checkpoint_sha256=digest((path/(label+'.pt')).read_bytes()),
            validation=score_logits(saved['validation'],validation[1]),
            test=score_logits(saved['test'],test[1]))
    atomic(path/'producer.json',receipt)


def eval_sets(domain,seed,splits,table):
    if domain=='text':
        return splits['val'],splits['test']
    sets=[]
    for split in (1,2):
        x,k=rule_batch(seed,split)
        sets.append((x,table[k]))
    return sets


@torch.inference_mode()
def evaluate(model,xy,ref=None):
    model.eval()
    x,y=xy
    logits=task_logits(model(x),y)
    return score_logits(logits,y,ref),logits


def scratch(path):
    skill=ImportedSkill()
    skill.core.load_state_dict(torch.load(path/'old.pt',weights_only=True))
    with torch.no_grad():
        skill.readout.weight.copy_(skill.core.emb.weight)
    return skill


def nbytes(skill):
    out=io.BytesIO()
    torch.save(skill.state_dict(),out)
    return out.tell()


def select_arm(path,label,method,arm):
    if arm=='scratch':
        return scratch(path)
    if arm=='parent':
        name='parent'
    elif arm=='wrong':
        name='B' if label=='A' else 'A'
    else:
        name=label
    return load_skill(path/(name+'.pt'),method)


def near(score,ref,domain):
    return (score['kl_bits']<=.02 and score['agreement']>=.99 and score['bpb']<=ref['bpb']+.02
            and (domain=='text' or score['accuracy']>=.95))


def hit(score,domain,val_target):
    return score['bpb']<=val_target and (domain=='text' or score['accuracy']>=.95)


def trial(path,seed,domain,label,method,arm,splits,ref,prod,screen):
    table=rules(seed)[label=='B']
    validation,test=eval_sets(domain,seed,splits,table)
    limit=.20 if domain=='rule' else prod['validation']['bpb']+.02
    # Same timing boundary for every arm. The base is an existing receiver.
    torch.manual_seed(seed+900)
    start=CLOCK()
    skill=select_arm(path,label,method,arm)
    prep=CLOCK()-start
    t=CLOCK()
    val,_=evaluate(skill,validation)
    timed=prep+CLOCK()-t
    zero,zlog=evaluate(skill,test,ref['test'])
    zero['near']=near(zero,prod['test'],domain)
    if domain=='rule':
        other=rules(seed)[label=='A']
        zero['opposite_accuracy']=float((zlog.argmax(-1)==other[rule_batch(seed,2)[1]]).double().mean())
    steps=0
    trace=[[0,val['bpb'],val['accuracy'],timed]]
    if not screen and not hit(val,domain,limit):
        t=CLOCK()
        opt=optim(skill)
        for steps in range(1,161):
            skill.train()
            x,y=batch(domain,splits['fit'],seed,steps,table)
            update(skill,opt,skill(x),y)
            if steps%8==0:
                val,_=evaluate(skill,validation)
                trace.append([steps,val['bpb'],val['accuracy'],timed+CLOCK()-t])
                if hit(val,domain,limit):
                    break
        timed+=CLOCK()-t
    selected,logits=evaluate(skill,test,ref['test'])
    selected['near']=near(selected,prod['test'],domain)
    test_target=.20 if domain=='rule' else prod['test']['bpb']+.02
    matched=hit(val,domain,limit) and hit(selected,domain,test_target)
    old=NexusExpanseByteCore(**CONFIG).eval()
    old.load_state_dict(torch.load(path/'old.pt',weights_only=True))
    # Verify against an independent old-mode receiver on all frozen Python probes.
    bank=SkillBank(copy.deepcopy(old),skill.eval()).eval()
    with torch.inference_mode():
        probes=splits['retention'][0]
        if not torch.equal(bank(probes),old.forward_neural(probes)[0]):
            raise RuntimeError('old-mode forecast changed')
        if not torch.equal(task_logits(bank(test[0],use_skill=True),test[1]),logits):
            raise RuntimeError('installed skill disagrees with evaluated skill')
    if not screen:
        torch.save(dict(state=bank.state_dict(),method='monarch' if arm=='scratch' else method,
                        label=label,test_sha=digest(logits.numpy().tobytes()),
                        old_sha=digest(
                            old.forward_neural(probes)[0].detach().numpy().tobytes())),
                   path/f'{label}-{arm}-{method}.restore')
    return dict(method=method,arm=arm,zero=zero,selected=selected,steps=steps,trace=trace,
                preparation_seconds=prep,total_receiver_seconds=timed,matched_target=matched,
                old_mode_exact=True,skill_bytes=nbytes(skill),prediction_sha256=digest(logits.numpy().tobytes()))


@torch.inference_mode()
def cold_batch(path,seed,domain,splits):
    checked=0
    for file in sorted(path.glob('*.restore')):
        saved=torch.load(file,weights_only=True)
        skill=ImportedSkill()
        method=saved['method']
        if method!='monarch':
            for layer in skill.core.layers:
                for name in ('fc1','fc2'):
                    old=getattr(layer.ffn,name)
                    setattr(layer.ffn,name,CorrectedLinear(old,torch.zeros(old.out_dim,old.in_dim),method))
        bank=SkillBank(NexusExpanseByteCore(**CONFIG),skill).eval()
        bank.load_state_dict(saved['state'])
        _,test=eval_sets(domain,seed,splits,rules(seed)[saved['label']=='B'])
        logits=task_logits(bank(test[0],use_skill=True),test[1])
        assert digest(logits.numpy().tobytes())==saved['test_sha']
        assert digest(bank(splits['retention'][0]).numpy().tobytes())==saved['old_sha']
        checked+=1
    print(checked)


def cross_parts(path,seed,label,method,splits,refs):
    opposite='B' if label=='A' else 'A'
    one=load_skill(path/(label+'.pt'),method)
    other=load_skill(path/(opposite+'.pt'),method)
    for a,b in zip(one.core.layers,other.core.layers):
        a.ffn,b.ffn=b.ffn,a.ffn
    out={}
    for name,m in [('own_rest_opposite_ffn',one),('opposite_rest_own_ffn',other)]:
        scores={}
        for idx,key in enumerate(('A','B')):
            _,test=eval_sets('rule',seed,splits,rules(seed)[idx])
            scores[key]=evaluate(m,test,refs[key]['test'])[0]
        out[name]=scores
    return out


def run_seed(domain,seed,stage,methods,splits):
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)
        subprocess.run([sys.executable,__file__,'producer',str(p),str(seed),domain],check=True)
        receipt=json.loads((p/'producer.json').read_text())
        labels=list(receipt['donors']) if stage!='screen' else ['A']
        refs={k:torch.load(p/(k+'.reference'),weights_only=True) for k in labels}
        results=[]
        with forbid_donor_forward() as guard:
            for label in labels:
                specs=[(m,'import') for m in methods]
                if stage!='screen':
                    specs+= [(methods[0],a) for a in (('scratch','parent','wrong') if domain=='rule'
                                                       else ('scratch','parent'))]
                    if domain=='text' and 'monarch' not in methods:
                        specs.append(('monarch','import'))
                random.Random(seed+ord(label)).shuffle(specs)
                for m,arm in specs:
                    reps=[trial(p,seed,domain,label,m,arm,splits,refs[label],receipt['donors'][label],stage=='screen')
                          for _ in range(2 if domain=='rule' and stage!='screen' else 1)]
                    r=reps[0]
                    if any(rep['prediction_sha256']!=r['prediction_sha256'] for rep in reps):
                        raise RuntimeError('nondeterministic timed repeat')
                    r['label']=label
                    r['timing_repetitions']=[rep['total_receiver_seconds'] for rep in reps]
                    r['total_receiver_seconds']=statistics.median(r['timing_repetitions'])
                    results.append(r)
            hybrids={label:{m:cross_parts(p,seed,label,m,splits,refs) for m in methods} for label in labels
                     if stage!='screen' and domain=='rule'}
        restored=0
        if stage!='screen':
            restored=int(subprocess.check_output([sys.executable,__file__,'cold',str(p),str(seed),domain],text=True))
            assert restored==len(results)
        return dict(seed=seed,domain=domain,cold_exact=restored,producer=receipt,donor_forward_calls_during_transfer=guard.donor_forward_calls,
                    runs=results,crossed_components=hybrids)


def summarize(rows,method):
    timings=[]
    fidelity=[]
    zero=[]
    swapped=[]
    donors=[]
    lower=[]
    for row in rows:
        for r in row['runs']:
            if r['method']!=method or r['arm']!='import':
                continue
            src=row['producer']['donors'][r['label']]['test']
            donors.append(row['domain']=='text' or hit(src,'rule',.20))
            fidelity.append(r['selected']['near'])
            zero.append(r['zero']['near'])
            if row['domain']=='rule':
                swapped.append(r['zero']['accuracy']-r['zero']['opposite_accuracy']>=.90)
            scratch_run=next(q for q in row['runs'] if q['arm']=='scratch' and q['label']==r['label'])
            diff=scratch_run['total_receiver_seconds']-r['total_receiver_seconds']
            frac=diff/scratch_run['total_receiver_seconds']
            if r['matched_target'] and scratch_run['matched_target']:
                donor_cost=(row['producer']['parent_creation_seconds']+
                            row['producer']['donors'][r['label']]['creation_and_reference_seconds'])
                timings.append(dict(seed=row['seed'],label=r['label'],seconds_saved=diff,fraction_saved=frac,
                                    donor_production_seconds=donor_cost,new_donor_net_seconds_saved=diff-donor_cost,
                                    receivers_to_amortize=math.ceil(donor_cost/diff) if diff>0 else None))
            elif r['matched_target'] and scratch_run['steps']==160 and not scratch_run['matched_target']:
                lower.append(dict(seed=row['seed'],label=r['label'],budget_time_difference=diff))
    return dict(cases=len(fidelity),qualified=sum(donors),zero_near=sum(zero),selected_near=sum(fidelity),
                swap_specific=sum(swapped),matched_times=timings,censored=lower,
                passed=all(donors) and all(fidelity) and all(swapped) and len(timings)==len(fidelity) and
                all(t['fraction_saved']>=.25 for t in timings))


def execute(stage,dest):
    splits,manifest=corpus()
    screenfile=OUT/'checkpoint-transfer-screen.json'
    confirmfile=OUT/'checkpoint-transfer-confirm.json'
    identity=git('rev-parse','HEAD').decode().strip()
    data=dict(stage=stage,source=identity,base=SOURCE,torch=str(torch.__version__),manifest=manifest,rules=[],text=[])
    if stage=='screen':
        data['rules']=[run_seed('rule',s,stage,METHODS,splits) for s in (101,102,103)]
        ranking=[]
        for m in METHODS:
            r=[next(v for v in b['runs'] if v['method']==m) for b in data['rules']]
            kl=[v['zero']['kl_bits'] for v in r]
            ranking.append((max(kl),statistics.mean(kl),r[0]['skill_bytes'],m))
        data['ranking']=sorted(ranking)
    else:
        winners=[row[-1] for row in json.loads(screenfile.read_text())['ranking'][:2]]
        offset=0
        if stage=='escalate':
            previous=json.loads(confirmfile.read_text())
            if previous['all_pass']:
                raise RuntimeError('escalation forbidden after success')
            winners=['residual16']
            offset=1000
        for category,seeds,methods in [('rules',RULE_SEEDS,winners),('text',TEXT_SEEDS,winners[:1])]:
            for s in seeds:
                print(stage,category,s+offset,flush=True)
                data[category].append(run_seed('rule' if category=='rules' else 'text',s+offset,stage,methods,splits))
                atomic(dest.with_suffix('.partial'),data)
        data['summary']={'rule':{m:summarize(data['rules'],m) for m in winners},
                         'text':summarize(data['text'],winners[0])}
        data['all_pass']=data['summary']['rule'][winners[0]]['passed'] and data['summary']['text']['passed']
    atomic(dest,data)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='cold':
        cold_batch(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4],corpus()[0])
    elif sys.argv[1]=='producer':
        produce(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4],corpus()[0])
    else:
        execute(sys.argv[1],Path(sys.argv[2]))
