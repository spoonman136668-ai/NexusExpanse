"""Independent same-byte, different-width donor swaps and Markdown timing."""
import copy
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks import checkpoint_transfer_study as study
from nexus_expanse.checkpoint_transport import CONFIG, ImportedSkill, CorrectedLinear, SkillBank, load_skill
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.donor_assimilation import forbid_donor_forward
import torch

WIDTH=64
CLOCK=time.perf_counter


def build_bank(path,width_method):
    old=NexusExpanseByteCore(**dict(CONFIG,dim=WIDTH))
    old.load_state_dict(torch.load(path/'old-wide.pt',weights_only=True))
    skill=ImportedSkill(WIDTH)
    if width_method!='scratch':
        for layer in skill.core.layers:
            for name in ('fc1','fc2'):
                base=getattr(layer.ffn,name)
                setattr(layer.ffn,name,CorrectedLinear(base,torch.zeros(base.out_dim,base.in_dim),width_method))
    return SkillBank(old,skill).eval()


def cold(path,seed,domain,method,splits):
    checked=0
    with torch.inference_mode():
        for f in sorted(path.glob('*.wide')):
            obj=torch.load(f,weights_only=True)
            bank=build_bank(path,method)
            bank.skill.load_state_dict(obj['state'])
            _,test=study.eval_sets(domain,seed,splits,study.rules(seed)[obj['label']=='B'])
            result=study.task_logits(bank(test[0],use_skill=True),test[1])
            assert study.digest(result.numpy().tobytes())==obj['hash']
            assert study.digest(bank(splits['retention'][0]).numpy().tobytes())==obj['old_hash']
            checked+=1
    print(checked)


def target(score,teacher,domain):
    return score['bpb']<=teacher['bpb']+.02 and (
        domain=='text' or score['accuracy']>=max(.95,teacher['accuracy']-.01))


def train_timed(path,arm,seed,domain,label,method,splits,receipt,ref,old):
    table=study.rules(seed)[label=='B']
    val,test=study.eval_sets(domain,seed,splits,table)
    torch.manual_seed(seed+173)
    start=CLOCK()
    if arm=='scratch':
        skill=ImportedSkill(WIDTH)
        skill.core.load_state_dict(torch.load(path/'old-wide.pt',weights_only=True))
        with torch.no_grad():
            skill.readout.weight.copy_(skill.core.emb.weight)
    else:
        src='parent' if arm=='parent' else label
        variant='monarch' if arm=='rank1' else method
        skill=load_skill(path/(src+'.pt'),variant,width_factor=2,scale_attention=arm!='naive')
    init=CLOCK()-start
    zero,z=study.evaluate(skill,test,ref['test'])
    zero['near']=study.near(zero,receipt['test'],domain)
    if domain=='rule':
        other=study.rules(seed)[label=='A'][study.rule_batch(seed,2)[1]]
        zero['opposite_accuracy']=float((z.argmax(-1)==other).double().mean())
    start=CLOCK()
    v,_=study.evaluate(skill,val)
    steps=0
    if arm in ('lift','scratch','parent') and not target(v,receipt['validation'],domain):
        opt=study.optim(skill)
        for steps in range(1,241):
            skill.train()
            x,y=study.batch(domain,splits['fit'],seed,steps,table)
            study.update(skill,opt,skill(x),y)
            if steps%8==0:
                v,_=study.evaluate(skill,val)
                if target(v,receipt['validation'],domain):
                    break
    elapsed=init+CLOCK()-start
    selected,z=study.evaluate(skill,test,ref['test'])
    selected['near']=study.near(selected,receipt['test'],domain)
    if arm=='lift':
        with torch.inference_mode():
            bank=SkillBank(copy.deepcopy(old),skill.eval()).eval()
            old_logits=old.forward_neural(splits['retention'][0])[0]
            assert torch.equal(old_logits,bank(splits['retention'][0]))
            assert torch.equal(z,study.task_logits(bank(test[0],use_skill=True),test[1]))
            torch.save(dict(state=skill.state_dict(),label=label,hash=study.digest(z.numpy().tobytes()),
                            old_hash=study.digest(old_logits.numpy().tobytes())),path/(label+'.wide'))
    return dict(arm=arm,zero=zero,selected=selected,matched=target(v,receipt['validation'],domain)
                and target(selected,receipt['test'],domain),steps=steps,conversion_seconds=init,
                total_seconds=elapsed,skill_bytes=study.nbytes(skill),hash=study.digest(z.numpy().tobytes()))


def run_seed(domain,seed,method,splits,demo):
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)
        subprocess.run([sys.executable,study.__file__,'producer',tmp,str(seed),domain],check=True)
        receipt=json.loads((path/'producer.json').read_text())
        torch.manual_seed(seed+301)
        old=NexusExpanseByteCore(**dict(CONFIG,dim=WIDTH))
        opt=study.optim(old)
        for step in range(32):
            x,y=study.batch('text',splits['python'],seed,step,None)
            study.update(old,opt,old.forward_neural(x)[0],y)
        torch.save(old.eval().state_dict(),path/'old-wide.pt')
        rows={}
        with forbid_donor_forward() as guard:
            for label,metrics in receipt['donors'].items():
                ref=torch.load(path/(label+'.reference'),weights_only=True)
                order=['lift','naive','rank1','parent','scratch']
                random.Random(seed+ord(label)).shuffle(order)
                rows[label]={}
                for arm in order:
                    repeats=[train_timed(path,arm,seed,domain,label,method,splits,metrics,ref,old)
                             for _ in range(2 if arm in ('lift','scratch','parent') else 1)]
                    r=repeats[0]
                    assert all(k['hash']==r['hash'] for k in repeats)
                    r['times']=[k['total_seconds'] for k in repeats]
                    r['total_seconds']=statistics.median(r['times'])
                    rows[label][arm]=r
        n=int(subprocess.check_output([sys.executable,__file__,'cold',tmp,str(seed),domain,method],text=True))
        assert n==len(rows)
        if demo:
            import shutil
            demo.mkdir(parents=True,exist_ok=True)
            for f in [path/'old-wide.pt',*path.glob('*.wide')]:
                shutil.copy(f,demo/f.name)
            study.atomic(demo/'example.json',dict(seed=seed,domain=domain,method=method,
                         command='python benchmarks/checkpoint_width_lift.py demo <demo-directory>',
                         source=study.git('rev-parse','HEAD').decode().strip()))
        return dict(seed=seed,domain=domain,donors=receipt['donors'],cases=rows,cold_exact=n,
                    forbidden_calls=guard.donor_forward_calls)


def run(dest,demo_path):
    splits,manifest=study.corpus()
    method=json.loads((study.OUT/'checkpoint-transfer-screen.json').read_text())['ranking'][0][-1]
    rows=[]
    for domain,seeds in [('rule',range(901,906)),('text',range(911,916))]:
        for seed in seeds:
            print(domain,seed,flush=True)
            rows.append(run_seed(domain,seed,method,splits,demo_path if seed==901 else None))
            study.atomic(dest.with_suffix('.partial'),dict(completed=rows))
    study.atomic(dest,dict(source=study.git('rev-parse','HEAD').decode().strip(),method=method,
                          donor_width=32,receiver_width=64,manifest=manifest,runs=rows))


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='cold':
        cold(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5],study.corpus()[0])
    elif sys.argv[1]=='demo':
        demo=Path(sys.argv[2])
        obj=json.loads((demo/'example.json').read_text())
        cold(demo,obj['seed'],obj['domain'],obj['method'],study.corpus()[0])
        prefix,_=study.rule_batch(obj['seed'],2)
        print('Raw prefix',prefix[0].tolist())
        for file in sorted(demo.glob('*.wide')):
            saved=torch.load(file,weights_only=True)
            bank=build_bank(demo,obj['method'])
            bank.skill.load_state_dict(saved['state'])
            with torch.inference_mode():
                print(saved['label'],'predicts byte',int(bank(prefix[:1],use_skill=True)[0,-1].argmax()))
    else:
        run(Path(sys.argv[1]),Path(sys.argv[2]))
