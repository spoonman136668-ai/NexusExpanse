"""Follow up a static FFN winner across a genuine multi-byte vocabulary boundary."""
import collections
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
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks import checkpoint_transfer_study as study
from nexus_expanse.donor_assimilation import DenseTokenDonorLM, learn_byte_bpe, canonical_byte_bpe, forbid_donor_forward
from nexus_expanse.checkpoint_transport import load_skill, SkillBank, CONFIG
from nexus_expanse.core import NexusExpanseByteCore

CLOCK=time.perf_counter
BRIDGES=('sum_head','suffix_lift','position_lift')


def data():
    # Separate file identities from the initial method-screen/text-confirmation suite.
    paths=dict(study.TEXT_FILES,fit=['docs/GATE17_PREREG.md','docs/GATE20_PREREG.md'],
               val=['docs/GATE22_PREREG.md'],test=['docs/GATE23_PREREG.md'])
    original=study.TEXT_FILES
    try:
        study.TEXT_FILES=paths
        return study.corpus()
    finally:
        study.TEXT_FILES=original


def blocks(xy):
    x,y=xy
    return [bytes(v) for v in torch.cat((x,y[:,-1:]),1).tolist()]


@torch.inference_mode()
def producer_reference(model,vocab,merges,xy):
    indices={pair:i+256 for i,pair in enumerate(merges)}
    first=torch.tensor([v[0] for v in vocab])
    x,_=xy
    queries=[canonical_byte_bpe(bytes(row[:t]),indices) for row in x.tolist() for t in range(1,x.size(1)+1)]
    outputs=[]
    for start in range(0,len(queries),64):
        group=queries[start:start+64]
        lengths=torch.tensor([len(q) for q in group])
        padded=torch.zeros(len(group),int(lengths.max()),dtype=torch.long)
        for i,q in enumerate(group):
            padded[i,:len(q)]=torch.tensor(q)
        logits=model(padded)[torch.arange(len(group)),lengths-1].double()
        probs=logits.softmax(-1)
        compact=torch.zeros(len(group),256,dtype=torch.float64)
        compact.scatter_add_(1,first.expand_as(probs),probs)
        outputs.append(compact.log())
    return torch.cat(outputs).view(len(x),x.size(1),256),sum(map(len,queries))/sum(range(1,x.size(1)+1))/len(x)


def producer(tmp,seed,splits):
    t=CLOCK()
    vocab,encoded,merges=learn_byte_bpe(blocks(splits['fit']),320)
    assert len(vocab)==320
    torch.manual_seed(seed)
    model=DenseTokenDonorLM(320,32,2,4,2,64)
    opt=study.optim(model)
    for step in range(96):
        rng=random.Random(seed*1000+step)
        group=rng.sample(encoded,min(8,len(encoded)))
        max_len=max(map(len,group))-1
        x=torch.zeros(len(group),max_len,dtype=torch.long)
        y=torch.full_like(x,-100)
        for i,row in enumerate(group):
            x[i,:len(row)-1]=torch.tensor(row[:-1])
            y[i,:len(row)-1]=torch.tensor(row[1:])
        logits=model(x)
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(logits.flatten(0,1),y.flatten(),ignore_index=-100).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    torch.save(dict(weights=model.state_dict(),byte_to_row=torch.arange(256),
                    spellings=[v.hex() for v in vocab]),tmp/'donor.pt')
    fit_seconds=CLOCK()-t
    model.eval()
    refs={}
    ratio={}
    for split in ('val','test'):
        refs[split],ratio[split]=producer_reference(model,vocab,merges,splits[split])
    torch.save(refs,tmp/'reference.pt')
    study.atomic(tmp/'producer.json',dict(fit_export_seconds=fit_seconds,total_seconds=CLOCK()-t,
        fit_steps=96,producer_forward_calls=model.forward_calls,vocab_size=len(vocab),
        fit_tokens_per_byte=sum(map(len,encoded))/sum(map(len,blocks(splits['fit']))),
        probe_tokens_per_byte=ratio,donor_sha=study.digest((tmp/'donor.pt').read_bytes()),
        test=study.score_logits(refs['test'],splits['test'][1])))


class SuffixEmbedding(nn.Module):
    def __init__(self,emb,keys,delta):
        super().__init__()
        self.weight=emb.weight
        self.register_buffer('keys',keys)
        self.register_buffer('delta',delta)

    def forward(self,x):
        value=F.embedding(x,self.weight)
        code=x[:,:-1]*256+x[:,1:]
        index=torch.searchsorted(self.keys,code).clamp_max(len(self.keys)-1)
        correction=self.delta[index]*(self.keys[index]==code).unsqueeze(-1)
        return value+F.pad(correction,(0,0,1,0))


class BoundarySkill(nn.Module):
    def __init__(self,tmp,method,bridge):
        super().__init__()
        with forbid_donor_forward(),torch.no_grad():
            self.skill=load_skill(tmp/'donor.pt',method)
            info=torch.load(tmp/'donor.pt',weights_only=True)
            vocab=[bytes.fromhex(s) for s in info['spellings']]
            if bridge.startswith('shuffled_'):
                rest=vocab[256:]
                random.Random(9341).shuffle(rest)
                vocab=vocab[:256]+rest
                bridge=bridge.removeprefix('shuffled_')
            state=info['weights']
            self.register_buffer('first',torch.tensor([v[0] for v in vocab]))
            self.skill.readout=nn.Linear(CONFIG['dim'],len(vocab),bias=False)
            self.skill.readout.weight.copy_(state['readout.weight'])
            if bridge in ('position_lift','suffix_lift'):
                residual=collections.defaultdict(list)
                for i,v in enumerate(vocab[256:],256):
                    key=256*v[-2]+v[-1]
                    residual[key].append(state['emb.weight'][i]-state['emb.weight'][v[-1]])
                keys=sorted(residual)
                delta=.75*torch.stack([torch.stack(residual[k]).mean(0) for k in keys])
                self.skill.core.emb=SuffixEmbedding(self.skill.core.emb,torch.tensor(keys),delta)
            if bridge=='position_lift':
                pos=self.skill.core.pos_emb.weight
                coord=torch.arange(len(pos))/statistics.mean(map(len,vocab))
                a=coord.long()
                pos.copy_(torch.lerp(pos[a],pos[(a+1).clamp_max(len(pos)-1)],(coord-a)[:,None]))

    def forward(self,x):
        p=self.skill(x).double().softmax(-1)
        out=p.new_zeros(*p.shape[:-1],256)
        return out.scatter_add(-1,self.first.expand_as(p),p).clamp_min(1e-300).log()


@torch.inference_mode()
def replay(tmp,method,splits):
    from nexus_expanse.checkpoint_transport import ImportedSkill,CorrectedLinear
    count=0
    for file in sorted(tmp.glob('*.cold')):
        saved=torch.load(file,weights_only=True)
        state=saved['state']
        skill=ImportedSkill()
        if saved['bridge']!='scratch':
            for layer in skill.core.layers:
                for name in ('fc1','fc2'):
                    base=getattr(layer.ffn,name)
                    if method!='monarch':
                        setattr(layer.ffn,name,CorrectedLinear(base,torch.zeros(base.out_dim,base.in_dim),method))
            if 'skill.core.emb.keys' in state:
                skill.core.emb=SuffixEmbedding(skill.core.emb,state['skill.core.emb.keys'],state['skill.core.emb.delta'])
            m=BoundarySkill.__new__(BoundarySkill)
            nn.Module.__init__(m)
            m.skill=skill
            m.skill.readout=nn.Linear(CONFIG['dim'],len(state['first']),bias=False)
            m.register_buffer('first',state['first'])
        else:
            m=skill
        m.load_state_dict(state)
        logits=m.eval()(splits['test'][0])
        assert study.digest(logits.numpy().tobytes())==saved['sha']
        count+=1
    print(count)


def run_seed(tmp,seed,method,bridges,screen,splits):
    subprocess.run([sys.executable,__file__,'producer',str(tmp),str(seed)],check=True)
    receipt=json.loads((tmp/'producer.json').read_text())
    refs=torch.load(tmp/'reference.pt',weights_only=True)
    torch.manual_seed(seed+301)
    old=NexusExpanseByteCore(**CONFIG)
    opt=study.optim(old)
    for step in range(32):
        x,y=study.batch('text',splits['python'],seed,step,None)
        study.update(old,opt,old.forward_neural(x)[0],y)
    torch.save(old.eval().state_dict(),tmp/'old.pt')
    limit=study.score_logits(refs['val'],splits['val'][1])['bpb']+.02
    evidence={}
    with forbid_donor_forward() as guard:
        order=list(bridges)
        random.Random(seed).shuffle(order)
        for bridge in order:
            torch.manual_seed(seed)
            t=CLOCK()
            model=study.scratch(tmp) if bridge=='scratch' else BoundarySkill(tmp,method,bridge)
            prep=CLOCK()-t
            pre,_=study.evaluate(model,splits['test'],refs['test'])
            start=CLOCK()
            val,_=study.evaluate(model,splits['val'])
            steps=0
            if not screen and val['bpb']>limit:
                opt=study.optim(model)
                for steps in range(1,161):
                    model.train()
                    x,y=study.batch('text',splits['fit'],seed,steps,None)
                    study.update(model,opt,model(x),y)
                    if steps%8==0:
                        val,_=study.evaluate(model,splits['val'])
                        if val['bpb']<=limit:
                            break
            total=prep+CLOCK()-start
            selected,logits=study.evaluate(model,splits['test'],refs['test'])
            with torch.inference_mode():
                probes=splits['retention'][0]
                assert torch.equal(SkillBank(old,model).eval()(probes),old.forward_neural(probes)[0])
            evidence[bridge]=dict(zero=pre,selected=selected,steps=steps,prepare_seconds=prep,total_seconds=total,
                old_exact=True,skill_bytes=study.nbytes(model),prediction_sha=study.digest(logits.numpy().tobytes()),
                matched=val['bpb']<=limit and selected['bpb']<=receipt['test']['bpb']+.02,
                near=study.near(selected,receipt['test'],'text'))
            if not screen:
                torch.save(dict(state=model.state_dict(),bridge=bridge,sha=evidence[bridge]['prediction_sha']),
                           tmp/(bridge+'.cold'))
    restored=0
    if not screen:
        restored=int(subprocess.check_output([sys.executable,__file__,'cold',str(tmp),method],text=True))
        assert restored==len(evidence)
    return dict(seed=seed,producer=receipt,arms=evidence,cold_exact=restored,forbidden_calls=guard.donor_forward_calls)


def run(stage,dest):
    splits,manifest=data()
    screen=json.loads((study.OUT/'checkpoint-transfer-screen.json').read_text())
    method=screen['ranking'][0][-1]
    if stage=='bpe-screen':
        bridges=BRIDGES
        seeds=(601,602,603)
    else:
        ranking=json.loads((study.OUT/'checkpoint-transfer-bpe-screen.json').read_text())['ranking']
        bridges=list(dict.fromkeys([ranking[0][-1],'sum_head','shuffled_'+ranking[0][-1],'scratch']))
        seeds=range(701,706)
    results=[]
    for seed in seeds:
        with tempfile.TemporaryDirectory() as d:
            results.append(run_seed(Path(d),seed,method,bridges,stage=='bpe-screen',splits))
    obj=dict(source=study.git('rev-parse','HEAD').decode().strip(),stage=stage,method=method,manifest=manifest,runs=results)
    if stage=='bpe-screen':
        obj['ranking']=sorted((max(r['arms'][b]['zero']['kl_bits'] for r in results),
            statistics.mean(r['arms'][b]['zero']['kl_bits'] for r in results),b) for b in BRIDGES)
    else:
        obj['winner']=bridges[0]
        obj['near_cases']=sum(r['arms'][bridges[0]]['near'] for r in results)
    study.atomic(dest,obj)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='cold':
        replay(Path(sys.argv[2]),sys.argv[3],data()[0])
    elif sys.argv[1]=='producer':
        producer(Path(sys.argv[2]),int(sys.argv[3]),data()[0])
    else:
        run(sys.argv[1],Path(sys.argv[2]))
