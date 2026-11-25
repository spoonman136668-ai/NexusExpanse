"""Frozen Gate 10 experiment; immutable per-epoch checkpoints, per-arm receipts."""
import argparse, json, os, platform, resource, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from benchmarks.gate9_jepa_dev import git,digest,atomic,pack,raw,fit,predict,noncollapse,features,state_hash,stream_latent,latent,encoder
from benchmarks.gate10_jepa_corpus import manifest,DOMAINS
from nexus_expanse.jepa_domain import DomainKuramotoJEPA
SEEDS=(42,43,44); ARMS=('untrained_kuramoto','next_byte_ce_kuramoto','kuramoto_jepa','hybrid_jepa_ce_kuramoto'); EPOCHS=10; BATCH=12; STEPS=300


def identity():
    if git('status','--porcelain'): raise ValueError('dirty tree')
    m=manifest(); assert m==json.loads((ROOT/'docs/GATE10_CORPUS_MANIFEST.json').read_text())
    return dict(source_sha=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),manifest_sha256=m['sha256'],torch=str(torch.__version__),platform=platform.platform(),threads=torch.get_num_threads())

def model_for(seed,arm):
    torch.manual_seed(seed)
    return DomainKuramotoJEPA() if arm in ARMS[2:] else encoder()

def base(m): return m.context if isinstance(m,DomainKuramotoJEPA) else m

def balanced_batches(rows,seed,epoch):
    rng=np.random.default_rng(seed+epoch)
    ids=[rng.permutation([i for i,r in enumerate(rows) if r['domain']==d]) for d in DOMAINS]
    assert all(len(i)==120 for i in ids)
    for j in range(0,120,4):
        yield [rows[i] for a in ids for i in a[j:j+4]]

def ce_loss(m,xs,ys):
    t,l=pack([x+y for x,y in zip(xs,ys)]); logits=m.forward_neural(t[:,:-1])[0]
    mask=torch.arange(t.shape[1]-1)[None,:]<l[:,None]-1
    return F.cross_entropy(logits[mask],t[:,1:][mask])

def retrieval(q,k,rows):
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12); k=k/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-12)
    ranks=[]; per={}
    for d in DOMAINS:
        ids=np.array([i for i,r in enumerate(rows) if r['domain']==d]); rr=[]
        assert len(ids)==40
        for j,i in enumerate(ids):
            # Contiguous neighboring examples include same-group counterfactuals.
            start=(j//4)*4; candidates=ids[(start+np.arange(16))%40]
            # Stable candidate order independent of the true target's position.
            candidates=np.sort(candidates); scores=k[candidates]@q[i]; order=np.argsort(-scores,kind='stable')
            rr.append(int(np.where(candidates[order]==i)[0][0])+1)
        ranks+=rr; per[d]=dict(top1=float(np.mean(np.array(rr)==1)),mrr=float(np.mean(1/np.array(rr))),ranks=rr)
    return dict(top1=float(np.mean(np.array(ranks)==1)),mrr=float(np.mean(1/np.array(ranks))),per_domain=per)

def hist(rows,key):
    return np.array([np.bincount(np.frombuffer(b,dtype=np.uint8),minlength=256) for b in raw(rows,key)],dtype=float)

def probe(xt,z,train,ev,domain,n):
    a=np.array([r['domain']==domain for r in train]); e=np.array([r['domain']==domain for r in ev]); mt=np.array([r['label'] for r in train]); me=np.array([r['label'] for r in ev])
    pred=predict(z[e],fit(xt[a],np.eye(n)[mt[a]])).argmax(1)
    return dict(accuracy=float(np.mean(pred==me[e])),predictions=pred.tolist(),labels=me[e].tolist())

@torch.inference_mode()
def evaluate(m,train,ev):
    m.eval(); b=base(m); before=state_hash(m)
    start=time.perf_counter(); z=features(b,ev,'ctx'); ms=(time.perf_counter()-start)*1000/len(ev)
    xt=features(b,train,'ctx'); yt=features(b,train,'fut'); y=features(b,ev,'fut')
    # Primary Task A is the same trained linear readout for all arms.
    out=dict(math=probe(xt,z,train,ev,'math',4),science=probe(xt,z,train,ev,'science',2),noncollapse={d:noncollapse(z[np.array([r['domain']==d for r in ev])]) for d in DOMAINS},retrieval=retrieval(predict(z,fit(xt,yt)),y,ev),direct_retrieval=retrieval(z,y,ev),inference_ms_span=ms,encoder_param_bytes=sum(p.numel()*p.element_size() for p in b.parameters()),whole_model_param_bytes=sum(p.numel()*p.element_size() for p in m.parameters()))
    if isinstance(m,DomainKuramotoJEPA):
        native=m.predictor(torch.from_numpy(z)).numpy(); target=features(m.target,ev,'fut'); out['native_retrieval']=retrieval(native,target,ev)
    diff=0.
    for r in ev[::40]:
        x=bytes.fromhex(r['ctx']); zz=latent(b,*pack([x]))[0]; diff=max(diff,float((zz-stream_latent(b,x)).abs().max()))
    out['max_stream_vs_batch_diff']=diff; assert diff==0.; assert before==state_hash(m)
    return out

def save_pt(path,obj):
    tmp=path.with_suffix('.partial'); assert not tmp.exists() and not path.exists()
    torch.save(obj,tmp)
    with tmp.open('rb') as f: os.fsync(f.fileno())
    os.link(tmp,path); tmp.unlink()

def run_arm(seed,arm,directory,ident,epoch_limit):
    train,ev=[manifest()['splits'][s] for s in ('train','eval')]; m=model_for(seed,arm)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    prefix=f'{seed}-{arm}'; receipt=directory/(prefix+'.json'); parent=None; elapsed=0.; losses=[]; initial=state_hash(base(m)); n=0 if arm==ARMS[0] else EPOCHS
    if receipt.exists():
        r=json.loads(receipt.read_text()); assert r['identity']==ident and r['seed']==seed and r['arm']==arm and digest(directory/r['checkpoint'])==r['checkpoint_sha256']; return r
    done=0; new=0
    for epoch in range(1,n+1):
        path=directory/f'{prefix}-epoch-{epoch}.pt'; meta=path.with_suffix('.json')
        if path.exists() or meta.exists():
            r=json.loads(meta.read_text()); assert r['identity']==ident and r['parent']==parent and r['sha256']==digest(path) and r['epoch']==epoch
            c=torch.load(path,weights_only=True); m.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); torch.set_rng_state(c['rng']); elapsed=r['elapsed']; losses=r['losses']; assert len(losses)==epoch*30; parent=r['sha256']; done=epoch; continue
        if new>=epoch_limit: break
        start=time.perf_counter(); m.train()
        for rows in balanced_batches(train,seed,epoch):
            xs=raw(rows,'ctx'); ys=raw(rows,'fut'); opt.zero_grad(set_to_none=True)
            if arm in ARMS[2:]:
                domains=torch.tensor([DOMAINS.index(r['domain']) for r in rows]); loss=m.loss(*pack(xs),*pack(ys),domains)
                if arm==ARMS[3]: loss=loss+ce_loss(m.context,xs,ys)
            else: loss=ce_loss(m,xs,ys)
            assert bool(torch.isfinite(loss)); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.); opt.step()
            if arm in ARMS[2:]: m.update_target(len(losses))
            losses.append(float(loss.detach()))
        elapsed+=time.perf_counter()-start; save_pt(path,dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())); h=digest(path)
        atomic(meta,dict(identity=ident,parent=parent,sha256=h,epoch=epoch,elapsed=elapsed,losses=losses)); parent=h; done=epoch; new+=1
        print(f'{prefix} epoch {epoch}/{n} optimizer seconds {elapsed:.2f}',flush=True)
    if done<n: return None
    final=directory/(prefix+'-final.pt'); save_pt(final,m.state_dict())
    cold=subprocess.check_output([sys.executable,__file__,'--cold',str(final),'--arm',arm,'--seed',str(seed)],text=True).strip(); assert cold==state_hash(m)
    r=dict(identity=ident,seed=seed,arm=arm,initial_encoder_hash=initial,final_hash=cold,checkpoint=final.name,checkpoint_sha256=digest(final),cold_equal=True,training_seconds=elapsed,steps=len(losses),losses=losses,**evaluate(m,train,ev)); r['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024; atomic(receipt,r); return r

def classify(arms,arm):
    u,c,j=arms[ARMS[0]],arms[ARMS[1]],arms[arm]; nc=all(v['effective_rank']>=8 and v['mean_std']>=.5 for v in j['noncollapse'].values()); floor=all(j[k]['accuracy']>u[k]['accuracy'] for k in ('math','science')) and j['retrieval']['top1']>u['retrieval']['top1']; strong=j['retrieval']['top1']>=c['retrieval']['top1']+.05-1e-12 and j['retrieval']['mrr']>c['retrieval']['mrr']; return dict(noncollapse=nc,beats_untrained=floor,beats_ce_bar=strong)

def merge(directory,ident,dest):
    runs=[]
    for seed in SEEDS:
        arms={}
        for arm in ARMS:
            r=json.loads((directory/f'{seed}-{arm}.json').read_text()); assert r['identity']==ident and r['seed']==seed and r['arm']==arm and r['steps']==(0 if arm==ARMS[0] else STEPS) and r['cold_equal']; assert digest(directory/r['checkpoint'])==r['checkpoint_sha256']; arms[arm]=r
        assert len({r['initial_encoder_hash'] for r in arms.values()})==1; runs.append(dict(seed=seed,arms=arms,checks={a:classify(arms,a) for a in ARMS[2:]}))
    statuses={}
    for a in ARMS[2:]:
        ok=all(r['checks'][a]['noncollapse'] and r['checks'][a]['beats_untrained'] for r in runs); statuses[a]='PASS' if ok and all(r['checks'][a]['beats_ce_bar'] for r in runs) else 'MIXED' if ok else 'FAIL'
    train,ev=[manifest()['splits'][s] for s in ('train','eval')]; xt,z=hist(train,'ctx'),hist(ev,'ctx'); yt,y=hist(train,'fut'),hist(ev,'fut')
    histogram=dict(math=probe(xt,z,train,ev,'math',4),science=probe(xt,z,train,ev,'science',2),retrieval=retrieval(predict(z,fit(xt,yt)),y,ev))
    sanity={str(r['seed']):dict(math_near_chance=r['arms'][ARMS[0]]['math']['accuracy']<=.40,science_near_chance=r['arms'][ARMS[0]]['science']['accuracy']<=.65,retrieval_near_chance=r['arms'][ARMS[0]]['retrieval']['top1']<=.15) for r in runs}
    status=max(statuses.values(),key=('FAIL','MIXED','PASS').index)
    atomic(dest,dict(schema='nexus-gate10-jepa-dev-v1',qualification='AUTHORED_RELATIONAL_DEV',identity=ident,status=status,arm_statuses=statuses,histogram_control=histogram,untrained_sanity=sanity,runs=runs)); print(status)

def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=SEEDS); p.add_argument('--arm',choices=ARMS); p.add_argument('--directory',type=Path); p.add_argument('--merge',type=Path); p.add_argument('--cold',type=Path); p.add_argument('--epoch-limit',type=int,default=2,choices=range(1,11)); a=p.parse_args()
    if a.cold:
        m=model_for(a.seed,a.arm); m.load_state_dict(torch.load(a.cold,weights_only=True)); print(state_hash(m)); return
    ident=identity(); a.directory.mkdir(parents=True,exist_ok=True)
    if a.merge: merge(a.directory,ident,a.merge); return
    if a.seed is None: p.error('seed required')
    if a.arm: run_arm(a.seed,a.arm,a.directory,ident,a.epoch_limit)
    else:
        for arm in ARMS: subprocess.run([sys.executable,__file__,'--seed',str(a.seed),'--arm',arm,'--directory',str(a.directory),'--epoch-limit',str(a.epoch_limit)],check=True,timeout=180)
if __name__=='__main__': main()
