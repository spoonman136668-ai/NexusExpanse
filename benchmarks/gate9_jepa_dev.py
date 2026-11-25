"""Per-seed frozen DEV experiment; checkpoint shards are outside source tree."""
import argparse,hashlib,json,os,platform,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from nexus_expanse.jepa import KuramotoJEPA,encoder,latent,stream_latent,state_hash
from benchmarks.gate9_jepa_corpus import manifest
SEEDS=(42,43,44); ARMS=('untrained_kuramoto','next_byte_ce_kuramoto','kuramoto_jepa'); DOMAINS=('math','code','science'); EPOCHS=4; BATCH=12

def git(*args): return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,obj):
    if p.exists(): raise FileExistsError(p)
    t=p.with_suffix(p.suffix+'.partial')
    with t.open('x') as f: json.dump(obj,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.link(t,p); t.unlink()
def identity():
    if git('status','--porcelain'): raise ValueError('dirty tree')
    m=manifest(); assert m==json.loads((ROOT/'docs/GATE9_CORPUS_MANIFEST.json').read_text())
    return dict(source_sha=git('rev-parse','HEAD'),tree=git('rev-parse','HEAD^{tree}'),manifest_sha256=m['sha256'],torch=str(torch.__version__),platform=platform.platform(),threads=torch.get_num_threads())
def model_for(seed,arm):
    torch.manual_seed(seed)
    return KuramotoJEPA() if arm==ARMS[2] else encoder()
def base(m): return m.context if isinstance(m,KuramotoJEPA) else m
def pack(raw):
    l=torch.tensor([len(x) for x in raw]); x=torch.zeros((len(raw),int(l.max())),dtype=torch.long)
    for i,b in enumerate(raw): x[i,:len(b)]=torch.tensor(list(b))
    return x,l
def raw(rows,key): return [bytes.fromhex(r[key]) for r in rows]
def fit(x,y):
    x=np.c_[x,np.ones(len(x))]; return np.linalg.solve(x.T@x+1.*np.eye(x.shape[1]),x.T@y)
def predict(x,w): return np.c_[x,np.ones(len(x))]@w
def noncollapse(z):
    s=np.linalg.svd(z-z.mean(0),compute_uv=False); p=s/max(s.sum(),1e-30); p=p[p>0]
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.,mean_std=float(z.std(0,ddof=1).mean()))
def retrieval(q,k,rows):
    q=q/np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-12); k=k/np.maximum(np.linalg.norm(k,axis=1,keepdims=True),1e-12)
    ranks=[]; per={}
    for d in DOMAINS:
        ids=np.array([i for i,r in enumerate(rows) if r['domain']==d]); rr=[]
        assert len(ids)==16
        for i in ids:
            scores=k[ids]@q[i]; order=np.argsort(-scores,kind='stable'); rr.append(int(np.where(ids[order]==i)[0][0])+1)
        ranks+=rr; per[d]=dict(top1=float(np.mean(np.array(rr)==1)),mrr=float(np.mean(1/np.array(rr))),ranks=rr)
    return dict(top1=float(np.mean(np.array(ranks)==1)),mrr=float(np.mean(1/np.array(ranks))),per_domain=per)
@torch.inference_mode()
def features(m,rows,key):
    # fixed batch geometry; no padding in selected hidden state
    return np.concatenate([latent(m,*pack(raw(rows[i:i+BATCH],key))).numpy() for i in range(0,len(rows),BATCH)])
@torch.inference_mode()
def evaluate(m,train,ev):
    m.eval(); b=base(m); before=state_hash(m)
    start=time.perf_counter(); z=features(b,ev,'ctx'); ms=(time.perf_counter()-start)*1000/len(ev)
    xt=features(b,train,'ctx'); yt=features(b,train,'fut'); y=features(b,ev,'fut');
    dlabels=lambda rows:np.array([DOMAINS.index(r['domain']) for r in rows]); a=dlabels(train); e=dlabels(ev)
    domain=float(np.mean(predict(z,fit(xt,np.eye(3)[a])).argmax(1)==e))
    mt=np.array([r['label'] for r in train]); me=np.array([r['label'] for r in ev]); math=float(np.mean(predict(z[e==0],fit(xt[a==0],np.eye(2)[mt[a==0]])).argmax(1)==me[e==0]))
    out=dict(domain_accuracy=domain,math_accuracy=math,noncollapse={d:noncollapse(z[e==i]) for i,d in enumerate(DOMAINS)},retrieval=retrieval(predict(z,fit(xt,yt)),y,ev),inference_ms_span=ms,encoder_param_bytes=sum(p.numel()*p.element_size() for p in b.parameters()),whole_model_param_bytes=sum(p.numel()*p.element_size() for p in m.parameters()))
    if isinstance(m,KuramotoJEPA):
        native=m.predictor(torch.from_numpy(z)).numpy(); target=features(m.target,ev,'fut'); out['native_retrieval_diagnostic']=retrieval(native,target,ev)
    diff=0.
    for r in ev[::16]:
        x=bytes.fromhex(r['ctx']); zz=latent(b,*pack([x]))[0]; diff=max(diff,float((zz-stream_latent(b,x)).abs().max()))
    out['max_stream_vs_batch_diff']=diff; assert diff==0.; assert before==state_hash(m)
    return out

def run_arm(seed,arm,directory,ident):
    train,ev=[manifest()['splits'][s] for s in ('train','eval')]; m=model_for(seed,arm)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    prefix=f'{seed}-{arm}'; receipt=directory/(prefix+'.json'); parent=None; elapsed=0.; losses=[]; initial=state_hash(base(m)); n=0 if arm==ARMS[0] else EPOCHS
    if receipt.exists():
        r=json.loads(receipt.read_text()); assert r['identity']==ident and r['seed']==seed and r['arm']==arm; assert digest(directory/r['checkpoint'])==r['checkpoint_sha256']; return r
    for epoch in range(1,n+1):
        path=directory/f'{prefix}-epoch-{epoch}.pt'; meta=path.with_suffix('.json')
        if path.exists() or meta.exists():
            r=json.loads(meta.read_text()); assert r['identity']==ident and r['parent']==parent and r['sha256']==digest(path) and r['epoch']==epoch
            c=torch.load(path,weights_only=True); m.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); torch.set_rng_state(c['rng']); elapsed=r['elapsed']; losses=r['losses']; parent=r['sha256']; continue
        start=time.perf_counter(); m.train(); order=np.random.default_rng(seed+epoch).permutation(len(train))
        for i in range(0,len(train),BATCH):
            rows=[train[j] for j in order[i:i+BATCH]]; xs=raw(rows,'ctx'); ys=raw(rows,'fut'); opt.zero_grad(set_to_none=True)
            if arm==ARMS[2]: loss=m.loss(*pack(xs),*pack(ys))
            else:
                t,l=pack([x+y for x,y in zip(xs,ys)]); logits= m.forward_neural(t[:,:-1])[0]; mask=torch.arange(t.shape[1]-1)[None,:]<l[:,None]-1
                loss=F.cross_entropy(logits[mask],t[:,1:][mask])
            loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.); opt.step()
            if arm==ARMS[2]: m.update_target()
            losses.append(float(loss.detach()))
        elapsed+=time.perf_counter()-start; tmp=path.with_suffix('.partial'); assert not tmp.exists(); torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()),tmp); os.link(tmp,path); tmp.unlink()
        h=digest(path); atomic(meta,dict(identity=ident,parent=parent,sha256=h,epoch=epoch,elapsed=elapsed,losses=losses)); parent=h
    final=directory/(prefix+'-final.pt'); tmp=final.with_suffix('.partial'); assert not final.exists() and not tmp.exists(); torch.save(m.state_dict(),tmp); os.link(tmp,final); tmp.unlink()
    cold=subprocess.check_output([sys.executable,__file__,'--cold',str(final),'--arm',arm,'--seed',str(seed)],text=True).strip(); assert cold==state_hash(m)
    r=dict(identity=ident,seed=seed,arm=arm,initial_encoder_hash=initial,final_hash=cold,checkpoint=final.name,checkpoint_sha256=digest(final),cold_equal=True,training_seconds=elapsed,steps=len(losses),losses=losses,**evaluate(m,train,ev)); r['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024; atomic(receipt,r); return r

def classify(arms):
    u,c,j=[arms[a] for a in ARMS]; nc=all(v['effective_rank']>=8 and v['mean_std']>=.5 for v in j['noncollapse'].values()); floor=all(j[k]>u[k] for k in ('domain_accuracy','math_accuracy')) and j['retrieval']['top1']>u['retrieval']['top1']; strong=j['retrieval']['top1']>=c['retrieval']['top1']+.05-1e-12 and all(j[k]>=c[k]-.02-1e-12 for k in ('domain_accuracy','math_accuracy')); return dict(noncollapse=nc,beats_untrained=floor,beats_ce_bar=strong)
def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,choices=SEEDS); p.add_argument('--arm',choices=ARMS); p.add_argument('--directory',type=Path); p.add_argument('--merge',type=Path); p.add_argument('--cold',type=Path); a=p.parse_args()
    if a.cold:
        m=model_for(a.seed,a.arm); m.load_state_dict(torch.load(a.cold,weights_only=True)); print(state_hash(m)); return
    ident=identity(); a.directory.mkdir(parents=True,exist_ok=True)
    if a.merge:
        runs=[]
        for seed in SEEDS:
            arms={}
            for arm in ARMS:
                r=json.loads((a.directory/f'{seed}-{arm}.json').read_text()); assert r['identity']==ident and r['seed']==seed and r['arm']==arm and r['steps']==(0 if arm==ARMS[0] else EPOCHS*6) and r['cold_equal']; assert digest(a.directory/r['checkpoint'])==r['checkpoint_sha256']; arms[arm]=r
            assert len({r['initial_encoder_hash'] for r in arms.values()})==1; runs.append(dict(seed=seed,arms=arms,checks=classify(arms)))
        ok=all(r['checks']['noncollapse'] and r['checks']['beats_untrained'] for r in runs); status='PASS' if ok and all(r['checks']['beats_ce_bar'] for r in runs) else 'MIXED' if ok else 'FAIL'
        atomic(a.merge,dict(schema='nexus-gate9-jepa-dev-v1',qualification='AUTHORED_MICRO_DEV',identity=ident,status=status,runs=runs)); print(status); return
    if a.seed is None: p.error('seed required')
    if a.arm: run_arm(a.seed,a.arm,a.directory,ident)
    else:
        for arm in ARMS: subprocess.run([sys.executable,__file__,'--seed',str(a.seed),'--arm',arm,'--directory',str(a.directory)],check=True,timeout=180)
if __name__=='__main__': main()
