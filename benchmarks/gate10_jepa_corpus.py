"""Deterministic raw-byte DEV corpus with exact histogram counterfactual groups."""
import hashlib, itertools, json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CODE={'train':['nexus_expanse/memory.py','nexus_expanse/arbitration.py','nexus_expanse/substrate.py'], 'eval':['nexus_expanse/hopfield_vsa.py','nexus_expanse/kuramoto.py','nexus_expanse/patching.py']}
DOMAINS=('math','code','science')

def corpus():
    splits={}
    for split,n,prefix in [('train',120,'1'),('eval',40,'6')]:
        rows=[]
        def add(domain,x,y,label,group,source):
            rows.append(dict(domain=domain,ctx=x.hex(),fut=y.hex(),label=label,group=group,source=source))
        for g in range(n//4):
            digits=str(g//10+4)+str(g%10)+ '0123'
            # Rotating last digit ensures four residues while preserving exact histogram.
            b=int(prefix+str(1000+g)); c=int(prefix+str(2000+g*3))
            for j in range(4):
                a=int(prefix+digits[:2]+''.join(str((k+j)%4) for k in range(4)))
                total=a+b-c; label=total%4
                add('math',f'(({a}+{b})-{c}) mod 4'.encode(),f'{a}+{b}-{c}={total}; mod4={label}'.encode(),label,g,'arithmetic')
        for k,path in enumerate(CODE[split]):
            data=(ROOT/path).read_bytes(); count=n//3+(k==0)*(n%3)
            assert len(data)>=count*48,(path,len(data),count)
            for i in range(count):
                pos=i*48
                add('code',data[pos:pos+24],data[pos+24:pos+48],-1,k*100+i,path)
        for g in range(n//4):
            # Same entity set, exactly two positive and two negative edges. Only
            # first two edges form the queried chain; other edges are disconnected.
            names=[f'{prefix}{g:02}{a}' for a in 'abcdefg']
            a,b,c,d,e,f,h=names
            for j,(s,t,u,v) in enumerate([('+','+','-','-'),('-','-','+','+'),('+','-','+','-'),('-','+','-','+')]):
                label=int(s==t)
                x=f'{a}{s}>{b};{b}{t}>{c};{d}{u}>{e};{f}{v}>{h}'
                # Unique continuation gives chain provenance, not merely class label.
                y=f'{a}{s}>{b}{t}>{c};net={"+" if label else "-"}'
                add('science',x.encode(),y.encode(),label,g,'synthetic_signed_logic')
        splits[split]=rows
    for k in ('ctx','fut'):
        assert not {r[k] for r in splits['train']}&{r[k] for r in splits['eval']}
    for rows in splits.values():
        assert len({r['ctx'] for r in rows})==len(rows)
        for d in ('math','science'):
            groups={g:[r for r in rows if r['domain']==d and r['group']==g] for g in {r['group'] for r in rows if r['domain']==d}}
            for a in groups.values():
                assert len(a)==4 and all(Counter(bytes.fromhex(r['ctx']))==Counter(bytes.fromhex(a[0]['ctx'])) for r in a)
                assert sorted(r['label'] for r in a)==([0,1,2,3] if d=='math' else [0,0,1,1])
    return splits

def manifest():
    s=corpus()
    return dict(schema='nexus-gate10-corpus-v1',splits=s,sha256=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest(),code_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for v in CODE.values() for p in v})
if __name__=='__main__': print(json.dumps(manifest(),indent=2))
