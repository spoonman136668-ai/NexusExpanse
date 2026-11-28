"""Frozen deterministic authored DEV corpus; no network, tokenizer or external data."""
from pathlib import Path
import hashlib,json
import subprocess
ROOT=Path(__file__).resolve().parents[1]
CODE_SOURCE = '447b8f0fb945014883334ca047cb32d25b25acea'
CODE={'train':['nexus_expanse/memory.py','nexus_expanse/arbitration.py','nexus_expanse/substrate.py'], 'eval':['nexus_expanse/hopfield_vsa.py','nexus_expanse/kuramoto.py','nexus_expanse/patching.py']}
# Topic labels are metadata, never passed to the model.
SCIENCE={
 'train':{
 'mechanics': [('A net force acts on a mass.','Acceleration equals net force divided by mass.'),('A ball falls in a vacuum.','Gravity increases its downward velocity.'),('Two carts collide without external force.','Total momentum is conserved.'),('A spring stretches under a load.','Its restoring force opposes displacement.'),('A lever pivots about a fulcrum.','Torque depends on force and moment arm.'),('A satellite stays in circular orbit.','Gravity provides centripetal acceleration.')],
 'cells':[('A cell membrane encloses cytoplasm.','Selective permeability regulates transport.'),('A ribosome reads messenger RNA.','It assembles an amino acid chain.'),('DNA is copied before cell division.','Each daughter can inherit genetic information.'),('An enzyme binds a substrate.','It lowers the activation energy of reaction.'),('Mitochondria oxidize nutrients.','They support ATP production.'),('Water crosses a semipermeable membrane.','Osmosis follows water potential gradients.')],
 'solutions':[('Salt dissolves in liquid water.','Solvated ions disperse through the liquid.'),('An aqueous acid donates protons.','Hydronium concentration can increase.'),('A dilute solution is concentrated.','Solute per unit volume increases.'),('A buffer receives a small acid dose.','Its conjugate base consumes some protons.'),('Two soluble salts yield insoluble ions.','A precipitate may form.'),('A saturated solution cools.','Reduced solubility may cause crystallization.')],
 'algorithms':[('Binary search uses a sorted array.','Each comparison halves the remaining interval.'),('A stack receives several entries.','Removal returns the most recent entry.'),('A queue receives several entries.','Removal returns the earliest waiting entry.'),('A hash table has colliding keys.','Collision handling distinguishes their entries.'),('A recursive procedure reaches its base case.','The recursion can terminate.'),('A stable sort compares equal keys.','Their original relative order is preserved.')]},
 'eval':{
 'waves':[('A sound wave enters a denser medium.','Its propagation speed depends on material properties.'),('Two coherent waves meet in phase.','Constructive interference increases amplitude.'),('Light enters glass from air obliquely.','Refraction changes its direction.'),('A wave passes through a narrow opening.','Diffraction spreads the wave beyond the opening.')],
 'ecology':[('A predator population loses its prey.','Food limitation can reduce predator survival.'),('A habitat becomes geographically isolated.','Gene flow between populations can decrease.'),('A producer fixes carbon using sunlight.','Chemical energy enters the food web.'),('Two species compete for scarce nutrients.','Resource limitation can reduce their growth.')],
 'redox':[('A metal atom loses an electron.','Oxidation raises its oxidation state.'),('An oxidant accepts electrons.','Reduction lowers its oxidation state.'),('A galvanic cell closes its circuit.','A spontaneous redox reaction drives current.'),('An electrolysis cell receives electrical work.','A nonspontaneous reaction can be driven.')],
 'networks':[('A packet exceeds a link transmission unit.','Fragmentation or packet rejection may follow.'),('A network route becomes unavailable.','Routing may select an alternative path.'),('A receiver verifies a packet checksum.','Transmission corruption may be detected.'),('A sender receives congestion feedback.','Reducing its rate can relieve network load.')]}}


def code_sources():
    # Read the preregistered source rather than mutable runtime code.
    return {path: subprocess.check_output(['git', 'show', f'{CODE_SOURCE}:{path}'], cwd=ROOT)
            for paths in CODE.values() for path in paths}


def corpus(code=None):
    if code is None:
        code = code_sources()
    splits={}
    for split,n,offset in [('train',24,3),('eval',16,103)]:
        rows=[]
        def add(domain,ctx,fut,label=-1,source=''):
            rows.append(dict(domain=domain,ctx=ctx.hex(),fut=fut.hex(),label=label,source=source))
        for i in range(n):
            a=offset+i; b=2+(i*7)%19; c=3+(i*3)%13
            if i%3==0: expr=f'({a}+{b})*{c}'; y=(a+b)*c
            elif i%3==1: expr=f'({a}*{b}+{c})%17'; y=(a*b+c)%17
            else: expr=f'({a}*{b}-{c})+2'; y=a*b-c+2
            add('math',f'Calculate {expr}. '.encode(),f'Result: {y}; compare with {a}: {y>a}.'.encode(),y%2,'arithmetic')
        for path in CODE[split]:
            raw=code[path]; count=n//3+(path==CODE[split][0])*(n%3)
            # Non-overlapping 96-byte windows, chosen without outcomes.
            for i in range(count):
                pos=(i+1)*(len(raw)-96)//(count+1)
                add('code',raw[pos:pos+48],raw[pos+48:pos+96],source=path)
        for topic,pairs in SCIENCE[split].items():
            for x,y in pairs: add('science',x.encode(),y.encode(),source=topic)
        splits[split]=rows
    assert not set(CODE['train'])&set(CODE['eval'])
    assert not set(SCIENCE['train'])&set(SCIENCE['eval'])
    for key in ('ctx','fut'):
        assert not {r[key] for r in splits['train']}&{r[key] for r in splits['eval']}
        for s in splits.values(): assert len({r[key] for r in s})==len(s)
    return splits


def manifest():
    code = code_sources()
    s=corpus(code)
    return dict(schema='nexus-gate9-corpus-v1',splits=s,sha256=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':')).encode()).hexdigest(),code_sha256={p:hashlib.sha256(raw).hexdigest() for p,raw in code.items()})

if __name__=='__main__': print(json.dumps(manifest(),indent=2))
