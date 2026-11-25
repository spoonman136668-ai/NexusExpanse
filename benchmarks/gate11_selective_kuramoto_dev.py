"""Frozen Gate 11 DEV protocol. Run on Linux CPU, with custody outside the checkout."""
import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.gate10_jepa_corpus import manifest
from benchmarks.gate10_jepa_dev import (balanced_batches, ce_loss, hist, probe,
                                       retrieval, save_pt)
from benchmarks.gate9_jepa_dev import (atomic, digest, features, fit, git,
                                      noncollapse, pack, predict, raw)
from nexus_expanse.core import NexusExpanseByteCore
from nexus_expanse.jepa import latent, state_hash, stream_latent
from nexus_expanse.kuramoto import KuramotoMonarchByteCore
from nexus_expanse.selective_kuramoto import SelectiveKuramotoByteCore, InfoNCEKuramotoJEPA

SEEDS = (42, 43, 44)
CAPACITY = ("ungated_kuramoto", "selective_kuramoto", "causal_attention_monarch")
OBJECTIVES = ("next_byte_ce", "infonce_jepa")
DOMAINS = ("math", "code", "science")
PHASES = ("capacity", "expanded", "objective")


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def identity():
    if git("status", "--porcelain"):
        raise ValueError("dirty source tree")
    m = manifest()
    if m != json.loads((ROOT / "docs/GATE10_CORPUS_MANIFEST.json").read_text()):
        raise ValueError("Gate 10 corpus drift")
    return dict(source_sha=git("rev-parse", "HEAD"), tree=git("rev-parse", "HEAD^{tree}"),
                manifest_sha256=m["sha256"], torch=str(torch.__version__),
                numpy=np.__version__, python=sys.version, platform=platform.platform(),
                machine=platform.machine(), threads=torch.get_num_threads())


def attention(dim=64, cached=False):
    model = NexusExpanseByteCore(n_layers=2, dim=dim, n_heads=4, ffn_mult=2,
                                enable_skip=cached, dropout=0., max_seq_len=256)
    if cached:
        with torch.no_grad():
            for layer in model.layers:
                layer.ffn.tau_eval.fill_(-float("inf"))
    return model


class Supervised(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.encoder = {"ungated_kuramoto": KuramotoMonarchByteCore,
                        "selective_kuramoto": SelectiveKuramotoByteCore,
                        "causal_attention_monarch": attention}[arm]()
        # Identical initial task readouts, regardless of encoder construction.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1011)
            self.heads = nn.ModuleDict({"math": nn.Linear(64, 4), "science": nn.Linear(64, 2)})

    def forward(self, x, lengths):
        h = self.encoder.forward_neural(x)[1]
        return h[torch.arange(len(x), device=x.device), lengths-1]


def model_for(seed, phase, arm):
    torch.manual_seed(seed)
    if phase in ("capacity", "expanded"):
        return Supervised(arm)
    return InfoNCEKuramotoJEPA() if arm == "infonce_jepa" else SelectiveKuramotoByteCore()


def base(model):
    if isinstance(model, Supervised):
        return model.encoder
    return model.context if isinstance(model, InfoNCEKuramotoJEPA) else model


def init_hash(model):
    return state_hash(base(model))


def expanded_train(train, ev):
    """Exactly 10x math/science training contexts, no evaluation rows."""
    rows = [r for r in train if r["domain"] != "code"]
    for prefix in ("2", "3", "4"):
        for g in range(90):
            b = int(prefix + str(1000+g))
            c = int(prefix + str(2000+g*3))
            names = [f"{prefix}{g:02}{a}" for a in "abcdefg"]
            a0,b0,c0,d0,e0,f0,h0 = names
            for j in range(4):
                a = int(prefix + f"{g:02}" + "".join(str((k+j)%4) for k in range(4)))
                total = a+b-c
                label = total%4
                rows.append(dict(domain="math", ctx=f"(({a}+{b})-{c}) mod 4".encode().hex(),
                                 fut=f"{a}+{b}-{c}={total}; mod4={label}".encode().hex(),
                                 label=label, group=int(prefix)*100+g, source="gate11_arithmetic"))
            for s,t,u,v in (("+","+","-","-"),("-","-","+","+"),
                            ("+","-","+","-"),("-","+","-","+")):
                label = int(s==t)
                x = f"{a0}{s}>{b0};{b0}{t}>{c0};{d0}{u}>{e0};{f0}{v}>{h0}"
                y = f'{a0}{s}>{b0}{t}>{c0};net={"+" if label else "-"}'
                rows.append(dict(domain="science", ctx=x.encode().hex(), fut=y.encode().hex(),
                                 label=label, group=int(prefix)*100+g, source="gate11_signed_logic"))
    assert len(rows) == 2400
    assert len({r["ctx"] for r in rows}) == len(rows)
    for key in ("ctx", "fut"):
        assert not {r[key] for r in rows} & {r[key] for r in ev}
    return rows


def capacity_batches(rows, seed, epoch):
    rng = np.random.default_rng(seed+epoch)
    ids = [rng.permutation([i for i,r in enumerate(rows) if r["domain"] == d])
           for d in ("math", "science")]
    assert len(ids[0]) == len(ids[1]) and len(ids[0]) % 6 == 0
    for j in range(0, len(ids[0]), 6):
        yield [rows[i] for a in ids for i in a[j:j+6]]


def capacity_loss(model, rows):
    z = model(*pack(raw(rows, "ctx")))
    return sum(F.cross_entropy(model.heads[d](z[sl]),
                              torch.tensor([r["label"] for r in rows[sl]]))
               for d,sl in (("math",slice(0,6)),("science",slice(6,12)))) / 2


@torch.inference_mode()
def capacity_eval(model, rows):
    model.eval()
    out = {}
    for d in ("math", "science"):
        data = [r for r in rows if r["domain"] == d]
        pred = []
        for i in range(0, len(data), 12):
            z = model(*pack(raw(data[i:i+12], "ctx")))
            pred.extend(model.heads[d](z).argmax(1).tolist())
        labels = [r["label"] for r in data]
        out[d] = dict(accuracy=float(np.mean(np.array(pred)==labels)),
                      predictions=pred, labels=labels)
    return out


@torch.inference_mode()
def parity(seeds=50):
    maximum = dict(prefix_logits=0., prefix_hidden=0., streaming_probabilities=0.,
                   streaming_hidden=0., phase_norm_error=0.)
    state_bytes = {}
    for dim in (64,128):
        for seed in range(seeds):
            torch.manual_seed(seed)
            m = SelectiveKuramotoByteCore(dim=dim).eval()
            x = torch.randint(0, 256, (1,32))
            full, hidden, _ = m.forward_neural(x)
            for cut in (1,7,13,25,31):
                p, h, _ = m.forward_neural(x[:,:cut])
                maximum["prefix_logits"] = max(maximum["prefix_logits"], float((p-full[:,:cut]).abs().max()))
                maximum["prefix_hidden"] = max(maximum["prefix_hidden"], float((h-hidden[:,:cut]).abs().max()))
            states = m.init_stream_state()
            initial_bytes = sum(z.numel()*z.element_size() for z in states)
            for i in range(x.shape[1]):
                prob, states = m.step_streaming(int(x[0,i]), states)
                expected = full[0,i].double().softmax(0)
                maximum["streaming_probabilities"] = max(maximum["streaming_probabilities"],
                                                         float((prob-expected).abs().max()))
                assert sum(z.numel()*z.element_size() for z in states) == initial_bytes
                assert all(z.shape == (1,dim//2,2) for z in states)
                maximum["phase_norm_error"] = max(maximum["phase_norm_error"],
                    max(float((z.norm(dim=-1)-1).abs().max()) for z in states))
            h = stream_latent(m, bytes(x[0].tolist()))
            maximum["streaming_hidden"] = max(maximum["streaming_hidden"], float((h-hidden[0,-1]).abs().max()))
            state_bytes[str(dim)] = initial_bytes
    passed = all(maximum[k] == 0. for k in maximum if k != "phase_norm_error") and maximum["phase_norm_error"] < 1e-10
    return dict(seeds=seeds, dimensions=[64,128], recurrent_state_bytes=state_bytes,
                zero_kv_cache=True, passed=passed, **maximum)


@torch.inference_mode()
def speed(seed, trained):
    """Compare real streaming softmax calls against repository always-on KV cache."""
    result = {}
    stream = np.random.default_rng(seed).integers(0,256,size=256).tolist()
    for dim in (64,128):
        torch.manual_seed(seed)
        osc = trained.eval() if dim == 64 else SelectiveKuramotoByteCore(dim=dim).eval()
        att = attention(dim, cached=True).eval()
        cache = att.new_kv_cache()
        reference = att.forward_neural(torch.tensor([stream]))[0][0]
        cached = torch.stack([cache.append(b) for b in stream])
        error = float((cached-reference).abs().max())
        if not torch.allclose(cached, reference, atol=2e-5, rtol=1e-5):
            raise ValueError(f"KV control mismatch: {error}")
        def oscillator():
            states = osc.init_stream_state()
            start = time.perf_counter()
            for b in stream:
                _, states = osc.step_streaming(b, states)
            return (time.perf_counter()-start)*1000/len(stream)
        def attention_cache():
            cache.reset()
            start = time.perf_counter()
            for b in stream:
                cache.append(b).double().softmax(0)
            return (time.perf_counter()-start)*1000/len(stream)
        oscillator(); attention_cache()
        a,b = [],[]
        for i in range(5):
            if i%2:
                b.append(attention_cache()); a.append(oscillator())
            else:
                a.append(oscillator()); b.append(attention_cache())
        aa,bb = statistics.median(a),statistics.median(b)
        result[str(dim)] = dict(selective_ms_byte=aa, attention_ms_byte=bb,
                               speedup=bb/aa, selective_trials=a, attention_trials=b,
                               attention_cache_max_logit_diff=error,
                               attention_cache_tensor_bytes=cache.tensor_bytes,
                               selective_state_bytes=sum(z.numel()*z.element_size() for z in osc.init_stream_state()),
                               selective_weights="trained" if dim==64 else "initialized")
    return result


@torch.inference_mode()
def objective_eval(model, train, ev):
    model.eval()
    b = base(model)
    xt,z = features(b, train, "ctx"),features(b, ev, "ctx")
    yt,y = features(b, train, "fut"),features(b, ev, "fut")
    before = state_hash(model)
    out = dict(math=probe(xt,z,train,ev,"math",4), science=probe(xt,z,train,ev,"science",2),
               retrieval=retrieval(predict(z,fit(xt,yt)),y,ev),
               direct_retrieval=retrieval(z,y,ev),
               noncollapse={d:noncollapse(z[np.array([r["domain"]==d for r in ev])]) for d in DOMAINS},
               encoder_param_bytes=sum(p.numel()*p.element_size() for p in b.parameters()))
    if isinstance(model, InfoNCEKuramotoJEPA):
        target = features(model.target,ev,"fut")
        out["native_retrieval_diagnostic"] = retrieval(model.predictor(torch.from_numpy(z)).numpy(), target, ev)
    diff=0.
    for r in ev[::40]:
        x=bytes.fromhex(r["ctx"])
        diff=max(diff,float((latent(b,*pack([x]))[0]-stream_latent(b,x)).abs().max()))
    out["max_stream_vs_batch_diff"]=diff
    assert state_hash(model)==before
    return out


def receipt_path(directory, seed, phase, arm):
    return directory/f"{seed}-{phase}-{arm}.json"


def read_receipt(directory, ident, seed, phase, arm):
    r = json.loads(receipt_path(directory,seed,phase,arm).read_text())
    expected_steps = {"capacity":300,"expanded":3000,"objective":300}[phase]
    assert r["identity"]==ident and r["seed"]==seed and r["phase"]==phase and r["arm"]==arm
    assert r["steps"]==expected_steps and r["cold_equal"]
    assert digest(directory/r["checkpoint"])==r["checkpoint_sha256"]
    return r


def all_receipts(directory, ident, phase):
    arms = OBJECTIVES if phase=="objective" else CAPACITY
    return {str(s):{a:read_receipt(directory,ident,s,phase,a) for a in arms} for s in SEEDS}


def needs_expansion(records):
    return any(r["train"][d]["accuracy"] < .95
               for arms in records.values() for r in arms.values() for d in ("math","science"))


def run_arm(seed,phase,arm,directory,ident,epoch_limit):
    receipt=receipt_path(directory,seed,phase,arm)
    if receipt.exists():
        return read_receipt(directory,ident,seed,phase,arm)
    train,ev=[manifest()["splits"][s] for s in ("train","eval")]
    if phase=="expanded":
        train=expanded_train(train,ev)
    model=model_for(seed,phase,arm)
    initial=init_hash(model)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.003,weight_decay=.01)
    epochs=10 if phase=="objective" else 15
    steps_epoch={"capacity":20,"expanded":200,"objective":30}[phase]
    prefix=f"{seed}-{phase}-{arm}"
    parent=None; elapsed=0.; losses=[]; new=0
    for epoch in range(1,epochs+1):
        ck=directory/f"{prefix}-epoch-{epoch}.pt"
        meta=ck.with_suffix(".json")
        intent=ck.with_suffix(".intent.json")
        if any(p.exists() for p in (ck,meta,intent)):
            if not all(p.exists() for p in (ck,meta,intent)):
                raise RuntimeError(f"ambiguous consumed epoch; do not replay: {ck}")
            record=json.loads(meta.read_text())
            assert json.loads(intent.read_text())==dict(identity=ident,epoch=epoch,parent=parent)
            assert record["identity"]==ident and record["epoch"]==epoch and record["parent"]==parent
            assert record["sha256"]==digest(ck)
            c=torch.load(ck,weights_only=True)
            model.load_state_dict(c["model"]); opt.load_state_dict(c["optimizer"])
            torch.set_rng_state(c["rng"]); elapsed=record["elapsed"]; losses=record["losses"]
            assert len(losses)==epoch*steps_epoch
            parent=record["sha256"]
            continue
        if new>=epoch_limit:
            return None
        atomic(intent,dict(identity=ident,epoch=epoch,parent=parent))
        start=time.perf_counter(); model.train()
        batches=balanced_batches(train,seed,epoch) if phase=="objective" else capacity_batches(train,seed,epoch)
        for rows in batches:
            opt.zero_grad(set_to_none=True)
            if phase!="objective":
                loss=capacity_loss(model,rows)
            elif arm=="infonce_jepa":
                loss=model.loss(*pack(raw(rows,"ctx")),*pack(raw(rows,"fut")))
            else:
                loss=ce_loss(model,raw(rows,"ctx"),raw(rows,"fut"))
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step()
            if arm=="infonce_jepa":
                model.update_target(len(losses))
            losses.append(float(loss.detach()))
        assert len(losses)==epoch*steps_epoch
        elapsed+=time.perf_counter()-start
        save_pt(ck,dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
        h=digest(ck)
        atomic(meta,dict(identity=ident,epoch=epoch,parent=parent,sha256=h,elapsed=elapsed,losses=losses))
        parent=h; new+=1
        print(f"{prefix} epoch {epoch}/{epochs}, optimizer seconds={elapsed:.2f}",flush=True)
    final=directory/(prefix+"-final.pt")
    # A complete final checkpoint may survive interruption during deterministic evaluation.
    if final.exists():
        restored=model_for(seed,phase,arm)
        restored.load_state_dict(torch.load(final,weights_only=True))
        assert state_hash(restored)==state_hash(model)
    else:
        save_pt(final,model.state_dict())
    cold=subprocess.check_output([sys.executable,__file__,"--cold",str(final),
                                  "--seed",str(seed),"--phase",phase,"--arm",arm],text=True).strip()
    assert cold==state_hash(model)
    if phase=="objective":
        metrics=objective_eval(model,train,ev)
        if arm=="infonce_jepa":
            metrics["speed"]=speed(seed,base(model))
    else:
        metrics=dict(train=capacity_eval(model,train),heldout=capacity_eval(model,ev))
    assert state_hash(model)==cold
    r=dict(identity=ident,seed=seed,phase=phase,arm=arm,steps=len(losses),losses=losses,
           initial_encoder_hash=initial,training_seconds=elapsed,checkpoint=final.name,
           checkpoint_sha256=digest(final),final_hash=cold,cold_equal=True,**metrics)
    atomic(receipt,r)
    return r


def classify(cap, obj, par, histogram):
    # The Gate 10 histogram is recomputed on exactly the original manifest.
    assert math.isclose(histogram["retrieval"]["top1"],.20,abs_tol=1e-12)
    parity_ok=par["passed"] and all(r["max_stream_vs_batch_diff"]==0. for a in obj.values() for r in a.values())
    improvement=all(a["selective_kuramoto"]["heldout"][d]["accuracy"] >
                    a["ungated_kuramoto"]["heldout"][d]["accuracy"]
                    for a in cap.values() for d in ("math","science"))
    capacity_bar=all(a["selective_kuramoto"]["heldout"]["math"]["accuracy"]>=.45 and
                     a["selective_kuramoto"]["heldout"]["science"]["accuracy"]>=.70
                     for a in cap.values())
    histogram_bar=all(a["infonce_jepa"]["retrieval"]["top1"]>.20 for a in obj.values())
    ce_bar=all(a["infonce_jepa"]["retrieval"]["top1"]>a["next_byte_ce"]["retrieval"]["top1"] for a in obj.values())
    speed_bar=all(a["infonce_jepa"]["speed"][str(d)]["speedup"]>=2. for a in obj.values() for d in (64,128))
    checks=dict(parity=parity_ok,beats_ungated=improvement,capacity_thresholds=capacity_bar,
                beats_histogram=histogram_bar,beats_ce=ce_bar,speed=speed_bar)
    if not (parity_ok and improvement and histogram_bar):
        status="FAIL"
    else:
        status="PASS" if all(checks.values()) else "MIXED"
    return status,checks


def read_parity(directory,ident):
    par=json.loads((directory/"parity.json").read_text())
    assert par["identity"]==ident and par["seeds"]==50 and par["passed"]
    return par


def main():
    setup()
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS)
    p.add_argument("--phase",choices=PHASES,default="capacity")
    p.add_argument("--arm",choices=CAPACITY+OBJECTIVES)
    p.add_argument("--directory",type=Path)
    p.add_argument("--epoch-limit",type=int,choices=range(1,16),default=15)
    p.add_argument("--cold",type=Path)
    p.add_argument("--parity",action="store_true")
    p.add_argument("--merge",type=Path)
    a=p.parse_args()
    if a.cold:
        m=model_for(a.seed,a.phase,a.arm)
        m.load_state_dict(torch.load(a.cold,weights_only=True))
        print(state_hash(m)); return
    if not a.directory:
        p.error("--directory required")
    directory=a.directory.resolve()
    if directory==ROOT or ROOT in directory.parents:
        p.error("checkpoints must be outside source tree")
    ident=identity(); directory.mkdir(parents=True,exist_ok=True)
    if a.parity:
        dest=directory/"parity.json"
        if dest.exists():
            print(json.dumps(read_parity(directory,ident))); return
        result=dict(identity=ident,**parity())
        atomic(dest,result); print(json.dumps(result))
        if not result["passed"]:
            raise SystemExit(1)
        return
    par=read_parity(directory,ident)
    if a.merge or a.phase!="capacity":
        initial=all_receipts(directory,ident,"capacity")
        expansion=needs_expansion(initial)
        if a.phase=="expanded" and not a.merge and not expansion:
            p.error("train-only expansion trigger not met")
        if a.merge or a.phase=="objective":
            selected=all_receipts(directory,ident,"expanded") if expansion else initial
    if a.merge:
        obj=all_receipts(directory,ident,"objective")
        for arms in obj.values():
            assert len({r["initial_encoder_hash"] for r in arms.values()})==1
        train,ev=[manifest()["splits"][s] for s in ("train","eval")]
        xt,z=hist(train,"ctx"),hist(ev,"ctx"); yt,y=hist(train,"fut"),hist(ev,"fut")
        histogram=dict(math=probe(xt,z,train,ev,"math",4),science=probe(xt,z,train,ev,"science",2),
                       retrieval=retrieval(predict(z,fit(xt,yt)),y,ev))
        status,checks=classify(selected,obj,par,histogram)
        atomic(a.merge,dict(schema="nexus-gate11-selective-kuramoto-dev-v1",
                            qualification="REUSED_AUTHORED_RELATIONAL_DEV",identity=ident,status=status,
                            checks=checks,parity=par,capacity_original=initial,
                            expansion_triggered=expansion,capacity_decisive=selected,
                            objectives=obj,histogram_control=histogram))
        print(status,json.dumps(checks)); return
    if a.seed is None:
        p.error("--seed required")
    arms=OBJECTIVES if a.phase=="objective" else CAPACITY
    if a.arm:
        if a.arm not in arms:
            p.error("invalid arm for phase")
        run_arm(a.seed,a.phase,a.arm,directory,ident,a.epoch_limit)
    else:
        for arm in arms:
            subprocess.run([sys.executable,__file__,"--seed",str(a.seed),"--phase",a.phase,
                            "--arm",arm,"--directory",str(directory),"--epoch-limit",str(a.epoch_limit)],
                           check=True)


if __name__=="__main__":
    main()
