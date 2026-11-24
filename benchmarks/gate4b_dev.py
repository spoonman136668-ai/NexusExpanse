"""Frozen code-to-prose streaming evaluation; each arm runs in a fresh process."""
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.ablation_dev import CONFIG, batches, shared_init, state_hash
from benchmarks.gate4_corpus import MANIFEST, read_split
from nexus_expanse import NexusExpanseByteCore, ReliabilityArbitrator
from nexus_expanse.consolidation import ConsolidatingMemory
from nexus_expanse.patching import BytePatchReader

BUDGET=40000


def prepare(directory, train):
    memory=ConsolidatingMemory(BUDGET,'sparse_support',8)
    counts=np.zeros((256,256), dtype='<u4')
    for block in train:
        for i,y in enumerate(block[1:],1):
            memory.observe_transition(block[max(0,i-8):i],y)
            counts[block[i-1],y]+=1
    (directory/'memory').write_bytes(memory.snapshot(max_bytes=BUDGET))
    np.save(directory/'counts.npy',counts)


def train_one(directory, seed, kind, train):
    ref=shared_init(seed,'monarch',False,None)
    model=shared_init(seed,kind,True,ref)
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01)
    bs=[batch for block in train for batch in batches(block)]
    start=time.perf_counter()
    for _ in range(2):
        model.train()
        for x,y in bs:
            opt.zero_grad(set_to_none=True)
            logits,_,aux,_=model.forward_neural(x,targets=y)
            (F.cross_entropy(logits.reshape(-1,256),y.flatten())+.1*aux).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        model.calibrate_gates(bs)
    torch.save(model.state_dict(),directory/'model')
    return dict(seed=seed,kind=kind,steps=2*len(bs),seconds=time.perf_counter()-start,
                state_hash=state_hash(model))


@torch.inference_mode()
def measure(directory,kind,policy,online):
    start=time.perf_counter()
    dev=read_split()['dev']
    mem=ConsolidatingMemory.restore((directory/'memory').read_bytes(),max_bytes=BUDGET)
    model=patch=mixer=None
    tensors=checkpoint=local_bytes=derived_buffer=0
    trained_hash=None
    if kind!='memory':
        model=NexusExpanseByteCore(**CONFIG,ffn_kind=kind,enable_skip=policy=='skip').eval()
        model.load_state_dict(torch.load(directory/'model',weights_only=True))
        trained_hash=state_hash(model)
        tensors=sum(v.numel()*v.element_size() for v in list(model.parameters())+list(model.buffers()))
        checkpoint=(directory/'model').stat().st_size
        mixer=ReliabilityArbitrator()
        if policy=='patch':
            patch=BytePatchReader(model,np.load(directory/'counts.npy'))
            local_bytes=(directory/'counts.npy').stat().st_size
            derived_buffer=patch.totals.nbytes
    loaded=time.perf_counter()
    total=0
    loss=0.
    per_block=[]
    calls=0
    peak_memory=mem.serialized_size()
    for block in dev:
        if patch:
            patch.reset_context()
        block_loss=0.
        for i,y in enumerate(block[1:],1):
            ctx=block[max(0,i-8):i]
            pm,confidence=mem.predict_probs(ctx)
            mp=float(pm[y])
            if model is None:
                prob=mp
            else:
                if patch:
                    neural=patch.forecast(ctx[-1])
                else:
                    raw=torch.tensor(list(block[max(0,i-32):i]))[None]
                    neural=model.forward_neural(raw)[0][0,-1].double().softmax(0)
                    calls+=1
                np_y=float(neural[y])
                w=mixer.weight(confidence)
                prob=(1-w)*np_y+w*mp
            block_loss-=math.log2(max(prob,1e-300))
            total+=1
            if online:
                if mixer:
                    mixer.observe(confidence,np_y,mp)
                mem.observe_transition(ctx,y)
                peak_memory=max(peak_memory,mem.serialized_size())
        per_block.append(block_loss/(len(block)-1))
        loss+=block_loss
    elapsed=time.perf_counter()-loaded
    # Fully specified binary scalars for the one-bin arbitrator and stream cursor.
    arbitration_bytes=len(struct.pack('<ddQ',mixer.share,mixer.weight(0.),mixer.updates)) if mixer else 0
    cursor=bytes(patch.events) if patch else b''
    cursor_bytes=len(struct.pack('<dHhh',patch.minimum_probability,len(cursor),
                               -1 if patch.previous is None else patch.previous,
                               -1 if patch.pending is None else patch.pending)+cursor) if patch else 0
    memory_bytes=len(mem.snapshot(max_bytes=BUDGET))
    raw_history_bytes=min(len(dev[-1])-1,32 if model and not patch else 8)
    return dict(kind=kind,policy=policy,mode='prequential' if online else 'frozen',transitions=total,
                bpb=loss/total,block_bpb=per_block,seconds=elapsed,ms_per_byte=1000*elapsed/total,
                load_seconds=loaded-start,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024/1e6,
                deep_calls=patch.deep_calls if patch else calls,trained_hash=trained_hash,
                parameter_buffer_bytes=tensors,neural_checkpoint_bytes=checkpoint,memory_bytes=memory_bytes,
                max_memory_bytes=peak_memory,patch_predictor_serialized_bytes=local_bytes,
                patch_derived_buffer_bytes=derived_buffer,arbitrator_bytes=arbitration_bytes,
                patch_cursor_bytes=cursor_bytes,raw_history_bytes=raw_history_bytes,
                serialized_total_bytes=checkpoint+memory_bytes+local_bytes+arbitration_bytes+cursor_bytes+raw_history_bytes)


def worker(path,kind,policy,mode):
    start=time.perf_counter()
    output=subprocess.check_output([sys.executable,__file__,'--arm',str(path),kind,policy,mode],text=True)
    result=json.loads(output)
    result["process_ms_per_byte"]=(time.perf_counter()-start)*1000/result["transitions"]
    return result


def run():
    assert not subprocess.check_output(['git','status','--porcelain'])
    data=read_split()
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp)
        prepare(directory,data['train'])
        memory=[worker(directory,'memory','memory_only',mode) for mode in ('static','online')]
        runs=[]
        for seed in (42,43,44):
            for kind in ('dense','monarch'):
                result=train_one(directory,seed,kind,data['train'])
                result['arms']=[worker(directory,kind,policy,mode) for policy in ('always','skip','patch')
                                for mode in ('static','online')]
                assert all(a['trained_hash']==result['state_hash'] for a in result['arms'])
                runs.append(result)
    return dict(qualification='REAL_CORPUS_DEV',source_sha=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),torch=str(torch.__version__),
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),memory_only=memory,runs=runs)


if __name__=='__main__':
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if sys.argv[1]=='--arm':
        print(json.dumps(measure(Path(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5]=='online')))
    else:
        with open(sys.argv[1],'x') as out:
            json.dump(run(),out,indent=2,allow_nan=False)
            out.write('\n')
