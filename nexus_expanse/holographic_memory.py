"""Fixed-size raw-byte bipolar associative memory for the Gate 6 DEV fixture."""
import math
import struct

import numpy as np
import torch


class HolographicRelationalMemory:
    dimension = 4096
    header = b'HDC1'

    def __init__(self, model):
        self.model = model
        rng = np.random.default_rng(601)
        self.symbols = torch.from_numpy((2*rng.integers(0,2,(256,self.dimension),
                                                        dtype=np.int8)-1).astype('float32'))
        # Distinct immutable role permutations prevent commutative role swaps.
        self.anchors = self.symbols.roll(137,1)
        self.cues = self.symbols.roll(811,1)
        self.bundle = torch.zeros(self.dimension)
        self.frequency = torch.zeros(256,dtype=torch.float64)
        self.outcomes = torch.zeros(256,dtype=torch.float64)
        self.records = 0

    @torch.inference_mode()
    def encode(self, prefix):
        logits = self.model.forward_neural(torch.tensor(list(prefix))[None])[0]
        return tuple(prefix), logits[0,-1].double().softmax(0)

    def query(self, anchor, cue):
        key = self.anchors[anchor]*self.cues[cue]
        scores = self.symbols @ (self.bundle*key) / self.dimension
        noise = self.bundle.norm().item()/self.dimension
        # Crosstalk estimated directly from the superposed vector.
        logits = scores.double()/max(.25, noise) + (self.outcomes+.1).log()
        return key, logits.softmax(0)

    def roles(self, prefix):
        # Outcomes observed earlier supply anchor evidence anywhere in the record.
        available = [(i,b) for i,b in enumerate(prefix) if self.outcomes[b]>0]
        if not available:
            return None, []
        i,anchor = max(available,key=lambda item:float(self.outcomes[item[1]]))
        candidates = []
        for j,b in enumerate(prefix[i+1:],i+1):
            weight = 1/(1+math.exp(16*(float(self.frequency[b])-.55)))
            candidates.append((j,b,weight))
        selected = sorted(candidates,key=lambda v:-v[2])[:2]
        return anchor, sorted(selected)

    def predict(self, prefix, prior):
        anchor, roles = self.roles(prefix)
        if not roles:
            return prior
        _,p1 = self.query(anchor,roles[0][1])
        result = p1
        if len(roles)==2:
            _,p2 = self.query(int(p1.argmax()),roles[1][1])
            w = roles[1][2]
            result = (1-w)*p1+w*p2
        w = .99*roles[0][2]
        return (1-w)*prior+w*result

    def observe(self, prefix, target):
        anchor, roles = self.roles(prefix)
        if roles:
            key,p1 = self.query(anchor,roles[0][1])
            if len(roles)==2:
                w = roles[1][2]
                next_key = self.anchors[int(p1.argmax())]*self.cues[roles[1][1]]
                key = (1-w)*key+w*next_key
            self.bundle.add_(roles[0][2]*key*self.symbols[target])
        self.frequency.mul_(.90)
        self.frequency[list(set(prefix))] += .10
        self.outcomes[target] += 1
        self.records += 1

    def snapshot(self):
        payload = (self.header + struct.pack('<Q',self.records) + self.bundle.numpy().astype('<f4').tobytes()
                   + self.frequency.numpy().astype('<f8').tobytes() + self.outcomes.numpy().astype('<f8').tobytes())
        if len(payload)>40000:
            raise ValueError('adaptive byte ceiling exceeded')
        return payload

    def restore(self, payload):
        if payload[:4]!=self.header or len(payload)!=12+4*self.dimension+4096:
            raise ValueError('invalid holographic snapshot')
        self.records, = struct.unpack_from('<Q',payload,4)
        start=12
        for tensor,dtype in [(self.bundle,'<f4'),(self.frequency,'<f8'),(self.outcomes,'<f8')]:
            values=np.frombuffer(payload,dtype,count=tensor.numel(),offset=start).copy()
            tensor.copy_(torch.from_numpy(values))
            start+=values.nbytes
