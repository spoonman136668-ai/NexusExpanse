"""Opt-in streaming patch reader for a fixed raw-byte neural backbone."""
from collections import deque
import numpy as np
import torch


class BytePatchReader:
    def __init__(self, model, counts, threshold_bits=4.0, kv_cache=None):
        self.model = model
        self.cache = kv_cache
        self.counts = counts
        self.totals = counts.sum(axis=1) + 256.
        self.minimum_probability = 2. ** -threshold_bits
        self.events = deque(maxlen=model.max_seq_len)
        self.previous = None
        self.pending = None
        self.deep_calls = 0

    def reset_context(self):
        self.events.clear()
        if self.cache is not None:
            self.cache.reset()
        self.previous = self.pending = None

    @torch.inference_mode()
    def forecast(self, observed):
        # Surprise belongs to the byte already observed, before the next target.
        wake = self.previous is None or (
            (self.counts[self.previous, observed] + 1.) / self.totals[self.previous]
            < self.minimum_probability)
        self.previous = observed
        if wake:
            if self.pending is not None:
                self.events.append(self.pending)
                if self.cache is not None:
                    self.cache.append(self.pending)
                self.pending = None
            self.events.append(observed)
            self.deep_calls += 1
            if self.cache is not None:
                return self.cache.append(observed).double().softmax(0)
            tokens = torch.tensor(list(self.events))[None]
            return self.model.forward_neural(tokens)[0][0,-1].double().softmax(0)
        # Fold a run into its latest raw byte. The local expert predicts throughout.
        self.pending = observed
        return torch.from_numpy((self.counts[observed].astype(np.float64)+1.) / self.totals[observed])


class CompactBigram:
    """Lossless sparse uint32 counts; row expansion is limited to 256 entries."""
    def __init__(self, keys, counts):
        self.keys = keys
        self.counts = counts
        self.offsets = np.searchsorted(keys.astype(np.int32), np.arange(257)*256)
        self.row_sums = np.asarray([counts[a:b].sum() for a,b in
                                    zip(self.offsets[:-1],self.offsets[1:])],dtype=np.uint64)

    @classmethod
    def from_dense(cls, counts):
        if counts.shape != (256,256) or counts.dtype != np.dtype('<u4'):
            raise ValueError('expected a 256 by 256 uint32 bigram table')
        flat = counts.ravel()
        indices = np.flatnonzero(flat)
        return cls(indices.astype('<u2'),flat[indices])

    def snapshot(self):
        return b'BG01'+self.keys.tobytes()+self.counts.tobytes()

    @classmethod
    def restore(cls, data):
        if data[:4] != b'BG01' or (len(data)-4)%6:
            raise ValueError('invalid bigram snapshot')
        n=(len(data)-4)//6
        keys=np.frombuffer(data,'<u2',n,4)
        counts=np.frombuffer(data,'<u4',n,4+2*n)
        if np.any(keys[1:]<=keys[:-1]) or np.any(counts==0):
            raise ValueError('noncanonical sparse bigram rows')
        return cls(keys,counts)

    def sum(self, axis):
        if axis != 1:
            raise ValueError('only row totals supported')
        return self.row_sums

    def __getitem__(self, index):
        if isinstance(index,tuple):
            a,b=index
            key=256*a+b
            i=self.keys.searchsorted(key)
            return self.counts[i] if i<len(self.keys) and self.keys[i]==key else 0
        start,end=self.offsets[index:index+2]
        row=np.zeros(256,dtype='<u4')
        row[self.keys[start:end]%256]=self.counts[start:end]
        return row
