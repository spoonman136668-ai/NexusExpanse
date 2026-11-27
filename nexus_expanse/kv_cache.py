"""Single-stream CPU inference cache for the fixed-geometry gated backbone."""
from collections import deque

import torch
import torch.nn.functional as F


class CausalKVCache:
    """One owner per stream. Freeze weights/calibration before constructing it.

    Fully open gates (tau=-inf) give always-on evaluation in the same numerical
    geometry. Learned absolute positions require a rebuild when the window rolls.
    """
    def __init__(self, model):
        if model.training or model.emb.weight.device.type != 'cpu' or model.emb.weight.dtype != torch.float32:
            raise ValueError('cache requires a frozen CPU FP32 eval model')
        if not torch.backends.mha.get_fastpath_enabled() or any(
            not l.ffn.enable_skip or l.attn.num_heads % 2 for l in model.layers
        ):
            raise ValueError('cache requires fixed evaluation geometry and native self-attention')
        self.model = model
        self.limit = model.max_seq_len
        self.history = deque(maxlen=self.limit)
        self.mask = torch.ones(self.limit,self.limit,dtype=torch.bool).triu(1)
        self.kv = []
        for layer in model.layers:
            heads = layer.attn.num_heads
            shape = (heads,self.limit,model.dim//heads)
            self.kv.append((torch.zeros(shape), torch.zeros(shape)))
        self.reset()

    def reset(self):
        self.history.clear()
        self.active = [0] * len(self.kv)
        for keys,values in self.kv:
            keys.zero_()
            values.zero_()

    @property
    def tensor_bytes(self):
        return self.mask.numel() + sum(t.numel()*t.element_size() for pair in self.kv for t in pair)

    @torch.inference_mode()
    def append(self, byte):
        if not isinstance(byte,int) or not 0 <= byte <= 255:
            raise ValueError('expected raw byte')
        if len(self.history) == self.limit:
            retained = list(self.history)[1:]
            self.reset()
            for previous in retained:
                self._step(previous)
        return self._step(byte)

    def _step(self, byte):
        pos = len(self.history)
        self.history.append(byte)
        m = self.model
        x = m.emb.weight[byte] + m.pos_emb.weight[pos]
        # Preserve the reference BLAS geometry at discrete gates. Scratch rows
        # are zeros; historical projected keys and values come only from cache.
        scratch = x.new_zeros(self.limit,m.dim)
        for i,layer in enumerate(m.layers):
            attn = layer.attn
            scratch.zero_()
            scratch[pos] = layer.attn_norm(x)
            packed = F.linear(scratch,attn.in_proj_weight)[None]
            q,k,v = torch.ops.aten._transform_bias_rescale_qkv(
                packed,attn.in_proj_bias,attn.num_heads)
            keys,values = self.kv[i]
            keys[:,pos] = k[0,:,pos]
            values[:,pos] = v[0,:,pos]
            scores = torch.bmm(q[0],keys.transpose(1,2))[None]
            mask = self.mask.expand_as(scores)
            weights = torch.ops.aten._masked_softmax(scores,mask,-1,2)[0]
            context = torch.bmm(weights,values)[:,pos].reshape(-1)
            scratch.zero_()
            scratch[pos] = context
            x = x + attn.out_proj(scratch)[pos]
            scratch.zero_()
            scratch[pos] = layer.ffn_norm(x)
            if layer.ffn.gate_scores(scratch)[pos] >= layer.ffn.tau_eval:
                slot = self.active[i] % 32
                tile = x.new_zeros(32,m.dim)
                tile[slot] = scratch[pos]
                x = x + layer.ffn.fc2(F.silu(layer.ffn.fc1(tile)))[slot]
                self.active[i] += 1
        scratch.zero_()
        scratch[pos] = m.out_norm(x)
        return F.linear(scratch,m.readout_weight)[pos].clone()
