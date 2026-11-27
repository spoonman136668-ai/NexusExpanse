"""Single-stream inference cache for the fixed-geometry gated backbone."""
from collections import deque

import torch
import torch.nn.functional as F


class CausalKVCache:
    """One owner per stream. Freeze weights/calibration before constructing it.

    Fully open gates (tau=-inf) give always-on evaluation in the same numerical
    geometry. Learned absolute positions require a rebuild when the window rolls.
    """
    def __init__(self, model, *, batched=False):
        weight = model.emb.weight
        if model.training or weight.dtype != torch.float32:
            raise ValueError('cache requires a frozen FP32 eval model')
        self.device = weight.device
        self.batched = batched
        if not torch.backends.mha.get_fastpath_enabled() or any(
            not l.ffn.enable_skip or l.attn.num_heads % 2 for l in model.layers
        ):
            raise ValueError('cache requires fixed evaluation geometry and native self-attention')
        self.model = model
        self.limit = model.max_seq_len
        self.history = deque(maxlen=self.limit)
        self.mask = torch.ones(self.limit,self.limit,dtype=torch.bool,device=self.device).triu(1)
        self.kv = []
        for layer in model.layers:
            heads = layer.attn.num_heads
            shape = (heads,self.limit,model.dim//heads)
            self.kv.append((weight.new_zeros(shape), weight.new_zeros(shape)))
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
            if self.batched:
                return self.rebuild([*retained, byte])
            self.reset()
            for previous in retained:
                self._step(previous)
        return self._step(byte)

    def _project(self, packed, attn):
        if self.device.type == "cpu":
            return torch.ops.aten._transform_bias_rescale_qkv(packed, attn.in_proj_bias, attn.num_heads)
        heads = attn.num_heads
        dim = attn.embed_dim // heads
        q, k, v = (packed + attn.in_proj_bias).unflatten(-1, (3, heads, dim)).permute(2, 0, 3, 1, 4)
        return q * dim ** -0.5, k, v

    @torch.inference_mode()
    def rebuild(self, events):
        """Rebase learned absolute positions and both KV and FFN cursors together."""
        length = len(events)
        if not 1 <= length <= self.limit:
            raise ValueError("expected a nonempty cache window")
        tokens = torch.tensor([list(events)], device=self.device)
        m = self.model
        x, mask = m._backbone_inputs(tokens)
        self.reset()
        self.history.extend(events)
        for i, layer in enumerate(m.layers):
            weight = layer.attn.in_proj_weight
            packed = F.linear(layer.attn_norm(x), weight)
            _, k, v = self._project(packed, layer.attn)
            keys, values = self.kv[i]
            keys[:, :length].copy_(k[0, :, :length])
            values[:, :length].copy_(v[0, :, :length])
            x = layer.attention_residual(x, mask)
            delta, _, deep = layer.ffn(layer.ffn_norm(x), valid_length=length)
            self.active[i] = int(deep.sum().item())
            x = x + layer.ffn_drop(delta)
        return F.linear(m.out_norm(x), m.readout_weight)[0, length - 1].clone()

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
            q,k,v = self._project(packed, attn)
            keys,values = self.kv[i]
            keys[:,pos] = k[0,:,pos]
            values[:,pos] = v[0,:,pos]
            scores = torch.bmm(q[0],keys.transpose(1,2))[None]
            mask = self.mask.expand_as(scores)
            if self.device.type == "cpu":
                weights = torch.ops.aten._masked_softmax(scores,mask,-1,2)[0]
            else:
                weights = scores.masked_fill(mask, -torch.inf).softmax(-1)[0]
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
