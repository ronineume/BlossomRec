# BlossomAttention
from __future__ import annotations

from copy import deepcopy
from math import ceil
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

from rotary_embedding_torch import RotaryEmbedding

# einstein notation
import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

BLOCK_SIZE = 1
SYMM_WINDOW = 8

def power_bool_mask(seq_len: int,
                    block_size: int = BLOCK_SIZE,
                    symm_window: int = SYMM_WINDOW,
                    device: torch.device = torch.device('cuda'),
                    causal: bool = False,
                    padding_mask: torch.Tensor = None):  

    q_idx  = torch.arange(seq_len, device=device).unsqueeze(1)   # [M, 1]
    kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)   # [1, N]
    blk_qk = q_idx // block_size - kv_idx // block_size          # [M, N]

    mask_window = blk_qk.abs() < symm_window
    mask_power  = (blk_qk & (blk_qk - 1)) == 0
    mask_sink   = kv_idx >= seq_len - block_size
    base = mask_window | mask_power | mask_sink                   # [M, N]

    if causal:
        base = base & (q_idx >= kv_idx)                           # [M, N]
    
    if padding_mask is not None:
        
        if base.dim() == 2:
            base = base.unsqueeze(0)  # [1, seq_len, seq_len]
        
        batch_size = padding_mask.size(0)
        
        # q_valid: [batch_size, seq_len, 1], kv_valid: [batch_size, 1, seq_len]
        q_valid = padding_mask.unsqueeze(2)   # [B, M, 1]
        kv_valid = padding_mask.unsqueeze(1)  # [B, 1, N]
        
        valid_mask = q_valid & kv_valid       # [B, M, N]
        
        base = base & valid_mask              # [B, M, N]
        
    
    return base  

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def round_down_mult(n, mult):
    return n // mult * mult

def round_up_mult(n, mult):
    return ceil(n / mult) * mult

def divisible_by(num, den):
    return (num % den) == 0

def is_empty(t):
    return t.numel() == 0

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def pack_one_with_inverse(t, pattern):
    packed, ps = pack([t], pattern)
    def inverse(out):
        return unpack(out, ps, pattern)[0]

    return packed, inverse

# tensor helpers

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def straight_through(t, target):
    return t + (target - t).detach()

# attend function

def C_attend(q, k, mask=None, scale=None):

    scale = default(scale, q.shape[-1] ** -0.5)
    q_heads, k_heads = q.shape[1], k.shape[1]
    num_grouped_queries = q_heads // k_heads
    
    q = rearrange(q, 'b (h qh) i d -> b h qh i d', qh=num_grouped_queries)
    
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    
    if mask is not None:
        
        if mask.dim() == 2:
            # [seq_len, num_blocks] -> [1, 1, 1, seq_len, num_blocks]
            mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            # [batch, seq_len, num_blocks] -> [batch, 1, 1, seq_len, num_blocks]
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        sim = sim.masked_fill(mask, float('-inf'))
    
    sim = rearrange(sim, 'b h qh i j -> b (h qh) i j')
    return sim



# classes

class SparseAttention(Module):
    def __init__(
        self,
        dim,
        dim_head,
        heads,
        compress_block_size,
        stride,
        selection_block_size,
        num_selected_blocks,
        kv_heads = None,
        num_compressed_mem_kv = 1,
        causal = False,
        use_diff_topk = False,
        use_triton_kernel = False,
        query_heads_share_selected_kv = True, # if set to True, importance score is averaged across query heads to select top-n buckets of kv per kv head - but can be set to False for each query head within a group to look at different sets of kv buckets. will be more memory and compute of course
        compress_mlp: Module | None = None,
        compress_mlp_expand_factor = 1.,
        strategy_combine_mlp: Module | None = None
    ):
        super().__init__()
        self.dim = dim
        # attention heads
        # handling gqa if `kv_heads` is set

        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)

        self.heads = heads
        self.dim_head = dim_head
        self.kv_heads = kv_heads
        self.num_grouped_queries = heads // kv_heads

        # scale

        self.scale = dim_head ** -0.5

        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads

        self.causal = causal

        # rotary

        self.rotary_emb = RotaryEmbedding(dim_head)

        # qkv

        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)
        self.qkv_split = qkv_split

        # compress strategy

        self.compress_block_size = compress_block_size
        self.stride = stride
        assert self.compress_block_size >= self.stride, 'compress_block_size must be >= stride'
        assert self.stride > 0, 'stride must be greater than 0'
        assert divisible_by(selection_block_size, self.stride), f'selection_block_size {selection_block_size} must be divisible by stride {self.stride}'

        # Compression window splitting
        self.split_compress_window = nn.Sequential(
            Rearrange('b h n d -> (b h) d 1 n'),
            nn.ZeroPad2d(((compress_block_size - stride), 0, 0, 0)),
            nn.Unfold(kernel_size=(1, self.compress_block_size), stride=(1, self.stride)),
            Rearrange('(b h) (d n) w -> b h w n d', d=dim_head, h=kv_heads, n=self.compress_block_size)
        )

        assert num_compressed_mem_kv > 0
        self.num_mem_compress_kv = num_compressed_mem_kv
        self.compress_mem_kv = nn.Parameter(torch.zeros(2, kv_heads, num_compressed_mem_kv, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = self.compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)

            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )

        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)

        # selection related

        self.use_diff_topk = use_diff_topk
        self.query_heads_share_selected_kv = query_heads_share_selected_kv
        self.selection_block_size = selection_block_size

        assert num_selected_blocks >= 0

        if num_selected_blocks == 0:
            print(f'`num_selected_blocks` should be set greater than 0, unless if you are ablating it for experimental purposes')

        self.num_selected_blocks = num_selected_blocks

        self.use_triton_kernel = use_triton_kernel

        # they combine the three sparse branches through a learned combine with sigmoid activation

        if not exists(strategy_combine_mlp):
            strategy_combine_mlp = nn.Linear(dim, 2 * heads)

            # init to power first, as network tends to pick up on local patterns first before distant ones

            nn.init.zeros_(strategy_combine_mlp.weight)
            strategy_combine_mlp.bias.data.copy_(tensor([ -2., 2.] * heads))
            

        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )

        # split and merging heads

        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        # combining heads

        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)


    def forward(
        self,
        inp,
        disable_triton_kernel = False,
        power_flex_mask = None,
        return_cache = False,
        attention_mask = None,
    ):
        
        batch, seq_len, scale, heads, kv_heads, device = *inp.shape[:2], self.scale, self.heads, self.kv_heads, inp.device

        compress_divisible_seq_len = round_down_mult(seq_len, self.stride)
        num_compress_blocks = compress_divisible_seq_len // self.stride

        compress_overlap_len = self.compress_block_size - self.stride
        has_compress_overlap = compress_overlap_len > 0

        fine_divisible_seq_len = round_up_mult(seq_len, self.selection_block_size)
        num_fine_blocks = fine_divisible_seq_len // self.selection_block_size

        # queries, keys, values

        q, k, v = self.to_qkv(inp).split(self.qkv_split, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)#RoPE
 
        # 1. LTIS

        k_compress_input= k[..., :compress_divisible_seq_len, :]

        if not is_empty(k_compress_input):
            k_compress_input = self.split_compress_window(k_compress_input)
        else:
            k_compress_input = tuple(t.reshape(batch, kv_heads, 0, self.compress_block_size, self.dim_head) for t in k_compress_input)

        # add the intra block positions

        if not is_empty(k_compress_input):
            k_compress_input = einx.add('b h w n d, h n d', k_compress_input, self.k_intrablock_positions)

        cq = q
        ck = self.k_compress(k_compress_input)   # Equation (7) of the Native Sparse Attention paper
        
        # coarse attention over compressed
        
        mem_ck = repeat(self.compress_mem_kv[0], '... -> b ...', b=batch)

        num_mem_compress_kv = mem_ck.shape[-2]

        ck = cat((mem_ck, ck), dim = -2)
        
        # compressed masking

        #  masking（causal + padding）
        cmask = None
        if self.causal or attention_mask is not None:
            cq_seq = arange(seq_len, device=device)
            ck_seq = ((arange(num_compress_blocks, device=device) + 1) * self.stride) - 1
            ck_seq = F.pad(ck_seq, (num_mem_compress_kv, 0), value=-1)

            
            if self.causal:
                
                cmask = einx.greater_equal('j, i -> i j', ck_seq, cq_seq)  # [seq_len, num_mem+num_blocks]
            else:
                cmask = torch.zeros(seq_len, num_mem_compress_kv + num_compress_blocks, 
                                device=device, dtype=torch.bool)

            if attention_mask is not None and compress_divisible_seq_len > 0:
                
                
            
                compress_divisible_mask = attention_mask[:, :compress_divisible_seq_len]  # [b, compress_len]
                
                block_mask_2d = compress_divisible_mask.reshape(batch, num_compress_blocks, self.stride)
                
            
                block_mask = block_mask_2d.any(dim=-1)  # [batch, num_compress_blocks]              
                
                full_block_mask = F.pad(block_mask, (num_mem_compress_kv, 0), value=True)  
            
                cmask = cmask.unsqueeze(0) | (~full_block_mask).unsqueeze(1)

        csim = C_attend(cq, ck, mask = cmask)

        # handle cache

        if return_cache:
            cache_kv = (k, v)
        
        # Importance Scoring

        importance_scores = csim[..., num_mem_compress_kv:]

        num_selected = min(self.num_selected_blocks, num_compress_blocks)
        has_selected_kv_for_fine_attn = num_selected > 0

        # maybe average the compressed attention across each grouped queries (per key / values)

        if self.query_heads_share_selected_kv:
            importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.num_grouped_queries)

            fine_num_grouped_queries = self.num_grouped_queries
        else:
            fine_num_grouped_queries = 1

        # handle if compress block size does not equal to the fine block size
        # cannot parse their equation, so will just improvise
        # first we expand all the compressed scores to the full sequence length, then average within each fine / selection block size - pad on the right to 0s, which should be fine as sliding window convers the local anyways

        if has_selected_kv_for_fine_attn:

            if self.stride != self.selection_block_size:

                num_compress_per_fine = self.selection_block_size // self.stride

                round_down_score_len = round_down_mult(importance_scores.shape[-1], num_compress_per_fine)
                importance_scores = importance_scores[..., :round_down_score_len]

                if not is_empty(importance_scores):
                    importance_scores = reduce(importance_scores, '... (j num_compress_per_fine) -> ... j', 'mean', num_compress_per_fine = num_compress_per_fine)

                    i, j = importance_scores.shape[-2:]

                    # mask out block diagonal

                    q_seq = arange(i, device = device) // self.selection_block_size
                    k_seq = arange(j, device = device)

                    block_diagonal_mask = einx.equal('i, j -> i j', q_seq, k_seq)

                    importance_scores = importance_scores.masked_fill(block_diagonal_mask, max_neg_value(csim))

            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]


        # handle if number of total blocks is less than number to select for fine attention

        fq = q
        fk = k
        fv = v

        num_selected = min(num_selected, importance_scores.shape[-1])
        has_selected_kv_for_fine_attn = num_selected > 0

        remainder = fine_divisible_seq_len - seq_len
        pad_to_multiple = partial(pad_at_dim, pad = (0, remainder), dim = -2)

        if has_selected_kv_for_fine_attn:

            # get the top-n kv segments for fine attention

            selected_importance_values, selected_block_indices = importance_scores.topk(num_selected, dim = -1)

            gates = straight_through(selected_importance_values, 1.) if self.use_diff_topk else None

            if self.use_triton_kernel and not disable_triton_kernel:

                from .tritonAttn import triton_sparse_attend

                fmask = selected_importance_values > 1e-10

                fine_attn_out = triton_sparse_attend(
                    fq, fk, fv,
                    self.selection_block_size,
                    selected_block_indices,
                    fmask,
                    sel_scale = gates,
                    include_block_causal = self.causal
                )

            else:
                fmask = selected_importance_values > 1e-10

                if seq_len < fine_divisible_seq_len:
                    fk, fv, fq = map(pad_to_multiple, (fk, fv, fq))

                    fmask = pad_at_dim(fmask, (0, remainder), value = False, dim = -2)

                    selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)

                    if exists(gates):
                        gates = pad_at_dim(gates, (0, remainder), value = 0, dim = -2)

                if self.causal:
                    # handle block causal diagonal in the diagram, but run experiments without to see

                    fine_window_seq = arange(fine_divisible_seq_len, device = device) // self.selection_block_size
                    fine_window_seq = repeat(fine_window_seq, 'n -> b h n 1', b = batch, h = selected_block_indices.shape[1])
                    selected_block_indices = cat((selected_block_indices, fine_window_seq), dim = -1) # for the block causal diagonal in fig2

                    fmask = repeat(fmask, 'b h i w -> b h i w j', j = self.selection_block_size)

                    causal_mask = torch.ones((self.selection_block_size,) * 2, device = device, dtype = torch.bool).tril()
                    causal_mask = repeat(causal_mask, 'i j -> b h (w i) 1 j', w = num_fine_blocks, b = batch, h = fmask.shape[1])

                    fmask = cat((fmask, causal_mask), dim = -2)
                    fmask = rearrange(fmask, 'b h i w j -> b h 1 i (w j)')

                else:
                    fmask = repeat(fmask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)

                # select out the spatial crops of keys / values for fine attention

                fk = rearrange(fk, 'b h (w n) d -> b h w n d', w = num_fine_blocks)
                fv = rearrange(fv, 'b h (w n) d -> b h w n d', w = num_fine_blocks)

                # get_at("b h [w] j d, b h i selected -> b h i selected j d", fkv, selected_block_indices)

                if self.query_heads_share_selected_kv:
                    fk = repeat(fk, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                    fv = repeat(fv, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
                else:
                    fk = repeat(fk, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)
                    fv = repeat(fv, 'b h w j d -> b (h qh) i w j d', i = selected_block_indices.shape[2], qh = self.num_grouped_queries)

                selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fk.shape[-2], d = fk.shape[-1])

                fk = fk.gather(3, selected_block_indices)
                fv = fv.gather(3, selected_block_indices)

                # differential topk gating

                if self.use_diff_topk:
                    if self.causal:
                        gates = F.pad(gates, (0, 1), value = 1.)

                    fk = einx.multiply('b h i sel, b h i sel j d -> b h i sel j d', gates, fk)

                # merge selected key values

                fk, fv = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fk, fv))

                # fine attention

                fq = rearrange(fq, 'b (h qh) ... -> b h qh ...', qh = fine_num_grouped_queries)

                fsim = einsum(fq, fk, 'b h qh i d, b h i j d -> b h qh i j') * self.scale

                mask_value = max_neg_value(fsim)

                fsim = fsim.masked_fill(~fmask, mask_value)

                fattn = fsim.softmax(dim = -1)
                
                fine_attn_out = einsum(fattn, fv, 'b h qh i j, b h i j d -> b h qh i d')

                fine_attn_out = rearrange(fine_attn_out, 'b h qh ... -> b (h qh) ...')

                fine_attn_out = fine_attn_out[..., :seq_len, :]

        else:
            # if only first block, just do a simple block causal

            seq_len = fk.shape[-2]
            fmask = None

            fk, fv, fq = map(pad_to_multiple, (fk, fv, fq))

            fq, fk, fv = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = self.selection_block_size) for t in (fq, fk, fv))

            if self.causal:
                fmask = causal_mask = torch.ones((self.selection_block_size, self.selection_block_size), device = device, dtype = torch.bool).tril()

            fine_attn_out = attend(fq, fk, fv, mask = fmask)

            fine_attn_out = rearrange(fine_attn_out, '(b w) h n d -> b h (w n) d', b = batch)
            fine_attn_out = fine_attn_out[..., :seq_len, :]

        # 3. overlapping power attention
        
        pq = q
        pk = k
        pv = v
        
        seq_len = pq.size(-2)          

        mask = power_bool_mask(seq_len, padding_mask=attention_mask, causal=self.causal)          
        #mask_f = mask.to(torch.float16).masked_fill(~mask, float("-inf"))  
        if mask.dim() == 3:      # (B, N, N)
            mask = mask.unsqueeze(1) 

        with torch.backends.cuda.sdp_kernel(enable_flash=True,
                                                enable_mem_efficient=True,
                                                enable_math=True,
                                                enable_cudnn= True ):
                
            power_attn_out = F.scaled_dot_product_attention(pq, pk, pv,
                                                        attn_mask=mask,
                                                        enable_gqa=True)
        # combine strategies

        strategy_weighted_combine = self.to_strategy_combine(inp)
        
        out = einsum(
            strategy_weighted_combine,  # (B, heads, N, 2)
            stack([fine_attn_out, power_attn_out]),  # (2, B, heads, N, d)
            'b h n s, s b h n d -> b h n d'
        )
        
        # merge heads and combine them
        
        out = self.merge_heads(out)

        out = self.combine_heads(out)
        
        if not return_cache:
            return out

        return out, (cache_kv, cache_compressed_kv)