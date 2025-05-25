"""
Fused Grouped Cross-Attention
===============
This is a Triton implementation of the Grouped Cross-Attention
Author: Xiang Hu
Extra Credits:
- Original flash attention2 paper (https://tridao.me/publications/flash2/flash2.pdf)
- OpenAI kernel team
"""

import pytest
import torch
import torch.nn.functional as F
import math
import triton
import triton.language as tl
from einops import rearrange, repeat


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, GROUP_NUM: tl.constexpr,
                    BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, Q_CTX: tl.constexpr, K_CTX: tl.constexpr, fp8_v: tl.constexpr):
    lo, hi = 0, K_CTX
    assert K_CTX % BLOCK_N == 0
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # loop over k, v and update accumulator
    # qk_mask = start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None] + tl.arange(0, BLOCK_N)[None, :] * 0 < Q_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)  # (dim, N)
        qk = tl.dot(q, k)  # (group_size * M, N)
        # qk = tl.where(qk_mask[None, :, None], qk, -float('inf'))  # (group_size, M, N)
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)  # (group_size, M)
        qk = qk * qk_scale - m_ij[:, None]  # (group_size, M, N) 
        p = tl.math.exp2(qk)  # （group_size * M, N)
        l_ij = tl.sum(p, 1)  # axis=1 （group_size, M）
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)  # (group_size, M) 
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.bfloat16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [32, 64] \
    for BN in [32, 64] \
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

inf_configs=[
    triton.Config({'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BN in [64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]

MIN_Q_CTX = 16
def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]

    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["GROUP_NUM", "Q_CTX", "K_CTX","HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, W, sm_scale, sm_n, M, Out, # M (Batch, Q_CTX, neighbor_num)
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kb, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vb, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_wz, stride_wq,
              stride_mz, stride_mh, stride_mm, stride_mk,
              Z, H, Q_CTX, K_CTX, GROUP_NUM: tl.constexpr, NEIGHBOR_NUM: tl.constexpr, # Z: batch_size, H, head_num
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(Q_CTX > BLOCK_M)
    start_m = tl.program_id(0)  # block_m id
    off_hz = tl.program_id(1)  # batch_size * head_num id
    off_z = off_hz // H  # batch_id
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + (off_h * GROUP_NUM).to(tl.int64) * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + (off_h * GROUP_NUM).to(tl.int64) * stride_oh
    vk_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    w_offset = off_z.to(tl.int64) * stride_wz  # share accross all heads
    m_offset = off_z.to(tl.int64) * stride_mz + (off_h * GROUP_NUM).to(tl.int64) * stride_mh

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_qh, stride_qm, stride_qk),
        offsets=(0, start_m * BLOCK_M, 0),
        block_shape=(GROUP_NUM, BLOCK_M, HEAD_DIM),
        order=(2, 1, 0),
    )
    # q_block_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    # Q_block_ptr = Q + q_offset + q_block_offsets
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_oh, stride_om, stride_on),
        offsets=(0, start_m * BLOCK_M, 0),
        block_shape=(GROUP_NUM, BLOCK_M, HEAD_DIM),
        order=(2, 1, 0),
    )
    
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.442695040888963  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option="nan")
    q = q.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)

    acc = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # outk_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_okm + tl.arange(0, HEAD_DIM)[None, :] * stride_okk

    for block_i in range(0, NEIGHBOR_NUM):
        V_block_ptr = tl.make_block_ptr(
            base=V + vk_offset + block_i * stride_kb,
            shape=(K_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + vk_offset + block_i * stride_vb,
            shape=(HEAD_DIM, K_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        m_block_ptr = tl.make_block_ptr(
            base=M + m_offset + block_i,
            shape=(GROUP_NUM, Q_CTX),
            strides=(stride_mh, stride_mm),
            offsets=(0, start_m * BLOCK_M),
            block_shape=(GROUP_NUM, BLOCK_M),
            order=(1, 0),
        )
        
        weight = tl.load(W + w_offset + block_i)

        # m_i = m_i * 0 - float("inf")
        # l_i = l_i * 0 + 1.0
        # tmp = tmp * 0
        tmp = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # initialize pointer to m and l    
        m_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) + sm_n


        tmp, l_i, m_i = _attn_fwd_inner(tmp, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, GROUP_NUM,
                                        BLOCK_N,  #
                                        4 - STAGE, Q_CTX, K_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
        # epilogue
        m_i += tl.math.log2(l_i)
        
        # m_ptrs = M + (off_hz * GROUP_NUM + tl.arange(0, GROUP_NUM))[:, None] * Q_CTX * NEIGHBOR_NUM + (offs_m * NEIGHBOR_NUM)[None, :] + block_i
        tmp = tmp / l_i[:, None]
        # tl.store(m_ptrs, m_i, mask=tl.arange(0, GROUP_NUM)[:, None] * 0.0 + offs_m[None, :] < Q_CTX)
        # tl.store(m_block_ptr, m_i.reshape(GROUP_NUM, BLOCK_M), boundary_check=(1,))
        # tl.store(Ok_block_ptr, tmp.reshape(GROUP_NUM, BLOCK_M, HEAD_DIM).to(Out_k.type.element_ty), boundary_check=(1,))
        tl.store(m_block_ptr, m_i.reshape(GROUP_NUM, BLOCK_M), boundary_check=(1,))
        acc += weight * tmp
        # acc += tmp / l_i[:, None]

    # initialize offsets
    # tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0)) #, mask=q_mask)
    acc = acc.reshape(GROUP_NUM, BLOCK_M, HEAD_DIM)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))


@triton.autotune(inf_configs, key=["K_CTX","HEAD_DIM"])
@triton.jit
def _attn_inf(Q, K, V, W, sm_scale, sm_n, Out,  # M (Batch, Q_CTX, neighbor_num)
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kb, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vb, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              stride_wz, stride_wq,
              Z, H, Q_CTX, K_CTX, NEIGHBOR_NUM: tl.constexpr, # Z: batch_size, H, head_num
              GROUP_NUM: tl.constexpr,
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr  #
              ):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    # tl.static_assert(Q_CTX > BLOCK_M)
    start_m = tl.program_id(0)  # block_m id
    off_hz = tl.program_id(1)  # batch_size * head_num id
    off_z = off_hz // H  # batch_id
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + (off_h * GROUP_NUM).to(tl.int64) * stride_qh
    o_offset = off_z.to(tl.int64) * stride_oz + (off_h * GROUP_NUM).to(tl.int64) * stride_oh
    vk_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
    w_offset = off_z.to(tl.int64) * stride_wz  # share accross all heads

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_qh, stride_qm, stride_qk),
        offsets=(0, start_m * BLOCK_M, 0),
        block_shape=(GROUP_NUM, BLOCK_M, HEAD_DIM),
        order=(2, 1, 0),
    )
    # q_block_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    # Q_block_ptr = Q + q_offset + q_block_offsets
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)

    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_oh, stride_om, stride_on),
        offsets=(0, start_m * BLOCK_M, 0),
        block_shape=(GROUP_NUM, BLOCK_M, HEAD_DIM),
        order=(2, 1, 0),
    )
    
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.442695040888963  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check=(1,), padding_option="nan")
    q = q.reshape(GROUP_NUM * BLOCK_M, HEAD_DIM)

    acc = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # outk_offsets = (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * stride_okm + tl.arange(0, HEAD_DIM)[None, :] * stride_okk

    for block_i in range(0, NEIGHBOR_NUM):
        V_block_ptr = tl.make_block_ptr(
            base=V + vk_offset + block_i * stride_kb,
            shape=(K_CTX, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + vk_offset + block_i * stride_vb,
            shape=(HEAD_DIM, K_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
        weight = tl.load(W + w_offset + block_i + tl.arange(0, 1))

        # m_i = m_i * 0 - float("inf")
        # l_i = l_i * 0 + 1.0
        # tmp = tmp * 0
        tmp = tl.zeros([GROUP_NUM * BLOCK_M, HEAD_DIM], dtype=tl.float32)
        # initialize pointer to m and l    
        m_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([GROUP_NUM * BLOCK_M], dtype=tl.float32) + sm_n


        tmp, l_i, m_i = _attn_fwd_inner(tmp, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, GROUP_NUM,
                                        BLOCK_N,  #
                                        4 - STAGE, Q_CTX, K_CTX, V.dtype.element_ty == tl.float8e5  #
                                        )
        # epilogue
        tmp = tmp / l_i[:, None]
        acc += weight * tmp
        # acc += tmp / l_i[:, None]

    # initialize offsets
    # tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0)) #, mask=q_mask)
    acc = acc.reshape(GROUP_NUM, BLOCK_M, HEAD_DIM)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(1,))


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _attn_bwd_dkdv(dk, dv,  #
                   Q, k, v, w, #
                   DO,  #
                   M, delta,
                   # shared by Q/K/V/DO.
                   stride_qh, stride_qm, stride_qk,  #
                   stride_dh, stride_dq,
                   stride_mh, stride_mq,
                   Q_CTX,
                   NEIGHBOR_NUM: tl.constexpr,
                   BLOCK_M1: tl.constexpr,  #
                   BLOCK_N1: tl.constexpr,  #
                   HEAD_DIM: tl.constexpr,  #
                   start_m, num_steps,
                   GROUP_NUM: tl.constexpr):  # start_m:Q起始位置, start_n: KV起始位置,num_steps: 遍历完整个qT需要的steps
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)

    for group_id in range(GROUP_NUM):
        qT_block_ptr = tl.make_block_ptr(
            base=Q + stride_qh * group_id,
            shape=(HEAD_DIM, Q_CTX),
            strides=(stride_qk, stride_qm),
            offsets=(0, start_m),
            block_shape=(HEAD_DIM, BLOCK_M1),
            order=(0, 1)
        )
        # q_mask = offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk < Q_CTX * stride_qm
        # do_ptrs = DO + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        do_block_ptr = tl.make_block_ptr(
            base=DO + stride_qh * group_id,
            shape=(Q_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(start_m, 0),
            block_shape=(BLOCK_M1, HEAD_DIM),
            order=(1, 0)
        )
        # M: (B, KHEADS, Q_CTX, K)
        m_block_ptr = tl.make_block_ptr(
            base=M + group_id * stride_mh,
            shape=(Q_CTX,),
            strides=(stride_mq,),
            offsets=(start_m,),
            block_shape=(BLOCK_M1,),
            order=(0,)
        )
        # Di: (BATCH, Q_HEADs, K, Q_CTX)
        Di_block_ptr = tl.make_block_ptr(
            base=delta + group_id * stride_dh,
            shape=(Q_CTX,),
            strides=(stride_dq,),
            offsets=(start_m,),
            block_shape=(BLOCK_M1,),
            order=(0,)
        )
        for _ in range(num_steps): 
            # q: (dim, M), k,v: (N, dim)
            qT = tl.load(qT_block_ptr, boundary_check=(1,))  # (dim, M)
            m = tl.load(m_block_ptr, boundary_check=(0,))
            do = tl.load(do_block_ptr, boundary_check=(0,))  # (M, dim)
            # m = tl.load(m_block_ptr, boundary_check=(0,))  # (M)
            qkT = tl.dot(k, qT)  # (BLOCK_SIZE, M)
            # TODO: add mask
            pT = tl.math.exp2(qkT - m[None, :])  # local softmax, (BLOCK_SIZE, M)
            o = tl.dot(tl.trans(v).to(tl.float32), pT)  # (dim, M)
            Di = tl.sum(tl.trans(o) * do, axis=1) # (M)
            # tl.store(Di_block_ptr, Di, boundary_check=(0,))
            tl.store(Di_block_ptr, Di, boundary_check=(0,))
            # qkT = tl.where(kqT_mask, qkT, -float('inf'))
            pT = w * pT  # (N, M)
            # pT = tl.where(pT_mask, pT, 0.0)
            # Autoregressive masking.
            # do = tl.load(do_ptrs, mask=q_mask, other=0.0)  # (M, dim)
            # Di = tl.sum(tl.trans(o) * do, axis=1)  # (M)
            # do = tl.zeros([BLOCK_M1, HEAD_DIM], dtype=tl.bfloat16)
            # Compute dV.
            ppT = pT  # (N, M)
            ppT = ppT.to(tl.bfloat16)
            dv += tl.dot(ppT, do)  # (N, dim)
            # D (= delta) is pre-divided by ds_scale.
            # Di = tl.load(D + offs_m, mask=offs_m < Q_CTX, other=0.0)
            # Di = tl.load(Di_block_ptr, boundary_check=(0,))  # (M)
            # Compute dP and dS.
            dpT = tl.dot(v, tl.trans(do)).to(tl.float32) # (N, M)
            dsT = pT * (dpT - Di[None, :])  # (N, M)
            dsT = dsT.to(tl.bfloat16)
            dk += tl.dot(dsT, tl.trans(qT))  # (BLOCK_SIZE, dim)
            # Increment pointers.
            qT_block_ptr = tl.advance(qT_block_ptr, (0, BLOCK_M1))
            do_block_ptr = tl.advance(do_block_ptr, (BLOCK_M1, 0))
            m_block_ptr = tl.advance(m_block_ptr, (BLOCK_M1,))
            Di_block_ptr = tl.advance(Di_block_ptr, (BLOCK_M1,))
    return dk, dv


# the main inner-loop logic for computing dQ
@triton.jit
def _attn_bwd_dq(dq, q, K, V, W, #
                 do, M, D,
                 # shared by Q/K/V/DO.
                 stride_kb, stride_kn, stride_kk,  #
                 stride_mh, stride_mq,
                 stride_dh, stride_dq,
                 Q_CTX,
                 NEIGHBOR_NUM: tl.constexpr,
                 GROUP_NUM: tl.constexpr,
                 BLOCK_M2: tl.constexpr,  #
                 BLOCK_N2: tl.constexpr,  #
                 HEAD_DIM: tl.constexpr,
                 # Filled in by the wrapper.
                 start_m):
    offs_n = tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)
    # weights = tl.load(W + tl.arange(0, NEIGHBOR_NUM))

    # D (= delta) is pre-divided by ds_scale.
    
    for chunk_i in range(NEIGHBOR_NUM):
        # dw =  tl.zeros([1,], dtype=tl.float32)
        # m = tl.load(M + offs_m * NEIGHBOR_NUM + chunk_i, mask=offs_m * NEIGHBOR_NUM + chunk_i < Q_CTX * NEIGHBOR_NUM, other=float('inf'))  # (M,)
        # m = m[:, None]
        m_ptr = tl.make_block_ptr(
            base=M + chunk_i,
            shape=(GROUP_NUM, Q_CTX),
            strides=(stride_mh, stride_mq),
            offsets=(0, start_m),
            block_shape=(GROUP_NUM, BLOCK_M2),
            order=(1, 0)
        )
        Di_block_ptr = tl.make_block_ptr(
            base=D + chunk_i * Q_CTX,
            shape=(GROUP_NUM, Q_CTX),
            strides=(stride_dh, stride_dq),
            offsets=(0, start_m),
            block_shape=(GROUP_NUM, BLOCK_M2),
            order=(1, 0)
        )
        weight = tl.load(W + chunk_i + tl.arange(0, 1))
        kT_ptrs = K + chunk_i * stride_kb + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        vT_ptrs = V + chunk_i * stride_kb + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
        # Di = tl.load(D + chunk_i * Q_CTX + offs_m)
        Di = tl.load(Di_block_ptr, boundary_check=(1,))
        Di = Di.reshape(GROUP_NUM * BLOCK_M2)
        m = tl.load(m_ptr, boundary_check=(1,))
        m = m.reshape(GROUP_NUM * BLOCK_M2)
        m = m[:, None]
        
        # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
        # tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)

        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)  # (G * M, N)
        p = tl.math.exp2(qk - m)  # (G * M, N)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32) # (G * M, N)
        # dw += tl.sum(p * dp)  # (1,)
        ds = weight * p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16) # (G * M, N)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))

    return dq


@triton.jit
def kernel_attn_bwd_dkdv(Q, K, V, weights, sm_scale, sm_n,  #
              DO,  #
              DQ, DK, DV, M, delta,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_qm, stride_qk,  # qz/qh: batch-level/head-level stride for query
              stride_kz, stride_kh, stride_kb, stride_kn, stride_kk,
              stride_mz, stride_mh, stride_mq, stride_mk,
              stride_dz, stride_dh, stride_dk, stride_dq,
              H, Q_CTX, K_CTX, neighbor_num: tl.constexpr,  #
              Q_PER_BLOCK: tl.constexpr,
              GROUP_NUM: tl.constexpr,
              BLOCK_M1: tl.constexpr,  # block size for iterating O, Q
              BLOCK_N1: tl.constexpr,  # block size for KV,
              BLOCK_M2: tl.constexpr,  # block size for iterating O, Q
              BLOCK_N2: tl.constexpr,  # block size for KV
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    # bhid = tl.program_id(1).to(tl.int64)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    chunk_id = tl.program_id(0).to(tl.int64)
    # off_chz = (bhid * Q_CTX * neighbor_num).to(tl.int64)
    off_chm = hid.to(tl.int64) * stride_mh * GROUP_NUM + bid.to(tl.int64) * stride_mz
    off_chd = hid.to(tl.int64) * stride_dh * GROUP_NUM + bid.to(tl.int64) * stride_dz
    # tl.static_assert(Q_CTX % BLOCK_M2 == 0)
    off_weights = bid.to(tl.int64) * neighbor_num
    off_chunkid = chunk_id.to(tl.int64)
    adj = stride_qh * hid.to(tl.int64) * GROUP_NUM + stride_qz * bid.to(tl.int64)
    kv_adj = stride_kh * hid.to(tl.int64) + stride_kz * bid.to(tl.int64)
    off_kv_chunki = chunk_id.to(tl.int64) * stride_kb

    # offset pointers for batch/head
    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    DK += kv_adj
    DV += kv_adj
    M += off_chm
    M_dkv = M + off_chunkid
    delta += off_chd
    D_dkv = delta + off_chunkid * Q_CTX
    weights += off_weights

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = 0 # start block for KV

    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + off_kv_chunki + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)  # (BLOCK_N1, HEAD_DIM)
    v = tl.load(V + off_kv_chunki + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    w = tl.load(weights + tl.arange(0, 1) + chunk_id) # (1,)

    num_steps = (Q_CTX + BLOCK_M1 - 1) // BLOCK_M1

    # Compute dK and dV for non-masked blocks.
    dk, dv = _attn_bwd_dkdv(  #
        dk, dv,  #
        Q, k, v, w, #
        DO,  #
        M_dkv, D_dkv,
        stride_qh, stride_qm, stride_qk,  #
        stride_dh, stride_dq,
        stride_mh, stride_mq,
        Q_CTX,
        neighbor_num,
        BLOCK_M1, BLOCK_N1, HEAD_DIM,  #
        0, num_steps,
        GROUP_NUM=GROUP_NUM
    )
    

    dv_ptrs = DV + off_kv_chunki + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + off_kv_chunki + offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk
    tl.store(dk_ptrs, dk)


@triton.jit
def kernel_attn_bwd_dq(Q, K, V, weights, sm_scale, sm_n,  #
              DO,  #
              DQ, M, D,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_qm, stride_qk,  # qz/qh: batch-level/head-level stride for query
              stride_kz, stride_kh, stride_kb, stride_kn, stride_kk,
              stride_mz, stride_mh, stride_mq, stride_mk,
              stride_dz, stride_dh, stride_dk, stride_dq,
              H, Q_CTX, K_CTX, neighbor_num: tl.constexpr,  #
              Q_PER_BLOCK: tl.constexpr,
              GROUP_NUM: tl.constexpr,
              BLOCK_M1: tl.constexpr,  # block size for iterating O, Q
              BLOCK_N1: tl.constexpr,  # block size for KV,
              BLOCK_M2: tl.constexpr,  # block size for iterating O, Q
              BLOCK_N2: tl.constexpr,  # block size for KV
              HEAD_DIM: tl.constexpr):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    m_id = tl.program_id(2)

    adj = stride_qh * (h_id % H).to(tl.int64) * GROUP_NUM + stride_qz * b_id.to(tl.int64)
    kv_adj = stride_kh * (h_id % H).to(tl.int64) + stride_kz * b_id.to(tl.int64)
    off_chm = h_id.to(tl.int64) * stride_mh * GROUP_NUM + b_id.to(tl.int64) * stride_mz
    off_chd = h_id.to(tl.int64) * stride_dh * GROUP_NUM + b_id.to(tl.int64) * stride_dz
    off_weights = b_id.to(tl.int64) * neighbor_num
    Q += adj
    K += kv_adj
    V += kv_adj
    DO += adj
    DQ += adj
    M += off_chm
    D += off_chd
    weights += off_weights

    # pid = pid // Q_PER_BLOCK
    # THIS BLOCK DOES DQ:
    # DW: (BATCH, N_HEAD * Q_CTX // BLOCK_M2 // GROUP_NUM, K)
    off_m = m_id.to(tl.int32)
    start_m = off_m * BLOCK_M2
    q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_qh, stride_qm, stride_qk),
        offsets=(0, start_m, 0),
        block_shape=(GROUP_NUM, BLOCK_M2, HEAD_DIM),
        order=(2,1,0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_qh, stride_qm, stride_qk),
        offsets=(0, start_m, 0),
        block_shape=(GROUP_NUM, BLOCK_M2, HEAD_DIM),
        order=(2,1,0)
    )
    dq_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(GROUP_NUM, Q_CTX, HEAD_DIM),
        strides=(stride_qh, stride_qm, stride_qk),
        offsets=(0, start_m, 0),
        block_shape=(GROUP_NUM, BLOCK_M2, HEAD_DIM),
        order=(2,1,0)
    )
    assert BLOCK_N1 % neighbor_num == 0
    # BLOCK_M2 = BLOCK_N1 // neighbor_num
    # BLOCK_N2 = BLOCK_M1
    # tl.static_assert(BLOCK_M2 * neighbor_num == BLOCK_N1)
    # start_m = ((pid * neighbor_num + chunk_id) // Q_PER_BLOCK) * BLOCK_M2

    # MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    # offs_m = start_m + tl.arange(0, BLOCK_M2)

    # q_mask = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk < Q_CTX * stride_qm
    # q = tl.load(Q + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, mask=q_mask, other=-float('inf'))
    q = tl.load(q_block_ptr, boundary_check=(0, 1, 2))
    q = q.reshape(GROUP_NUM * BLOCK_M2, HEAD_DIM)
    dq = tl.zeros([GROUP_NUM * BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(do_block_ptr, boundary_check=(0, 1, 2))
    do = do.reshape(GROUP_NUM * BLOCK_M2, HEAD_DIM)
    # do = tl.load(DO + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk, mask=q_mask, other=-float('inf'))

    # m = tl.load(M + (offs_m * neighbor_num)[:, None] + tl.arange(0, neighbor_num)[None, :])  # (BLOCK_M2, neighbor_num)
    # m = m[:, :, None]
    dq = _attn_bwd_dq(dq, q, K, V, weights, #
                        do, M, D, #
                        stride_kb, stride_kn, stride_kk,  #
                        stride_mh, stride_mq,
                        stride_dh, stride_dq,
                        Q_CTX,
                        neighbor_num, #
                        GROUP_NUM,
                        BLOCK_M2, BLOCK_N2, HEAD_DIM,  #
                        start_m
                        )
    # # Write back dQ.
    # dq_ptrs = DQ + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    dq *= LN2
    dq = dq.reshape(GROUP_NUM, BLOCK_M2, HEAD_DIM)
    tl.store(dq_block_ptr, dq.to(DQ.type.element_ty), boundary_check=(0, 1, 2))


def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, weights, sm_scale, sm_n):
        # shape constraints
        # k (batch_size, head_num, K, L, head_dim)
        # q (Batch, Head, Q_CTX, Head_dim)
        assert q.is_contiguous()
        assert k.is_contiguous()
        # assert q.shape[2] % 64 == 0
        # assert v.is_contiguous()
        assert k.shape[3] >= 64
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        neighbor_num = k.shape[2]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}, f'current head dim: {HEAD_DIM_K}'
        assert len(weights.shape) == 2
        o = torch.empty_like(q)
        # o_k = torch.empty(q.shape[0], q.shape[1], neighbor_num, q.shape[2], q.shape[3], dtype=q.dtype, device=q.device)
        stage = 3
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        # grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        assert q.shape[1] % k.shape[1] == 0
        group_size = q.shape[1] // k.shape[1]
        assert group_size >= 1
        # print(f'group size: {group_size}')
        grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * k.shape[1], 1)
        # print(f'grid: {triton.cdiv(q.shape[2], 32)}, Q_CTX: {q.shape[2]}') 
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2], neighbor_num), device=q.device, dtype=torch.float32)
        assert q.shape[2] >= 64
        assert k.shape[3] >= 64
        _attn_fwd[grid](
            q, k, v, weights, sm_scale, sm_n, M, o, #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4), # (B, head_num, K, L, dim)
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4), #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            weights.stride(0), weights.stride(1),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            Z=q.shape[0], H=k.shape[1],  #
            Q_CTX=q.shape[2], K_CTX=k.shape[3], GROUP_NUM=group_size, NEIGHBOR_NUM=neighbor_num, #
            HEAD_DIM=HEAD_DIM_K,  #
            STAGE=stage,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, weights, M) # default call contiguous?
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.sm_n = sm_n
        ctx.HEAD_DIM = HEAD_DIM_K
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, weights, M = ctx.saved_tensors
        BATCH, N_HEAD, Q_CTX = q.shape[:3]
        K_HEAD = k.shape[1]
        GROUP_NUM = q.shape[1] // k.shape[1]
        assert GROUP_NUM >= 1
        K = k.shape[2]
        K_CTX = k.shape[3]  # (B, head, K, L, d)
        assert K_CTX % 16 == 0
        # print(do.shape)
        # print(weights.shape)
        do = do.contiguous()  # (B, H, L, d)
        assert do.is_contiguous()
        # assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        CHUNK_SIZE = BATCH // weights.shape[0]
        next_round_q_len = next_power_of_2(Q_CTX)
        # print(v.stride())
        # print(dw.shape)
        # print(M.shape)
        NUM_WARPS, NUM_STAGES = 4, 5
        # assert K * K_CTX % Q_CTX == 0, f'{K}, {K_CTX}, {Q_CTX}'
        # R = K * K_CTX // Q_CTX  # TODO: reconsider how to implement
        assert 128 % K == 0
        BLOCK_M1, BLOCK_N1 = 32, K_CTX
        assert BLOCK_N1 % 8 == 0
        BLOCK_M2 = max(next_round_q_len // K, 16)
        Q_PER_BLOCK = K // (next_round_q_len // BLOCK_M2)  # default: next_round_q_len 64, Q_PER_BLOCK is K // 4 (8 for K=32)
        BLOCK_N2 = K_CTX  # M2 FOR Q_CTX

        # dw = torch.zeros((BATCH * N_HEAD, Q_CTX // BLOCK_M2 , K), device=weights.device)
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)
        # PRE_BLOCK = min(Q_CTX, 128)
        PRE_BLOCK = min(next_round_q_len, 128)
        assert next_round_q_len % PRE_BLOCK == 0
        
        # dw = torch.zeros(BATCH, N_HEAD // GROUP_NUM, Q_CTX // BLOCK_M2, K, device=q.device, dtype=torch.float32)
        assert K * K_CTX // BLOCK_N1 * BLOCK_M2 // Q_PER_BLOCK == next_round_q_len
        grid = (K, BATCH, K_HEAD)  # K, neighbor size
        delta = torch.zeros(BATCH, N_HEAD, K, Q_CTX, device=q.device, dtype=torch.float32)
        # _attn_bwd[grid](
        #     q, arg_k, v, weights, ctx.sm_scale, ctx.sm_n,
        #     do, 
        #     dq, dk, dv, dw,#
        #     q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        #     k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
        #     dw.stride(0), dw.stride(1), dw.stride(2), dw.stride(3),
        #     K_HEAD, Q_CTX, K_CTX, K, #
        #     Q_PER_BLOCK,
        #     GROUP_NUM=GROUP_NUM,
        #     BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
        #     BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
        #     HEAD_DIM=ctx.HEAD_DIM,  #
        #     num_warps=NUM_WARPS,  #
        #     num_stages=NUM_STAGES  #
        # )

        kernel_attn_bwd_dkdv[grid](
            q, arg_k, v, weights, ctx.sm_scale, ctx.sm_n,
            do, 
            dq, dk, dv, M, delta,#
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2), delta.stride(3),
            K_HEAD, Q_CTX, K_CTX, K, #
            Q_PER_BLOCK,
            GROUP_NUM=GROUP_NUM,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        grid = (BATCH, K_HEAD, Q_CTX // BLOCK_M2)
        kernel_attn_bwd_dq[grid](
            q, arg_k, v, weights, ctx.sm_scale, ctx.sm_n,
            do, 
            dq, M, delta,#
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
            M.stride(0), M.stride(1), M.stride(2), M.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2), delta.stride(3),
            K_HEAD, Q_CTX, K_CTX, K, #
            Q_PER_BLOCK,
            GROUP_NUM=GROUP_NUM,
            BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,  #
            BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,  #
            HEAD_DIM=ctx.HEAD_DIM,  #
            num_warps=NUM_WARPS,  #
            num_stages=NUM_STAGES  #
        )

        delta_w = rearrange(delta, 'B N K Q -> B (N Q) K')
        dw = delta_w.sum(dim=1)  # (N, K)
        return dq, dk, dv, dw, None, None


attention = _attention.apply

def gca_kv_cache(q, k, v, weights, sm_scale, sm_n):

    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    # when v is in float8_e5m2 it is transposed.
    neighbor_num = k.shape[2]
    GROUP_NUM = q.shape[1] // k.shape[1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}, f'current head dim: {HEAD_DIM_K}'
    assert len(weights.shape) == 2
    o = torch.empty_like(q)

    # o_k = torch.zeros(q.shape[0], q.shape[1], neighbor_num, q.shape[2], q.shape[3], dtype=q.dtype, device=q.device)
    stage = 3 if causal else 1
    extra_kern_args = {}
    # Tuning for AMD target
    if is_hip():
        waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
        extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

    grid = lambda args: (triton.cdiv(q.shape[2], args["BLOCK_M"]), q.shape[0] * k.shape[1], 1)
    # print(f'grid: {triton.cdiv(q.shape[2], 32)}, Q_CTX: {q.shape[2]}') 
    # M = torch.empty((q.shape[0] * q.shape[1], q.shape[2], neighbor_num), device=q.device, dtype=torch.float32)
    BLOCK_M = 16
    for blk_m in [16, 32, 64]:
        if q.shape[2] % blk_m == 0:
            BLOCK_M = blk_m
    _attn_inf[grid](
        q, k, v, weights, sm_scale, sm_n, o, #
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4), # (B, head_num, K, L, dim)
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4), #
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
        weights.stride(0), weights.stride(1),
        Z=q.shape[0], H=k.shape[1],  #
        Q_CTX=q.shape[2], K_CTX=k.shape[3], NEIGHBOR_NUM=neighbor_num, #
        GROUP_NUM=GROUP_NUM,
        HEAD_DIM=HEAD_DIM_K,  #
        STAGE=stage,  #
        BLOCK_M = BLOCK_M,
        **extra_kern_args)

    return o


@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM, K", [(512, 40, 64, 64, 32)])
@pytest.mark.parametrize("causal", [False])
def test_group_qa(Z, H, N_CTX, HEAD_DIM, K, causal, dtype=torch.bfloat16):
    import torch.nn.functional as F
    torch.manual_seed(22)
    # q = (torch.empty((Z + 1, H * (N_CTX + 1), HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    # q = rearrange(q, 'Z (H L) D -> Z H L D ', H=H)
    # # print(q.stride())
    # org_q = q
    # q = q[:-1, :, :-1, :]
    # q_ = q.contiguous()
    q = (torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    k = (torch.empty((Z, H // 4, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    v = (torch.empty((Z, H // 4, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
    weights = F.softmax(torch.tensor(torch.rand(Z, K), dtype=dtype, device='cuda'), dim=-1).requires_grad_()
    # weights = torch.ones((Z, K), dtype=dtype, device='cuda').requires_grad_()
    sm_scale = 1 / 8
    dout = torch.randn_like(q)
    # reference implementation
    # M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    # p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    k_ = torch.repeat_interleave(k, dim=1, repeats=4)
    v_ = torch.repeat_interleave(v, dim=1, repeats=4)
    p = torch.einsum('Z H M h, Z H K N h->Z H K M N', q, k_) * sm_scale
    # if causal:
    #     p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p, dim=-1).to(dtype)
    # p = torch.exp(p)
    # ref_out = torch.matmul(p, v)
    o_k = torch.einsum('Z H K M N, Z H K N h->Z H K M h', p, v_)
    ref_out = torch.einsum('Z H K M h, Z K->Z H M h', o_k, weights)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dw, weights.grad = weights.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, weights, sm_scale, 0)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    tri_dw, weights.grad = weights.grad.clone(), None
    # compare
    assert not torch.any(torch.isnan(tri_out))
    # print(ref_out[:, -1, -5:, :5])
    # print(tri_out[:, -1, -5:, :5])
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0), (ref_out - tri_out).abs().max()
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For details see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    # print(tri_dv[])

    # delta_ref = torch.einsum(
    #     'N H S D, N H K S D->N H K S', dout.to(torch.float32), o_k.to(torch.float32)
    # )
    # delta_w = rearrange(delta_ref, 'N H K Q -> N (H Q) K').sum(dim=1).to(torch.bfloat16)
    # assert torch.allclose(delta_w, ref_dw, atol=1e-2, rtol=0),  (delta_w - ref_dw).abs().max()
    

    # print(ref_dv[0, :,-1, -5:, 0])
    # print(tri_dv[0, :,-1, -5:, 0])

    # print(ref_dv[1, :, -5:, 0])
    # print(tri_dv[1, :, -5:, 0])
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol), (ref_dv - tri_dv).abs().max()
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol), (ref_dk - tri_dk).abs().max()
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol), (ref_dq - tri_dq).abs().max()
    assert torch.allclose(ref_dw, tri_dw, atol=0.5, rtol=rtol), (ref_dw - tri_dw).abs().max()

    with torch.no_grad():
        # K = 6
        # HEAD_DIM = 32
        # Z = 1
        q = (torch.empty((Z, H, 2, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        weights = F.softmax(torch.tensor(torch.rand(Z, K), dtype=dtype, device='cuda'), dim=-1).requires_grad_()
        sm_scale = 0.5
        k_ = torch.repeat_interleave(k, dim=1, repeats=2)
        v_ = torch.repeat_interleave(v, dim=1, repeats=2)
        p = torch.einsum('Z H M h, Z H K N h->Z H K M N', q, k_) * sm_scale
        # if causal:
        #     p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p, dim=-1)
        # p = torch.exp(p)
        # ref_out = torch.matmul(p, v)
        ref_out = torch.einsum('Z H K M N, Z H K N h->Z H K M h', p, v_)
        ref_out = torch.einsum('Z H K M h, Z K->Z H M h', ref_out, weights)
        tri_out = gca_kv_cache(q, k, v, weights, sm_scale, 0.0)  # (Z H M dim)
        # print(ref_out[0, 0, 0, :5])
        # print(tri_out[0, 0, 0, :5])
        # print(ref_out[0, 1, 0, :5])
        # print(tri_out[0, 1, 0, :5])
        assert ref_out.shape == tri_out.shape
        assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

        q = (torch.empty((Z, H, 1, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        weights = F.softmax(torch.tensor(torch.rand(Z, K), dtype=dtype, device='cuda'), dim=-1).requires_grad_()
        sm_scale = 0.5
        k_ = torch.repeat_interleave(k, dim=1, repeats=2)
        v_ = torch.repeat_interleave(v, dim=1, repeats=2)
        p = torch.einsum('Z H M h, Z H K N h->Z H K M N', q, k_) * sm_scale
        # if causal:
        #     p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p, dim=-1)
        # p = torch.exp(p)
        # ref_out = torch.matmul(p, v)
        ref_out = torch.einsum('Z H K M N, Z H K N h->Z H K M h', p, v_)
        ref_out = torch.einsum('Z H K M h, Z K->Z H M h', ref_out, weights)
        tri_out = gca_kv_cache(q, k, v, weights, sm_scale, 0.0)  # (Z H M dim)
        # print(ref_out[0, 0, 0, :5])
        # print(tri_out[0, 0, 0, :5])
        # print(ref_out[0, 1, 0, :5])
        # print(tri_out[0, 1, 0, :5])
        assert ref_out.shape == tri_out.shape
        assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, HEAD_DIM = 128, 32, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd", "bwd"]:
    for causal in [False]:
        configs.append(
            triton.testing.Benchmark(
                x_names=["N_CTX"],
                x_vals=[2**i for i in range(6, 7)],
                line_arg="provider",
                line_vals=["triton-fp16"] + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                (["flash"] if HAS_FLASH else []),
                line_names=["Triton [FP16]"] + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                (["Flash-2"] if HAS_FLASH else []),
                styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                ylabel="TFLOPS",
                plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}",
                args={
                    "H": N_HEADS,
                    "BATCH": BATCH,
                    "HEAD_DIM": HEAD_DIM,
                    "mode": mode,
                    "causal": causal,
                },
            ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device="cuda"):
    import torch.nn.functional as F
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 500
    dtype = torch.bfloat16
    K = 8
    # Q_CTX, K_CTX = N_CTX, N_CTX // K
    if "triton" in provider:
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H // 2, K, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        weights = F.softmax(torch.randn((BATCH, K), dtype=dtype, device=device, requires_grad=True), dim=-1)
        if mode == "fwd" and "fp8" in provider:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 2, 4, 3).contiguous()
            v = v.permute(0, 1, 2, 4, 3)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, weights, sm_scale, 0.0)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        # qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        q = torch.randn((BATCH, N_CTX, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, K * N_CTX, H // 2, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, K * N_CTX, H // 2, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops * 1e-12 / (ms * 1e-3)


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)