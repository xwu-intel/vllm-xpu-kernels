# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from tests.ops.fp8_quant_op import scaled_fp8_quant
from tests.register_ops import fp8_block_scaled_gemm, fp8_gemm, fp8_gemm_w8a16

BATCHES = [1]
MNK_FACTORS = [
    (1, 4096, 1),
    (1, 32, 1024),
    (4, 16, 1024),
    (8, 32, 1024),
    (8, 512, 1024),
]

MINI_MNK_FACTORS = [
    (1, 4, 8),
    (2, 4, 8),
    (4, 32, 16),
]

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "test_fp8_gemm_w8a16": {
        "mnk_factors": MINI_MNK_FACTORS[:1],
    },
    "test_fp8_gemm_per_tensor": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
    "test_fp8_gemm_per_channel": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
    "test_fp8_gemm_w8a16_per_channel": {
        "mnk_factors": MINI_MNK_FACTORS[:1],
    },
    "test_fp8_block_scaled_gemm": {
        "mnk_factors": MINI_MNK_FACTORS,
    },
}


def _expand_block_scales(scale: torch.Tensor, block_0: int, block_1: int,
                         dim_0: int, dim_1: int) -> torch.Tensor:
    dense = scale.repeat_interleave(block_0, dim=0).repeat_interleave(
        block_1, dim=1)
    return dense[:dim_0, :dim_1]


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("trans_wei", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16(fp8_dtype, dtype, trans_wei, is_mbk, batch,
                        mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn([batch, m, k], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    if trans_wei:
        weight = torch.ones([n, k], dtype=dtype).xpu()
    else:
        weight = torch.ones([k, n], dtype=dtype).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()
    scale_shape = None

    weight_fp8, _ = scaled_fp8_quant(weight, scale_wei, False, False,
                                     fp8_dtype, scale_shape)

    # reference fp16 gemm
    if trans_wei:
        output_ref = torch.matmul(input, weight.t())
    else:
        output_ref = torch.matmul(input, weight)

    # onednn fp8 gemm
    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8.transpose(0, 1) if trans_wei else weight_fp8,
        scale_wei,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_tensor(fp8_dtype, dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch * m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=dtype).xpu() / 10.0

    scale_src = (torch.ones(batch) * 4).xpu()
    scale_wei = (torch.ones(batch) * 4).xpu()

    input_fp8, _ = scaled_fp8_quant(input.reshape(-1, k),
                                    scale_src,
                                    False,
                                    False,
                                    fp8_dtype=fp8_dtype)

    weight_fp8, _ = scaled_fp8_quant(weight,
                                     scale_wei,
                                     False,
                                     False,
                                     fp8_dtype=fp8_dtype)

    # reference fp16 gemm
    output_ref = torch.matmul(input, weight.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        dtype,
        scale_src,
        scale_wei,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_per_channel(fp8_dtype, dtype, is_nt, batch, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn(
        [batch * m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=dtype).xpu() / 10.0

    input_fp8, scale_src_fp8 = scaled_fp8_quant(input.reshape(-1, k),
                                                use_per_token_if_dynamic=True,
                                                fp8_dtype=fp8_dtype)

    weight_fp8, scale_wei_fp8 = scaled_fp8_quant(weight,
                                                 use_per_token_if_dynamic=True,
                                                 fp8_dtype=fp8_dtype)

    # reference fp16 gemm
    output_ref = torch.matmul(input, weight.t())

    weight_fp8 = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8 = weight_fp8.contiguous()

    output_fp8 = fp8_gemm(
        input_fp8,
        weight_fp8,
        dtype,
        scale_src_fp8,
        scale_wei_fp8,
        torch.Tensor(),
    )

    torch.testing.assert_close(output_fp8, output_ref, atol=6e-2, rtol=6e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_nt", [True, False])
@pytest.mark.parametrize("is_mbk", [True, False])
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_gemm_w8a16_per_channel(fp8_dtype, dtype, is_nt, is_mbk, batch,
                                    mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors

    input = torch.randn([batch, m, k], dtype=dtype,
                        device=torch.device("xpu")) / 10.0
    weight = torch.randn([n, k], dtype=dtype).xpu() / 10.0

    weight_fp8, scale_wei_fp8 = scaled_fp8_quant(weight,
                                                 use_per_token_if_dynamic=True,
                                                 fp8_dtype=fp8_dtype)
    # scale_wei_fp8 is [n, 1], flatten to [n] for per-channel scale
    scale_wei_flat = scale_wei_fp8.flatten()

    # reference: dequantize weight then fp16/bf16 matmul
    weight_dequant = weight_fp8.to(dtype) * scale_wei_fp8.to(dtype)
    output_ref = torch.matmul(input, weight_dequant.t())

    weight_fp8_t = weight_fp8.transpose(0, 1)
    if is_nt:
        weight_fp8_t = weight_fp8_t.contiguous()

    if is_mbk:
        input = input.transpose(0, 1)
    output_fp8 = fp8_gemm_w8a16(
        input,
        weight_fp8_t,
        scale_wei_flat,
        torch.Tensor(),
    )
    output_fp8 = output_fp8.transpose(0, 1) if is_mbk else output_fp8

    torch.testing.assert_close(output_fp8, output_ref, atol=5e-2, rtol=5e-2)


@pytest.mark.parametrize("fp8_dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mnk_factors", MNK_FACTORS)
def test_fp8_block_scaled_gemm(fp8_dtype, dtype, mnk_factors):
    seed = 1234
    torch.manual_seed(seed)

    m, n, k = mnk_factors
    block_n = 128
    block_k = 128

    # The op expects A:[M,K], B:[N,K].
    input_fp = torch.randn([m, k], dtype=dtype, device=torch.device("xpu")) / 10.0
    weight_fp = torch.randn([n, k], dtype=dtype, device=torch.device("xpu")) / 10.0

    scale_shape_a = (m, (k + block_k - 1) // block_k)
    scale_shape_b = ((n + block_n - 1) // block_n, (k + block_k - 1) // block_k)

    a_scale_blocks = torch.rand(scale_shape_a, dtype=torch.float32,
                                device=torch.device("xpu")) * 0.02 + 0.01
    b_scale_blocks = torch.rand(scale_shape_b, dtype=torch.float32,
                                device=torch.device("xpu")) * 0.02 + 0.01

    # Quantize against dense elementwise scale maps derived from block scales.
    a_scale_dense = _expand_block_scales(a_scale_blocks, 1, block_k, m, k)
    b_scale_dense = _expand_block_scales(b_scale_blocks, block_n, block_k, n, k)

    input_q = (input_fp / a_scale_dense).to(fp8_dtype)
    weight_q = (weight_fp / b_scale_dense).to(fp8_dtype)

    output_fp8 = fp8_block_scaled_gemm(
        input_q,
        weight_q,
        a_scale_blocks,
        b_scale_blocks,
        block_n,
        block_k,
        dtype,
        None,
    )

    input_deq = input_q.to(torch.float32) * a_scale_dense
    weight_deq = weight_q.to(torch.float32) * b_scale_dense
    output_ref = torch.matmul(input_deq, weight_deq.t()).to(dtype)

    torch.testing.assert_close(output_fp8, output_ref, atol=8e-2, rtol=8e-2)
