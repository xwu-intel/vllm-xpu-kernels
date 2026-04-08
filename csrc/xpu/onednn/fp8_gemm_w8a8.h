#pragma once

#include <c10/xpu/XPUStream.h>
#include <dnnl.hpp>
#include <torch/torch.h>

#include "../ops.h"
#include "onednn_ext.h"
#include "onednn_runtime.h"

namespace oneDNN {

static inline torch::Tensor dnnl_matmul_w8a8_block_fp8(
  const torch::Tensor& mat1,  // [m, k], fp8
  const torch::Tensor& mat2,  // [n, k], fp8
  const torch::Tensor& m1_sc, // [m, ceil(k / block_k)]
  const torch::Tensor& m2_sc, // [ceil(n / block_n), ceil(k / block_k)]
  int64_t block_n,
  int64_t block_k,
  std::optional<c10::ScalarType> out_dtype,
  const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat1.dim() == 2, "A must be 2D [M, K].");
  TORCH_CHECK(mat2.dim() == 2, "B must be 2D [N, K].");
  TORCH_CHECK(block_n > 0 && block_k > 0, "block_n and block_k must be > 0.");

  const int64_t m = mat1.size(0);
  const int64_t n = mat2.size(0);
  const int64_t k = mat1.size(1);

  TORCH_CHECK(mat2.size(1) == k, "A and B must have the same K dimension.");

  const int64_t k_tiles = (k + block_k - 1) / block_k;
  const int64_t n_tiles = (n + block_n - 1) / block_n;

  TORCH_CHECK(
    m1_sc.dim() == 2 && m1_sc.size(0) == m && m1_sc.size(1) == k_tiles,
    "A_scale must have shape [M, ceil(K / block_k)].");
  TORCH_CHECK(
    m2_sc.dim() == 2 && m2_sc.size(0) == n_tiles && m2_sc.size(1) == k_tiles,
    "B_scale must have shape [ceil(N / block_n), ceil(K / block_k)].");
  return fp8_gemm(
    mat1,
    mat2.transpose(0, 1),
    out_dtype,
    std::optional<torch::Tensor>(m1_sc),
    std::optional<torch::Tensor>(m2_sc),
    bias);
}

static inline void dnnl_matmul_w8a8_fp8(
    torch::Tensor& result,      // dst, [b, m, n]
    const torch::Tensor& mat1,  // src, [b, m, k]
    const torch::Tensor& mat2,  // quantized weight, [k, n] transpose
    bool is_nt,
    const std::optional<torch::Tensor>& bias,
    const torch::Tensor& m1_sc,
    const torch::Tensor& m2_sc) {
  auto src_sz = mat1.sizes();
  auto o_sz = result.sizes();

  const int m = std::reduce(
      src_sz.begin(), src_sz.end() - 1, 1, std::multiplies<int64_t>());
  const int n = o_sz.back();  // presume channel last format
  const int k = *(src_sz.end() - 1);

  bool src_block_scales = false;
  bool wei_block_scales = false;
  int64_t src_group_k = k;
  int64_t wei_group_k = k;
  int64_t wei_group_n = 1;
  torch::Tensor wei_scale_attr = m2_sc;

  if (m1_sc.numel() > 1) {
    TORCH_CHECK(
        m1_sc.dim() == 2 && m1_sc.size(0) == m,
        "A_scale must be scalar or 2D [M, num_k_tiles].");
    const int64_t src_k_tiles = m1_sc.size(1);
    TORCH_CHECK(src_k_tiles > 0, "A_scale K-tiles must be > 0.");
    src_group_k = (k + src_k_tiles - 1) / src_k_tiles;
    TORCH_CHECK(
        src_group_k > 0 && (k + src_group_k - 1) / src_group_k == src_k_tiles,
        "A_scale has incompatible K tile shape.");
    src_block_scales = (src_group_k != k);
  }

  if (m2_sc.numel() > 1) {
    if (m2_sc.dim() == 1) {
      TORCH_CHECK(
          m2_sc.size(0) == n,
          "B_scale 1D shape must be [N] for per-channel scaling.");
    } else {
      TORCH_CHECK(
          m2_sc.dim() == 2,
          "B_scale must be scalar, 1D [N], or 2D [num_n_tiles, num_k_tiles].");
      const int64_t wei_n_tiles = m2_sc.size(0);
      const int64_t wei_k_tiles = m2_sc.size(1);
      TORCH_CHECK(
          wei_n_tiles > 0 && wei_k_tiles > 0,
          "B_scale tile dimensions must be > 0.");

      wei_group_n = (n + wei_n_tiles - 1) / wei_n_tiles;
      wei_group_k = (k + wei_k_tiles - 1) / wei_k_tiles;
      TORCH_CHECK(
          wei_group_n > 0 &&
              (n + wei_group_n - 1) / wei_group_n == wei_n_tiles,
          "B_scale has incompatible N tile shape.");
      TORCH_CHECK(
          wei_group_k > 0 &&
              (k + wei_group_k - 1) / wei_group_k == wei_k_tiles,
          "B_scale has incompatible K tile shape.");

      wei_block_scales = true;
      // oneDNN weights dimensions are [K, N], so block scales must be [Kt, Nt]
      // when using mask (1<<0) + (1<<1).
      wei_scale_attr = m2_sc.transpose(0, 1).contiguous();
    }
  }

  // get joint dtypes
  joint_dtypes_t jd;
  auto in_dtype = mat1.scalar_type();
  auto wei_dtype = mat2.scalar_type();
  auto out_dtype = result.scalar_type();

  if (in_dtype == at::ScalarType::Float8_e5m2) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e5m2_bf16
                                               : joint_dtypes_t::f8_e5m2_f16;
  } else if (in_dtype == at::ScalarType::Float8_e4m3fn) {
    jd = out_dtype == at::ScalarType::BFloat16 ? joint_dtypes_t::f8_e4m3_bf16
                                               : joint_dtypes_t::f8_e4m3_f16;
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported data type for fp8 matmul: ", mat1.scalar_type());
  }

  // get bias type
  bias_type_t b_type = get_bias_type(bias, m, n);

  trans_type_t tt = trans_type_t::nn;
  if (is_nt) {
    // transpose mat2
    tt = trans_type_t::nt;
  }

  // get lda ldb and ldc
  auto mat1_strides = mat1.strides();
  int64_t leading_dim = -1;
  if (mat1.dim() == 2) {
    leading_dim = 0;
  } else if (mat1.dim() == 3) {
    leading_dim = mat1_strides[0] < mat1_strides[1] ? 0 : 1;
  } else {
    TORCH_CHECK(
        false, "Unsupported input dimension for fp8 matmul: ", mat1.dim());
  }
  int64_t lda = mat1_strides[leading_dim];
  int64_t ldb = mat2.strides()[mat2.dim() - 1] == 1
                    ? mat2.strides()[mat2.dim() - 2]
                    : mat2.strides()[mat2.dim() - 1];
  int64_t ldc = result.strides()[leading_dim];

  auto f_attr = [&](dnnl::primitive_attr& pattr) {
    pattr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    if (m1_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ 0,
          {},
          get_onednn_dtype(m1_sc));
      /* per tensor quant */
    } else {
      pattr.set_scales(
          DNNL_ARG_SRC,
          /* mask */ (1 << 0) + (1 << 1),
          {1, src_group_k},
          get_onednn_dtype(m1_sc));
      /* per token quant */
    }

    if (m2_sc.numel() == 1) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ 0,
          {},
          get_onednn_dtype(m2_sc));
      /* per tensor quant */
    } else if (wei_block_scales) {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 0) + (1 << 1),
          {wei_group_k, wei_group_n},
          get_onednn_dtype(wei_scale_attr));
      /* block-wise quant */
    } else {
      pattr.set_scales(
          DNNL_ARG_WEIGHTS,
          /* mask */ (1 << 1),
          {},
          get_onednn_dtype(m2_sc));
      /* per channel quant */
    }
  };

  int arg_off = 0;

  // ************************************************************
  // get device, engine, stream
  const int dev_id = c10::xpu::getCurrentXPUStream().device_index();
  at::Device curDevice = at::Device(at::kXPU, dev_id);
  auto engine = GpuEngineManager::Instance().get_engine(curDevice);

    int m1_sc_group_size = m1_sc.numel();
    int m2_sc_group_size = wei_scale_attr.numel();
    int sc_group_size =
      ((src_group_k & 0x3FF) << 22) | ((wei_group_k & 0x3FF) << 12) |
      ((wei_group_n & 0x3FF) << 2) |
      ((src_block_scales ? 1 : 0) << 1) | (wei_block_scales ? 1 : 0);
    sc_group_size ^= (m1_sc_group_size & 0x7FF) << 11;
    sc_group_size ^= (m2_sc_group_size & 0x7FF);
  auto& matmul_ext = matmul_primitive_create_and_cache(
      jd, tt, b_type, m, n, k, lda, ldb, ldc, dev_id, f_attr, sc_group_size);

  matmul_ext.set_attribute(
      arg_off++,
      DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
      wei_scale_attr.data_ptr(),
      [&]() {
        return make_onednn_memory(
        get_onednn_md(wei_scale_attr), engine, wei_scale_attr.data_ptr());
      });
  matmul_ext.set_attribute(
      arg_off++, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, m1_sc.data_ptr(), [&]() {
        return make_onednn_memory(
            get_onednn_md(m1_sc), engine, m1_sc.data_ptr());
      });

  std::vector<std::pair<int, void*>> arg_handles;
  arg_handles.reserve(8);

  arg_handles.emplace_back(DNNL_ARG_SRC, mat1.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_WEIGHTS, mat2.data_ptr());
  arg_handles.emplace_back(DNNL_ARG_DST, result.data_ptr());
  if (get_shape(b_type) != bias_shape_t::none) {
    arg_handles.emplace_back(DNNL_ARG_BIAS, bias.value().data_ptr());
  }
  int scratchpad_size = matmul_ext.get_scratchpad_size();
  torch::Tensor scratchpad_tensor = at::empty(
      {scratchpad_size}, mat1.options().dtype(at::kByte), c10::nullopt);
  arg_handles.emplace_back(DNNL_ARG_SCRATCHPAD, scratchpad_tensor.data_ptr());

  auto& strm = GpuStreamManager::Instance().get_stream();
  auto qfp8_matmul_event =
      matmul_ext.execute(strm, engine, std::move(arg_handles), arg_off);
}
}  // namespace oneDNN