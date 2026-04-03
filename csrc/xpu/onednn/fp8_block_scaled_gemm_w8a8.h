#pragma once

#include <torch/torch.h>

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

  // Expand block scales into dense per-element scales.
  auto m1_sc_dense = m1_sc.repeat_interleave(block_k, /*dim=*/1);
  if (m1_sc_dense.size(1) > k) {
    m1_sc_dense = m1_sc_dense.narrow(/*dim=*/1, /*start=*/0, /*length=*/k);
  }

  auto m2_sc_dense =
      m2_sc.repeat_interleave(block_n, /*dim=*/0).repeat_interleave(block_k, /*dim=*/1);
  if (m2_sc_dense.size(0) > n) {
    m2_sc_dense = m2_sc_dense.narrow(/*dim=*/0, /*start=*/0, /*length=*/n);
  }
  if (m2_sc_dense.size(1) > k) {
    m2_sc_dense = m2_sc_dense.narrow(/*dim=*/1, /*start=*/0, /*length=*/k);
  }

  // Dequantize and invoke XPU GEMM. On XPU, torch.matmul maps to oneDNN-backed
  // kernels for these dtypes.
  auto a_deq = mat1.to(torch::kFloat) * m1_sc_dense.to(torch::kFloat);
  auto b_deq = mat2.to(torch::kFloat) * m2_sc_dense.to(torch::kFloat);

  auto out = at::matmul(a_deq, b_deq.transpose(0, 1));

  const auto out_dtype_ = out_dtype.value_or(torch::kHalf);
  out = out.to(out_dtype_);

  if (bias.has_value() && bias.value().defined() && bias.value().numel() > 0) {
    TORCH_CHECK(
        bias.value().numel() == n,
        "bias must be 1D with N elements if provided.");
    out = out + bias.value();
  }

  return out;
}

}  // namespace oneDNN
