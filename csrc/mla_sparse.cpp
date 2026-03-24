#include <sycl/sycl.hpp>

#include <cmath>
#include <limits>
#include <tuple>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"

namespace {

constexpr float kLogE2 = 0.6931471805599453f;

template <typename cache_t, Fp8KVCacheDataType kv_dt>
inline float load_kv_value(cache_t raw, float scale) {
	if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
		return static_cast<float>(raw);
	} else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
		auto fp8_val = sycl::bit_cast<at::Float8_e5m2>(raw);
		return static_cast<float>(fp8_val) * scale;
	} else {
		auto fp8_val = sycl::bit_cast<at::Float8_e4m3fn>(raw);
		return static_cast<float>(fp8_val) * scale;
	}
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class mla_sparse_decode_kernel {
 public:
	mla_sparse_decode_kernel(
			const scalar_t* q_ptr,
			const cache_t* kv_ptr,
			const int32_t* indices_ptr,
			scalar_t* out_ptr,
			float* max_logits_ptr,
			float* lse_ptr,
			int64_t num_tokens,
			int64_t num_heads,
			int64_t dim_qk,
			int64_t d_v,
			int64_t topk,
			int64_t q_stride_token,
			int64_t q_stride_head,
			int64_t q_stride_d,
			int64_t kv_stride_token,
			int64_t kv_stride_head,
			int64_t kv_stride_d,
			int64_t indices_stride_token,
			int64_t indices_stride_head,
			int64_t indices_stride_topk,
			int64_t out_stride_token,
			int64_t out_stride_head,
			int64_t out_stride_d,
			int64_t logits_stride_token,
			int64_t logits_stride_head,
			float sm_scale,
			float k_scale,
			float v_scale)
			: q_ptr_(q_ptr),
				kv_ptr_(kv_ptr),
				indices_ptr_(indices_ptr),
				out_ptr_(out_ptr),
				max_logits_ptr_(max_logits_ptr),
				lse_ptr_(lse_ptr),
				num_tokens_(num_tokens),
				num_heads_(num_heads),
				dim_qk_(dim_qk),
				d_v_(d_v),
				topk_(topk),
				q_stride_token_(q_stride_token),
				q_stride_head_(q_stride_head),
				q_stride_d_(q_stride_d),
				kv_stride_token_(kv_stride_token),
				kv_stride_head_(kv_stride_head),
				kv_stride_d_(kv_stride_d),
				indices_stride_token_(indices_stride_token),
				indices_stride_head_(indices_stride_head),
				indices_stride_topk_(indices_stride_topk),
				out_stride_token_(out_stride_token),
				out_stride_head_(out_stride_head),
				out_stride_d_(out_stride_d),
				logits_stride_token_(logits_stride_token),
				logits_stride_head_(logits_stride_head),
				sm_scale_(sm_scale),
				k_scale_(k_scale),
				v_scale_(v_scale) {}

	void operator()(const sycl::nd_item<2>& item) const {
		const int64_t token_idx = item.get_global_id(0);
		const int64_t head_idx = item.get_global_id(1);
		if (token_idx >= num_tokens_ || head_idx >= num_heads_) {
			return;
		}

		float e_max = -std::numeric_limits<float>::infinity();
		int valid_cnt = 0;

		// Pass 1: find max logit.
		for (int64_t n = 0; n < topk_; ++n) {
			const int64_t idx_offset = token_idx * indices_stride_token_ +
																 0 * indices_stride_head_ +
																 n * indices_stride_topk_;
			const int32_t kv_idx = indices_ptr_[idx_offset];
			if (kv_idx < 0 || kv_idx >= kv_stride_token_) {
				continue;
			}

			float qk = 0.0f;
			for (int64_t d = 0; d < dim_qk_; ++d) {
				const int64_t q_off = token_idx * q_stride_token_ +
															head_idx * q_stride_head_ + d * q_stride_d_;
				const int64_t k_off = kv_idx * kv_stride_token_ +
															0 * kv_stride_head_ + d * kv_stride_d_;
				const float qv = static_cast<float>(q_ptr_[q_off]);
				const float kv = load_kv_value<cache_t, kv_dt>(kv_ptr_[k_off], k_scale_);
				qk += qv * kv;
			}
			qk *= sm_scale_;
			e_max = sycl::fmax(e_max, qk);
			++valid_cnt;
		}

		const int64_t logits_off = token_idx * logits_stride_token_ +
															 head_idx * logits_stride_head_;
		max_logits_ptr_[logits_off] = e_max * kLogE2;

		if (valid_cnt == 0) {
			for (int64_t d = 0; d < d_v_; ++d) {
				const int64_t out_off = token_idx * out_stride_token_ +
																head_idx * out_stride_head_ + d * out_stride_d_;
				out_ptr_[out_off] = static_cast<scalar_t>(0.0f);
			}
			lse_ptr_[logits_off] = std::numeric_limits<float>::infinity();
			return;
		}

		float e_sum = 0.0f;

		// Pass 2: denominator.
		for (int64_t n = 0; n < topk_; ++n) {
			const int64_t idx_offset = token_idx * indices_stride_token_ +
																 0 * indices_stride_head_ +
																 n * indices_stride_topk_;
			const int32_t kv_idx = indices_ptr_[idx_offset];
			if (kv_idx < 0 || kv_idx >= kv_stride_token_) {
				continue;
			}

			float qk = 0.0f;
			for (int64_t d = 0; d < dim_qk_; ++d) {
				const int64_t q_off = token_idx * q_stride_token_ +
															head_idx * q_stride_head_ + d * q_stride_d_;
				const int64_t k_off = kv_idx * kv_stride_token_ +
															0 * kv_stride_head_ + d * kv_stride_d_;
				const float qv = static_cast<float>(q_ptr_[q_off]);
				const float kv = load_kv_value<cache_t, kv_dt>(kv_ptr_[k_off], k_scale_);
				qk += qv * kv;
			}
			qk *= sm_scale_;
			e_sum += sycl::exp2(qk - e_max);
		}

		// Pass 3: numerator for each output dim.
		for (int64_t d = 0; d < d_v_; ++d) {
			float acc = 0.0f;
			for (int64_t n = 0; n < topk_; ++n) {
				const int64_t idx_offset = token_idx * indices_stride_token_ +
																	 0 * indices_stride_head_ +
																	 n * indices_stride_topk_;
				const int32_t kv_idx = indices_ptr_[idx_offset];
				if (kv_idx < 0 || kv_idx >= kv_stride_token_) {
					continue;
				}

				float qk = 0.0f;
				for (int64_t qd = 0; qd < dim_qk_; ++qd) {
					const int64_t q_off = token_idx * q_stride_token_ +
																head_idx * q_stride_head_ + qd * q_stride_d_;
					const int64_t k_off = kv_idx * kv_stride_token_ +
																0 * kv_stride_head_ + qd * kv_stride_d_;
					const float qv = static_cast<float>(q_ptr_[q_off]);
					const float kv =
							load_kv_value<cache_t, kv_dt>(kv_ptr_[k_off], k_scale_);
					qk += qv * kv;
				}
				qk *= sm_scale_;
				const float p = sycl::exp2(qk - e_max);
				const int64_t v_off = kv_idx * kv_stride_token_ +
															0 * kv_stride_head_ + d * kv_stride_d_;
				const float vv = load_kv_value<cache_t, kv_dt>(kv_ptr_[v_off], v_scale_);
				acc += p * vv;
			}
			const float out_val = acc / e_sum;
			const int64_t out_off = token_idx * out_stride_token_ +
															head_idx * out_stride_head_ + d * out_stride_d_;
			out_ptr_[out_off] = static_cast<scalar_t>(out_val);
		}

		lse_ptr_[logits_off] = e_max * kLogE2 + sycl::log2(e_sum) * kLogE2;
	}

 private:
	const scalar_t* q_ptr_;
	const cache_t* kv_ptr_;
	const int32_t* indices_ptr_;
	scalar_t* out_ptr_;
	float* max_logits_ptr_;
	float* lse_ptr_;
	const int64_t num_tokens_;
	const int64_t num_heads_;
	const int64_t dim_qk_;
	const int64_t d_v_;
	const int64_t topk_;
	const int64_t q_stride_token_;
	const int64_t q_stride_head_;
	const int64_t q_stride_d_;
	const int64_t kv_stride_token_;
	const int64_t kv_stride_head_;
	const int64_t kv_stride_d_;
	const int64_t indices_stride_token_;
	const int64_t indices_stride_head_;
	const int64_t indices_stride_topk_;
	const int64_t out_stride_token_;
	const int64_t out_stride_head_;
	const int64_t out_stride_d_;
	const int64_t logits_stride_token_;
	const int64_t logits_stride_head_;
	const float sm_scale_;
	const float k_scale_;
	const float v_scale_;
};

}  // namespace

#define CALL_MLA_SPARSE_DECODE(SCALAR_T, CACHE_T, KV_DTYPE)                    \
	queue.submit([&](sycl::handler& cgh) {                                       \
		cgh.parallel_for(                                                           \
				sycl::nd_range<2>(global, local),                                       \
				mla_sparse_decode_kernel<SCALAR_T, CACHE_T, KV_DTYPE>(                  \
						reinterpret_cast<SCALAR_T*>(q.data_ptr()),                          \
						reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
						indices.data_ptr<int32_t>(),                                        \
						reinterpret_cast<SCALAR_T*>(out.data_ptr()),                        \
						max_logits.data_ptr<float>(),                                       \
						lse.data_ptr<float>(),                                              \
						num_tokens,                                                         \
						num_heads,                                                          \
						dim_qk,                                                             \
						d_v,                                                                \
						topk,                                                               \
						q_stride_token,                                                     \
						q_stride_head,                                                      \
						q_stride_d,                                                         \
						kv_stride_token,                                                    \
						kv_stride_head,                                                     \
						kv_stride_d,                                                        \
						indices_stride_token,                                               \
						indices_stride_head,                                                \
						indices_stride_topk,                                                \
						out_stride_token,                                                   \
						out_stride_head,                                                    \
						out_stride_d,                                                       \
						logits_stride_token,                                                \
						logits_stride_head,                                                 \
						static_cast<float>(sm_scale),                                       \
						static_cast<float>(k_scale_scalar),                                 \
						static_cast<float>(v_scale_scalar)));                               \
	});

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mla_sparse_decode(
		torch::Tensor& q,
		torch::Tensor& kv_cache,
		torch::Tensor& indices,
		double sm_scale,
		int64_t d_v,
		torch::Tensor& k_scale,
		torch::Tensor& v_scale,
		const std::string& kv_cache_dtype) {
	CHECK_DEVICE(q);
	CHECK_DEVICE(kv_cache);
	CHECK_DEVICE(indices);
	TORCH_CHECK(indices.dtype() == at::kInt, "indices must be int32");
	TORCH_CHECK(q.dim() == 3, "q must be [num_tokens, num_heads, dim_qk]");
	TORCH_CHECK(
			kv_cache.dim() == 3,
			"kv_cache must be [num_kv_tokens, num_kv_heads, dim_qk]");
	TORCH_CHECK(indices.dim() == 3, "indices must be [num_tokens, num_kv_heads, topk]");
	TORCH_CHECK(kv_cache.size(1) == 1, "only num_kv_heads=1 is supported");
	TORCH_CHECK(indices.size(1) == 1, "only num_kv_heads=1 is supported");

	const int64_t num_tokens = q.size(0);
	const int64_t num_heads = q.size(1);
	const int64_t dim_qk = q.size(2);
	const int64_t topk = indices.size(2);
	TORCH_CHECK(dim_qk == kv_cache.size(2), "q and kv_cache dim_qk mismatch");
	TORCH_CHECK(d_v > 0 && d_v <= dim_qk, "invalid d_v");

	auto out = torch::zeros({num_tokens, num_heads, d_v}, q.options());
	auto max_logits = torch::zeros({num_tokens, num_heads}, q.options().dtype(torch::kFloat));
	auto lse = torch::zeros({num_tokens, num_heads}, q.options().dtype(torch::kFloat));

	const int64_t q_stride_token = q.stride(0);
	const int64_t q_stride_head = q.stride(1);
	const int64_t q_stride_d = q.stride(2);

	const int64_t kv_stride_token = kv_cache.stride(0);
	const int64_t kv_stride_head = kv_cache.stride(1);
	const int64_t kv_stride_d = kv_cache.stride(2);

	const int64_t indices_stride_token = indices.stride(0);
	const int64_t indices_stride_head = indices.stride(1);
	const int64_t indices_stride_topk = indices.stride(2);

	const int64_t out_stride_token = out.stride(0);
	const int64_t out_stride_head = out.stride(1);
	const int64_t out_stride_d = out.stride(2);

	const int64_t logits_stride_token = max_logits.stride(0);
	const int64_t logits_stride_head = max_logits.stride(1);

	const double k_scale_scalar = k_scale.item<double>();
	const double v_scale_scalar = v_scale.item<double>();

	const at::DeviceGuard device_guard(q.device());
	auto& queue = vllm::xpu::vllmGetQueue();

	sycl::range<2> global(
			static_cast<size_t>(num_tokens), static_cast<size_t>(num_heads));
	sycl::range<2> local(1, 1);

	DISPATCH_BY_KV_CACHE_DTYPE(
			q.scalar_type(), kv_cache_dtype, CALL_MLA_SPARSE_DECODE);

	return {out, max_logits, lse};
}

