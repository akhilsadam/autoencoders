#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// relu device function
template<typename T>
static __device__ __forceinline__ T relu_fwd(const T& x) {
    return x > T(0) ? x : T(0);
}

template<typename T>
static __device__ __forceinline__ T relu_bwd(const T& g, const T& y) {
    return y > T(0) ? g : T(0);
}


// Specialization for bf16_2
template<>
__device__ __forceinline__ bf16_2 relu_fwd<bf16_2>(const bf16_2& x) {
    bf16_2 y;
    y.x = __hgt(x.x, __float2bfloat16_rn(0.0f)) ? x.x : __float2bfloat16_rn(0.0f);
    y.y = __hgt(x.y, __float2bfloat16_rn(0.0f)) ? x.y : __float2bfloat16_rn(0.0f);
    return y;
}

template<>
__device__ __forceinline__ bf16_2 relu_bwd<bf16_2>(const bf16_2& g, const bf16_2& y) {
    bf16_2 grad;
    grad.x = __hgt(y.x, __float2bfloat16_rn(0.0f)) ? g.x : __float2bfloat16_rn(0.0f);
    grad.y = __hgt(y.y, __float2bfloat16_rn(0.0f)) ? g.y : __float2bfloat16_rn(0.0f);
    return grad;
}