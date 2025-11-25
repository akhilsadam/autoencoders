#include "kittens.cuh"
using namespace kittens;

// relu device function
// Forward ReLU
struct relu_fwd {
    template<typename T>
    __device__ __forceinline__ static T op(const T& x) {
        T y;
        y.x = x.x > T(0) ? x.x : T(0);
        y.y = x.y > T(0) ? x.y : T(0);
        return y;
        // return x > T(0) ? x : T(0);
    }
};

// Specializations
template<>
__device__ __forceinline__ bf16_2 relu_fwd::op<bf16_2>(const bf16_2& x) {
    bf16_2 y;
    y.x = __hgt(x.x, __float2bfloat16_rn(0.0f)) ? x.x : __float2bfloat16_rn(0.0f);
    y.y = __hgt(x.y, __float2bfloat16_rn(0.0f)) ? x.y : __float2bfloat16_rn(0.0f);
    return y;
}

struct relu_bwd {
    template<typename T>
    __device__ __forceinline__ static T op(const T& g, const T& y) {
        // return y > T(0) ? g : T(0);
        T out;
        out.x = y.x > T(0) ? g.x : T(0);
        out.y = y.y > T(0) ? g.y : T(0);
        return out;
    }
};

// Specialization for bf16_2
template<>
__device__ __forceinline__ bf16_2 relu_bwd::op<bf16_2>(const bf16_2& g, const bf16_2& y) {
    bf16_2 out;
    out.x = __hgt(y.x, __float2bfloat16_rn(0.0f)) ? g.x : __float2bfloat16_rn(0.0f);
    out.y = __hgt(y.y, __float2bfloat16_rn(0.0f)) ? g.y : __float2bfloat16_rn(0.0f);
    return out;
}
