#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// relu device function
template<typename T>
__device__ __forceinline__ T relu_fwd(T x) {
    return x > T(0) ? x : T(0);
}

template<typename T>
__device__ __forceinline__ T relu_bwd(T g, T y) {
    return y > T(0) ? g : T(0);
}