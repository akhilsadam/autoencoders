#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// relu device function
__device__ __forceinline__ float relu_fwd(float x) {
    return x > 0.0f ? x : 0.0f;
}

__device__ __forceinline__ float relu_bwd(float g, float y) {
    return y > 0.0f ? g : 0.0f;
}