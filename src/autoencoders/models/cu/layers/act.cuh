#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// relu device function
struct relu_fwd
{
    template<typename T>
    static __device__ __forceinline__ T op(T& x) {
        return x > T(0) ? x : T(0);
    }
};

struct relu_bwd
{
    template<typename T>
    static __device__ __forceinline__ T op(T& g, T& y) {
        return y > T(0) ? g : T(0);
    }
};