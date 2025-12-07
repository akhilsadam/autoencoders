
#include "ops/vectorization.cuh"

#ifndef OPS_BASIC_CUH_INCLUDED
#define OPS_BASIC_CUH_INCLUDED

struct scale {
    template<class T>
    __device__ __forceinline__ static T op(T& out, const T& in, const ftype& weight) 
    { return in * weight; }
};

#endif // OPS_BASIC_CUH_INCLUDED