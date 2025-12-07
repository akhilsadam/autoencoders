#include "kittens.cuh"
using namespace kittens;


// partial ops that need to be collected with a scan across the warp
#ifndef FRAG_CUH_INCLUDED
#define FRAG_CUH_INCLUDED

template<typename op, ducks::st::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void frag_dot(typename T::dtype &dst, const T &A, const T &B) {
    #pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst += A.data[i] * B.data[i];
    }
}





#endif // FRAG_CUH_INCLUDED