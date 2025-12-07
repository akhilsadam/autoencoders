#include "kittens.cuh"
using namespace kittens;


// partial ops that need to be collected with a scan across the warp
#ifndef FRAG_CUH_INCLUDED
#define FRAG_CUH_INCLUDED

template<typename op, ducks::rt::all T, typename D> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void frag_dot(D &dst, const T &A, const T &B) {
    #pragma unroll
    for(int i = 0; i < dst.height; i++) {
        #pragma unroll
        for(int j = 0; j < dst.width; j++) {
            #pragma unroll
            for(int k = 0; k < dst.packed_per_tile; k++) {
                dst += (A.tiles[i][j].data[k] * B.tiles[i][j].data[k]);
            }
        }
    }
}





#endif // FRAG_CUH_INCLUDED