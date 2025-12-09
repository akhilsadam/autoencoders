#include "kittens.cuh"
using namespace kittens;

// #include "ops/vectorization.cuh"
// partial ops that need to be collected with a scan across the warp
#ifndef ACT_CUH_INCLUDED
#define ACT_CUH_INCLUDED


template<float w,ducks::rt::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void act_sine(T &A, const T &X) {
    #pragma unroll
    for(int i = 0; i < A.height; i++) {
        #pragma unroll
        for(int j = 0; j < A.width; j++) {
            #pragma unroll
            for(int k = 0; k < A.packed_per_tile; k++) {
                // sine(x)
                A.tiles[i][j].data[k].x = sinf(w * X.tiles[i][j].data[k].x);
                A.tiles[i][j].data[k].y = sinf(w * X.tiles[i][j].data[k].y);
            }
        }
    }
}

template<float w, ducks::rt::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void act_sine_bwd(T &A, const T &X, const T &GY) {
    #pragma unroll
    for(int i = 0; i < A.height; i++) {
        #pragma unroll
        for(int j = 0; j < A.width; j++) {
            #pragma unroll
            for(int k = 0; k < A.packed_per_tile; k++) {
                // sine(x)
                A.tiles[i][j].data[k].x = w * cosf(w * X.tiles[i][j].data[k].x) * GY.tiles[i][j].data[k].x;
                A.tiles[i][j].data[k].y = w * cosf(w * X.tiles[i][j].data[k].y) * GY.tiles[i][j].data[k].y;
            }
        }
    }
}



#endif