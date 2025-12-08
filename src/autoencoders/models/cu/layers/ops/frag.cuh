#include "kittens.cuh"
using namespace kittens;

// #include "ops/vectorization.cuh"
// partial ops that need to be collected with a scan across the warp
#ifndef FRAG_CUH_INCLUDED
#define FRAG_CUH_INCLUDED

template<ducks::rt::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void frag_dot(ftype &dst, const T &A, const T &B) {
    #pragma unroll
    for(int i = 0; i < A.height; i++) {
        #pragma unroll
        for(int j = 0; j < A.width; j++) {
            #pragma unroll
            for(int k = 0; k < A.packed_per_tile; k++) {
                // dst += vector_sum(A.tiles[i][j].data[k] * B.tiles[i][j].data[k]);
                dst += A.tiles[i][j].data[k].x * B.tiles[i][j].data[k].x
                     + A.tiles[i][j].data[k].y * B.tiles[i][j].data[k].y;
            }
        }
    }
}




template<ducks::st::all T, ducks::rt::all U, int32_t c_in, int32_t k_in>
__device__ static inline void tile_to_flat(U &A_flat, const T* A) {
        
    const int y_tiles = A[0].height / k_in;
    const int x_tiles = A[0].width / k_in;

    #pragma unroll
    for (int c = 0; c < c_in; ++c) {
        #pragma unroll
        for(int j = 0; j < A[0].height; j++) {
            #pragma unroll
            for(int i = 0; i < A[0].width; i++) {
                // make 4x4 blocks from 16x16 input

                int n = (j / k_in) * x_tiles + (i / k_in);
                int l = c * k_in * k_in + (j % k_in) * k_in + (i % k_in);

                #pragma unroll
                for(int k = 0; k < A[0].packed_per_tile; k++) {
                    A_flat.tiles[n][l].data[k].x = A[c].tiles[j][i].data[k].x;
                    A_flat.tiles[n][l].data[k].y = A[c].tiles[j][i].data[k].y;
                }
            }
        }
    }
}

template<ducks::st::all T, ducks::rt::all U, int32_t c_in, int32_t k_in>
__device__ static inline void flat_to_tile(T* A, const U &A_flat) {
        
    const int y_tiles = A[0].height / k_in;
    const int x_tiles = A[0].width / k_in;

    #pragma unroll
    for (int c = 0; c < c_in; ++c) {
        #pragma unroll
        for(int j = 0; j < A[0].height; j++) {
            #pragma unroll
            for(int i = 0; i < A[0].width; i++) {
                // make 4x4 blocks from 16x16 input

                int n = (j / k_in) * x_tiles + (i / k_in);
                int l = c * k_in * k_in + (j % k_in) * k_in + (i % k_in);

                #pragma unroll
                for(int k = 0; k < A[0].packed_per_tile; k++) {
                    A[c].tiles[j][i].data[k].x = A_flat.tiles[n][l].data[k].x;
                    A[c].tiles[j][i].data[k].y = A_flat.tiles[n][l].data[k].y;
                }
            }
        }
    }
}





#endif // FRAG_CUH_INCLUDED