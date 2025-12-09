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

template<ducks::rt::all T> // T2, w, h can be inferred from dst as long as op is specialized
__device__ static inline void debug_mult(T &A, const T &B, ftype &w) {
    if (threadIdx.x==0)
    {
        #pragma unroll
        for(int i = 0; i < A.height; i++) {
            #pragma unroll
            for(int j = 0; j < A.width; j++) {
                #pragma unroll
                for(int k = 0; k < A.packed_per_tile; k++) {
                    // dst += vector_sum(A.tiles[i][j].data[k] * B.tiles[i][j].data[k]);
                    A.tiles[i][j].data[k].x = B.tiles[i][j].data[k].x * w;
                    A.tiles[i][j].data[k].y = B.tiles[i][j].data[k].y * w;
                }
            }
        }
    } // neeed to fix, still wrong
}



template<int32_t c_in, int32_t k_in, ducks::rt::all T, ducks::rt::all U>
__device__ static inline void tile_to_flat(U &A_flat, const T (&A)[c_in]) {
        
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

template<int32_t c_in, int32_t k_in, ducks::rt::all T, ducks::rt::all U>
__device__ static inline void cast_tile_to_flat(U &A_flat, const T (&A)[c_in]) {
        
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
                    A_flat.tiles[n][l].data[k].x = __float2bfloat16(A[c].tiles[j][i].data[k].x);
                    A_flat.tiles[n][l].data[k].y = __float2bfloat16(A[c].tiles[j][i].data[k].y);
                }
            }
        }
    }
}

template<int32_t c_in, int32_t k_in, ducks::rt::all T, ducks::rt::all U>
__device__ static inline void flat_to_tile(T (&A)[c_in], const U &A_flat) {
        
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



// template<ducks::rt::all A,  ducks::st::all T>
// __device__ static inline void to_st(T &S, ftype &w) {
//     constexpr int n = A.height * A.width

//     // float4 loads
//     constexpr int words = (A.packed_per_tile * 2 * sizeof(ftype) + sizeof(float4) - 1) / sizeof(float4);

//     for (int u = threadIdx.x; u < A.height * A.width; u += blockDim.x) 
//     {
//         int j = u / A.width;
//         int i = u % A.width;

//         // load vectorized
//         float4 vec_load[words] = {};

//             #pragma unroll
//             for(int k = 0; k < A.packed_per_tile; k++) 
//             {
//                 A.tiles[j][i].data[k].x = B.tiles[j][i].data[k].x * w;
//                 A.tiles[j][i].data[k].y = B.tiles[j][i].data[k].y * w;
//             }
//     }
// }


template<typename qtype, int cols, ducks::st::all ST, int N_THREADS=kittens::WARP_THREADS>
__device__ static inline void aligned_load_to_st(ST &dst, qtype* src_ptr) {

    using T = typename ST::dtype;
    const int row_stride = cols; // src.template stride<axis>(); // axis is 2 by default, so cols..
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height*ST::width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up
    constexpr int total_rows = ST::height*ST::width;

    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % dst.cols;

        float4 tmp;
        move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
        move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
    }
}



template<typename qtype, int cols, ducks::st::all ST, int N_THREADS=kittens::WARP_THREADS>
__device__ static inline void aligned_store_to_gl(qtype* dst_ptr, const ST &src) {
    using T = typename ST::dtype;
    const int row_stride = cols; // dst.template stride<axis>(); // same reasoning as above
    // we can handle this many rows each time we run a memcpy_async
    constexpr int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::height*ST::width * kittens::TILE_ROW_DIM<T>*kittens::TILE_COL_DIM<T> + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy); // round up

    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    int laneid = threadIdx.x % N_THREADS;

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int load_idx = i * N_THREADS + laneid;
        
        int row = load_idx / memcpy_per_row;
        int col = (load_idx*elem_per_memcpy) % src.cols;
    
        float4 tmp;
        move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
        move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
    }
}


template<typename op, ducks::rt::all T>
__device__ static inline void inplace_bin_map(T &lhs, const T &rhs) {
    #pragma unroll
    for(int i = 0; i < lhs.height; i++) {
        #pragma unroll
        for(int j = 0; j < lhs.width; j++) {
            #pragma unroll
            for(int k = 0; k < lhs.packed_per_tile; k++) {
                op::template op<typename T::dtype>(lhs.tiles[i][j].data[k], rhs.tiles[i][j].data[k]);
            }
        }
    }
}

template<typename op, ducks::rt::all R, ducks::st::all T>
__device__ static inline void inplace_bin_map_st(T &lhs, const T &rhs) {
    R l, r;
    load(l, lhs);
    load(r, rhs);
    inplace_bin_map<op>(l, r);
    store(lhs, l);
}


#endif // FRAG_CUH_INCLUDED