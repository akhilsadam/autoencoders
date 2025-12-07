#include "kittens.cuh"
using namespace kittens;

constexpr unsigned int warp_size = kittens::WARP_THREADS;
constexpr unsigned int warp_size_p2 = 5;

namespace scan 
{

template<typename Data>
__device__ __forceinline__ Data shfl_up_custom(Data val, int offset) {
    static_assert(sizeof(Data) % 4 == 0, "Data type size must be a multiple of 4 bytes for shuffle");
    __builtin_assume_aligned(&val, 4);
    
    constexpr int num_words = sizeof(Data) / 4; // number of 32-bit words
    uint32_t* words = reinterpret_cast<uint32_t*>(&val); // data as array of uint32_t
    
    // shuffled output object
    Data shfl_val;
    uint32_t* result_words = reinterpret_cast<uint32_t*>(&shfl_val);
    
    // shuffle words
    #pragma unroll
    for (int i = 0; i < num_words; i++) {
        result_words[i] = __shfl_up_sync(0xFFFFFFFF, words[i], offset);
    }
    
    return shfl_val;
}

template <typename T>
__device__ __forceinline__ void frag_collect(T& val) {
    uint8_t lane = threadIdx.x % warp_size;
    #pragma unroll
    for (int p2 = 0; p2 < warp_size_p2; p2++) {
        int offset = 1 << p2; // 1, 2, 4, 8, 16
        T shfl_val = shfl_up_custom(val, offset);
        if (lane >= offset) {
            val = shfl_val + val;
        }
    }
}

template <typename T>
__device__ __forceinline__ void atomic_store(T& refval, T& val) {
    uint8_t lane = threadIdx.x % warp_size;
    if (lane == warp_size - 1) {
        atomicAdd(&refval, val);
    }
}

// template <
//     typename T,
//     uint32_t ks_steps,
//     uint32_t bound,
//     uint32_t _stride,
//     uint32_t _offset>
// __device__ __forceinline__ void warp_shuffle(uint8_t lane, T *src) {
//     constexpr auto n_steps = (bound + warp_size - 1) / warp_size;

//     T sum = 0;
//     #pragma unroll
//     for (uint32_t i = 0; i < n_steps; i++) {
//         auto index = lane * _stride + _offset + i * warp_size;

//         T val = src[index];

//         #pragma unroll
//         for (int p2 = 0; p2 < ks_steps; p2++) {
//             int offset = 1 << p2; // 1, 2, 4, 8, 16
//             T shfl_val = __shfl_up_sync(0xFFFFFFFF, val, offset);
//             if (lane >= offset) {
//                 val = shfl_val + val;
//             }
//         }

//         if (index < bound) {
//             src[index] = sum + val;
//         }

//         sum = src[(warp_size - 1) * _stride + _offset + i * warp_size];
//     }
// }

// template <typename T, uint32_t reduction_width>
// __device__ __forceinline__ void warp_add(uint8_t lane, T *src) {
//     T base = src[-1];
//     #pragma unroll
//     for (int i = lane; i < reduction_width; i += warp_size) {
//         src[i] = i < reduction_width - 1 ? (base + src[i]) : src[i];
//         // no need for base sum
//     }
// }

// template <typename T, uint32_t reduction_width>
// __device__ __forceinline__ void warp_scan_one_level(T *shared_mem) {
//     auto tid = threadIdx.x;
//     auto warp = tid / warp_size;
//     auto lane = tid % warp_size;

//     // only first warp reduce
//     if (warp == 0) {
//         warp_shuffle<T, warp_size_p2, reduction_width, 1, 0>(
//             lane,
//             shared_mem + warp * reduction_width);
//     }
//     __syncthreads();
// }

// template <typename T, uint32_t reduction_width, uint32_t block_width>
// __device__ __forceinline__ void warp_scan_two_level(T *shared_mem) {
//     auto tid = threadIdx.x;
//     auto warp = tid / warp_size;
//     auto lane = tid % warp_size;

//     // warp-level downsweep
//     warp_shuffle<T, warp_size_p2, reduction_width, 1, 0>(
//         lane,
//         shared_mem + warp * reduction_width); // reduce by reduction_width
//     __syncthreads();

//     // only first warp reduce
//     if (warp == 0) {
//         warp_shuffle<T, warp_size_p2, block_width, reduction_width, reduction_width - 1>(
//             lane,
//             shared_mem); // reduce by 32
//     }
//     __syncthreads();

//     // warp-level upsweep
//     warp_add<T, reduction_width>(lane, shared_mem + warp * reduction_width);
//     __syncthreads();
// }

// template <typename T, uint32_t reduction_width, uint8_t factor>
// __device__ __forceinline__ void warp_scan_auto(T *shared_mem) {
//     if constexpr (reduction_width <= 4 * warp_size) {
//         warp_scan_one_level<T, reduction_width>(shared_mem);
//     } else {
//         constexpr auto warp_reduction_width = warp_size * factor;
//         constexpr auto block_width =
//             (reduction_width + warp_reduction_width - 1) / warp_reduction_width;
//         warp_scan_two_level<T, warp_reduction_width, block_width>(shared_mem);
//     }
// }

} 

// namespace scan