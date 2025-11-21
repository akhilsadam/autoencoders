#include "kittens.cuh"
using namespace kittens;

// template<typename F, typename T>
// static __device__ __forceinline__ void unary_wmap(T& WARP_y, const T& WARP_x) {
//     #pragma unroll
//     for (int i = 0; i < WARP_x.rows; i++) {
//         #pragma unroll
//         for (int j = 0; j < WARP_x.cols; j++) {
//             WARP_y(i,j) = F(WARP_x(i,j));
//         }
//     }
// }

// template<typename F, typename T>
// static __device__ __forceinline__ void binary_wmap(T& WARP_y, const T& WARP_x, const T& WARP_x2) {
//     #pragma unroll
//     for (int i = 0; i < WARP_x.rows; i++) {
//         #pragma unroll
//         for (int j = 0; j < WARP_x.cols; j++) {
//             WARP_y(i,j) = F(WARP_x(i,j), WARP_x2(i,j));
//         }
//     }
// }

// template<typename F, typename T>
// static __device__ __forceinline__ void ternary_wmap(T& WARP_y, const T& WARP_x, const T& WARP_x2, const T& WARP_x3) {
//     #pragma unroll
//     for (int i = 0; i < WARP_x.rows; i++) {
//         #pragma unroll
//         for (int j = 0; j < WARP_x.cols; j++) {
//             WARP_y(i,j) = F(WARP_x(i,j), WARP_x2(i,j), WARP_x3(i,j));
//         }
//     }
// }