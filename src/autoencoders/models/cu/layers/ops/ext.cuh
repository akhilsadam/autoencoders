#include "kittens.cuh"
using namespace kittens;

// extensions for kittens operations

/**
 * @brief Performs a uniform ternary operation on a tile with a scalar parameter.
 * 
 * This function applies a given ternary operation to each element of the source tile and a scalar parameter, then stores the result in the destination tile.
 * The operation is applied independently to each element, treating the scalar parameter as the second operand for each operation.
 * 
 * @tparam op The ternary operation to be applied. Must be specialized to support operation on the data type of T and the scalar parameter.
 * @tparam T The type of the tile. Must satisfy the `ducks::st::all` concept.
 * @param[out] dst The destination tile where the results are stored.
 * @param[in] src The source tile to which the ternary operation is applied.
 * @param[in] param The scalar parameter to be used as the third operand in the ternary operation.
 */
template<typename op, ducks::st::all T>
__device__ static inline void tern_map(T &dst, const T &src, const T &src2, const typename T::dtype &param) {
    #pragma unroll
    for(int i = laneid(); i < dst.num_elements; i += GROUP_THREADS) {
        dst.data[i] = op::template op<typename T::dtype>(src.data[i], src2.data[i], param);
    }
}