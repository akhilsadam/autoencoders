#include "kittens.cuh"
using namespace kittens;

#ifndef TILE_CUH_INCLUDED
#include "tile.cuh"
#endif

template<int N>
struct mse_bwd {
    template<typename T>
    __device__ __forceinline__ static T op(const T& y_hat, const T& y) {
        // assume T is a vector 2 type
        T diff;
        diff.x = (y_hat.x - y.x) / N;
        diff.y = (y_hat.y - y.y) / N;
        return diff; 

        // 1/2N * (y_hat - y).T @ (y_hat - y) derivative
        // is (y_hat - y) / N
    }
};

struct MSELoss{
    template<class L>
    static __device__ __forceinline__ void op(
        const L::shmem_array* y_hat,
        const L::shmem_array* y,
        L::shmem_array* grad_y
    ) {
        // BCHW again, and we work on [WT.y, WT.x, C] arrays of type ST[Wp.y, Wp.x].

        L::reg_wp WARP_y, WARP_y_hat, WARP_grad_y;

        for (int wave = 0; wave < L::warpwaves; ++wave) 
        {
            int2 ij = L::warptile_ixy(wave);
            for (int c = 0; c < L::C; ++c) {

                load(WARP_y_hat, this->y_hat[0][ij.y][ij.x][c]);
                load(WARP_y, this->y[0][ij.y][ij.x][c]); 
                // TODO make it additive for multiple losses
                bin_map<mse_bwd<L::N>>(WARP_grad_y, WARP_y_hat, WARP_y);
                store(this->grad_y[0][ij.y][ij.x][c], WARP_grad_y);

                __syncwarp();
            }
        }
    }
};
