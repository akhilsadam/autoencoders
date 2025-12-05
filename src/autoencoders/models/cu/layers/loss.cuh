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

template<class Layout>
__device__ __forceinline__ void MSE(
    int32_t channels,
    int32_t batch,
    const shmem<Layout.By, Layout.Bx>* y_hat,
    const shmem<Layout.By, Layout.Bx>* y,
    shmem<Layout.By, Layout.Bx>* grad_y
) {
    // BCHW again, and we work on [C] arrays of type ST[H,W].

    rt<ftype, Layout.Wy, Layout.Wx> WARP_y, WARP_y_hat, WARP_grad_y; // register tiles
    
    for(int32_t chan = 0; chan < channels; chan++) {
        for (int32_t wave = 0; wave < Layout.warpwaves; wave++) {

            int2 p = Layout.warptile_xy(wave);
            coord<> idx(batch, chan, p.y, p.x);

            load(WARP_y_hat, y_hat, idx);
            load(WARP_y, y, idx);
            bin_map<mse_bwd<Layout.N>>(WARP_grad_y, WARP_y_hat, WARP_y);
            store(grad_y, WARP_grad_y, idx);
        }
    }

    __syncthreads();
}
