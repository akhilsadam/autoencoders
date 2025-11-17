#include "kittens.cuh"
#include "ops/warp/warp.cuh" // for load/store/map_xy
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"

struct fwd_data {
    tiled_layout x, y;
    dim3 grid()  { return dim3(x.batch()); } // b
    dim3 block() { return dim3(x.cols(), x.rows(), x.depth()); } // whc
};
struct bwd_data {
    tiled_layout grad_y, y, grad_x;
    dim3 grid()  { return dim3(grad_y.batch()); } // b
    dim3 block() { return dim3(grad_y.cols(), grad_y.rows(), grad_y.depth()); } // whc
};

__global__ void relu_fwd_kernel(const __grid_constant__ fwd_data g) {
    rt_reg<float, Tile::W.y, Tile::W.x> WARP_y, WARP_x; // register tiles
    load(WARP_x, g.x, blockIdx.x); // load to register tile
    map_xy(WARP_y, WARP_x, relu_fwd);  // apply relu in registers
    store(WARP_y, g.y, blockIdx.x); // store to global memory

    // g.y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}] 
    // = relu_fwd(
    //     g.x[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}]
    // );
}
__global__ void relu_bwd_kernel(const __grid_constant__ bwd_data g) {
        g.grad_x[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}] 
        = relu_bwd(
            g.grad_y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}],
            g.y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}]
        );
}

void run_relu_fwd_kernel(fwd_data g) {
    relu_fwd_kernel<<<g.grid(), g.block()>>>(g);
}
void run_relu_bwd_kernel(bwd_data g) {
    relu_bwd_kernel<<<g.grid(), g.block()>>>(g);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_kernel<relu_fwd_kernel>(m, "relu_fwd_kernel", &fwd_data::x, &fwd_data::y);
    py::bind_kernel<relu_bwd_kernel>(m, "relu_bwd_kernel", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
}