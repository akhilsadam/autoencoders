#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"

using fwd_data = BCHW_fwd<tiled_layout>;
using bwd_data = BCHW_bwd_stateless<tiled_layout>;

__global__ void relu_fwd_kernel(const __grid_constant__ fwd_data g) {
    reg_tile_dt WARP_y, WARP_x; // register tiles
    
    // Loop over all channels
    for(uint32_t channel = 0; channel < g.tiles_depth(); channel++) {
        load(WARP_x, g.x, {g.tile_batch(), channel, g.idx_row(), g.idx_col()});
        map(WARP_y, WARP_x, relu_fwd);
        store(WARP_y, g.y, {g.tile_batch(), channel, g.idx_row(), g.idx_col()});
    }
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