#include "kittens.cuh"
// #include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"

template<typename Layout>
__global__ void _relu_fwd_kernel(const __grid_constant__ Layout g) {
    reg_tile_dt WARP_y, WARP_x; // register tiles
    
    for(int32_t channel = 0; channel < g.channels(); channel++) {
        auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
        load(WARP_x, g.x, warptile_x);
        unary_map<relu_fwd>(WARP_y, WARP_x);
        store(g.y, WARP_y, warptile_x);
    }
}

template<typename Layout>
__global__ void _relu_bwd_kernel(const __grid_constant__ Layout g) {
    reg_tile_dt WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t channel = 0; channel < g.channels(); channel++) {
        auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
        load(WARP_grad_y, g.grad_y, warptile_x);
        load(WARP_y, g.y, warptile_x);
        bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
        store(g.grad_x, WARP_grad_x, warptile_x);
    }
}

struct ReLU {
    using fwd = _relu_fwd_kernel;
    using bwd = _relu_bwd_kernel;
};

void run_relu_fwd_kernel(fwd_data g) {
    auto kernel = dispatch_fwd_kernel(ReLU, g);
    kernel<<<g.grid(), g.block()>>>(g); // no need for shared memory
}
void run_relu_bwd_kernel(bwd_data g) {
    auto kernel = dispatch_bwd_sl_kernel(ReLU, g);
    kernel<<<g.grid(), g.block()>>>(g); // no need for shared memory
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
    // no need for x since relu is stateless
}