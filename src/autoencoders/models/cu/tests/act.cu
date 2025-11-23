#include "kittens.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"

template<typename Layout, typename TileType>
static __global__ void _relu_fwd_kernel(const __grid_constant__ Layout g) {
    reg_tile_dt<TileType> WARP_y, WARP_x; // register tiles
    
    for(int32_t channel = 0; channel < g.channels(); channel++) {
        auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
        load(WARP_x, g.x, warptile_x);
        unary_map<relu_fwd>(WARP_y, WARP_x);
        store(g.y, WARP_y, warptile_x);
    }
}

template<typename Layout, typename TileType>
static __global__ void _relu_bwd_kernel(const __grid_constant__ Layout g) {
    reg_tile_dt<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t channel = 0; channel < g.channels(); channel++) {
        auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
        load(WARP_grad_y, g.grad_y, warptile_x);
        load(WARP_y, g.y, warptile_x);
        bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
        store(g.grad_x, WARP_grad_x, warptile_x);
    }
}

struct ReLU {
    // list of types supported

    inline static const void(*layout_fwd[3])(fwd_data) = 
    {
        +[](fwd_data d) -> void* {return new BCHW_fwd<Tile28>(d);},
        +[](fwd_data d) -> void* {return new BCHW_fwd<Tile64>(d);},
        +[](fwd_data d) -> void* {return new BCHW_fwd<Tile128>(d);},
    };

    inline static const void(*layout_bwd[3])(bwd_data) = 
    {
        +[](bwd_data d) -> void* {return new BCHW_bwd_stateless<Tile28>(d);},
        +[](bwd_data d) -> void* {return new BCHW_bwd_stateless<Tile64>(d);},
        +[](bwd_data d) -> void* {return new BCHW_bwd_stateless<Tile128>(d);},
    };

    static constexpr void(*relu_fwd[3]) = {
        (void*)_relu_fwd_kernel<BCHW_fwd<Tile28>, Tile28>,
        (void*)_relu_fwd_kernel<BCHW_fwd<Tile64>, Tile64>,
        (void*)_relu_fwd_kernel<BCHW_fwd<Tile128>, Tile128>
    };

    static constexpr void(*relu_bwd[3]) = {
        (void*)_relu_bwd_kernel<BCHW_bwd_stateless<Tile28>, Tile28>,
        (void*)_relu_bwd_kernel<BCHW_bwd_stateless<Tile64>, Tile64>,
        (void*)_relu_bwd_kernel<BCHW_bwd_stateless<Tile128>, Tile128>
    };
};

void run_relu_fwd_kernel(fwd_data g) {
    auto tile_idx = TileIndex(g);
    void* data = (void*) ReLU::layout_fwd[tile_idx](g);
    void* kernel = ReLU::relu_fwd[tile_idx];
    void* args[] = { &data };
    cudaLaunchKernel(kernel, data.grid(), data.block(), args, data.shmem_size, nullptr);
}
void run_relu_bwd_kernel(bwd_data g) {
    auto tile_idx = TileIndex(g);
    auto data = ReLU::layout_bwd[tile_idx](g);
    auto kernel = ReLU::relu_bwd[tile_idx];
    kernel<<<data.grid(), data.block()>>>(data);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
    // no need for x since relu is stateless
}