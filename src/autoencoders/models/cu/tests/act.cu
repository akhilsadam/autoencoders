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
    template<typename Tile>
    BCHW_fwd<Tile> fwd_layout(const fwd_data& g) {
        return BCHW_fwd<Tile>{g};
    }

    template<typename Tile>
    BCHW_bwd_stateless<Tile> bwd_layout(const bwd_data& g) {
        return BCHW_bwd_stateless<Tile>{g};
    }

    static constexpr auto layout_fwd = std::array{
        fwd_layout<Tile28>,
        fwd_layout<Tile64>,
        fwd_layout<Tile128>};
    static constexpr auto layout_bwd = std::array{
        bwd_layout<Tile28>,
        bwd_layout<Tile64>,
        bwd_layout<Tile128>};
    static constexpr auto relu_fwd = std::array{
        _relu_fwd_kernel<BCHW_fwd<Tile28>, Tile28>,
        _relu_fwd_kernel<BCHW_fwd<Tile64>, Tile64>,
        _relu_fwd_kernel<BCHW_fwd<Tile128>, Tile128>};
    static constexpr auto relu_bwd = std::array{
        _relu_bwd_kernel<BCHW_bwd_stateless<Tile28>, Tile28>,
        _relu_bwd_kernel<BCHW_bwd_stateless<Tile64>, Tile64>,
        _relu_bwd_kernel<BCHW_bwd_stateless<Tile128>, Tile128>};
};

void run_relu_fwd_kernel(fwd_data g) {
    auto tile_idx = TileIndex(g);
    auto data = ReLU::layout_fwd[tile_idx](g);
    auto kernel = ReLU::relu_fwd[tile_idx];
    kernel<<<data.grid(), data.block()>>>(data);
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