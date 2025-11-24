#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


// template<typename Layout, typename TileType>
// static __global__ void _relu_fwd_kernel(const __grid_constant__ Layout g) {
//     reg_tile_dt<TileType> WARP_y, WARP_x; // register tiles
    
//     for(int32_t channel = 0; channel < g.channels(); channel++) {
//         auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
//         load(WARP_x, g.x, warptile_x);
//         unary_map<relu_fwd>(WARP_y, WARP_x);
//         store(g.y, WARP_y, warptile_x);
//     }
// }

// template<typename Layout, typename TileType>
// static __global__ void _relu_bwd_kernel(const __grid_constant__ Layout g) {
//     reg_tile_dt<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

//     for(int32_t channel = 0; channel < g.channels(); channel++) {
//         auto warptile_x = {g.batch(), channel, g.warptile_gx(), g.warptile_gy()};
//         load(WARP_grad_y, g.grad_y, warptile_x);
//         load(WARP_y, g.y, warptile_x);
//         bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
//         store(g.grad_x, WARP_grad_x, warptile_x);
//     }
// }

void run_relu_fwd_kernel(fwd_data g) {

    printf("g.x shape: (%d, %d, %d, %d)\n", g.x.batch(), g.x.depth(), g.x.rows(), g.x.cols());

    // layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    // std::visit([&](auto& layout) {
    //     using Layout = std::decay_t<decltype(layout)>;
    //     using Tile   = typename Layout::tile_type;
    //     // printf("Layout and Tile are %s and %s\n", typeid(Layout).name(), typeid(Tile).name());
    //     // printf("Running ReLU forward with tile size %dx%d\n", Tile::B.x, Tile::B.y);

    //     // auto* kernel = _relu_fwd_kernel<Layout, Tile>;
    //     // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
    //     // kernel<<<layout.grid(), layout.block()>>>(layout);
    // }, layout);
}
void run_relu_bwd_kernel(bwd_data g) {
    // layout_variant<BCHW_bwd_stateless> layout = create_layout<BCHW_bwd_stateless>(g);
    // std::visit([&](auto& layout) {
    //     using Layout = std::decay_t<decltype(layout)>;
    //     using Tile   = typename Layout::tile_type;
    //     auto* kernel = _relu_bwd_kernel<Layout, Tile>;
    //     // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
    //     kernel<<<layout.grid(), layout.block()>>>(layout);
    // }, layout);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
    // no need for x since relu is stateless
}