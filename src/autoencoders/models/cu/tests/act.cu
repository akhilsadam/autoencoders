#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


template<typename DataLayout, typename TileType>
static __global__ void _relu_fwd_kernel(const __grid_constant__ DataLayout g) {
    reg_tile_dt<TileType> WARP_y, WARP_x; // register tiles
    
    for(int32_t channel = 0; channel < g.channels(); channel++) {
        load(WARP_x, g.x, {g.batch(), channel, g.warptile_gx(), g.warptile_gy()});
        unary_map<relu_fwd>(WARP_y, WARP_x);
        store(g.y, WARP_y, {g.batch(), channel, g.warptile_gx(), g.warptile_gy()});
    }
}

template<typename DataLayout, typename TileType>
static __global__ void _relu_bwd_kernel(const __grid_constant__ DataLayout g) {
    reg_tile_dt<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t channel = 0; channel < g.channels(); channel++) {
        load(WARP_grad_y, g.grad_y, {g.batch(), channel, g.warptile_gx(), g.warptile_gy()});
        load(WARP_y, g.y, {g.batch(), channel, g.warptile_gx(), g.warptile_gy()});
        bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
        store(g.grad_x, WARP_grad_x, {g.batch(), channel, g.warptile_gx(), g.warptile_gy()});
    }
}

void run_relu_fwd_kernel(fwd_data g) {

    // printf("true g.x shape: (%d, %d, %d, %d)\n", g.x.batch(), g.x.depth(), g.x.rows(), g.x.cols());

    layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        printf("Layout and Tile are %s and %s\n", typeid(Layout).name(), typeid(Tile).name());
        printf("Running ReLU forward with tile size %dx%d\n", Tile::B.x, Tile::B.y);

        // show what's inside layout (passes)
        printf("Layout grid: (%d, %d, %d)\n", layout.grid().x, layout.grid().y, layout.grid().z);
        printf("Layout block: (%d, %d, %d)\n", layout.block().x, layout.block().y, layout.block().z);
        printf("Layout mem: %d\n", layout.mem());

        // print the tensor shapes (passes)
        printf("g.x shape: (%d, %d, %d, %d)\n", g.x.batch(), g.x.depth(), g.x.rows(), g.x.cols());
        printf("g.y shape: (%d, %d, %d, %d)\n", g.y.batch(), g.y.depth(), g.y.rows(), g.y.cols());

        // now from layout print the tensor shapes (passes as well)
        printf("layout.x shape: (%d, %d, %d, %d)\n", layout.x.batch(), layout.x.depth(), layout.x.rows(), layout.x.cols());
        printf("layout.y shape: (%d, %d, %d, %d)\n", layout.y.batch(), layout.y.depth(), layout.y.rows(), layout.y.cols());

        auto* kernel = _relu_fwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        // kernel<<<layout.grid(), layout.block()>>>(layout);
    }, layout);
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
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x, &bwd_data::x);
    // no need for x since relu is stateless
}