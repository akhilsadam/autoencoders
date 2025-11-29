#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


template<typename DataLayout, typename TileType>
static __global__ void _kernel(const DataLayout g) {
    reg_tile_ft<TileType> WARP_y, WARP_x; // register tiles
    
    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {

            int2 p = g.warptile_gxy(wave);
            coord<> idx(g.batch(), chan, p.y, p.x);

            load(WARP_x, g.x, idx);
            unary_map>(WARP_y, WARP_x);
            store(g.y, WARP_y, idx);
        }
    }
}

template<typename DataLayout, typename TileType>
static __global__ void _kernel(const DataLayout g) {
    reg_tile_ft<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            
            int2 p = g.warptile_gxy(wave);
            coord<> idx(g.batch(), chan, p.y, p.x);

            load(WARP_grad_y, g.grad_y, idx);
            load(WARP_y, g.y, idx);
            bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
            store(g.grad_x, WARP_grad_x, idx);
        }
    }
}

void eval_kernel(fwd_data g) {
    layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = eval_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout);
    }, layout);
}
void train_kernel(bwd_data g) {
    layout_variant<BCHW_bwd> layout = create_layout<BCHW_bwd>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = train_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout);
    }, layout);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_function<eval_kernel>(m, "eval", &fwd_data::x, &fwd_data::y, &fwd_data::mem_ptr);
    py::bind_function<train_kernel>(m, "train", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x, &bwd_data::x, &bwd_data::mem_ptr);
}