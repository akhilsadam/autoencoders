#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


#define in_chan 64
#define out_chan 32


using A_layout = gl<ftype, 1, 1, out_chan, in_chan>;
using b_layout = gl<ftype, 1, 1, 1, out_chan>;

struct weights{
    A_layout A;
    b_layout b;
}

template<typename DataLayout, typename TileType>
static __global__ void _linear_fwd_kernel(const __grid_constant__ DataLayout g, const weights w) {
    reg_tile_ft<TileType> WARP_y, WARP_x; // register tiles
    
    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            // if (!g.warptile_active(wave)) { continue; }
            int2 p = g.warptile_gxy(wave);
            coord<> idx(g.batch(), chan, p.y, p.x);

            load(WARP_x, g.x, idx);
            unary_map<linear_fwd>(WARP_y, WARP_x);
            store(g.y, WARP_y, idx);
        }
    }
}

template<typename DataLayout, typename TileType>
static __global__ void _linear_bwd_kernel(const __grid_constant__ DataLayout g, const weights w) {
    reg_tile_ft<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            int2 p = g.warptile_gxy(wave);
            coord<> idx(g.batch(), chan, p.y, p.x);

            load(WARP_grad_y, g.grad_y, idx);
            load(WARP_y, g.y, idx);
            bin_map<linear_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
            store(g.grad_x, WARP_grad_x, idx);
        }
    }
}

void run_linear_fwd_kernel(fwd_data g, weights w) {
    layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = _linear_fwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout, w);
    }, layout);
}
void run_linear_bwd_kernel(bwd_data g, weights w) {
    layout_variant<BCHW_bwd> layout = create_layout<BCHW_bwd>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = _linear_bwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout, w);
    }, layout);
}

PYBIND11_MODULE(linear, m) {
    m.doc() = "linear functions python module";
    py::bind_function<run_linear_fwd_kernel>(m, "linear_fwd", &fwd_data::x, &fwd_data::y, &weights::A, &weights::b);
    py::bind_function<run_linear_bwd_kernel>(m, "linear_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x, &bwd_data::x, &weights::A, &weights::b);
    // no need for x since linear is stateless
}