#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


#define _chan 16
#define k 5


using K_layout = gl<ftype, 1, _chan, 5, 5, st_fl<64, 64>>;

struct weights
{
    K_layout K;
    int32_t stride;
    int32_t padding;
};

struct fwd_weights {
    base_layout_ x, y;
    K_layout K;
    int32_t stride;
    int32_t padding;
};

struct bwd_weights {
    base_layout_ x, y, grad_x, grad_y;
    K_layout K;
    int32_t stride;
    int32_t padding;
};

template<typename DataLayout, typename TileType>
static __global__ void _conv_fwd_kernel(const __grid_constant__ DataLayout g, const weights w) {
    reg_tile_ft<TileType> WARP_y, WARP_x; // register tiles
    
    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            // if (!g.warptile_active(wave)) { continue; }
            int2 p = g.warptile_gxy(wave);
            coord<> idx(g.batch(), chan, p.y, p.x);

            load(WARP_x, g.x, idx);
            unary_map<relu_fwd>(WARP_y, WARP_x);
            store(g.y, WARP_y, idx);
        }
    }
}

template<typename DataLayout, typename TileType>
static __global__ void _conv_bwd_kernel(const __grid_constant__ DataLayout g, const weights w) {
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

void run_conv_fwd_kernel(fwd_weights gw) {
    fwd_data g{gw.x, gw.y};
    weights w{gw.K, gw.stride, gw.padding};

    layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = _conv_fwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout, w);
    }, layout);
}
void run_conv_bwd_kernel(bwd_weights gw) {
    bwd_data g{gw.x, gw.y, gw.grad_x, gw.grad_y};
    weights w{gw.K, gw.stride, gw.padding};

    layout_variant<BCHW_bwd> layout = create_layout<BCHW_bwd, bwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = _conv_bwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout, w);
    }, layout);
}

PYBIND11_MODULE(conv, m) {
    m.doc() = "conv functions python module";
    py::bind_function<run_conv_fwd_kernel>(m, "conv_fwd", &fwd_weights::x, &fwd_weights::y, &fwd_weights::K, &fwd_weights::stride, &fwd_weights::padding);
    py::bind_function<run_conv_bwd_kernel>(m, "conv_bwd", &bwd_weights::grad_y, &bwd_weights::y, &bwd_weights::grad_x, &bwd_weights::x, &bwd_weights::K, &bwd_weights::stride, &bwd_weights::padding);
    // no need for x since conv is stateless
}