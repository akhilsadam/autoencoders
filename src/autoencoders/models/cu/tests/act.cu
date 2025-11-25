#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"
#include "tile.cuh"


template<typename DataLayout, typename TileType>
static __global__ void _relu_fwd_kernel(const __grid_constant__ DataLayout g) {
    reg_tile_ft<TileType> WARP_y, WARP_x; // register tiles
    
    // Debug: print block info once per block
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("Block (%d,%d,%d), warp %d, warpwaves=%d\n", 
               blockIdx.x, blockIdx.y, blockIdx.z, g.warp_id(), DataLayout::warpwaves);
    }
    
    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            int2 p = g.warptile_gxy(wave);
            
            // Debug: only print for first block, first channel
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && chan == 0) {
                printf("  wave %d: global p=(%d, %d), tile_offset=(%d,%d)\n", 
                       wave, p.x, p.y, g.tile_x(), g.tile_y());
            }
            load(WARP_x, g.x, {g.batch(), chan, p.y, p.x});
            // unary_map<relu_fwd>(WARP_y, WARP_x);
            store(g.y, WARP_y, {g.batch(), chan, p.y, p.x});
            __syncwarp();
        }
    }
}

template<typename DataLayout, typename TileType>
static __global__ void _relu_bwd_kernel(const __grid_constant__ DataLayout g) {
    reg_tile_ft<TileType> WARP_grad_y, WARP_y, WARP_grad_x; // register tiles

    for(int32_t chan = 0; chan < g.channels(); chan++) {
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) {
            int2 p = g.warptile_gxy(wave);
            load(WARP_grad_y, g.grad_y, {g.batch(), chan, p.y, p.x});
            load(WARP_y, g.y, {g.batch(), chan, p.y, p.x});
            bin_map<relu_bwd>(WARP_grad_x, WARP_grad_y, WARP_y);
            store(g.grad_x, WARP_grad_x, {g.batch(), chan, p.y, p.x});
        }
    }
}

void run_relu_fwd_kernel(fwd_data g) {

    // printf("true g.x shape: (%d, %d, %d, %d)\n", g.x.batch(), g.x.depth(), g.x.rows(), g.x.cols());

    layout_variant<BCHW_fwd> layout = create_layout<BCHW_fwd, fwd_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        // printf("Layout and Tile are %s and %s\n", typeid(Layout).name(), typeid(Tile).name());
        // printf("Running ReLU forward with tile size %dx%d\n", Tile::B.x, Tile::B.y);

        // // show what's inside layout (passes)
        // printf("Layout grid: (%d, %d, %d)\n", layout.grid().x, layout.grid().y, layout.grid().z);
        // printf("Layout block: (%d, %d, %d)\n", layout.block().x, layout.block().y, layout.block().z);
        // printf("Layout mem: %d\n", layout.mem());

        // // print the tensor shapes (passes)
        // printf("g.x shape: (%d, %d, %d, %d)\n", g.x.batch(), g.x.depth(), g.x.rows(), g.x.cols());
        // printf("g.y shape: (%d, %d, %d, %d)\n", g.y.batch(), g.y.depth(), g.y.rows(), g.y.cols());

        // // now from layout print the tensor shapes (passes as well)
        // printf("layout.x shape: (%d, %d, %d, %d)\n", layout.x.batch(), layout.x.depth(), layout.x.rows(), layout.x.cols());
        // printf("layout.y shape: (%d, %d, %d, %d)\n", layout.y.batch(), layout.y.depth(), layout.y.rows(), layout.y.cols());
        // // printf("layout.ref() shape: (%d, %d, %d, %d)\n", layout.ref().batch(), layout.ref().depth(), layout.ref().rows(), layout.ref().cols());

        auto* kernel = _relu_fwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout);
    }, layout);
}
void run_relu_bwd_kernel(bwd_data g) {
    layout_variant<BCHW_bwd> layout = create_layout<BCHW_bwd>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        auto* kernel = _relu_bwd_kernel<Layout, Tile>;
        // cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout);
    }, layout);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x, &bwd_data::x);
    // no need for x since relu is stateless
}