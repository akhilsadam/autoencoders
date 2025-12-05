#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "nn.cuh"
#include "loss.cuh"
#include "opt.cuh"
#include "modules/scale.cuh"
#include "bind_w_return.cuh"

template<class L>
using network = module_chain<L, SGD, ScaleModule>;
using Loss = MSELoss;

uint64_t train(train_data& g) {
    channel_variant chan_var = channel_var(g);
    std::visit([&](auto& chan_var) {
        using Chan = std::decay_t<decltype(chan_var)>;
    
        layout_variant<BCHW_train> layout = create_layout<BCHW_train, train_data>(g);
        std::visit([&](auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            using Tile   = typename Layout::tile_type;
            using WarpTile = CHW<Chan::C, Tile::B.y, Tile::B.x, Tile::W.y, Tile::W.x>;
            using Net = network<WarpTile>;

            size_t total_weights = Net::total_weight_bytes();
            // malloc weights
            void* weight_mem_ptr;
            cudaMalloc(&weight_mem_ptr, total_weights);

            printf("Train @ C=%d, Tile=%dx%d, with weight bytes %zu @ %p\n", Chan::C, Tile::B.x, Tile::B.y, total_weights, weight_mem_ptr);

            auto* kernel = train_kernel<Layout, Tile, WarpTile, Net, Loss>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
            kernel<<<layout.grid(), layout.block()>>>(layout);

            g.weight_mem_ptr = reinterpret_cast<uint64_t>(weight_mem_ptr);

        }, layout);
    }, chan_var);
    
    return g.weight_mem_ptr;
}

PYBIND11_MODULE(sanity, m) {
    m.doc() = "nn test python module";
    // py::bind_function<eval_kernel>(m, "eval", &fwd_data::x, &fwd_data::y, &fwd_data::mem_ptr);
    py::bind_function_with_return<train>(m, "train", &train_data::x, &train_data::y, &train_data::iterations, &train_data::weight_mem_ptr);
}