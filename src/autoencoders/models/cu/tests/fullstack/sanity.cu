#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "nn.cuh"
#include "loss.cuh"
#include "opt.cuh"
#include "modules/scale.cuh"

template<class L>
using Net = module_chain<L, SGD, ScaleModule>;
using Loss = MSELoss;

void train(train_data g) {
    channel_variant chan_var = channel_var(g);
    std::visit([&](auto& chan_var) {
        using Chan = std::decay_t<decltype(chan_var)>;
    
        layout_variant<BCHW_train> layout = create_layout<BCHW_train, train_data>(g);
        std::visit([&](auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            using Tile   = typename Layout::tile_type;
            using WarpTile = CHW<Chan::C, Tile::B.y, Tile::B.x, Tile::W.y, Tile::W.x>;

            printf("Running training with C=%d, Tile=%dx%d\n", Chan.C, Tile::B.x, Tile::B.y);

            auto* kernel = train_kernel<Layout, Tile, WarpTile, Net, Loss>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
            kernel<<<layout.grid(), layout.block()>>>(layout);

        }, layout);
    }, chan_var);
}


PYBIND11_MODULE(sanity, m) {
    m.doc() = "nn test python module";
    // py::bind_function<eval_kernel>(m, "eval", &fwd_data::x, &fwd_data::y, &fwd_data::mem_ptr);
    py::bind_function<train>(m, "train", &train_data::x, &train_data::y);
}