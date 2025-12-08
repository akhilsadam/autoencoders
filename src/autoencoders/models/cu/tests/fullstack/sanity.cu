#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "nn.cuh"
#include "loss.cuh"
#include "opt.cuh"
#include "modules/scale.cuh"
#include "util/bind_w_return.cuh"

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
            using WarpTile = CHW<Chan::C, Tile>;
            using Net = network<WarpTile>;

            size_t total_weights = Net::total_weight_bytes();
            // malloc weights
            if (g.weight_mem_ptr == 0)
            {
                printf("Allocating weight memory of size %zu\n", total_weights);
                void* weight_mem_ptr;
                cudaMalloc(&weight_mem_ptr, total_weights);
                std::vector<float> h_weights(total_weights / sizeof(float), 1.0f); // initialize all to 1.0
                cudaMemcpy(weight_mem_ptr, h_weights.data(), total_weights, cudaMemcpyHostToDevice); // to init values
                g.weight_mem_ptr = reinterpret_cast<uint64_t>(weight_mem_ptr);
                
            }

            printf("Train @ C=%d, Tile=%dx%d, with weight bytes %zu @ %p\n", Chan::C, Tile::B.x, Tile::B.y, total_weights, g.weight_mem_ptr);

            auto* kernel = train_kernel<Layout, Tile, Net, Loss>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
            kernel<<<layout.grid(), layout.block(), layout.mem()>>>(layout);


        }, layout);
    }, chan_var);
    
    return g.weight_mem_ptr;
}

void eval(train_data& g) {
    channel_variant chan_var = channel_var(g);
    std::visit([&](auto& chan_var) {
        using Chan = std::decay_t<decltype(chan_var)>;
    
        layout_variant<BCHW_train> layout = create_layout<BCHW_train, train_data>(g);
        std::visit([&](auto& layout) {
            using Layout = std::decay_t<decltype(layout)>;
            using Tile   = typename Layout::tile_type;
            using WarpTile = CHW<Chan::C, Tile>;
            using Net = network<WarpTile>;

            size_t total_weights = Net::total_weight_bytes();
            // printf("uint64_t mem_ptr = %llu\n", g.weight_mem_ptr);
            // void* weight_ptr = reinterpret_cast<void*>(g.weight_mem_ptr);

            printf("Eval @ C=%d, Tile=%dx%d, with weight bytes %zu\n", Chan::C, Tile::B.x, Tile::B.y, total_weights);

            auto* kernel = eval_kernel<Layout, Tile, Net, Loss>;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
            kernel<<<layout.grid(), layout.block(), layout.mem()>>>(layout);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA ERROR after eval kernel: %s\n", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();

        }, layout);
    }, chan_var);
}

PYBIND11_MODULE(sanity, m) {
    m.doc() = "nn test python module";
    py::bind_function<eval>(m, "eval", &train_data::x, &train_data::y, &train_data::weight_mem_ptr, &train_data::iterations);
    py::bind_function_with_return<train>(m, "train", &train_data::x, &train_data::y, &train_data::weight_mem_ptr, &train_data::iterations);
    // order matters! this needs to match the train_data struct layout as well, for some reason
}