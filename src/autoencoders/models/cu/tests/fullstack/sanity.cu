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

template<typename DataLayout, typename TileType, class L>
static __global__ void train_kernel(const DataLayout data)
{
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    Net<L> net;

    // allocate memory
    using shmem_tile = shmem<DataLayout::tile_type::B.y, DataLayout::tile_type::B.x>;
    shmem_tile* x_array = al.allocate<shmem_tile, L::C>();
    shmem_tile* y_array = al.allocate<shmem_tile, L::C>();
    shmem_tile* grad_y_array = al.allocate<shmem_tile, L::C>();

    shmem_tile* y_hat_array = reinterpret_cast<shmem_tile*>
    (
        net.eval(al,
            reinterpret_cast<uint64_t>(x_array))
    );
    net.train(al, reinterpret_cast<uint64_t>(grad_y_array));
   
    // -----------------------
    // Init weights ONCE
    net.__init_weights__(al);
    __syncthreads();
    if (data.weight_mem_array != 0)
    {
        net.__load_weights__(data.weight_mem_array);
        __syncthreads();        
    } // optional: load weights from global memory

    // -----------------------
    // training loop
    for (int batch = 0; batch < data.batch_size; batch++)
    {
        for (int iter = 0; iter < data.iterations; iter++)
        {            
            // load input data for this batch item
            for (int c = 0; c < data.x.depth(); c++)
            {
                coord<> idx(batch, c, 0, 0);
                load(x_array[c], data.x, idx);
                load(y_array[c], data.y, idx);
            }
            __syncthreads();

            // net.fwd();
            // MSE<L>(data.depth, batch, y_hat_array, y_array, grad_y_array);
            // net.bwd();

            __syncthreads();
        }
    }

    // --------------------------------------
    // Save weights back to global
    if (data.weight_mem_array != 0)
        net.__save_weights__();
}

void train(train_data g) {
    layout_variant<BCHW_train> layout = create_layout<BCHW_train, train_data>(g);
    std::visit([&](auto& layout) {
        using Layout = std::decay_t<decltype(layout)>;
        using Tile   = typename Layout::tile_type;
        using WarpTile = CHW<3, Tile::B.y, Tile::B.x, Tile::W.y, Tile::W.x>;

        auto* kernel = train_kernel<Layout, Tile, WarpTile>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, layout.mem());
        kernel<<<layout.grid(), layout.block()>>>(layout);

    }, layout);
}


PYBIND11_MODULE(sanity, m) {
    m.doc() = "nn test python module";
    // py::bind_function<eval_kernel>(m, "eval", &fwd_data::x, &fwd_data::y, &fwd_data::mem_array);
    py::bind_function<train>(m, "train", &train_data::x, &train_data::y);
}