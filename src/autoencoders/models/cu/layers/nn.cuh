#include "kittens.cuh"
using namespace kittens;

#ifndef TILE_CUH_INCLUDED
#include "tile.cuh"
#endif

#define NN_CUH_INCLUDED

template<class _IN, template<class> class Transform, class Opt>
class module {
    public:
        // BCHW 
        using IN = _IN;
        using OUT = Transform<IN>;
        IN::shmem_array* x;
        OUT::shmem_array* y;
        OUT::shmem_array* grad_y;
        IN::shmem_array* grad_x;

        static constexpr size_t weight_bytes = 0;

        // some unknown parameters in templated classes
        /// TODO fix xy pointers in list below and concurrently only allocate
        /// the intermediate xy (since y of one module is x of next)
        /// and grad_x, grad_y similarly

        template <typename T>
        __device__ __forceinline__ uint64_t eval(T& al, const uint64_t x_ptr) {
            // allocate
            if (x_ptr != 0) 
            {
                x = reinterpret_cast<IN::shmem_array*>(x_ptr);
            }
            else
            {
                IN::template salloc(al, x);
            }
            OUT::template salloc(al, y);
            return reinterpret_cast<uint64_t>(y);
        }

        template <typename T>
        __device__ __forceinline__ uint64_t train(T& al, const uint64_t grad_y_ptr) {
            // allocate
            if (grad_y_ptr != 0) 
            {
                grad_y = reinterpret_cast<OUT::shmem_array*>(grad_y_ptr);
            }
            else
            {
                OUT::template salloc(al, grad_y);
            }
            IN::template salloc(al, grad_x);
            return reinterpret_cast<uint64_t>(grad_x);
        }
        
        template <typename T>
        __device__ __forceinline__ void __init_weights__(T& al) {
            // allocate weights if any
            // and initialize in shared memory
        }

        virtual __device__ __forceinline__ void __load_weights__(const uint64_t mem_ptr) {
            // load weights from global memory to shared memory
        }

        virtual __device__ __forceinline__ void __save_weights__() {
            // save weights from shared memory to global memory
        }

        virtual __device__ __forceinline__ void fwd() {
            // run forward pass
        }
        
        virtual __device__ __forceinline__ void bwd() {
            // run backward pass
        }

};

template <template<class, template<class> class, class> class ModuleType, template<class> class Transform>
struct ModuleSpec {
    template<class IN, class Opt>
    using type = ModuleType<IN, Transform, Opt>;
};

// size, optimizer, chained modules
template<class _IN, class Opt, class ModuleSpec, class... Rest>
struct module_chain {
    // Instantiate the actual module using its internal transform
    // and then recursively instantiate the next module in the chain
    using IN = _IN;

    using CurrentModule = typename ModuleSpec::template type<IN, Opt>;
    using NextIN = CurrentModule::OUT;
    using NextModule = module_chain<NextIN, Opt, Rest...>;

    using OUT = NextModule::OUT;

    CurrentModule current;
    NextModule next;

    // for the first and last modules, handle input/output pointers

    // iterate forward through the chain for y_ptr
    template <typename T>
    __device__ inline uint64_t eval(T& al, const uint64_t x_ptr = 0) {
        auto* next_x_ptr = current.eval(al, x_ptr);
        return next.eval(al, next_x_ptr);
    }

    // iterate backward through the chain for grad_x_ptr
    template <typename T>
    __device__ inline uint64_t train(T& al, const uint64_t grad_y_ptr = 0) {
        auto* curr_grad_y_ptr = next.train(al, grad_y_ptr);
        return current.train(al, curr_grad_y_ptr);
    }

    // iterate through to init weights
    template <typename T>
    __device__ inline void init_weights(T& al) {
        current.init_weights(al);
        next.init_weights(al);
    }

    __device__ inline void __load_weights__(const uint64_t mem_ptr) {
        current.__load_weights__(mem_ptr);
        next.__load_weights__(mem_ptr + CurrentModule::weight_bytes());
    }

    // __device__ inline void _save_weights(uint64_t mem_ptr) {
    //     current._save_weights(mem_ptr);
    //     next._save_weights(mem_ptr + CurrentModule::weight_bytes());
    // }
    __device__ inline void __save_weights__() {
        current.__save_weights__();
        next.__save_weights__();
    }

    __device__ inline void fwd() {
        current.fwd();
        __syncthreads(); // since in L1
        next.fwd();
    }

    __device__ inline void bwd() {
        next.bwd();
        __syncthreads(); // since in L1
        current.bwd();
    }

    static size_t total_weight_bytes() {
        if (sizeof...(Rest) == 0)
            return CurrentModule::weight_bytes;
        else
            return CurrentModule::weight_bytes + module_chain<NextIN, Rest...>::total_weight_bytes();
    }

};


// base case
template<class _IN, class Opt, class ModuleSpec>
struct module_chain<_IN, Opt, ModuleSpec> 
{
    using IN = _IN;

    using CurrentModule = typename ModuleSpec::template type<IN, Opt>;
    CurrentModule current;

    using OUT = CurrentModule::OUT;

    template <typename T>
    __device__ inline uint64_t eval(T& al, const uint64_t x_ptr = 0) {
        return current.eval(al, x_ptr);
    }

    template <typename T>
    __device__ inline uint64_t train(T& al, const uint64_t grad_y_ptr = 0) {
        return current.train(al, grad_y_ptr);
    }

    __device__ inline void fwd() { current.fwd(); __syncthreads(); }
    __device__ inline void bwd() { current.bwd(); } // no sync needed here
    
    template <typename T>
    __device__ inline void __init_weights__(T& al) { current.__init_weights__(al); }
    __device__ inline void __load_weights__(uint64_t mem_ptr) { current.__load_weights__(mem_ptr); }
    __device__ inline void __save_weights__() { current.__save_weights__(); }

    static size_t total_weight_bytes() { return CurrentModule::weight_bytes; }
};


template<typename DataLayout, typename TileType, class Net, class Loss>
static __global__ void train_kernel(const DataLayout data)
{
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    Net net;

    // allocate memory
    using IN = Net::IN;
    using OUT = Net::OUT;

    auto* x_array = IN::template salloc(al);
    auto* y_array = OUT::template salloc(al);
    auto* grad_y_array = OUT::template salloc(al);

    auto* y_hat_array = reinterpret_cast<OUT::shmem_wp*>
    (
        net.eval(al,
            reinterpret_cast<uint64_t>(x_array))
    );
    net.train(al, reinterpret_cast<uint64_t>(grad_y_array));
    // --------------------------------------
    // // weight initialization
    net.__init_weights__(al);
    __syncthreads();

    if (data.weight_mem_ptr != 0)
    {   // optional: load weights from global memory
        net.__load_weights__(data.weight_mem_ptr);
        __syncthreads();        
    } 
    
    // --------------------------------------
    // training loop, one batch (across blocks)
    for (int iter = 0; iter < data.iterations; iter++)
    {            
        // load input data for this batch item
        for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) 
        {
            int2 wt = data.warptile_ixy(wave);
            int2 p = data.warptile_ixy_to_gxy(wt);
            for (int c = 0; c < data.x.depth(); c++)
            {
                coord<> idx(data.batch(), c, p.y, p.x);
                load((*x_array)[wt.y][wt.x][c], data.x, idx);
                load((*y_array)[wt.y][wt.x][c], data.y, idx);

            }
        }
        __syncthreads();

        net.fwd();
        Loss::template op<OUT>(y_hat_array, y_array, grad_y_array);
        net.bwd();

        __syncthreads();
    }
    // --------------------------------------
    // Save weights back to global
    if (data.weight_mem_ptr != 0)
    {
        net.__save_weights__();
    }

    // printf("Training done.\n");
}


template<typename DataLayout, typename TileType, class Net, class Loss>
static __global__ void eval_kernel(const DataLayout data)
{
    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    Net net;

    using IN = Net::IN;
    using OUT = Net::OUT;

    // allocate memory
    auto* x_array = IN::template salloc(al);
    auto* y_hat_array = reinterpret_cast<OUT::shmem_wp*>
    (
        net.eval(al,
            reinterpret_cast<uint64_t>(x_array))
    );
    // --------------------------------------
    // weight initialization
    net.__init_weights__(al);
    __syncthreads();

    if (data.weight_mem_ptr != 0)
    {   // optional: load weights from global memory
        net.__load_weights__(data.weight_mem_ptr);
        __syncthreads();        
    } 
    // --------------------------------------  
    // load input data for this batch item
    for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) 
    {
        int2 wt = data.warptile_ixy(wave);
        int2 p = data.warptile_ixy_to_gxy(wt);
        for (int c = 0; c < data.x.depth(); c++)
        {
            coord<> idx(data.batch(), c, p.y, p.x);
            load((*x_array)[wt.y][wt.x][c], data.x, idx);
        }
    }
    __syncthreads();
    net.fwd(); // does syncthreads internally
    // --------------------------------------
    // Save y_hat back to global
    for (int32_t wave = 0; wave < DataLayout::warpwaves; wave++) 
    {
        int2 wt = data.warptile_ixy(wave);
        int2 p = data.warptile_ixy_to_gxy(wt);
        for (int c = 0; c < data.y.depth(); c++)
        {
            coord<> idx(data.batch(), c, p.y, p.x);
            store(data.y, (*y_hat_array)[wt.y][wt.x][c], idx);
        }
    }

}