#include "tile.cuh"


struct train_data {
    base_layout_ y, x;
    uint64_t iterations = 100;
} 

template<int By, int Bx>
using shmem = st<ftype, By, Bx>;

template<CHW IN, class Transform, class Opt>
struct module {
    // BCHW 
    static constexpr CHW OUT = Transform::apply(IN);
    shmem<IN.By, IN.Bx> x[IN.C];
    shmem<OUT.By, OUT.Bx> y[OUT.C];
    shmem<OUT.By, OUT.Bx> grad_y[OUT.C];
    shmem<IN.By, IN.Bx> grad_x[IN.C];

    // some unknown parameters in templated classes
    /// TODO fix xy pointers in list below and concurrently only allocate
    /// the intermediate xy (since y of one module is x of next)
    /// and grad_x, grad_y similarly

    __device__ __forceinline__ uint64_t eval(shared_allocator al, const uint64_t x_ptr) {
        // allocate
        if (x_ptr != 0) 
        {
            x = reinterpret_cast<shmem<IN.By, IN.Bx>*>(x_ptr);
        }
        else
        {
            auto* x = al.allocate<shmem<IN.By, IN.Bx>>(IN.C);
        }
        auto* y = al.allocate<shmem<OUT.By, OUT.Bx>>(OUT.C);
        return reinterpret_cast<uint64_t>(y);
    }

    __device__ __forceinline__ uint64_t train(shared_allocator al, const uint64_t grad_y_ptr) {
        // allocate
        if const (grad_y_ptr != 0) 
        {
            grad_y = reinterpret_cast<shmem<OUT.By, OUT.Bx>*>(grad_y_ptr);
        }
        else
        {
            auto* grad_y = al.allocate<shmem<OUT.By, OUT.Bx>>(OUT.C);
        }
        auto* grad_x = al.allocate<shmem<IN.By, IN.Bx>>(IN.C);
        return reinterpret_cast<uint64_t>(grad_x);
    }
    
    virtual __device__ __forceinline__ void init_weights(shared_allocator al) {
        // allocate weights if any
        // and initialize in shared memory
    }

    virtual static constexpr size_t weight_bytes() {
        // // Example -- replace with real layer parameters
        // return OUT.C * IN.C * sizeof(ftype);
    }

    virtual __device__ __forceinline__ void load_weights(const uint64_t mem_ptr) {
        // load weights from global memory to shared memory
    }

    virtual __device__ __forceinline__ void save_weights(uint64_t mem_ptr) {
        // save weights from shared memory to global memory
    }

    virtual __device__ __forceinline__ void fwd() {
        // run forward pass
    }
    
    virtual __device__ __forceinline__ void bwd() {
        // run backward pass
    }

}

template <template<CHW, class> class ModuleType, class Transform, class Opt>
struct ModuleSpec {
    template<CHW IN>
    using type = ModuleType<IN, Transform, Opt>;
};

template<CHW IN, class ModuleSpec, class... Rest>
struct module_chain {
    // Instantiate the actual module using its internal transform
    // and then recursively instantiate the next module in the chain
    using CurrentModule = typename ModuleSpec::template type<IN>;
    CurrentModule current;
    static constexpr CHW NextIN = CurrentModule::OUT;
    module_chain<NextIN, Rest...> next;

    // for the first and last modules, handle input/output pointers

    // iterate forward through the chain for y_ptr
    __device__ inline uint64_t eval(shared_allocator al, const uint64_t x_ptr = 0) {
        auto* next_x_ptr = current.eval(al, x_ptr);
        return next.eval(al, next_x_ptr);
    }

    // iterate backward through the chain for grad_x_ptr
    __device__ inline uint64_t train(shared_allocator al, const uint64_t grad_y_ptr = 0) {
        auto* curr_grad_y_ptr = next.train(al, grad_y_ptr);
        return current.train(al, curr_grad_y_ptr);
    }

    // iterate through to init weights
    __device__ inline void init_weights(shared_allocator al) {
        current.init_weights(al);
        next.init_weights(al);
    }

    __device__ inline void _load_weights(const uint64_t mem_ptr) {
        current._load_weights(mem_ptr);
        next._load_weights(mem_ptr + CurrentModule::weight_bytes());
    }

    __device__ inline void _save_weights(uint64_t mem_ptr) {
        current._save_weights(mem_ptr);
        next._save_weights(mem_ptr + CurrentModule::weight_bytes());
    }

    __device__ inline void fwd() {
        current.fwd();
        next.fwd();
    }

    __device__ inline void bwd() {
        next.bwd();
        current.bwd();
    }

    static constexpr size_t total_weight_bytes() {
        if constexpr (sizeof...(Rest) == 0)
            return curr_w_bytes;
        else
            return curr_w_bytes + module_chain<NextIN, Rest...>::total_weight_bytes();
    }

};
