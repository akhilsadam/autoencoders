#include "kittens.cuh"
using namespace kittens;
#include "tile.cuh"
#include "nn.cuh"

struct IdentityTransform {
    template<CHW IN>
    static constexpr CHW apply(IN) {
        return IN;
    }
};

template<CHW IN, class Transform, class Opt>
struct scale_module : public module<IN, Transform, Opt> {
   
    // one shared weight
    shmem<1,1>* w;        // lives in shared memory
    shmem<1,1>* grad_w;   // gradient accumulator

    // ------------------ weights ----------------------
    virtual __device__ __forceinline__
    void init_weights(shared_allocator al) {
        w = al.allocate<shmem<1,1>>(1);
        grad_w = al.allocate<shmem<1,1>>(1);
        w.at(0,0) = 1.0f;  // initialize weight to 1.0
        grad_w.at(0,0) = 0.0f;
    }

    static constexpr size_t weight_bytes() { return sizeof(ftype); }

    virtual __device__ __forceinline__
    void load_weights(uint64_t mem_ptr) {
        *w = *reinterpret_cast<ftype*>(mem_ptr);
    }

    virtual __device__ __forceinline__
    void save_weights(uint64_t mem_ptr) {
        *reinterpret_cast<ftype*>(mem_ptr) = *w;
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd(int32_t batch) {
        reg_wtile_ft<IN> X, Y;

        for (int c = 0; c < IN.C; ++c) {
            for (int wave = 0; wave < IN.warpwaves; ++wave) {

                int2 ij = IN.warptile_xy(wave);
                coord<> idx(batch, c, ij.y, ij.x);
                load(X, x, idx);

                #pragma unroll
                for (int i = 0; i < X.num_elems; i++)
                    Y.data[i] = X.data[i] * (*w);

                store(y, Y, idx);
            }
        }
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd(int32_t batch) {
        reg_wtile_ft<IN> GX, GY, X;

        ftype local_grad_w = 0.0f;

        for (int c = 0; c < IN.C; ++c) {
            for (int wave = 0; wave < IN.warpwaves; ++wave) {

                int2 ij = IN.warptile_xy(wave);
                coord<> idx(batch, c, ij.y, ij.x);

                load(X, x, idx);
                load(GY, grad_y, idx);

                #pragma unroll
                for (int i=0;i<X.num_elems;i++) {
                    GX.data[i] = GY.data[i] * (*w);
                    local_grad_w += GY.data[i] * X.data[i];
                }

                store(grad_x, GX, idx);
            }
        }

        atomicAdd(grad_w, local_grad_w);  // should do parallel scan over warps

        // Apply SGD update
        *w = Opt::update(*w, *grad_w);
        *grad_w = 0.0f;
    }
};

using ScaleModule = ModuleSpec<scale_module, IdentityTransform>;
