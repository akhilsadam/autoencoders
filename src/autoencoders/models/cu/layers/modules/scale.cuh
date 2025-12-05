#include "kittens.cuh"
using namespace kittens;

#ifndef TILE_CUH_INCLUDED
#include "tile.cuh"
#endif

#ifndef NN_CUH_INCLUDED
#include "nn.cuh"
#endif

struct IdentityTransform {
    template<HW IN>
    static constexpr HW apply(IN) {
        return IN;
    }
};

template<HW IN, class Transform, class Opt>
struct scale_module : public module<IN, Transform, Opt> {
   
    // one shared weight
    using wtile = shmem<1,1>;
    wtile* weight;        // lives in shared memory
    wtile* grad_weight;   // gradient accumulator

    // ------------------ weights ----------------------
    __device__ __forceinline__
    void init_weights(shared_allocator al) {
        coord<> _id(0,0,0,0);
        weight = al.allocate<shmem<1,1>>(1);
        grad_weight = al.allocate<shmem<1,1>>(1);
        weight[_id] = 1.0f;
        grad_weight[_id] = 0.0f;
    }

    constexpr size_t weight_bytes = sizeof(ftype);

    __device__ __forceinline__
    void load_weights(uint64_t mem_ptr) {
        coord<> _id(0,0,0,0);
        weight[_id] = *reinterpret_cast<ftype*>(mem_ptr);
    }

    virtual __device__ __forceinline__
    void save_weights(uint64_t mem_ptr) {
        coord<> _id(0,0,0,0);
        *reinterpret_cast<ftype*>(mem_ptr) = weight[_id];
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd(int32_t batch) {
        reg_wtile_ft<IN> X, Y;
        reg_wtile_ft<1,1> W;
        load(W, weight, {0,0,0,0});

        auto w = W.tiles[0][0].data[0].x;

        for (int c = 0; c < IN.C; ++c) {
            for (int wave = 0; wave < IN.warpwaves; ++wave) {

                int2 ij = IN.warptile_xy(wave);
                coord<> idx(batch, c, ij.y, ij.x);
                load(X, x, idx);

                // #pragma unroll
                // for (int i = 0; i < X.num_elems; i++)
                //     Y.data[i] = X.data[i] * w;

                store(y, X, idx);
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

                // #pragma unroll
                // for (int i=0;i<X.num_elems;i++) {
                //     GX.data[i] = GY.data[i] * w[0].at(0,0);
                //     local_grad_w += GY.data[i] * X.data[i];
                // }

                store(grad_x, GY, idx);
            }
        }

        // atomicAdd(grad_weight[0].at(0,0), local_grad_w);  // should do parallel scan over warps
        // // Apply SGD update only if we are the first thread
        // if (threadIdx.x == 0)
        // {
        //     w[0].at(0,0) = Opt::update(w[0].at(0,0), grad_w[0].at(0,0));
        //     grad_w[0].at(0,0) = 0.0f;
        // }
    }
};

using ScaleModule = ModuleSpec<scale_module, IdentityTransform>;
