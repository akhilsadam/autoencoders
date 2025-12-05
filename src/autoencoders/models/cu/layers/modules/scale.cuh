#include "kittens.cuh"
using namespace kittens;

#ifndef TILE_CUH_INCLUDED
#include "tile.cuh"
#endif

#ifndef NN_CUH_INCLUDED
#include "nn.cuh"
#endif

template<class IN>
using IdentityTransform = IN;

template<class IN, template<class> class Transform, class Opt>
struct scale_module : public module<IN, Transform, Opt> {
   
    // one shared weight
    using wgl = ftype; // gl<ftype,1,1,1,1>;
    using wtile = ftype; // shmem<1,1>;
    wgl* g_weight;
    wgl* g_grad_weight;
    wtile* weight;        // lives in shared memory
    wtile* grad_weight;   // gradient accumulator

    size_t weight_bytes = sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void __init_weights__(T& al) {
        // weight = al.template allocate<wtile, 1>();
        // grad_weight = al.template allocate<wtile, 1>();
        // one(*weight);
        // zero(*grad_weight);
        __shared__ wtile shm_weight;
        __shared__ wtile shm_grad_weight;
        if (threadIdx.x == 0) {
            shm_weight = 1.0f;
            shm_grad_weight = 0.0f;
        }
        // syncthreads happens outside automatically
        weight = &shm_weight;
        grad_weight = &shm_grad_weight;
    }


    __device__ __forceinline__
    void __load_weights__(uint64_t mem_ptr) {
        // g_weight = reinterpret_cast<wgl*>(mem_ptr);
        // load(*weight, *g_weight, {0,0,0,0});
        g_weight = reinterpret_cast<wgl*>(mem_ptr);
        weight[0] = *g_weight;
    }

    virtual __device__ __forceinline__
    void __save_weights__() {
        *g_weight = weight[0];
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd(int32_t batch) {
        rt<ftype, IN::By, IN::Bx> X, Y;
        // rt<ftype,1,1> W;
        // load(W, *weight);
        // auto w = W.tiles[0][0].data[0].x;

        ftype w = weight[0];

        for (int c = 0; c < IN::C; ++c) {
            // for (int wave = 0; wave < IN::warpwaves; ++wave) {

            //     int2 ij = IN::warptile_xy(wave);
            //     coord<> idx(batch, c, ij.y, ij.x);
                load(X, *(this->x));

                // #pragma unroll
                // for (int i = 0; i < X.num_elems; i++)
                //     Y.data[i] = X.data[i] * w;

                store(*(this->y), X);
            // }
        }
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd(int32_t batch) {
        rt<ftype, IN::By, IN::Bx> GX, GY, X;

        ftype local_grad_w = 0.0f;

        for (int c = 0; c < IN::C; ++c) {
            // for (int wave = 0; wave < IN::warpwaves; ++wave) {

                // int2 ij = IN::warptile_xy(wave);
                // coord<> idx(batch, c, ij.y, ij.x);

                load(X, *(this->x));
                load(GY, *(this->grad_y));

                // #pragma unroll
                // for (int i=0;i<X.num_elems;i++) {
                //     GX.data[i] = GY.data[i] * w[0].at(0,0);
                //     local_grad_w += GY.data[i] * X.data[i];
                // }

                store(*(this->grad_x), GY);
            // }
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
