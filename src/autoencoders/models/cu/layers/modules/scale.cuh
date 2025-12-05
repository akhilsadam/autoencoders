#include "kittens.cuh"
using namespace kittens;

#ifndef TILE_CUH_INCLUDED
#include "tile.cuh"
#endif

#ifndef NN_CUH_INCLUDED
#include "nn.cuh"
#endif

struct IdentityTransform {
    template<typename _HW>
    static constexpr _HW apply(_HW IN) {
        return IN;
    }
};

template<HW IN, class Transform, class Opt>
struct scale_module : public module<IN, Transform, Opt> {
   
    // one shared weight
    using wgl = gl<ftype,1,1,1,1>;
    using wtile = shmem<1,1>;
    wgl* g_weight;
    wgl* g_grad_weight;
    wtile* weight;        // lives in shared memory
    wtile* grad_weight;   // gradient accumulator

    size_t weight_bytes = sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void init_weights(T& al) {
        weight = al.template allocate<shmem<1,1>, 1>();
        grad_weight = al.template allocate<shmem<1,1>, 1>();
        one(*weight);
        zero(*grad_weight);
    }


    __device__ __forceinline__
    void load_weights(uint64_t mem_ptr) {
        g_weight = reinterpret_cast<wgl*>(mem_ptr);
        load(*weight, *g_weight, {0,0,0,0});
    }

    virtual __device__ __forceinline__
    void save_weights() {
        store(*g_weight, *weight);
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd(int32_t batch) {
        rt<ftype, IN.Wy, IN.Wx> X, Y;
        rt<ftype,1,1> W;
        load(W, *weight);

        auto w = W.tiles[0][0].data[0].x;

        for (int c = 0; c < IN.C; ++c) {
            for (int wave = 0; wave < IN.warpwaves; ++wave) {

                int2 ij = IN.warptile_xy(wave);
                coord<> idx(batch, c, ij.y, ij.x);
                load(X, this->x, idx);

                // #pragma unroll
                // for (int i = 0; i < X.num_elems; i++)
                //     Y.data[i] = X.data[i] * w;

                store(this->y, X, idx);
            }
        }
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd(int32_t batch) {
        rt<ftype, IN.Wy, IN.Wx> GX, GY, X;

        ftype local_grad_w = 0.0f;

        for (int c = 0; c < IN.C; ++c) {
            for (int wave = 0; wave < IN.warpwaves; ++wave) {

                int2 ij = IN.warptile_xy(wave);
                coord<> idx(batch, c, ij.y, ij.x);

                load(X, this->x, idx);
                load(GY, this->grad_y, idx);

                // #pragma unroll
                // for (int i=0;i<X.num_elems;i++) {
                //     GX.data[i] = GY.data[i] * w[0].at(0,0);
                //     local_grad_w += GY.data[i] * X.data[i];
                // }

                store(this->grad_x, GY, idx);
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
