#include "kittens.cuh"
using namespace kittens;

#include "tile.cuh"
#include "nn.cuh"
#include "ops/frag.cuh"
#include "ops/scan.cuh"

template<class _IN, template<class> class Transform, class Opt>
struct ChannelModuleBase : public module<_IN, Transform, Opt> {
   
    using IN = _IN;
    using OUT = Transform<IN>;

    // one shared weight
    using wgl = ftype[OUT::C][IN::C + 1]; 
    using wmat = ftype[OUT::C][IN::C + 1]; // +1 for bias
    wgl* g_weight;        // pointer to global memory    
    wmat* weight;        // pointer to shared memory
    wmat* grad_weight;   // pointer to shared memory for gradient

    static constexpr size_t weight_bytes = sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void __init_weights__(T& al) {
        // weight = al.template allocate<wtile, 1>();
        // grad_weight = al.template allocate<wtile, 1>();
        // one(*weight);
        // zero(*grad_weight);
        __shared__ wmat shm_weight;
        __shared__ wmat shm_grad_weight;
        // if (threadIdx.x == 0) {
        //     shm_weight = 1.0f;
        //     shm_grad_weight = 0.0f;
        // }
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

    __device__ __forceinline__
    void __save_weights__() {
        *g_weight = weight[0];
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd() {
        typename IN::reg_array X;
        typename OUT::reg_array Y;
        
        wmat w;
        for (int oc = 0; oc < OUT::C; ++oc) 
        {
            for (int ic = 0; ic < IN::C + 1; ++ic) 
            {
                w[oc][ic] = weight[0][oc][ic];
            }
        }


        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            // // expecting a tile of size 16x16(xPx2 pack, p=1 for now)

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
            }

            // Y = X * w
            for (int oc = 0; oc < OUT::C; ++oc) 
            {
                zero(Y[oc]);
                for (int ic = 0; ic < IN::C; ++ic) 
                {   
                    // Y[oc] += X[ic] * w[oc][ic] + w[oc][IN::C]; // bias
                    scalar_fma_map(Y[oc], X[ic], w[oc][ic], Y[oc]);
                }
                bin_map<base_ops::sum>(Y[oc], w[oc][IN::C]); // bias
            }
  

            for (int c = 0; c < OUT::C; ++c) 
            {
                store(this->y[0][ij.y][ij.x][c], Y[c]);
            }

            __syncwarp();
        }
        
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd() {
        typename IN::reg_array GX, X;
        typename OUT::reg_array GY;

        wmat w;
        wmat reg_grad_w;

        for (int oc = 0; oc < OUT::C; ++oc) 
        {
            for (int ic = 0; ic < IN::C + 1; ++ic) 
            {
                w[oc][ic] = weight[0][oc][ic];
                reg_grad_w[oc][ic] = 0.0f;
            }
        }

        
        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            // // expecting a tile of size 16x16(xPx2 pack, p=1 for now)

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
            }
            for (int c = 0; c < OUT::C; ++c)    
            {   
                load(GY[c], this->grad_y[0][ij.y][ij.x][c]);
            }

            // GX = GY * W^T
            // GW = GY^T * [X,1]

            for (int ic = 0; ic < IN::C; ++ic) 
            {
                zero(GX[ic]);
                for (int oc = 0; oc < OUT::C; ++oc) 
                {   
                    // GX[ic] += GY[oc] * w[oc][ic];
                    scalar_fma_map(GX[ic], GY[oc], w[oc][ic], GX[ic]);
                    
                    // reg_grad_w[oc][ic] += frag_dot(GY[oc], X[ic]);
                    frag_dot(reg_grad_w[oc][ic], GY[oc], X[ic]);    
                }
            }
  
            // bias gradient
            for (int oc = 0; oc < OUT::C; ++oc) 
            {
                // reg_grad_w[oc][IN::C] += frag_sum(GY[oc]);
                frag_sum(reg_grad_w[oc][IN::C], GY[oc]);
            }

            for (int c = 0; c < OUT::C; ++c) 
            {
                store(this->grad_x[0][ij.y][ij.x][c], GX[c]);
            }

            __syncwarp();
        }

        // accumulate gradient (parallel scan) across warps
        // frag_collect collects fragments across lanes in a warp
        for (int oc = 0; oc < OUT::C; ++oc)
        {
            for (int ic = 0; ic < IN::C + 1; ++ic)
            {
                scan::frag_collect(reg_grad_w[oc][ic]);
                scan::atomic_store(grad_weight[0][oc][ic], reg_grad_w);
                // Apply SGD update
                if(threadIdx.x == 0) 
                {
                    Opt::update(weight[0][oc][ic], grad_weight[0][oc][ic]);
                }
            }
        }


    }
};

template<int32_t C>
struct ChannelTransform
{
    template<class _IN>
    using T = CHW<C, typename _IN::TT>;
};


template<int32_t C>
using ChannelModule = ModuleSpec<ChannelModuleBase, ChannelTransform<C>::template T>;