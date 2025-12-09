#include "kittens.cuh"
using namespace kittens;

#include "tile.cuh"
#include "nn.cuh"
#include "ops/frag.cuh"
#include "ops/scan.cuh"

template<class _IN>
using IdentityTransform = _IN;

template<class _IN, template<class> class Transform, class Opt>
struct PixelDNModule : public module<_IN, Transform, Opt> {
   
    using IN = _IN;
    using OUT = Transform<IN>;

    static constexpr uint32_t k_in = 4; // template this
    static constexpr uint32_t k_out = 4; // half of k_in



    static constexpr uint32_t l_in = IN::C * k_in * k_in;
    static constexpr uint32_t l_out = OUT::C * k_out * k_out;
    static constexpr uint32_t n_in = IN::N_WT / l_in;
    static constexpr uint32_t n_out = OUT::N_WT / l_out;

    // matmul is n,l * Ll -> n,L

    // one shared weight
    // using wgl = ftype; // gl<ftype,1,1,1,1>;
    // using wtile = ftype; // shmem<1,1>;
    // wgl* g_weight;        // pointer to global memory    
    // wtile* weight;        // pointer to shared memory
    // wtile* grad_weight;   // pointer to shared memory for gradient


    using wtile_mat = st<ftype,l_out,l_in>;
    ftype* g_weight_mat;          // pointer to global memory    
    wtile_mat* weight_mat;        // pointer to shared memory
    wtile_mat* grad_weight_mat;   // pointer to shared memory for gradient

    // static constexpr uint32_t n_weights = 1;
    static constexpr uint32_t n_weights = (l_out * l_in);
    static constexpr size_t weight_bytes =  n_weights * sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void __init_weights__(T& al) {

        // __shared__ wtile shm_weight;
        // __shared__ wtile shm_grad_weight;

        // if (threadIdx.x == 0) {
        //     shm_weight = 1.0f;
        //     shm_grad_weight = 0.0f;
        // }
        // // syncthreads happens outside automatically
        // weight = &shm_weight;
        // grad_weight = &shm_grad_weight;



        weight_mat = &al.template allocate<wtile_mat>();
        grad_weight_mat = &al.template allocate<wtile_mat>();


    }


    __device__ __forceinline__
    void __load_weights__(uint64_t mem_ptr) {
        // g_weight = reinterpret_cast<wgl*>(mem_ptr);
        // load(*weight, *g_weight, {0,0,0,0});

        // g_weight = reinterpret_cast<wgl*>(mem_ptr);// + (l_out * l_in) * sizeof(ftype));
        // weight[0] = *g_weight;

        // g_weight_mat = reinterpret_cast<wgl_mat*>(mem_ptr);
        // g_weight_mat[0].raw_ptr = reinterpret_cast<ftype*>(mem_ptr);
        
        
        // if (threadIdx.x == 0)
        // {
        //     printf("pointer %p %p -> %p\n", mem_ptr, g_weight_mat, weight_mat);
        // }

        g_weight_mat = reinterpret_cast<ftype*>(mem_ptr);
        aligned_load_to_st<l_in, wtile_mat>(weight_mat[0], g_weight_mat);

    
    }

    __device__ __forceinline__
    void __save_weights__() {
        aligned_store_to_gl<l_out, wtile_mat>(g_weight_mat, weight_mat[0]);
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd() {
        typename IN::reg_array X;
        typename OUT::reg_array Y;
        rt<ftype, n_in, l_in> X_flat; // n,l layout
        rt<ftype, n_out, l_out> Y_flat;
        rt<ftype, n_out, l_out> ZY_flat; // n,l layout

        zero(ZY_flat);

        // typename IN::reg_wp X;
        // typename OUT::reg_wp Y;
        
        // ftype w = weight[0];

        rt<ftype,l_out,l_in,ducks::rt_layout::col> W_flat;
        load(W_flat, *weight_mat);


        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        {
            printf("weight mat 0,0: %f\n", W_flat.tiles[0][0].data[0].x);
            // printf("Scale weight: %f\n", w);
            // printf("N_in: %d, l_in: %d, N_out: %d, l_out: %d\n", n_in, l_in, n_out, l_out);
        }

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            // // expecting a tile of size 16x16(xPx2 pack, p=1 for now)

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
            }
            tile_to_flat<IN::C, k_in>(X_flat, X);

            // Y (n,L) <- X (n,l) * W (L,l)^T
            mma_ABt(Y_flat, X_flat, W_flat, ZY_flat);  

            flat_to_tile<IN::C, k_in>(Y, Y_flat);
            for (int c = 0; c < OUT::C; ++c) 
            {
                store(this->y[0][ij.y][ij.x][c], Y[c]);
            }

            __syncwarp();

            // for (int c = 0; c < IN::C; ++c) 
            // {
            //     load(X, this->x[0][ij.y][ij.x][c]);
            //     bin_map<base_ops::mul>(Y, X, w);
            //     store(this->y[0][ij.y][ij.x][c], Y);
            //     __syncwarp();
            // }
        }
        
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd() {
        // typename IN::reg_wp GX, X;
        // typename OUT::reg_wp GY;

        typename IN::reg_array GX, X;
        typename OUT::reg_array GY;
        rt<ftype, n_in, l_in> GX_flat;
        rt<ftype, n_in, l_in, ducks::rt_layout::col> X_flat; // n,l layout
        rt<ftype, n_out, l_out> GY_flat; // n,l layout

        rt<ftype,l_out,l_in,ducks::rt_layout::col> W_flat;
        rt<ftype,l_out,l_in> GW_flat;
        load(W_flat, *weight_mat);

        zero(GX_flat);
        zero(GW_flat);


        // ftype w = weight[0];
        // ftype reg_grad_w = 0.0f;

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
                load(GY[c], this->grad_y[0][ij.y][ij.x][c]);
            }
            tile_to_flat<IN::C, k_in>(X_flat, X);
            tile_to_flat<IN::C, k_in>(GY_flat, GY);

            /////
            
            // GX (n,l) += GY (n,L) * W (L,l)
            mma_AB(GX_flat, GY_flat, W_flat, GX_flat);  

            // GA += GY (n,L)^T * X (n,l)
            mma_AtB(GW_flat, GY_flat, X_flat, GW_flat);

            /////
            flat_to_tile<IN::C, k_in>(GX, GX_flat);
            for (int c = 0; c < IN::C; ++c)
            {
                store(this->grad_x[0][ij.y][ij.x][c], GX[c]);
            }
            __syncwarp();

        }

        // accumulate gradient (parallel scan) across warps
        // frag_collect collects fragments across lanes in a warp
        // scan::frag_collect(reg_grad_w); 
        // scan::atomic_store(grad_weight[0], reg_grad_w);

        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        // {
        //     printf("grad weight: %f\n", grad_weight[0]);
        //     // printf("N_in: %d, l_in: %d, N_out: %d, l_out: %d\n", n_in, l_in, n_out, l_out);
        // }

        // // Apply SGD update 
        // Opt::update(weight[0], grad_weight[0]);
    }
};

using PixelDown = ModuleSpec<PixelDNModule, IdentityTransform>;
