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
    static constexpr uint32_t n_in = IN::N_subtile / l_in;
    static constexpr uint32_t n_out = OUT::N_subtile / l_out;

    // matmul is n,l * Ll -> n,L

    // one shared weight
    using wgl = ftype; // gl<ftype,1,1,1,1>;
    using wtile = ftype; // shmem<1,1>;
    wgl* g_weight;        // pointer to global memory    
    wtile* weight;        // pointer to shared memory
    wtile* grad_weight;   // pointer to shared memory for gradient


    using wgl_mat = gl<ftype,1,1,l_out,l_in>; // weight for matmul
    using wtile_mat = st<ftype,l_out,l_in>;
    wgl_mat* g_weight_mat;        // pointer to global memory    
    wtile_mat* weight_mat;        // pointer to shared memory
    wtile_mat* grad_weight_mat;   // pointer to shared memory for gradient

    static constexpr uint32_t n_weights = 1 + 3 + (l_out * l_in); // scale + (align) + matmul
    static constexpr size_t weight_bytes =  n_weights * sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void __init_weights__(T& al) {

        __shared__ wtile shm_weight;
        __shared__ wtile shm_grad_weight;

        weight_mat = &al.template allocate<wtile_mat>();
        grad_weight_mat = &al.template allocate<wtile_mat>();

        if (threadIdx.x == 0) {
            shm_weight = 1.0f;
            shm_grad_weight = 0.0f;
        }
        // one(shm_weight_mat);
        // zero(shm_grad_weight_mat);
        // syncthreads happens outside automatically
        weight = &shm_weight;
        grad_weight = &shm_grad_weight;
    }


    __device__ __forceinline__
    void __load_weights__(uint64_t mem_ptr) {
        // g_weight = reinterpret_cast<wgl*>(mem_ptr);
        // load(*weight, *g_weight, {0,0,0,0});
        g_weight = reinterpret_cast<wgl*>(mem_ptr + (l_out * l_in) * sizeof(ftype));
        weight[0] = *g_weight;

        // g_weight_mat = reinterpret_cast<wgl_mat*>(mem_ptr);
        // load(*weight_mat, *g_weight_mat, {0,0,0,0});
    }

    __device__ __forceinline__
    void __save_weights__() {
        *g_weight = weight[0];

        // store(*g_weight_mat, *weight_mat, {0,0,0,0});
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd() {
        typename IN::reg_wp X;
        typename OUT::reg_wp Y;
        rt<ftype, n_in, l_in> X_flat; // n,l layout
        rt<ftype, n_out, l_out> Y_flat; // n,l layout
        
        ftype w = weight[0];

        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        //{
        //     printf("Scale weight: %f\n", w);
        // }

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            // expecting a tile of size 16x16(xPx2 pack, p=1 for now)
            tile_to_flat<IN::C, k_in>(X_flat, this->x[0][ij.y][ij.x]);

            bin_map<base_ops::mul>(Y_flat, X_flat, w);

            flat_to_tile<IN::C, k_in>(this->y[0][ij.y][ij.x], Y_flat);

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
        typename IN::reg_wp GX, X;
        typename OUT::reg_wp GY;

        ftype w = weight[0];
        ftype reg_grad_w = 0.0f;

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            for (int c = 0; c < IN::C; ++c) 
            {

                load(X, this->x[0][ij.y][ij.x][c]);
                load(GY, this->grad_y[0][ij.y][ij.x][c]);

                bin_map<base_ops::mul>(GX, GY, w); // GX.data[i] = GY.data[i] * w;
                frag_dot(reg_grad_w, GY, X); // reg_grad_w += GY.data[i] * X.data[i];

                store(this->grad_x[0][ij.y][ij.x][c], GX);
                __syncwarp();
            }
        }

        // accumulate gradient (parallel scan) across warps
        // frag_collect collects fragments across lanes in a warp
        scan::frag_collect(reg_grad_w); 
        scan::atomic_store(grad_weight[0], reg_grad_w);

        // Apply SGD update 
        Opt::update(weight[0], grad_weight[0]);
    }
};

using PixelDown = ModuleSpec<PixelDNModule, IdentityTransform>;
