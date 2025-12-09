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
    // static constexpr uint32_t n_in = OUT::N_WT / l_out;
    // will be the same

    // matmul is n,l * Ll -> n,L

    // one shared weight
    // using wgl = ftype; // gl<ftype,1,1,1,1>;
    // using wtile = ftype; // shmem<1,1>;
    // wgl* g_weight;        // pointer to global memory    
    // wtile* weight;        // pointer to shared memory
    // wtile* grad_weight;   // pointer to shared memory for gradient


    using wtile_mat = st<smtype,l_out,l_in>;
    using wmarray = wtile_mat[NUM_WORKERS];
    smtype* g_weight_mat;          // pointer to global memory    
    wtile_mat* weight_mat;        // pointer to shared memory
    wmarray* grad_weight_mat;   // pointer to shared memory for gradient

    // static constexpr uint32_t n_weights = 1;
    static constexpr uint32_t n_weights = (l_out * l_in);
    static constexpr size_t weight_bytes =  n_weights * sizeof(ftype);

    // ------------------ weights ----------------------
    template <typename T>
    __device__ __forceinline__ void __init_weights__(T& al) {
        weight_mat = &al.template allocate<wtile_mat>();
        grad_weight_mat = &al.template allocate<wtile_mat, NUM_WORKERS>();
        // unfortunately, need one per warp...
        if (warpid() == 0)
            zero(weight_mat[0]);
    }

    __device__ __forceinline__
    void __load_weights__(uint64_t mem_ptr) {
        g_weight_mat = reinterpret_cast<smtype*>(mem_ptr);
        aligned_load_to_st<smtype, l_in, wtile_mat>(weight_mat[0], g_weight_mat);
    }

    __device__ __forceinline__
    void __save_weights__() {
        aligned_store_to_gl<smtype, l_out, wtile_mat>(g_weight_mat, weight_mat[0]);
    }

    // ------------------ fwd() ----------------------
    __device__ __forceinline__ void fwd() {
        typename IN::reg_array X;
        typename OUT::reg_array Y;
        rt<smtype, n_in, l_in> X_flat; // n,l layout

        rt<smtype, l_out, l_in> W_flat;
        load(W_flat, *weight_mat);

        rt<ftype, n_in, l_out> Y_flat;
        rt<ftype, n_in, l_out> ZY_flat; // n,l layout


        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        {
            // printf("weight mat 0,0: %d\n", W_flat.tiles[0][0].data[0].x);
            // printf("Scale weight: %f\n", w);
            // printf("N_in: %d, l_in: %d, N_out: %d, l_out: %d\n", n_in, l_in, n_in, l_out);
        }

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);
            // // expecting a tile of size 16x16(xPx2 pack, p=1 for now)

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
            }
            cast_tile_to_flat<IN::C, k_in>(X_flat, X);

            // Y (n,L) = X (n,l) * W (L,l)^T
            // row, row, row, row
            zero(Y_flat);
            zero(ZY_flat);
            mma_ABt(Y_flat, X_flat, W_flat, ZY_flat);  

            flat_to_tile<OUT::C, k_out>(Y, Y_flat);  // Fixed: should use OUT::C and k_out
            for (int c = 0; c < OUT::C; ++c) 
            {
                store(this->y[0][ij.y][ij.x][c], Y[c]);
            }

            __syncwarp();


            // now check that the MMA is correct
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            {
                int yi = 4;
                int xi = 3;
                float val_hat = 0.0f;
                for(int u=0; u < l_in; u++){
                    val_hat += __bfloat162float(X_flat.tiles[yi][u].data[0].x * W_flat.tiles[xi][u].data[0].x);
                }

                printf("\n");
                for (int u=0; u < l_in; u++){
                    // float x_val = __bfloat162float_rn(X_flat.tiles[yi][u].data[0].x);
                    float w_val = __bfloat162float(W_flat.tiles[xi][u].data[0].x);
                    printf("%f ", w_val);
                }
                printf("\n");
                float err = val_hat - Y_flat.tiles[yi][xi].data[0].x;
                // if (abs(err) > 0.1f){
                    printf("FWD Mma err at y=%d,x=%d: %f (val_hat: %f, val: %f)\n", yi, xi, err, val_hat, Y_flat.tiles[yi][xi].data[0].x);
                // }

            }

        }
        
    }

    // ------------------ bwd() ----------------------
    __device__ __forceinline__ void bwd() {
        // typename IN::reg_wp GX, X;
        // typename OUT::reg_wp GY;

        typename IN::reg_array GX, X;
        typename OUT::reg_array GY;
        
        rt<smtype, n_in, l_in, ducks::rt_layout::col> X_flat; // n,l layout
        rt<smtype, n_in, l_out> GY_flat; // n,l layout
        rt<smtype, n_in, l_out, ducks::rt_layout::col> GY_flat_col; // n,l layout
        rt<smtype, l_out, l_in, ducks::rt_layout::col> W_flat;
        load(W_flat, *weight_mat);

        rt<ftype, n_in, l_in> GX_flat;
        rt<ftype, l_out, l_in> GW_flat;
        zero(GW_flat);


        // ftype w = weight[0];
        // ftype reg_grad_w = 0.0f;

        for (int wave = 0; wave < IN::warpwaves; ++wave) 
        {
            int2 ij = IN::warptile_ixy(wave);

            for (int c = 0; c < IN::C; ++c) 
            {
                load(X[c], this->x[0][ij.y][ij.x][c]);
            }
            for (int c = 0; c < OUT::C; ++c)
            {
                load(GY[c], this->grad_y[0][ij.y][ij.x][c]);
            }
            cast_tile_to_flat<IN::C, k_in>(X_flat, X);
            cast_tile_to_flat<OUT::C, k_out>(GY_flat, GY);
            cast_tile_to_flat<OUT::C, k_out>(GY_flat_col, GY);

            /////
            
            // GX (n,l) = GY (n,L) * W (L,l)
            // row, row, col, row
            zero(GX_flat);
            mma_AB(GX_flat, GY_flat, W_flat, GX_flat);  


           // now check that the MMA is correct
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
            {
                int ci = 1;
                int yi = 4;
                int xi = 3;
                float val_hat = 0.0f;
                for(int u=0; u < l_out; u++){
                    val_hat += __bfloat162float(GY_flat.tiles[yi][u].data[0].x) * __bfloat162float(W_flat.tiles[u][xi].data[0].x);
                }

                float err = val_hat - GX_flat.tiles[yi][xi].data[0].x;
                if (abs(err) > 0.1f){
                    printf("FWD Mma err at c=%d,y=%d,x=%d: %f (val_hat: %f, val: %f)\n", ci, yi, xi, err, val_hat, GX_flat.tiles[yi][xi].data[0].x);
                }

            }




            // // need to check if this is equiv.
            // rt<smtype, l_in, l_out> W_flat_T;  // Transposed dimensions
            // transpose(W_flat_T, W_flat);
            // mma_ABt(GX_flat, GY_flat, W_flat_T, GX_flat);

            // GA += GY (n,L)^T * X (n,l)
            // row, col, col, row
            mma_AtB(GW_flat, GY_flat_col, X_flat, GW_flat);
            // mismatch -> maybe use transposes instead
            // then row row col row
            // transpose_inplace(GY_flat); does not quite work

            /////
            flat_to_tile<IN::C, k_in>(GX, GX_flat);
            for (int c = 0; c < IN::C; ++c)
            {
                store(this->grad_x[0][ij.y][ij.x][c], GX[c]);
            }
            __syncwarp();

        }

        // accumulate gradient (TODO parallel scan) across warps
        // for now serial + slow add 
        store(grad_weight_mat[0][warpid()], GW_flat);
        __syncthreads();
        // if (warpid() == 0) {
        //     for (int w = 1; w < NUM_WORKERS; ++w)
        //     {
        //         add(grad_weight_mat[0][0], grad_weight_mat[0][0], grad_weight_mat[0][w]);
        //     }
        // }
        
        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) 
        // {
        //     printf("grad weight: %f\n", grad_weight[0]);
        //     // printf("N_in: %d, l_in: %d, N_out: %d, l_out: %d\n", n_in, l_in, n_in, l_out);
        // }

        // Apply SGD update 
        if (warpid() == 0){
            update_bin_map_st<Opt, rt<smtype, l_out, l_in>>(weight_mat[0], grad_weight_mat[0][0]);
        }
    }
};

using PixelDown = ModuleSpec<PixelDNModule, IdentityTransform>;
