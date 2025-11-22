#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

#ifndef NUM_WORKERS
#define NUM_WORKERS (1)
#endif

#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

#ifndef G_BATCH
#define G_BATCH -1
#endif

#ifndef G_CHANNEL
#define G_CHANNEL -1
#endif

// #ifndef GTILE_XY
// #define GTILE_XY {-1, -1}
// #endif

// #ifndef BLOCKTILE_XY
// #define BLOCKTILE_XY {128, 128}
// #endif

// #ifndef WARPTILE_XY
// #define WARPTILE_XY {16, 16}
// #endif

#ifndef ftype
#define ftype float
#endif

#ifndef dtype
#define dtype bf16
#endif

template<int2 G, int2 B, int2 W>
struct Tile {
    // static constexpr int2 G = GTILE_XY;
    // static constexpr int2 B = BLOCKTILE_XY;
    // static constexpr int2 W = WARPTILE_XY;
    
    // Tile hierarchy ratios
    static constexpr uint32_t warps_x = B.x / W.x;
    static constexpr uint32_t warps_y = B.y / W.y;
};

template<typename Layout, typename TileType>
struct TileBCHW : public TileType {
    // equivalently Bzyx for threads
    // and zCyx for blocks
    // and column-major:  BC rows, cols
    const Layout* reference;

    __host__ __device__ const Layout& ref() const { 
        return *reference;
    }

    // Grid dimensions for kernel launch
    dim3 grid()  { return dim3(tiles_x(), tiles_y(), ref().batch()); }
    dim3 block() { return dim3(NUM_THREADS); }
    unsigned long mem_size = 100000; // 100 KB default shared memory size
    
    // Number of block tiles in each dimension
    __host__ __device__ int32_t tiles_x() const { return (ref().cols() + B.x - 1) / B.x; }
    __host__ __device__ int32_t tiles_y() const { return (ref().rows() + B.y - 1) / B.y; }
    __host__ __device__ int32_t channels() const { return ref().depth(); }
    
    // Block-level indices
    __device__ int32_t tile_x()  const { return blockIdx.x; }
    __device__ int32_t tile_y()  const { return blockIdx.y; }
    __device__ int32_t batch() const { return blockIdx.z; }
    
    // Warp-level indices
    __device__ int32_t warp_id()  const { return threadIdx.x / kittens::WARP_THREADS; }
    __device__ int32_t warp_idx() const { return warp_id() % warps_x; }
    __device__ int32_t warp_idy() const { return warp_id() / warps_x; }
    
    // Global warp tile indices (for load/store operations)
    __device__ int32_t warptile_gx() const { return tile_x() * warps_x + warp_idx(); }
    __device__ int32_t warptile_gy() const { return tile_y() * warps_y + warp_idy(); }
};

// Forward pass data structure (x, y)
template<typename Layout, typename TileType>
struct _BCHW_fwd : public TileBCHW<Layout, TileType> {
    Layout x, y;
    _BCHW_fwd(
        const Layout& x_, 
        const Layout& y_)
        :
        x(x_),
        y(y_) 
    {
        this->reference = &x;
    }
};

template<typename Layout, typename TileType>
struct _BCHW_bwd_stateless : public TileBCHW<Layout, TileType> {
    Layout grad_y, y, grad_x;
    _BCHW_bwd_stateless(
        const Layout& grad_y_,
        const Layout& y_,
        const Layout& grad_x_)
        :
        grad_y(grad_y_), y(y_),
        grad_x(grad_x_)
    {
        this->reference = &y;
    }
};

template<typename Layout, typename TileType>
struct _BCHW_bwd : public TileBCHW<Layout, TileType> {
    Layout grad_y, y, grad_x, x;
    _BCHW_bwd(const Layout& grad_y_,
        const Layout& y_,
        const Layout& x_,
        const Layout& grad_x_)
        :
        grad_y(grad_y_), y(y_), x(x_),
        grad_x(grad_x_)
    {
        this->reference = &x;
    }
};

// ------------------- Type aliases -------------------

template<typename TileType>
using tiled_layout = gl<dtype, G_BATCH, G_CHANNEL,
 TileType::G.y, TileType::G.x,
 st_fl<TileType::B.y, TileType::B.x>>; // bchw layout

template<typename TileType>
using reg_tile_ft = rt<ftype, TileType::W.y, TileType::W.x>;

template<typename TileType>
using reg_tile_dt = rt<dtype, TileType::W.y, TileType::W.x>;

template<typename TileType>
using fwd_data = _BCHW_fwd<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_bwd_stateless = _BCHW_bwd_stateless<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_bwd = _BCHW_bwd<tiled_layout<TileType>, TileType>;

using Tile28 = Tile<{-1, -1}, {28, 28}, {28, 28}>;
using Tile64 = Tile<{-1, -1}, {64, 64}, {16, 16}>;
using Tile128 = Tile<{-1, -1}, {128, 128}, {16, 16}>;

// select tile based on width

template<typename T, typename Data>
void dispatch_fwd_kernel(T kernel, const Data& g) {
    int W = g.x.cols();
    if(W == 28) {
        kernel.template fwd<Tile28>(g);
    } 
    else if(W == 64) {
        kernel.template fwd<Tile64>(g);
    }
    else {
        kernel.template fwd<Tile128>(g);
    }
}

template<typename T, typename Data>
void dispatch_bwd_kernel(T kernel, const Data& g) {
    int W = g.y.cols();
    if(W == 28) {
        kernel.template bwd<Tile28>(g);
    } 
    else if(W == 64) {
        kernel.template bwd<Tile64>(g);
    }
    else {
        kernel.template bwd<Tile128>(g);
    }
}