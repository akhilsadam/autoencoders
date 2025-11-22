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

template<int Gx, int Gy, int Bx, int By, int Wx, int Wy>
struct Tile {
    static constexpr int2 G = {Gx, Gy};
    static constexpr int2 B = {Bx, By};
    static constexpr int2 W = {Wx, Wy};
    
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
    __host__ __device__ int32_t tiles_x() const { return (ref().cols() + TileType::B.x - 1) / TileType::B.x; }
    __host__ __device__ int32_t tiles_y() const { return (ref().rows() + TileType::B.y - 1) / TileType::B.y; }
    __host__ __device__ int32_t channels() const { return ref().depth(); }
    
    // Block-level indices
    __device__ int32_t tile_x()  const { return blockIdx.x; }
    __device__ int32_t tile_y()  const { return blockIdx.y; }
    __device__ int32_t batch() const { return blockIdx.z; }
    
    // Warp-level indices
    __device__ int32_t warp_id()  const { return threadIdx.x / kittens::WARP_THREADS; }
    __device__ int32_t warp_idx() const { return warp_id() % TileType::warps_x; }
    __device__ int32_t warp_idy() const { return warp_id() / TileType::warps_x; }
    
    // Global warp tile indices (for load/store operations)
    __device__ int32_t warptile_gx() const { return tile_x() * TileType::warps_x + warp_idx(); }
    __device__ int32_t warptile_gy() const { return tile_y() * TileType::warps_y + warp_idy(); }
};

// blank fwd, bwd data structures
using base_layout = gl<dtype, -1, -1, -1, -1, st_fl<1,1>>;
struct fwd_data
{
    const base_layout& x, y;
};
struct bwd_data
{
    const base_layout& grad_y, y, grad_x, x;
};

template<typename Layout, typename TileType>
struct _BCHW_fwd : public TileBCHW<Layout, TileType> {
    Layout x, y;
    _BCHW_fwd(const fwd_data& g)
        :
        x(g.x),
        y(g.y) 
    {
        this->reference = &x;
    }
};

template<typename Layout, typename TileType>
struct _BCHW_bwd_stateless : public TileBCHW<Layout, TileType> {
    Layout grad_y, y, grad_x;
    _BCHW_bwd_stateless(const bwd_data& g)
        :
        grad_y(g.grad_y), y(g.y),
        grad_x(g.grad_x)
    {
        this->reference = &y;
    }
};

template<typename Layout, typename TileType>
struct _BCHW_bwd : public TileBCHW<Layout, TileType> {
    Layout grad_y, y, grad_x, x;
    _BCHW_bwd(const bwd_data& g)
        :
        grad_y(g.grad_y), y(g.y), x(g.x),
        grad_x(g.grad_x)
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
using BCHW_fwd = _BCHW_fwd<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_bwd_stateless = _BCHW_bwd_stateless<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_bwd = _BCHW_bwd<tiled_layout<TileType>, TileType>;

using Tile28 = Tile<-1, -1, 28, 28, 28, 28>;
using Tile64 = Tile<-1, -1, 64, 64, 16, 16>;
using Tile128 = Tile<-1, -1, 128, 128, 16, 16>;

// select tile based on width

template<typename T, typename BaseData>
void dispatch_fwd_kernel(T kernel, const BaseData& g) {
    int W = g.x.cols();
    if(W == 28) {
        using Data = BCHW_fwd<Tile28>;
        Data g_cast = Data(g);
        kernel.template fwd<Data, Tile28>(g_cast);
    } 
    else if(W == 64) {
        using Data = BCHW_fwd<Tile64>;
        Data g_cast = Data(g);
        kernel.template fwd<Data, Tile64>(g_cast);
    }
    else {
        using Data = BCHW_fwd<Tile128>;
        Data g_cast = Data(g);
        kernel.template fwd<Data, Tile128>(g_cast);
    }
}

template<typename T, typename BaseData>
void dispatch_bwd_sl_kernel(T kernel, const BaseData& g) {
    int W = g.y.cols();
    if(W == 28) {
        using Data = BCHW_bwd_stateless<Tile28>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile28>(g_cast);
    } 
    else if(W == 64) {
        using Data = BCHW_bwd_stateless<Tile64>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile64>(g_cast);
    }
    else {
        using Data = BCHW_bwd_stateless<Tile128>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile128>(g_cast);
    }
}

template<typename T, typename BaseData>
void dispatch_bwd_kernel(T kernel, const BaseData& g) {
    int W = g.y.cols();
    if(W == 28) {
        using Data = BCHW_bwd<Tile28>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile28>(g_cast);
    } 
    else if(W == 64) {
        using Data = BCHW_bwd<Tile64>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile64>(g_cast);
    }
    else {
        using Data = BCHW_bwd<Tile128>;
        Data g_cast = Data(g);
        kernel.template bwd<Data, Tile128>(g_cast);
    }
}