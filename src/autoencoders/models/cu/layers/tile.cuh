// #include <torch/extension.h> 
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <array>
#include <functional>
#include <variant>

using namespace kittens;

#define TILE_CUH_INCLUDED

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
};

template<typename Layout, typename TileType>
struct TileBCHW : public TileType {
    using tile_type = TileType; // for external access

    // equivalently Bzyx for threads
    // and zCyx for blocks
    // and column-major:  BC rows, cols
    Layout x, y;

    TileBCHW(const Layout& x_, const Layout& y_) : x(x_), y(y_) {}

    // Grid dimensions for kernel launch
    dim3 grid()  { return dim3(tile_nx(), tile_ny(), x.batch()); }
    dim3 block() { return dim3(NUM_THREADS); }
    unsigned long shmem_size = 100000; // 100 KB default shared memory size
    long mem() {return shmem_size;}

    // Number of block tiles in each dimension // each block handles only one tile
    __host__ __device__ int32_t tile_nx() const { return (x.cols() + TileType::B.x - 1) / TileType::B.x; }
    __host__ __device__ int32_t tile_ny() const { return (x.rows() + TileType::B.y - 1) / TileType::B.y; }
    __host__ __device__ int32_t channels() const { return x.depth(); }
    
    // Block-level indices
    __device__ __forceinline__ int32_t tile_idx() const { return blockIdx.x; }
    __device__ __forceinline__ int32_t tile_idy() const { return blockIdx.y; }
    __device__ __forceinline__ int32_t tile_x() const { return tile_idx() * TileType::B.x; }
    __device__ __forceinline__ int32_t tile_y() const { return tile_idy() * TileType::B.y; }
    __device__ __forceinline__ int32_t batch() const { return blockIdx.z; }
    
    // Warp-level indices
    __device__ __forceinline__ int32_t warp_id()  const { return threadIdx.x / kittens::WARP_THREADS; }
    static constexpr int32_t warptile_nx = TileType::B.x / TileType::W.x;
    static constexpr int32_t warptile_ny = TileType::B.y / TileType::W.y;
    static constexpr int32_t warptiles = warptile_nx * warptile_ny;
    static constexpr int32_t warpwaves = (warptiles + NUM_WORKERS - 1) / NUM_WORKERS;

    __device__ __forceinline__ int32_t warptile_linear_id(int32_t wave) const {
        return wave * NUM_WORKERS + warp_id();
    }

    __device__ __forceinline__ bool warptile_active(int32_t wave) const {
        return warptile_linear_id(wave) < warptiles;
    }

    __device__ __forceinline__ int2 warptile_ixy(int32_t wave) const {
        int32_t warptile_id = warptile_linear_id(wave);
        int32_t warptile_ix = warptile_id % warptile_nx; // iterate x (cols) first
        int32_t warptile_iy = warptile_id / warptile_nx;
        return make_int2(warptile_ix, warptile_iy);
    }
    
    __device__ __forceinline__ int2 warptile_xy(int32_t wave) const {
        int2 ij = warptile_ixy(wave);
        return make_int2(ij.x * TileType::W.x,
                    ij.y * TileType::W.y);
    }

    __device__ __forceinline__ int2 warptile_gxy(int32_t wave) const {
        int2 xy = warptile_xy(wave);
        xy.x = tile_x() + xy.x;
        xy.y = tile_y() + xy.y;
        return xy;
    }

};

template<int32_t _By, int32_t _Bx, int32_t _Wy, int32_t _Wx>
struct HW{
    static constexpr int32_t By = _By; 
    static constexpr int32_t Bx = _Bx;
    static constexpr int32_t Wy = _Wy;
    static constexpr int32_t Wx = _Wx;

    // Warp-level indices
    __device__ __forceinline__ int32_t warp_id()  const { return threadIdx.x / kittens::WARP_THREADS; }
    static constexpr int32_t warptile_nx = Bx / Wx;
    static constexpr int32_t warptile_ny = By / Wy;
    static constexpr int32_t warptiles = warptile_nx * warptile_ny;
    static constexpr int32_t warpwaves = (warptiles + NUM_WORKERS - 1) / NUM_WORKERS;

    __device__ __forceinline__ int32_t warptile_linear_id(int32_t wave) const {
        return wave * NUM_WORKERS + warp_id();
    }

    __device__ __forceinline__ bool warptile_active(int32_t wave) const {
        return warptile_linear_id(wave) < warptiles;
    }

    __device__ __forceinline__ int2 warptile_ixy(int32_t wave) const {
        int32_t warptile_id = warptile_linear_id(wave);
        int32_t warptile_ix = warptile_id % warptile_nx; // iterate x (cols) first
        int32_t warptile_iy = warptile_id / warptile_nx;
        return make_int2(warptile_ix, warptile_iy);
    }
    
    __device__ __forceinline__ int2 warptile_xy(int32_t wave) const {
        int2 ij = warptile_ixy(wave);
        return make_int2(ij.x * Wx,
                    ij.y * Wy);
    }
};

// blank fwd, bwd data structures
// struct base_layout {
//     torch::Tensor tensor;
//     void* raw_ptr() const { return reinterpret_cast<uint64_t>(tensor.data_ptr()); }
//     int64_t batch() const { return tensor.size(0); }
//     int64_t depth() const { return tensor.size(1); }
//     int64_t rows()  const { return tensor.size(2); }
//     int64_t cols()  const { return tensor.size(3); }
// };
///// notice raw_ptr is now ()

// forward facing temporaries to parse tensor

using base_layout_ = gl<ftype, -1, -1, -1, -1, st_fl<64, 64>>;

struct fwd_data
{
    base_layout_ x, y;
};
struct bwd_data
{
    base_layout_ grad_y, y, grad_x, x;
};

struct train_data {
    base_layout_ y, x;
    uint64_t weight_mem_ptr;
    uint64_t iterations;
};

// now for backward facing stuff

template<typename L, typename BL>
__host__ L LYC(BL base) {
    return make_gl<L>(
        reinterpret_cast<uint64_t>(base.raw_ptr),
        base.batch(),
        base.depth(),
        base.rows(),
        base.cols()
    );
    // layout constructor (short LYC)
}

template<typename Layout, typename TileType>
struct _BCHW_fwd : public TileBCHW<Layout, TileType> {
    _BCHW_fwd(const fwd_data& g):
        TileBCHW<Layout, TileType>(LYC<Layout>(g.x), LYC<Layout>(g.y))
    {}
};

template<typename Layout, typename TileType>
struct _BCHW_train : public TileBCHW<Layout, TileType> {
    uint64_t weight_mem_ptr;
    uint64_t iterations;

    _BCHW_train(const train_data& g):
        TileBCHW<Layout, TileType>(LYC<Layout>(g.x), LYC<Layout>(g.y)),
        weight_mem_ptr(g.weight_mem_ptr),
        iterations(g.iterations)
    {}
};

template<typename Layout, typename TileType>
struct _BCHW_bwd : public TileBCHW<Layout, TileType> {
    Layout grad_y, grad_x;
    _BCHW_bwd(const bwd_data& g):
        TileBCHW<Layout, TileType>(LYC<Layout>(g.x), LYC<Layout>(g.y)),
        grad_y(LYC<Layout>(g.grad_y)),
        grad_x(LYC<Layout>(g.grad_x))
    {}
};

// ------------------- Type aliases -------------------
template<typename TileType>
using tiled_layout = gl<ftype, G_BATCH, G_CHANNEL,
 TileType::G.y, TileType::G.x,
 st_fl<TileType::B.y, TileType::B.x>>; // bchw layout

template<typename TileType>
using reg_tile_ft = rt<ftype, TileType::W.y, TileType::W.x>;

template<typename TileType>
using reg_tile_dt = rt<dtype, TileType::W.y, TileType::W.x>;

// template<class _HW>
// using reg_wtile_ft = rt<ftype, _HW::Wy, _HW::Wx>;

// template<class _HW>
// using reg_wtile_dt = rt<dtype, _HW::Wy, _HW::Wx>;

template<typename TileType>
using BCHW_fwd = _BCHW_fwd<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_bwd = _BCHW_bwd<tiled_layout<TileType>, TileType>;

template<typename TileType>
using BCHW_train = _BCHW_train<tiled_layout<TileType>, TileType>;

using Tile28 = Tile<-1, -1, 32, 16, 32, 16>;
using Tile64 = Tile<-1, -1, 64, 32, 32, 16>; /// second is 1/2 since packed
using Tile128 = Tile<-1, -1, 128, 64, 32, 16>;

// select tile based on width
template<typename BaseData>
__host__ size_t TileIndex(const BaseData& g) {
    int W = g.x.cols();
    size_t idx = (W == 28) ? 0 : (W == 64) ? 1 : 2;
    return idx;
}

template<template<class> class DataLayout>
using layout_variant = std::variant<
    DataLayout<Tile28>,
    DataLayout<Tile64>,
    DataLayout<Tile128>
>;

template<template<class> class DataLayout, typename Data>
__host__ layout_variant<DataLayout> create_layout(Data g) {
    auto tile_idx = TileIndex(g);
    switch (tile_idx) {
        case 0: return DataLayout<Tile28>(g);
        case 1: return DataLayout<Tile64>(g);
        case 2: return DataLayout<Tile128>(g);
        default:
            throw std::runtime_error("Unsupported tile size");
    }
}

