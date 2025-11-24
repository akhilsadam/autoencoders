// #include <torch/extension.h> 
#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
#include <array>
#include <functional>
#include <variant>

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
    using tile_type = TileType; // for external access

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
    unsigned long shmem_size = 100000; // 100 KB default shared memory size
    long mem() {return shmem_size;}

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

// now for backward facing stuff

// template<typename L, typename BL>
// __host__ L LYC(BL base) {
//     return make_gl<L>(
//         reinterpret_cast<uint64_t>(base.raw_ptr),
//         base.batch(),
//         base.depth(),
//         base.rows(),
//         base.cols()
//     );
//     // layout constructor (short LYC)
// }

// template<typename Layout, typename TileType>
// struct _BCHW_fwd : public TileBCHW<Layout, TileType> {
//     Layout x, y;
//     _BCHW_fwd(const fwd_data& g):
//         x(LYC<Layout>(g.x)),
//         y(LYC<Layout>(g.y))
//     {
//         this->reference = &x;
//         // printf("GL parts of x: B=%d C=%d R=%d C=%d\n", g.x.batch_internal, g.x.depth_internal, g.x.rows_internal, g.x.cols_internal);
//     }
// };

// template<typename Layout, typename TileType>
// struct _BCHW_bwd_stateless : public TileBCHW<Layout, TileType> {
//     Layout grad_y, y, grad_x;
//     _BCHW_bwd_stateless(const bwd_data& g):
//         grad_y(LYC<Layout>(g.grad_y)),
//              y(LYC<Layout>(g.y)),
//         grad_x(LYC<Layout>(g.grad_x))
//     {
//         this->reference = &y;
//     }
// };

// template<typename Layout, typename TileType>
// struct _BCHW_bwd : public TileBCHW<Layout, TileType> {
//     Layout grad_y, y, grad_x, x;
//     _BCHW_bwd(const bwd_data& g):
//         grad_y(LYC<Layout>(g.grad_y)),
//              y(LYC<Layout>(g.y)),
//              x(LYC<Layout>(g.x)),
//         grad_x(LYC<Layout>(g.grad_x))
//     {
//         this->reference = &x;
//     }
// };

// // ------------------- Type aliases -------------------
// template<typename TileType>
// using tiled_layout = gl<dtype, G_BATCH, G_CHANNEL,
//  TileType::G.y, TileType::G.x,
//  st_fl<TileType::B.y, TileType::B.x>>; // bchw layout

// template<typename TileType>
// using reg_tile_ft = rt<ftype, TileType::W.y, TileType::W.x>;

// template<typename TileType>
// using reg_tile_dt = rt<dtype, TileType::W.y, TileType::W.x>;

// template<typename TileType>
// using BCHW_fwd = _BCHW_fwd<tiled_layout<TileType>, TileType>;

// template<typename TileType>
// using BCHW_bwd_stateless = _BCHW_bwd_stateless<tiled_layout<TileType>, TileType>;

// template<typename TileType>
// using BCHW_bwd = _BCHW_bwd<tiled_layout<TileType>, TileType>;

// using Tile28 = Tile<-1, -1, 32, 32, 16, 16>;
// using Tile64 = Tile<-1, -1, 64, 64, 16, 16>;
// using Tile128 = Tile<-1, -1, 128, 128, 16, 16>;

// // select tile based on width
// template<typename BaseData>
// __host__ size_t TileIndex(const BaseData& g) {
//     int W = g.x.cols();
//     size_t idx = (W == 28) ? 0 : (W == 64) ? 1 : 2;
//     return idx;
// }

// template<template<typename TileType> class layoutv>
// using layout_variant = std::variant<
//     layoutv<Tile28>,
//     layoutv<Tile64>,
//     layoutv<Tile128>
// >;

// template<template<typename TileType> class layoutv, typename _data>
// __host__ layout_variant<layoutv> create_layout(_data g) {
//     auto tile_idx = TileIndex(g);
//     switch (tile_idx) {
//         case 0:
//             return layoutv<Tile28>(g);
//         case 1:
//             return layoutv<Tile64>(g);
//         case 2:
//             return layoutv<Tile128>(g);
//         default:
//             throw std::runtime_error("Unsupported tile size");
//     }
// }


