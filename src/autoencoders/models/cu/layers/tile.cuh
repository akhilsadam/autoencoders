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

#ifndef GTILE_XY
#define GTILE_XY {-1, -1}
#endif

#ifndef BLOCKTILE_XY
#define BLOCKTILE_XY {128, 128}
#endif

#ifndef WARPTILE_XY
#define WARPTILE_XY {16, 16}
#endif

#ifndef ftype
#define ftype float
#endif

#ifndef dtype
#define dtype bf16
#endif

struct Tile {
    static constexpr int2 G = GTILE_XY;
    static constexpr int2 B = BLOCKTILE_XY;
    static constexpr int2 W = WARPTILE_XY;
    
    // Tile hierarchy ratios
    static constexpr uint32_t warps_x = B.x / W.x;
    static constexpr uint32_t warps_y = B.y / W.y;
};


template<typename Layout>
struct TileBCHW {

    const Layout& reference;

    __host__ __device__ const Layout& ref() const { 
        return reference;
    }

    // Grid dimensions for kernel launch
    dim3 grid()  { return dim3(ref().batch(), tiles_rows(), tiles_cols()); }
    dim3 block() { return dim3(NUM_THREADS); }
    unsigned long mem_size = 100000; // 100 KB default shared memory size
    
    // Number of block tiles in each dimension
    __host__ __device__ uint32_t tiles_cols() const { return (ref().cols() + Tile::B.x - 1) / Tile::B.x; }
    __host__ __device__ uint32_t tiles_rows() const { return (ref().rows() + Tile::B.y - 1) / Tile::B.y; }
    __host__ __device__ uint32_t tiles_depth() const { return ref().depth(); }
    
    // Block-level indices
    __device__ uint32_t tile_col()  const { return blockIdx.z; }
    __device__ uint32_t tile_row()  const { return blockIdx.y; }
    __device__ uint32_t tile_batch() const { return blockIdx.x; }
    
    // Warp-level indices
    __device__ uint32_t warp_id()  const { return threadIdx.x / kittens::WARP_THREADS; }
    __device__ uint32_t warp_col() const { return warp_id() % Tile::warps_x; }
    __device__ uint32_t warp_row() const { return warp_id() / Tile::warps_x; }
    
    // Global warp tile indices (for load/store operations)
    __device__ uint32_t idx_col() const { return tile_col() * Tile::warps_x + warp_col(); }
    __device__ uint32_t idx_row() const { return tile_row() * Tile::warps_y + warp_row(); }
};

// Forward pass data structure (x, y)
template<typename Layout>
struct BCHW_fwd : public TileBCHW<Layout> {
    Layout x, y;
    BCHW_fwd() : TileBCHW<Layout>(&x) {}
};

template<typename Layout>
struct BCHW_bwd_stateless : public TileBCHW<Layout> {
    Layout grad_y, y, grad_x;
    BCHW_bwd_stateless() : TileBCHW<Layout>(&y) {}
};

template<typename Layout>
struct BCHW_bwd : public TileBCHW<Layout> {
    Layout grad_y, y, grad_x, x;
    BCHW_bwd() : TileBCHW<Layout>(&x) {}
};

using tiled_layout = gl<dtype, G_BATCH, G_CHANNEL, Tile::G.y, Tile::G.x, st_fl<Tile::B.y, Tile::B.x>>; // bchw layout
using reg_tile_ft = rt<ftype, Tile::W.y, Tile::W.x>;
using reg_tile_dt = rt<dtype, Tile::W.y, Tile::W.x>;