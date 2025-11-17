#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

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

struct Tile {
    static constexpr int2 G = GTILE_XY;
    static constexpr int2 B = BLOCKTILE_XY;
    static constexpr int2 W = WARPTILE_XY;
};

using tiled_layout = gl<float, G_BATCH, G_CHANNEL, Tile::G.y, Tile::G.x, st_fl<Tile::G.y, Tile::G.x>>; // bchw layout
