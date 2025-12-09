#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "nn.cuh"
#include "loss.cuh"
#include "opt.cuh"
// #include "modules/scale.cuh"
// #include "modules/pixel_down.cuh"
#include "modules/chan.cuh"
#include "util/bind_w_return.cuh"

template<class L>
using network = module_chain<L, SGD,
 ChannelModule<3>,
//  ChannelModule<3>,
//  ChannelModule<10>,
 ChannelModule<3>,
>;

using Loss = MSELoss;


PYBIND11_MODULE(siren, m) {
    m.doc() = "nn test python module";
    py::bind_function<basic_eval<network,Loss>>(m, "eval", &train_data::x, &train_data::y, &train_data::weight_mem_ptr, &train_data::iterations);
    py::bind_function_with_return<basic_train<network,Loss>>(m, "train", &train_data::x, &train_data::y, &train_data::weight_mem_ptr, &train_data::iterations);
    // order matters! this needs to match the train_data struct layout as well, for some reason
}