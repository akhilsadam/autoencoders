#include "kittens.cuh"
#include "pyutils/pyutils.cuh"
using namespace kittens;

// from layers
#include "act.cuh"

// need to think about templating with defines
using my_layout = gl<float, -1, -1, -1, -1, st_fl<64,64>>; // bchw layout
struct fwd_data {
    my_layout x, y;
    dim3 grid()  { return dim3(x.batch()); } // b
    dim3 block() { return dim3(x.depth(), x.cols(), x.rows()); } // whc
};
struct bwd_data {
    my_layout grad_y, y, grad_x;
    dim3 grid()  { return dim3(grad_y.batch()); } // b
    dim3 block() { return dim3(grad_y.depth(), grad_y.cols(), grad_y.rows()); } // whc
};

__global__ void relu_fwd_kernel(const __grid_constant__ fwd_data g) {
        g.y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}] 
        = relu_fwd(
            g.x[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}]
        );
}
__global__ void relu_bwd_kernel(const __grid_constant__ bwd_data g) {
        g.grad_x[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}] 
        = relu_bwd(
            g.grad_y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}],
            g.y[{blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x}]
        );
}

void run_relu_fwd_kernel(fwd_data g) {
    relu_fwd_kernel<<<g.grid(), g.block()>>>(g);
}
void run_relu_bwd_kernel(bwd_data g) {
    relu_bwd_kernel<<<g.grid(), g.block()>>>(g);
}

PYBIND11_MODULE(act, m) {
    m.doc() = "activation functions python module";
    py::bind_kernel<relu_fwd_kernel>(m, "relu_fwd_kernel", &fwd_data::x, &fwd_data::y);
    py::bind_kernel<relu_bwd_kernel>(m, "relu_bwd_kernel", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
    py::bind_function<run_relu_fwd_kernel>(m, "relu_fwd", &fwd_data::x, &fwd_data::y);
    py::bind_function<run_relu_bwd_kernel>(m, "relu_bwd", &bwd_data::grad_y, &bwd_data::y, &bwd_data::grad_x);
}