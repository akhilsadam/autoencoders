# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils.cpp_extension import load
import torch
import sysconfig
import os
import pybind11

THUNDERKITTENS_ROOT = os.path.join(os.getcwd(), "lib/ext/tk")

GPU = os.environ.get("GPU", "4090")
if GPU == "4090":
    GPU_FLAGS = ["-DKITTENS_4090", "-arch=sm_89"]
elif GPU == "A100":
    GPU_FLAGS = ["-DKITTENS_A100", "-arch=sm_80"]
else:
    GPU_FLAGS = ["-DKITTENS_HOPPER", "-arch=sm_90a"]

NVCC_FLAGS = [
    "-DNDEBUG",
    "-lineinfo",
    "--expt-extended-lambda",
    "--expt-relaxed-constexpr",
    "-Xcompiler=-Wno-psabi",
    "-Xcompiler=-fno-strict-aliasing",
    "-forward-unknown-to-host-compiler",
    "-ftemplate-backtrace-limit=0",
    "-std=c++20",
    "-O3",
    "--use_fast_math",
    # "--dlto",
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
    f"-I{THUNDERKITTENS_ROOT}/include",
    f"-I{THUNDERKITTENS_ROOT}/prototype",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-diag-suppress=3189",
] + GPU_FLAGS

# Add PyTorch / Python includes
NVCC_FLAGS += [f"-I{inc}" for inc in torch.utils.cpp_extension.include_paths()]
NVCC_FLAGS.append(f"-I{sysconfig.get_path('include')}")
NVCC_FLAGS += [f"-I{inc}" for inc in pybind11.get_include().split()]

CXX_FLAGS = ["-O2", "-g"]

def compile(name, sources, build_dir):
    try:
        module = load(
            name=name,
            sources=sources,
            verbose=True,
            extra_cflags=CXX_FLAGS,
            extra_cuda_cflags=NVCC_FLAGS,
            build_directory=build_dir,
        )
        return module
    except Exception as e:
        print(f"Error compiling CUDA extension {name}: {e}")
        raise e
    
    
if __name__ == "__main__":
    build_dir = os.path.dirname(os.path.abspath(__file__))
    compile(
        name="cuda_extension",
        sources=[
            os.path.join(build_dir, "example_bind.cu"),
            # "extension_kernel.cu",
            # "extension_device.cu",
            # "extension_device2.cu",
        ],
        build_dir=build_dir,
    )

# setup(
#     name="cuda_extension",
#     ext_modules=[
#         CUDAExtension(
#             name="cuda_extension",
#             sources=[
#                 "src/autoencoders/models/kernels/layers/cu/example_bind.cu",
#                 # "extension_kernel.cu",
#                 # "extension_device.cu",
#                 # "extension_device2.cu",
#             ],
#             extra_compile_args={
#                 "cxx": CXX_FLAGS,
#                 "nvcc": NVCC_FLAGS,
#             },
#             include_dirs=[THUNDERKITTENS_ROOT],  # Optional additional include
#         )
#     ],
#     cmdclass={"build_ext": BuildExtension},
# )
