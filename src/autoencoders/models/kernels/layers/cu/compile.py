from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch
import sysconfig
import sys
import os
import pybind11

TK_DIR = 'lib/ext/tk'
THUNDERKITTENS_ROOT = os.path.join(os.getcwd(),  TK_DIR)

# Detect Python + PyTorch paths
PYTHON_INCLUDE = sysconfig.get_path("include")
PYTHON_LIBDIR = sysconfig.get_config_var("LIBDIR")
PYTHON_VERSION = sysconfig.get_config_var("LDVERSION")

TORCH_INCLUDE_DIRS = torch.utils.cpp_extension.include_paths()
TORCH_LIBRARY_DIRS = torch.utils.cpp_extension.library_paths()

# Detect GPU architecture (can be set via environment variable)
GPU = os.environ.get("GPU", "4090")
if GPU == "4090":
    GPU_FLAGS = ["-DKITTENS_4090", "-arch=sm_89"]
elif GPU == "A100":
    GPU_FLAGS = ["-DKITTENS_A100", "-arch=sm_80"]
else:
    GPU_FLAGS = ["-DKITTENS_HOPPER", "-arch=sm_90a"]

# Base NVCC flags
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
    "--dlto",  # Device LTO for cross-file optimization
    "-Xnvlink=--verbose",
    "-Xptxas=--verbose",
    "-Xptxas=--warn-on-spills",
     f"-I{THUNDERKITTENS_ROOT}/include -I{THUNDERKITTENS_ROOT}/prototype",
    "-D__CUDA_NO_HALF_OPERATORS__",
    "-D__CUDA_NO_HALF_CONVERSIONS__",
    "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-D__CUDA_NO_HALF2_OPERATORS__",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-diag-suppress=3189",
]

# Add GPU architecture flags
NVCC_FLAGS += GPU_FLAGS

# Add include and library paths
for inc in TORCH_INCLUDE_DIRS:
    NVCC_FLAGS.append(f"-I{inc}")
NVCC_FLAGS.append(f"-I{PYTHON_INCLUDE}")
NVCC_FLAGS += pybind11.get_include().split()

for lib in TORCH_LIBRARY_DIRS:
    NVCC_FLAGS.append(f"-L{lib}")
NVCC_FLAGS.append(f"-L{PYTHON_LIBDIR}")

# Add necessary libraries (same as Makefile)
NVCC_FLAGS += [
    "-lcuda",
    "-lcudadevrt",
    "-lcudart_static",
    "-lcublas",
    "-ltorch_python",
    "-ltorch_cuda",
    "-ltorch_cpu",
    "-ltorch",
    "-lc10_cuda",
    "-lc10",
    f"-lpython{PYTHON_VERSION}",
    "-lrt",
    "-lpthread",
    "-ldl",
]

setup(
    name="cuda_extension",
    ext_modules=[
        CUDAExtension(
            name="cuda_extension",
            sources=[
                "example_bind.cu"
            ],
            extra_compile_args={
                "cxx": ["-O2", "-g"],
                "nvcc": NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
