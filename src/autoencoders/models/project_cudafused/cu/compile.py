
from torch.utils.cpp_extension import load
# from functools import lru_cache
import torch
import sysconfig
import os
import pybind11
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
os.environ["CUDAHOSTCXX"] = "g++"

THUNDERKITTENS_ROOT = os.path.join(os.getcwd(), "lib/ext/tk")
CU_LAYERS_ROOT = os.path.join(os.getcwd(), "src/autoencoders/models/cu/layers")

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
    f"-I{CU_LAYERS_ROOT}",
    # "-D__CUDA_NO_HALF_OPERATORS__",
    # "-D__CUDA_NO_HALF_CONVERSIONS__",
    # "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    # "-D__CUDA_NO_HALF2_OPERATORS__",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-D_GLIBCXX_USE_CXX11_ABI=1",
    "-diag-suppress=3189",
] + GPU_FLAGS

# Add PyTorch / Python includes
NVCC_FLAGS += [f"-I{inc}" for inc in torch.utils.cpp_extension.include_paths()]
NVCC_FLAGS.append(f"-I{sysconfig.get_path('include')}")
NVCC_FLAGS += [f"-I{inc}" for inc in pybind11.get_include().split()]

CXX_FLAGS = ["-O2", "-g"]

# @lru_cache(maxsize=128)
def compile(kernel, build_dir = None, template_kwargs = {}):
    device_functions = []
    try:
        name = kernel.split('/')[-1].replace('.cu','')
        
        kwargs = {}
        if build_dir is not None:
            kernel_dir = os.path.join(build_dir, name)
            os.makedirs(kernel_dir, exist_ok=True)
            kwargs['build_directory'] = kernel_dir
            # else PyTorch will use a default temp dir
        
        # convert kwargs to defines
        for k, v in template_kwargs.items():
            define_flag = f"-D{k}={v}"
            NVCC_FLAGS.append(define_flag)
        
        module = load(
            name=name,
            sources=[kernel, *device_functions],
            verbose=True,
            extra_cflags=CXX_FLAGS,
            extra_cuda_cflags=NVCC_FLAGS,
            **kwargs
        )
        print(f"Successfully compiled CUDA extension {name}")
        return module
    except Exception as e:
        print(f"Error compiling CUDA extension {name}: {e}")
        raise e
    
    
if __name__ == "__main__":
    compile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "copy_example.cu"
            )
    )