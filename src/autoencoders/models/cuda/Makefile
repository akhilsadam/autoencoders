# Compiler
NVCC?=nvcc
GPU=4090

PYTHON=python3
PYCONFIG=python3-config

TARGET=example_bind
SRC=example_bind.cu

PYTHON_VERSION ?= $(shell ${PYTHON} -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES ?= $(shell ${PYTHON} -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYBIND_INCLUDES ?= $(shell ${PYTHON} -m pybind11 --includes)
PYCONFIG_LDFLAGS ?= $(shell ${PYCONFIG} --ldflags)
PYTORCH_INCLUDES ?= $(shell ${PYTHON} -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTHON_LIBDIR ?= $(shell ${PYTHON} -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYTORCH_LIBDIR ?= $(shell ${PYTHON} -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
PYTORCH_INCLUDES ?= $(shell ${PYTHON} -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTORCH_LIBDIR ?= $(shell ${PYTHON} -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS += -O3 --use_fast_math # VERY important to include this flag
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
NVCCFLAGS += -shared -fPIC # make shared object

NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DPy_LIMITED_API=0x03120000
NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C -D_GLIBCXX_USE_CXX11_ABI=1
NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES)
NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10

NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES) $(PYBIND_INCLUDES) ${PYCONFIG_LDFLAGS}
NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}

NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
NVCCFLAGS += -diag-suppress 3189

# Conditional setup based on the target GPU
ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	echo "Compiling ThunderKittens CUDA extension for $(TARGET)"
	echo "Using Python interpreter: $(PYTHON)"
	echo "Using Python config tool: $(PYCONFIG)"
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell $(PYCONFIG) --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell $(PYCONFIG) --extension-suffix)