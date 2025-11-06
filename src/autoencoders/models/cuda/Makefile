# Compiler
NVCC?=nvcc
GPU=4090

PYTHON?=python3
PYCONFIG?=python3-config

echo "Compiling ThunderKittens CUDA extension for GPU: $(GPU)"
echo "Using Python interpreter: $(PYTHON)"
echo "Using Python config tool: $(PYCONFIG)"

TARGET=example_bind
SRC=example_bind.cu

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell ${PYTHON} -m pybind11 --includes) $(shell ${PYCONFIG} --ldflags) -shared -fPIC -lpython3.11

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
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell $(PYCONFIG) --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell $(PYCONFIG) --extension-suffix)