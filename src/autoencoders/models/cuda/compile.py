# compile all kernels in cuda directory 
# following the TK example with TK library
import os
import sys

TK_DIR = 'lib/ext/tk'
# BIND_DIR = 'kernels/example_bind'
#{cu_folder = os.path.join(TK_DIR, BIND_DIR)

root_dir = os.getcwd()
TK_root = os.path.join(root_dir,  TK_DIR)

flags = f'THUNDERKITTENS_ROOT={TK_root} ' 

if len(sys.argv) > 1:
    venv = os.path.abspath(sys.argv[1])
    flags += f"""
PYTHON={venv}/bin/python3
PYCONFIG={venv}/bin/python3-config """

flags = flags.replace('\n', ' ')

def build():
    print("Compiling CUDA kernels...")
    cu_folder = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            
            module_path = os.path.join(cu_folder, module_name + '.cu')
            TK_compile_command = f'make -C {cu_folder} all TARGET={module_name} SRC={module_path} {flags}'
            os.system(TK_compile_command)
            
def compile_module(module_name, **kwargs):
    cu_folder = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(cu_folder, module_name + '.cu')
    
    nvcc_flags = 'NVCCFLAGS="'
    for key, value in kwargs.items():
        nvcc_flags += f'-D{key.upper()}={value} '
    nvcc_flags += '"'
    
    TK_compile_command = f'make -C {cu_folder} all TARGET={module_name} SRC={module_path} {flags} {nvcc_flags}'
    os.system(TK_compile_command)
    
    # maybe compile to run directory OR cache directory with UUID.
    # Latter preferred.
    # actually, let's save the template values with the filename.
        
def clean():
    cu_folder = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            module_path = os.path.join(cu_folder, file)
            TK_clean_command = f'make -C {cu_folder} clean TARGET={module_name} SRC={module_path} {flags}'
            os.system(TK_clean_command)
            
# if __name__ == "__main__":
#     build()