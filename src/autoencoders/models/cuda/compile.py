# compile all kernels in cuda directory 
# following the TK example with TK library
import os

TK_DIR = 'lib/ext/TK'
BIND_DIR = 'kernels/example_bind'
Makefile_path = os.path.join(TK_DIR, BIND_DIR, 'Makefile')

cu_folder = os.path.dirname(__file__)

def build():
    print(f"Compiling CUDA kernels...{os.getcwd()}")
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            module_path = os.path.join(cu_folder, file)
            
            TK_compile_command = f'make -f {Makefile_path} TARGET={module_name} SRC={module_path} all'
            os.system(TK_compile_command)
            
def clean():
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            TK_clean_command = f'make -f {Makefile_path} TARGET={module_name} clean'
            os.system(TK_clean_command)
            
if __name__ == "__main__":
    build()