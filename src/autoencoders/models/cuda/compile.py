# compile all kernels in cuda directory 
# following the TK example with TK library
import os
import sys

TK_DIR = 'lib/ext/tk'
# BIND_DIR = 'kernels/example_bind'
#{cu_folder = os.path.join(TK_DIR, BIND_DIR)

cu_folder = os.path.dirname(os.path.abspath(__file__))
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
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            module_path = os.path.join(cu_folder, file)
            
            TK_compile_command = f'make -C {cu_folder} all TARGET={module_name} SRC={module_path} {flags}'
            os.system(TK_compile_command)
            
def clean():
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            TK_clean_command = f'make -C {cu_folder} clean TARGET={module_name} SRC={module_path} {flags}'
            os.system(TK_clean_command)
            
if __name__ == "__main__":
    build()