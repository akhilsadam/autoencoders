# compile all kernels in cuda directory 
# following the TK example with TK library
import os

TK_DIR = 'lib/ext/tk'
# BIND_DIR = 'kernels/example_bind'
# makefile_path = os.path.join(TK_DIR, BIND_DIR)
makefile_path = '.'
cu_folder = os.path.dirname(__file__)
root_dir = os.getcwd()
TK_root = os.path.join(root_dir,  TK_DIR)


flags = f'THUNDERKITTENS_ROOT={TK_root} ' 

if venv:=os.environ.get('VENV', False):
    flags += f"""
PYTHON={venv}/bin/python3
PYTHON_CONFIG={venv}/bin/python3-config """

flags = flags.replace('\n', ' ')

def build():
    print("Compiling CUDA kernels...")
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            module_path = os.path.join(cu_folder, file)
            
            TK_compile_command = f'make -C {makefile_path} all TARGET={module_name} SRC={module_path} '
            os.system(TK_compile_command)
            
def clean():
    for file in os.listdir(cu_folder):
        if file.endswith('.cu'):
            module_name = file[:-3]
            TK_clean_command = f'make -C {makefile_path} clean TARGET={module_name} SRC={module_path} '
            os.system(TK_clean_command)
            
if __name__ == "__main__":
    build()