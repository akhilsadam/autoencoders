mkdir tmp
cd tmp
wget https://developer.download.nvidia.com/compute/nvidia-driver/580.105.08/local_installers/nvidia-driver-local-repo-rhel8-580.105.08-1.0-1.x86_64.rpm
rpm2cpio nvidia-driver-local-repo-rhel8-580.105.08-1.0-1.x86_64.rpm | cpio -idmv
cp var/nvidia-driver-local-repo-rhel8-580.105.08/cuda-compat-13-0-580.105.08-1.el8.x86_64.rpm cuda-compat-13.rpm
mkdir other_cuda
mv etc other_cuda/etc
mv var other_cuda/var
mv nvidia-driver-local-repo-rhel8-580.105.08-1.0-1.x86_64.rpm other_cuda/nvidia.rpm
rpm2cpio cuda-compat-13.rpm | cpio -idmv
# Now make a software directory in home
mkdir -p /home/$USER/software/cuda-13-compact/lib64
cp -r ./usr/local/cuda-13.0/compat/* /home/$USER/software/cuda-13-compact/lib64/
# add it to bashrc
echo 'export LD_LIBRARY_PATH=/home/$USER/software/cuda-13-compact/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
## cleanup
cd ..
rm -r tmp