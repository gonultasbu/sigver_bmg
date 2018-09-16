pip install "scipy==0.18.0" "pillow==3.0.0"
pip install opencv-python
pip install "Theano==0.9"
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
git clone https://github.com/luizgh/sigver_wiwd.git
cd sigver_wiwd/models
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_models.zip"
wget "https://storage.googleapis.com/luizgh-datasets/models/signet_spp_models.zip"
unzip signet_models.zip
unzip signet_spp_models.zip
cd ~/Downloads
# TensorFlow installation
# Adds NVIDIA package repository.
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
sudo dpkg -i nvidia-machine-learning-repo-ubuntu1604_1.0.0-1_amd64.deb
sudo apt-get update
# Includes optional NCCL 2.x.
sudo apt-get install cuda9.0 cuda-cublas-9-0 cuda-cufft-9-0 cuda-curand-9-0 \
  cuda-cusolver-9-0 cuda-cusparse-9-0 libcudnn7=7.2.1.38-1+cuda9.0 \
   libnccl2=2.2.13-1+cuda9.0 cuda-command-line-tools-9-0
# Optionally install TensorRT runtime, must be done after above CUDA install.
sudo apt-get update
sudo apt-get install libnvinfer4=4.1.2-1+cuda9.0
