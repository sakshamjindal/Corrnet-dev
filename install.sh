docker run -it \
        -p $1:8800 \
        --gpus all \
        --shm-size="16g"\
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -v $XAUTH:/root/.Xauthority \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $HOME/work/DPC:/workspace/DPC \
        -v $HOME/work/CorrNet3D:/workspace/CorrNet3D \
        -v $HOME/work/Pointnet2_PyTorch:/workspace/Pointnet2_PyTorch \
        pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel



apt-get install ssh nano
rm /etc/apt/sources.list.d/cuda.list
rm /etc/apt/sources.list.d/nvidia-ml.list
apt-get update
apt-get -y install cmake

# pip install pytorch-lightning==1.2.8
# conda install -c pytorch torchvision

# conda install -c plotly psutil requests python-kaleido --yes
# pip install cython==0.29.20 autowrap ninja tables ply ilock
# pip install h5py pydocstyle plotly psutil xvfbwrapper yapf mypy openmesh plyfile neuralnet-pytorch imageio pyinstrument pairing robust_laplacian pymesh trimesh cmake "ray[tune]" "pytorch-lightning-bolts>=0.2.5" pyrr gdist neptune-client neptune-contrib iopath sklearn autowrap py-goicp opencv-python torchsummary gdown
# conda install "notebook>=5.3" "ipywidgets>=7.2" flake8 black flake8 -y
# conda install pytorch-metric-learning -c metric-learning -c pytorch -y
# pip install addict

apt-get update && apt-get install libgl1
pip install open3d-python

pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git 
cd Pointnet2_PyTorch
pip install -r requirements.txt 
cd ..

pip install hydra-core==1.0
pip install omegaconf==2.0.1
pip install pytorch-lightning==1.1.6
pip install h5py
pip install tables
pip install matplotlib
