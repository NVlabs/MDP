cp -r /workspace/data/nuscenes/ /raid/
conda create -n streampetr python=3.8 -y
conda init bash
source ~/.bashrc
conda activate streampetr
conda update -n base conda -y
# #conda install -n base conda-libmamba-solver -y
# #conda config --set solver libmamba

pip install --upgrade pip
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.6.0
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
cd /workspace/alex/StreamPETR
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
cd /workspace/alex/StreamPETR/mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
pip install mmdet==2.28.2

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
pip install numpy==1.23.1
pip install einops
pip install setuptools==59.5.0

pip install pyomo
apt install -y gcc libglpk-dev glpk-utils
conda install -c conda-forge ipopt
pip install pulp
cp -r /workspace/alex/*temporal*.pkl /raid
# cd /workspace/alex/StreamPETR
# python tools/create_data_nusc.py --root-path /raid/nuscenes/ --out-dir /raid/nuscenes/ --extra-tag nuscenes2d --version v1.0