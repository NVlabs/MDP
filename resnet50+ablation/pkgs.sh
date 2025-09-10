sudo apt update
cp -r /workspace/alex/saved_models/*.pth /home
cp -r /workspace/alex/saved_models/*.pkl /home
tar -xf /workspace/data/imagenet2012.tar.gz -C /raid
pip install pulp
pip install pyomo
apt install -y gcc libglpk-dev glpk-utils
conda install -c conda-forge ipopt
