sudo apt update
cp -r /workspace/alex/saved_models/*.pth /home
cp -r /workspace/alex/saved_models/*.pkl /home
# cp /home/xinglongs/alex-space/unpruned_SSD
cp /workspace/alex/dense_ssd512.pth /home
tar -xf /workspace/data/VOC_2007_2012.tar.gz -C /raid
cd /workspace/alex/lookahead-pascal
python scripts/create_voc_data_lists.py /raid/VOCdevkit/VOC2007/ /raid/VOCdevkit/VOC2012/ /raid/
cd /raid
mkdir json_path
mv *.json json_path/
cp -r json_path/*.json ./

pip install pulp
pip install pyomo
apt install -y gcc libglpk-dev glpk-utils
conda install -c conda-forge ipopt
rge ipopt
