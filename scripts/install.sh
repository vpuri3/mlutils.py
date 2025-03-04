#
conda create --name MLUtils python=3.11

conda activate MLUtils
pip install ipython
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install tqdm jsonargparse
pip install timm einops
pip install matplotlib # scipy pandas seaborn
#