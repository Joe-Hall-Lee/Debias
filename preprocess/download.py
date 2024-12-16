import os
# 模型下载
from modelscope import snapshot_download

# Get the current working directory
current_dir = os.getcwd()

# Download the model to the current working directory
model_dir = snapshot_download('Xorbits/vicuna-7b-v1.3', cache_dir=current_dir)
