# This is the official code for DualCast
we intergate the DualCast model framework into three baseline models, so there are three folders, each folder corresponding to a model with DualCast framework.

# Dataset
The link is: https://drive.google.com/drive/folders/1ACUoAE6h-RKoTq9C5XrTJDHQd6k6znpg?usp=sharing
download the dataset, and put the file into corresponsing folder (Folder located in drive - file located in computer)
1. DualCast-P/raw_data - DualCastP/raw_data/PeMS08; DualCast-P/data_cache - DualCastP/cache/data_cache
2. DualCast-G/data - DualCastG/data
2. DualCast-S/data - DualCastS/data

# Environment
1. DualCastS and DualCastG have same environment (python3.8.13 pytorch1.12.1)
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
[Website] if has other cuda version, can check through https://pytorch.org/get-started/previous-versions/

2. DualCastP has an environment (python3.7.16 pytorch1.10.1 cuda11.1)
Following the requirement file. Go to DualCastG (or DualCastP) to install the env. for them

pip install -r requirments.txt

[Q] The installing may be failed when directly pip install torch==xxx+cuxxx, same as torch-scatter and torch-sparse. 
[Solution] 
That is a common issue for old pytorch version, we should adding the wheel link
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
same for torch-scatter torch-geometric and torch-sparse
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html

[Q] hint to install ray[tune] and ray[]
[Solution] pip install ray[tune], and then reinstall pip install protobuf==3.20.0; due to the ray will install latest the protobuf version

[Others]
1. If hint other package missing, following the hint and the version in requirement.txt to install them
2. IF pip install -r requirement is interrupted soma package may be faild installed


If the GPU do not have enough space, please reduce the batch_size in .sh file

# Running code
chmod +x ./main_08.sh
./main_08.sh
