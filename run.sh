#!/bin/bash

pip install virtualenv
pyv="$(virtualenv --version)"
echo "$pyv"
virtualenv DS_VAE
source DS_VAE/bin/activate
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
#python train.py 
python generate.py