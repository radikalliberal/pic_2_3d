cd "/home/jscholz/Code/pic23d/"
source activate tensorflow
tensorboard --logdir="./cnnlog/"&
chromium-browser&
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/extras/CUPTI/lib64"
python ./src/train.py
