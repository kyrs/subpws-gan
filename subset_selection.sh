WSGAN_PATH=./
export PYTHONPATH=$WSGAN_PATH



python $WSGAN_PATH/lightning_subset/embedding_lfs.py\
 --dataset GTSRB\
 --gpus 0\
 --lffname $WSGAN_PATH/lf_files/ssl_lfs.pth\
 --data_path downloaded_data


