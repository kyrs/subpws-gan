WSGAN_PATH=./

export PYTHONPATH=$WSGAN_PATH

python lightning_subset/main_labelmodel.py\
 --dataset GTSRB\
 --gpus 1\
 --batch_size 16\
 --lffname $WSGAN_PATH/lf_files/ssl_GTSRB_complete_features.pt\
 --numlfs 100\
 --max_epochs 200\
 --whichmodule GANLabelModel\
 --ganenctype encoderX\
 --data_path downloaded_data\
 --storedir outputs/GTSRB\
 --decaylossterm 1\
 --dclr 0.00018\
 --gclr 0.00001\
 --budget 0.8\
 --beta_weight 0.7\
 --alpha_weight 0.3\
 --alpha_weight_subModular 3.0\
