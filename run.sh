WSGAN_PATH=/home/ece/hdd/weak_supervision/wsgan

export PYTHONPATH=$WSGAN_PATH

# gpus is not number of gpus, but the ids of gpus. thats why gpus is set to 0 1


# ============ MNIST ====================

python lightning/main_labelmodel.py\
 --dataset MNIST\
 --gpus 0 1\
 --batch_size 16\
 --lffname $WSGAN_PATH/data/MNIST/ssl_lfs.pth\
 --numlfs 29\
 --max_epochs 200\
 --whichmodule GANLabelModel\
 --ganenctype encoderX\
 --data_path downloaded_data\
 --storedir outputs/mnist_outputs_multiple_gpu_test/\
 --decaylossterm 1.0\
 --num_workers 8\


# ============ CIFAR ====================

# python lightning/main_labelmodel.py\
#  --dataset CIFAR10\
#  --gpus 0 1\
#  --batch_size 16\
#  --lffname $WSGAN_PATH/data/CIFAR10/ssl_lfs.pth\
#  --numlfs 20\
#  --max_epochs 200\
#  --whichmodule GANLabelModel\
#  --ganenctype encoderX\
#  --data_path downloaded_data\
#  --storedir outputs/cifar_outputs_multiple_gpu_decayloss/\
#  --decaylossterm\
#  --load_ckpt last\
