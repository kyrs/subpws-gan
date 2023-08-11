import os
import sys 
from datetime import datetime
from embeddings import *
from pytorch_lightning import loggers as pl_loggers
from lightning_subset.module import GANLabelModel
from lightning_subset.datamodule import JSONImageDataset
from pathlib import Path
from lightning_subset.lutils import (
    get_datasets,
    load_domainnet_lfs,
    load_awa_lfs,
    create_L_ind
)
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import warnings
import torchvision
import tqdm
def get_embedder(embedding):
    if embedding == 'inceptionv3':
        embedder = Inceptionv3Embedding().eval().cuda()
    

    if torch.cuda.current_device() > 1:
        embedder = nn.DataParallel(embedder)
    return embedder

def get_embeddings_from_loader(dataloader,
                               embedder,
                               return_labels=False,
                               verbose=True):
    embeddings = []
    labels = []

    with torch.no_grad():
        if verbose:
            dataloader = tqdm(dataloader, desc='Extracting embeddings')
        i=0
        for data in dataloader:
            i+=1
            print(f"Processed image : {i}")
            if len(data) == 2:
                images, label = data
                images = images.cuda()  
            else:
                images = data.cuda()
                labels.append(torch.zeros(len(images)))

            embed = embedder(images)
            embeddings.append(embed.cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)
    # labels = torch.cat(labels, dim=0)
    print(embeddings.shape)
    return embeddings


def main(args: Namespace) -> None:
    # name for subfolder to save to
    datasetsavename = args.dataset
    
    train_idxs = None
    drop_last = args.droplast
    architecture = args.architecture
    latent_dim = args.latent_dim
    if args.dataset.lower() in ["awa2", "awa", "domainnet"]:
        if args.imgsize == 32:
            img_shape = (3, 32, 32)
            augment_style = "cifar"
        elif args.imgsize == 64:
            img_shape = (3, 64, 64)
            augment_style = "cifar64"
            architecture += "64"
        else:
            raise NotImplementedError("img size selected not implemented. Check StyleWSGAN repository instead.")

        if args.dataset.lower() in "awa2":
            img_root = args.data_path  # path to images
            dset_lfs = args.lffname # labeling function file, saved as json
            if not dset_lfs.endswith(".json"):
                dset_lfs = os.path.join(dset_lfs, "train.json")
            datasetsavename = "awa2"
            n_classes = 10
            Lambdas, Ytrue, train_idxs = load_awa_lfs(dset_lfs)
            
        elif args.dataset.lower() == "domainnet":
            img_root = args.data_path  # path to images
            dset_lfs = args.lffname # labeling function file, saved as json
            if not dset_lfs.endswith(".json"):
                dset_lfs = os.path.join(dset_lfs, "train.json")
            datasetsavename = "domainnet"
            n_classes = 10
            drop_last = True  # number of train images leads to last batch of size 1
            # load train LFs
            Lambdas, Ytrue, train_idxs = load_domainnet_lfs(dset_lfs, applythreshold=False)
        else:
            raise NotImplementedError("dataset not found")

        num_LFs = Lambdas.shape[1]


        val = all(train_idxs[i] <= train_idxs[i+1] for i in range(len(train_idxs) - 1)) ## check if the lisit is sorted
            #print(val)
        if(val):
            pass 
        else:
            raise Exception("list index is not sorted, embedding mapping can get wrong..")
        dirAdd = os.path.dirname(dset_lfs)
        baseName = os.path.basename(dset_lfs)
        print(baseName)
        assert(baseName == "train.json")
        embedPath = os.path.join(dirAdd, "train_embedding.pt")
        print(embedPath)
        # input()

        # set up datasets
        trainset_sub = JSONImageDataset(
            jsonpath=dset_lfs,
            img_root=img_root,
            transform=None,
            size=img_shape[1:],
        )

        embedder = get_embedder(args.embedding)
        print("extracting the features of dataset")
        out = get_embeddings_from_loader(trainset_sub,
                               embedder,
                               return_labels=False,
                               verbose=False)
        torch.save([out], embedPath)
    else:
        # ------------------------
        # pytorch vision dataset
        # ------------------------

        traindataset, testdataset, img_shape, n_classes, augment_style = get_datasets(
            args, basetransforms = True
        ) ### base transform = True is needed to extract features from the inception model used in the code. 
        if args.dataset.lower() == "gtsrb":
            architecture += "64"

        # ------------------------
        # Load fixed LFs
        # ------------------------
        print("loading fixed LFs from %s" % args.lffname)
        try:
            (
                train_idxs,
                val_idxs,
                Lambdas,
                LF_accuracies,
                LF_propensity,
                LF_labels,
                ValLambdas,
            ) = torch.load(args.lffname, map_location=lambda storage, loc: storage)
        except ValueError:
            (
                train_idxs,
                val_idxs,
                Lambdas,
                LF_accuracies,
                LF_propensity,
                LF_labels,
            ) = torch.load(args.lffname, map_location=lambda storage, loc: storage)

        num_LFs = Lambdas.shape[1]
        print(f"shape of lambdas : {num_LFs}")
        datasetsavename = datasetsavename + "_%dlfs" % num_LFs
        trainset_sub = torch.utils.data.Subset(traindataset, train_idxs)
        
        embedder = get_embedder(args.embedding)
        out = get_embeddings_from_loader(trainset_sub,
                               embedder,
                               return_labels=False,
                               verbose=False)

        objDetails = [train_idxs,
                val_idxs,
                Lambdas,
                LF_accuracies,
                LF_propensity,
                LF_labels,out]

        saveAdd = os.path.join(os.getcwd(), "subset_filter")
        fileBaseName = os.path.basename(args.lffname).split(".")[0]
        finalFileName = fileBaseName+f"{args.dataset}_{args.embedding}_complete_features.pt"
        Path(saveAdd).mkdir(parents=True, exist_ok=True)
        completeAdd = os.path.join(saveAdd, finalFileName)
        torch.save(objDetails, completeAdd)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", help="dataset to load")
    parser.add_argument(
        "--data_path", type=str, default=os.getcwd(), help="dataset to load"
    )
    parser.add_argument(
        "--lffname",
        type=str,
        default=None,
        required=False,
        help="path to precomputed LFs and training indexes",
    )

    parser.add_argument(
        "--embedding",
        type=str,
        default="inceptionv3",
        help="Which WSGAN model to run: InfoGAN or GANLabelModel",
    )
    parser.add_argument(
        "--imgsize",
        type=int,
        default=32,
        help="Size of images. 32x32 or 64x64. Will only be changed for AwA2 and Domainnet",
    )


    parser.add_argument(
        "--droplast", default=False, action="store_true", help="Drop last uneven batch"
    )


    # ------------------------
    # Add DataLoader args
    # parser = LFDataModule.add_argparse_args(parser)

    # # Add model specific args
    parser = GANLabelModel.add_argparse_args(parser)

    ##########################
    # GAN trainer args
    ##########################
    parser.add_argument("--gpus", nargs="+", type=int, help="GPU ids", required=True)
    parser.add_argument("--max_epochs", type=int, help="Number of training epochs", default=150)

    

    #########################
    # path parameters
    #########################
    parser.add_argument(
        "--storedir",
        type=str,
        default = "",
        required=False,
        help="path to save logs, fake images , and checkpoints to",
    )

    parser.add_argument(
        "--save_suffix",
        type=str,
        default="",
        help="Suffix to append to logging folder for results",
    )

    # Parse all arguments
    args = parser.parse_args()

    if len(args.gpus)>1:
        torch.multiprocessing.set_start_method('spawn')

    main(args)
