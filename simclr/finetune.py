import argparse
import os
from pprint import pformat, pprint

import numpy as np
import tensorflow as tf

import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib
from utils import get_files_and_labels, read_class_label_map
import config
from config import num_classes, image_size, train_mode

parser = argparse.ArgumentParser()

data_args = parser.add_argument_group("Data args")
data_args.add_argument(
    "data_dir",
    type=str,
    help="Directory in which to recursively search for images/metadata",
)
data_args.add_argument(
    "--file_list",
    type=str,
    default=None,
    help=".txt file containing a list of images and their corresponding labels",
)
data_args.add_argument(
    "--class_map_csv",
    type=str,
    default="class_map.csv",
    help="CSV file that maps int labels to class names, e.g., 0, Arm cover\n1, Other rover part",
)
data_args.add_argument(
    "--ext", type=str, default="jpg", help="Extension of the images in data_dir"
)
data_args.add_argument(
    "--image_size",
    type=int,
    default=224,
    help="Will preprocess all images to match this size.",
)
data_args.add_argument(
    "--color_jitter_strength",
    type=float,
    default=1.0,
    help="Strength of color jittering",
)

train_args = parser.add_argument_group("Training params")
train_args.add_argument("--batch_size", type=int, default=32, help="Batch size")
train_args.add_argument(
    "--epochs", type=int, default=30, help="Number of epochs to train for"
)
train_args.add_argument(
    "--checkpoint_epochs",
    type=int,
    default=1,
    help="Number of epochs between checkpoints/summaries",
)
train_args.add_argument(
    "--gpu_ids", type=int, nargs="+", default=[], help="List of GPU IDs to use in parallel"
)

if __name__ == "__main__":
    """
    Usage:
    ======
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/Mars_Classification/msl-labeled-data-set-v2.1/ --file_list train-set-v2.1.txt
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/UCMerced_LandUse_PNG/Images --ext png --gpu_ids [4,5,6,7]
    """

    global train_mode, image_size, num_classes


    args = parser.parse_args()

    if args.file_list is not None:
        file_list = os.path.join(args.data_dir, args.file_list)
        img_folder = os.path.join(args.data_dir, "images")
        class_mapping = read_class_label_map(
            os.path.join(args.data_dir, args.class_map_csv)
        )
    else:
        file_list = None
        img_folder = args.data_dir
        class_mapping = None

    X, num_files, num_classes_, class_mapping = get_files_and_labels(
        img_folder, ext=args.ext, metadata_file=file_list, mapping=class_mapping
    )
    config.num_classes = num_classes_
    classes = list(class_mapping.keys())

    if len(args.gpu_ids) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{int(d)}" for d in args.gpu_ids]
        )
        print(f"Running using MirroredStrategy on {strategy.num_replicas_in_sync} replicas")
    else:
        d = np.random.randint(len(tf.config.list_physical_devices('GPU')))
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{d}")

    print(
        f"img_folder = {img_folder}\n"
        f"file_list = {file_list}\n"
        f"class_map_csv = {os.path.join(args.data_dir, args.class_map_csv)}\n"
        f"num_files = {num_files}\n"
        f"num_classes = {num_classes}\n"
        f"classes = {pformat(classes, compact=True)}"
    )

    image_size = args.image_size
    # with strategy.scope():
    #     model = model_lib.Model()

    model_lib.test_model_lib()