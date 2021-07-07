import argparse
import os
import json
from glob import glob
from datetime import datetime
from pprint import pformat, pprint
import time

import numpy as np
import tensorflow as tf

import model as model_lib
import objective as obj_lib
from utils import get_files_and_labels, read_class_label_map, get_tf_dataset

from absl import logging

logging.set_verbosity(logging.ERROR)

parser = argparse.ArgumentParser()

data_args = parser.add_argument_group("Data args")
data_args.add_argument(
    "data_dir",
    type=str,
    help="Directory in which to recursively search for images/metadata",
)
data_args.add_argument(
    "--dataset",
    default="",
    type=str,
    help="Name of the current dataset; used for tensorboard and checkpointing",
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
train_args.add_argument("--batch_size", type=int, default=256, help="Batch size")
train_args.add_argument(
    "--epochs", type=int, default=30, help="Number of epochs to train for"
)
train_args.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Folder containing a .pb file from which a checkpoint can be restored.",
)
train_args.add_argument(
    "--model_dir", type=str, default=None, help="Model directory for training"
)
train_args.add_argument(
    "--checkpoint_epochs",
    type=int,
    default=1,
    help="Number of epochs between checkpoints/summaries",
)
train_args.add_argument(
    "--keep_checkpoint_max",
    type=int,
    default=5,
    help="Maximum number of checkpoints to keep.",
)
train_args.add_argument(
    "--gpu_ids",
    type=int,
    nargs="+",
    default=[],
    help="List of GPU IDs to use in parallel",
)
train_args.add_argument("--resnet_depth", type=int, default=50, help="Resnet depth")
train_args.add_argument(
    "--width_multiplier", type=int, default=2, help="Multiplier for resnet width"
)
train_args.add_argument(
    "--sk_ratio",
    type=float,
    default=0.0625,
    help="Enable selective kernels if > 0; recommendation: 0.0625",
)
train_args.add_argument(
    "--proj_out_dim", type=int, default=128, help="Number of head projection dimension"
)
train_args.add_argument(
    "--num_proj_layers",
    type=int,
    default=3,
    help="Number of nonlinear head layers after resnet",
)
train_args.add_argument(
    "--ft_proj_selector",
    type=int,
    default=0,
    help=(
        "Which layer of the proj head to use during finetuning. "
        "0 means no projection head, and -1 means the final layer."
        "The SimCLR v2 paper used 0 when finetuning with all labels, 1 when finetuning from 1 and 10 percent."
    ),
)
train_args.add_argument(
    "--optimizer",
    type=str,
    default="lars",
    help="Optimizer to use. Choices: ['adam', 'lars', 'SGD']",
)
train_args.add_argument(
    "--learning_rate",
    type=float,
    default=0.3,
    help="Initial learning rate per batch size of 256 (0.02 * sqrt(256))",
)
train_args.add_argument(
    "--learning_rate_scaling",
    type=str,
    default="linear",
    help="How to scale the learning rate as a function of batch size. Can be `linear` or `sqrt`",
)
train_args.add_argument(
    "--warmup_epochs", type=int, default=0, help="Number of epochs of warmup"
)
train_args.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="Momentum param for lars and SGD optimizers",
)
train_args.add_argument(
    "--weight_decay", type=float, default=0.0, help="Amount of weight decay to use"
)
train_args.add_argument(
    "--fine_tune_after_block",
    type=int,
    default=-1,
    help=(
        "The layers after which block that we will fine-tune. -1 means fine-tuning "
        "everything. 0 means fine-tuning after stem block. 4 means fine-tuning "
        "just the linear head."
    ),
)

train_args.add_argument(
    "--run_eagerly", action="store_true", help="Set eager tracing to true for debugging"
)


def try_restore_from_checkpoint(
    model,
    model_dir,
    keep_checkpoint_max,
    checkpoint,  # e.g., '/home/goh/Documents/D3M/simclr_tf2_models/pretrained/r50_2x_sk0/saved_model/'
    global_step,
    optimizer,
    zero_init_logits_layer,
):
    """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
    ckpt = tf.train.Checkpoint(
        model=model, global_step=global_step, optimizer=optimizer
    )
    checkpoint_manager = tf.train.CheckpointManager(
        ckpt, directory=model_dir, max_to_keep=keep_checkpoint_max
    )
    latest_ckpt = checkpoint_manager.latest_checkpoint
    if latest_ckpt:
        # Restore model weights, global step, optimizer states
        print("Restoring from latest checkpoint: %s", latest_ckpt)
        checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
    elif checkpoint:
        # Restore model weights only, but not global step and optimizer states
        if os.path.isfile(checkpoint) and os.path.splitext(checkpoint)[-1] == ".h5":
            print(f"Loading model weights from {checkpoint}")
            model.load_weights(checkpoint, by_name=True, skip_mismatch=True)
        elif os.path.isdir(checkpoint):
            h5_files = glob(os.path.join(checkpoint, "*.h5"))  # look for model weights
            if len(h5_files) > 0:
                print(
                    f"Found the following .h5 files: {pformat(h5_files, compact=True)}\n"
                    f"Restore model weights from {h5_files[0]}"
                )
                model.load_weights(
                    os.path.join(checkpoint, h5_files[0]),
                    by_name=True,
                    skip_mismatch=True,
                )
            else:
                print(
                    f"Could not find any .h5 files... "
                    f"Will try to convert {checkpoint} to .h5 and restore weights.\n"
                    f"Note that you cannot do this from within a mirrored strategy scope."
                )
                temp_model = tf.keras.models.load_model(checkpoint)
                h5_file = os.path.join(checkpoint, "model_weights.h5")
                temp_model.model.save_weights(h5_file)
                time.sleep(2)
                model.load_weights(h5_file, by_name=True, skip_mismatch=True)

        # print("Restoring from given checkpoint: %s", checkpoint)
        # checkpoint_manager2 = tf.train.CheckpointManager(
        #     tf.train.Checkpoint(model=model),
        #     directory=model_dir,
        #     max_to_keep=keep_checkpoint_max,
        # )
        # checkpoint_manager2.checkpoint.restore(checkpoint).expect_partial()
        # if zero_init_logits_layer:
        #     model = checkpoint_manager2.checkpoint.model
        #     output_layer_parameters = model.supervised_head.trainable_weights
        #     print(
        #         "Initializing output layer parameters %s to zero",
        #         [x.op.name for x in output_layer_parameters],
        #     )
        #     for x in output_layer_parameters:
        #         x.assign(tf.zeros_like(x))

    return checkpoint_manager


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_manager):
        super(CheckpointCallback, self).__init__()
        self.checkpoint_manager = checkpoint_manager

    def on_epoch_end(self, epoch, logs=None):
        self.checkpoint_manager.save()


if __name__ == "__main__":
    """
    Usage:
    ======
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/Mars_Classification/msl-labeled-data-set-v2.1/ --file_list train-set-v2.1.txt
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/UCMerced_LandUse_PNG/Images --ext png --gpu_ids [4,5,6,7]
    """

    args = parser.parse_args()

    print(f"Argparse parsed the following command-line arguments:\n")
    pprint(vars(args), indent=2, compact=True)

    image_size = args.image_size

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

    X, num_files, num_classes, class_mapping = get_files_and_labels(
        img_folder, ext=args.ext, metadata_file=file_list, mapping=class_mapping
    )

    classes = list(class_mapping.keys())

    ds = get_tf_dataset(
        X, args.ext, preprocess=True, width=image_size, height=image_size
    )

    if len(args.gpu_ids) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{int(d)}" for d in args.gpu_ids]
        )
        print(
            f"Running using MirroredStrategy on {strategy.num_replicas_in_sync} replicas"
        )
    else:
        if len(args.gpu_ids) == 1:
            d = args.gpu_ids[0]
        else:
            d = np.random.randint(len(tf.config.list_physical_devices("GPU")))
        strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:{d}")

    print(
        f"img_folder = {img_folder}\n"
        f"file_list = {file_list}\n"
        f"class_map_csv = {os.path.join(args.data_dir, args.class_map_csv)}\n"
        f"num_files = {num_files}\n"
        f"num_classes = {num_classes}\n"
        f"classes = {pformat(classes, compact=True)}"
    )

    # Set up checkpoint directory
    model_substr = (
        f"r{args.resnet_depth}_{args.width_multiplier}x_sk{int(args.sk_ratio > 0)}"
    )
    model_str = f"finetune_{args.dataset}_{model_substr}_proj{args.proj_out_dim}_{args.num_proj_layers}projLayers"

    print(f"{model_str}")
    log_name = f"{model_str}_{datetime.now().strftime('%Y-%m-%d_%H%M')}"

    if args.model_dir:
        model_dir = os.path.join(args.model_dir, log_name)
    elif args.checkpoint:
        model_dir = os.path.join(os.path.dirname(args.checkpoint), log_name)
    else:
        repo_root = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(repo_root, "model_checkpoints")
        model_dir = os.path.join(model_dir, log_name)

    os.makedirs(model_dir, exist_ok=True)
    # Dump provided command-line args
    json.dump(vars(args), open(os.path.join(model_dir, "config.json"), "w"), indent=2)

    with strategy.scope():

        # Build LR schedule and optimizer.
        # learning_rate = model_lib.WarmUpAndCosineDecay(
        #     args.learning_rate,
        #     num_files,
        #     args.warmup_epochs,
        #     args.epochs,
        #     args.batch_size,
        #     args.learning_rate_scaling,
        # )
        learning_rate = args.learning_rate
        optimizer = model_lib.build_optimizer(
            learning_rate, args.optimizer, args.momentum
        )

        # Build metrics
        weight_decay_metric = tf.keras.metrics.Mean("train/weight_decay")
        total_loss_metric = tf.keras.metrics.Mean("train/total_loss")
        supervised_loss_metric = tf.keras.metrics.Mean("train/supervised_loss")
        supervised_acc_metric = tf.keras.metrics.Mean("train/supervised_acc")
        all_metrics = [
            weight_decay_metric,
            total_loss_metric,
            supervised_loss_metric,
            supervised_acc_metric,
        ]
        metrics_dict = {
            "weight_decay": weight_decay_metric,
            "total_loss": total_loss_metric,
            "supervised_loss": supervised_loss_metric,
            "supervised_acc": supervised_acc_metric,
        }

        model = model_lib.Model(
            num_classes,
            image_size,
            train_mode="finetune",
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            resnet_depth=args.resnet_depth,
            sk_ratio=args.sk_ratio,
            width_multiplier=args.width_multiplier,
            proj_out_dim=args.proj_out_dim,
            num_proj_layers=args.num_proj_layers,
            ft_proj_selector=args.ft_proj_selector,
            head_mode="nonlinear",
            use_bn=True,  # whether to use batch norm in projection head
            fine_tune_after_block=args.fine_tune_after_block,
            linear_eval_while_pretraining=False,
        )
        model.build((None, args.image_size, args.image_size, 3))
        print(model.summary())

        # Restore from checkpoint if provided
        if args.checkpoint:

            # checkpoint_manager = tf.train.CheckpointManager(
            #     tf.train.Checkpoint(optimizer = optimizer, model=model),
            #     directory=model_dir,
            #     max_to_keep=args.keep_checkpoint_max,
            # )
            # checkpoint_manager.checkpoint.restore(args.checkpoint).expect_partial()
            checkpoint_manager = try_restore_from_checkpoint(
                model,
                model_dir,
                args.keep_checkpoint_max,
                args.checkpoint,
                tf.Variable(0, dtype=tf.int64),
                optimizer,
                False,
            )

        model.compile(
            loss=obj_lib.add_supervised_loss,
            metrics=all_metrics,
            optimizer=optimizer,
            run_eagerly=args.run_eagerly
        )

        temp_ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="train/supervised_acc", patience=6
        )
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(
                os.path.expanduser("~"), "Documents", "tensorboard", log_name
            ),
            write_graph=False,
            write_images=False,
            write_steps_per_second=True,
        )
        callbacks = [early_stop, tensorboard]
        if args.checkpoint:
            callbacks.append(CheckpointCallback(checkpoint_manager))
        model.fit(temp_ds, epochs=args.epochs, callbacks=callbacks)

    print(model.summary())
    # tf.keras.utils.plot_model(model.model(images.shape[1:]), "model2.png", show_shapes=True, expand_nested=True)
