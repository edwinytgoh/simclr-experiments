import argparse
import json
import os
import time
from datetime import datetime
from glob import glob
from pprint import pformat, pprint

import numpy as np
import tensorflow as tf
from absl import logging

from arguments import parser
import simclr.model as model_lib
import simclr.objective as obj_lib
from simclr.utils import get_files_and_labels, get_tf_dataset, read_class_label_map

logging.set_verbosity(logging.ERROR)

def try_restore_from_checkpoint(
    model,
    model_dir,
    keep_checkpoint_max,
    global_step,
    optimizer,
    checkpoint=None,  # e.g., '/home/goh/Documents/D3M/simclr_tf2_models/pretrained/r50_2x_sk0/saved_model/'
    zero_init_logits_layer=False,
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
        print("Restoring from latest checkpoint: ", latest_ckpt)
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
                model.load_weights(h5_file, by_name=True, skip_mismatch=True)
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
