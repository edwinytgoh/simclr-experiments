#!/usr/bin/env python
# coding: utf-8
import copy
import json
import os
import sys
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf
from absl import logging
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.training.tracking import data_structures

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


[
    tf.config.experimental.set_memory_growth(d, True)
    for d in tf.config.list_physical_devices()
    if d.device_type == "GPU"
]

# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, MeanIoU, SparseCategoricalAccuracy, Sum

from simclr.data_util import CROP_PROPORTION, center_crop, preprocess_image
from simclr.segmentation_model import SegModel

# tf.config.run_functions_eagerly(True)
# tf.compat.v1.enable_eager_execution()
# tf.data.experimental.enable_debug_mode()

sys.path.insert(0, "../simclr")
from arguments import parser

import simclr.objective as obj_lib
from simclr.finetune import CheckpointCallback, try_restore_from_checkpoint
from simclr.model import Model, build_optimizer
from simclr.utils import get_ai4mars_tfds_masks_as_labels, get_msl_seg_tfds

logging.set_verbosity(logging.ERROR)

home = os.path.expanduser("~")


if __name__ == "__main__":

    args = parser.parse_args()
    # Config params
    imsize = 512
    batch_size = 8
    learning_rate = 0.002 * np.sqrt(batch_size)
    resnet_depth = 50
    width_multiplier = 2
    sk_ratio = 0.0625
    fine_tune_after_block = -1
    ft_proj_selector = 0
    num_proj_layers = 3
    proj_out_dim = 128
    weighted_loss = False
    model_dir = os.path.join(home, "Documents/D3M/ai4mars_models/")
    msl_folder = os.path.join(
        home,
        "Documents/D3M/ai4mars-dataset-merged-0.1/msl_preprocessed_512px_masks_as_classes/",
    )
    dataset = "ai4mars_512px_exclude_null"
    txt_file = "paper_2900_train.txt"
    print(f"Batch size = {batch_size}")
    print(f"Learning rate = {learning_rate:.3f}")

    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.MirroredStrategy(["/gpu:1", "/gpu:2"])
    # strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
    # strategy = tf.distribute.MirroredStrategy(["/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    # strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:2")

    ai4mars_ds = get_ai4mars_tfds_masks_as_labels(msl_folder, split="train")

    num_images = tf.data.experimental.cardinality(ai4mars_ds).numpy()
    num_classes = 6

    train_pct = args.train_pct
    train_pct = train_pct / 100 if 1 < train_pct <= 100 else train_pct
    num_train = min(int(train_pct * num_images), num_images)
    train_ds = ai4mars_ds.take(num_train)

    # get 20% val
    num_val = int(0.2 * num_images)
    val_ds = ai4mars_ds.shuffle(num_images, seed=123).take(num_val)

    # # Set up model params and paths
    # Set up checkpoint directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    model_substr = f"r{resnet_depth}_{width_multiplier}x_sk{int(sk_ratio > 0)}"
    model_str = f"finetune_{dataset}_{model_substr}"
    if args.checkpoint is not None:
        checkpoint = os.path.join(
            home,
            f"Documents/D3M/simclr_tf2_models/pretrained/{model_substr}/saved_model",
        )
    else:
        checkpoint = None

    if checkpoint is None:
        pretrain_or_supervised = "supervised"
    elif "supervised" in checkpoint:
        pretrain_or_supervised = "transfer"
    elif "pretrained" in checkpoint:
        pretrain_or_supervised = "pretrained"
    else:
        pretrain_or_supervised = "unknown"

    log_name = (
        f"{model_str}_{pretrain_or_supervised}_{train_pct*100:.0f}pctTrain_{timestamp}"
    )

    model_dir = os.path.join(model_dir, log_name)
    print(
        f"timestamp = {timestamp}\n"
        f"model_substr = {model_substr}\n"
        f"model_str = {model_str}\n"
        f"model_dir = {model_dir}\n"
        f"log_name = {log_name}\n"
        f"checkpoint = {checkpoint}"
    )

    # # Initialize Model
    with strategy.scope():
        optimizer = build_optimizer(learning_rate, "lars", 0.9, weight_decay=0.0)

        # Build metrics
        all_metrics = [
            tf.keras.metrics.SparseCategoricalCrossentropy(
                name="crossentropy_loss", from_logits=True
            ),
            MeanIoU(num_classes, name="iou"),
            SparseCategoricalAccuracy("accuracy"),  # "sparse_cat_acc"
        ]

        model = SegModel(
            num_classes,
            imsize,
            train_mode="finetune",
            optimizer_name="lars",
            weight_decay=0,
            resnet_depth=resnet_depth,
            sk_ratio=sk_ratio,
            width_multiplier=width_multiplier,
            proj_out_dim=proj_out_dim,
            num_proj_layers=num_proj_layers,
            ft_proj_selector=ft_proj_selector,
            head_mode="nonlinear",
            fine_tune_after_block=fine_tune_after_block,
            linear_eval_while_pretraining=False,
        )

        model.build((None, imsize, imsize, 3))

        model.summary()

        checkpoint_manager = try_restore_from_checkpoint(
            model,
            model_dir,
            40,
            tf.Variable(0, dtype=tf.int64),
            optimizer,
            # checkpoint=checkpoint,
            checkpoint=checkpoint,
            zero_init_logits_layer=False,
        )

        model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=all_metrics,
            optimizer=optimizer,
            run_eagerly=False,
        )


    os.makedirs(model_dir, exist_ok=True)

    # Keep track of hyperparameters in Tensorboard
    config = model.get_config()
    checkpoint_str = "None" if checkpoint is None else checkpoint
    config.update(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "momentum": 0.9,
            "dataset": dataset,
            "checkpoint": checkpoint_str,
            "num_train": int(tf.data.experimental.cardinality(train_ds).numpy()),
        }
    )

    json.dump(
        config,
        open(os.path.join(model_dir, f"model_config_{timestamp}.json"), "w"),
        indent=2,
    )

    temp_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds_batched = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    monitor = "supervised_loss" if weighted_loss else "accuracy"
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=7)

    logdir = os.path.join(os.path.expanduser("~"), "Documents", "tensorboard", log_name)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=False,
        write_images=False,
        # write_steps_per_second=False,
    )
    # remove unncessary/str hparams
    callbacks = [early_stop, tensorboard, hp.KerasCallback(logdir, config)]
    callbacks.append(CheckpointCallback(checkpoint_manager))
    model.fit(temp_ds, epochs=50, callbacks=callbacks, validation_data=val_ds_batched)

    # model.save_weights(os.path.join(model_dir, "full_model_116_epochs.h5"), save_format="h5")
