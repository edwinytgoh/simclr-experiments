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
import tensorflow_datasets as tfds
from absl import logging
from sklearn.preprocessing import MinMaxScaler

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
import simclr.objective as obj_lib
from simclr.finetune import CheckpointCallback, try_restore_from_checkpoint
from simclr.model import Model, WarmUpAndCosineDecay, build_optimizer, generate_model
from simclr.utils import get_ai4mars_tfds_masks_as_labels, get_msl_seg_tfds

logging.set_verbosity(logging.ERROR)

home = os.path.expanduser("~")


def run(config):
    weighted_loss = False

    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.MirroredStrategy(["/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
    # strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:2")

    train_ds, val_ds, num_classes, num_train, num_val = get_ai4mars_train_val(config["data_config"])

    with strategy.scope():
        model = instantiate_and_build_seg_model(config)
        compile_model_with_metrics_and_optimizer(model, config)

    checkpoint, experiment_dir, tb_logdir, log_name = set_up_dirs(config)

    checkpoint_manager = try_restore_from_checkpoint(
        model,
        config["data_config"]["experiment_dir"],
        config["train_config"]["num_checkpoints_max"],
        tf.Variable(0, dtype=tf.int64),
        model.optimizer,
        checkpoint=config["data_config"]["checkpoint_dir"],
        zero_init_logits_layer=False,
    )

    tb_config = get_hparams_config_and_save(config)
    callbacks = set_up_callbacks(checkpoint_manager, tb_config, tb_logdir)

    num_epochs = config["train_config"]["num_epochs"]
    model.fit(train_ds, epochs=num_epochs, callbacks=callbacks, validation_data=val_ds)


def get_ai4mars_train_val(data_config):
    # ai4mars_ds = get_msl_seg_tfds(msl_folder, split="train")
    msl_folder = data_config["msl_folder"]
    batch_size = data_config["batch_size"]
    ai4mars_ds = get_ai4mars_tfds_masks_as_labels(msl_folder, split="train")
    num_images = int(tf.data.experimental.cardinality(ai4mars_ds).numpy())
    num_classes = 7
    num_val = int(0.2 * num_images)
    val_ds = ai4mars_ds.take(num_val)
    train_ds = ai4mars_ds.skip(num_val)
    num_train = int(tf.data.experimental.cardinality(train_ds).numpy())
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    data_config.update({
        "num_images": num_images,
        "num_classes": num_classes,
        "num_train": num_train,
        "num_val": num_val
    })

    return train_ds, val_ds


def instantiate_and_build_seg_model(config):
    model = SegModel(config)
    image_size = config["data_config"]["image_size"]
    model.build((None, image_size, image_size, 3))
    model.summary()
    return model


def compile_model_with_metrics_and_optimizer(model, config):
    # Build LR schedule and optimizer.
    optimizer = build_optimizer(
        WarmUpAndCosineDecay(config),
        config["optimizer_config"]["optimizer_name"],
        config["optimizer_config"]["momentum"],
        weight_decay=config["optimizer_config"]["weight_decay"]
    )

    # Build metrics
    loss = tf.keras.metrics.SparseCategoricalCrossentropy
    all_metrics = [
        loss(name="crossentropy_loss", from_logits=True),
        MeanIoU(config["data_config"]["num_classes"], name="iou"),
        SparseCategoricalAccuracy("accuracy"),  # "sparse_cat_acc"
    ]

    # Compile
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=all_metrics,
        optimizer=optimizer,
        run_eagerly=config["model_config"]["compile_config"]["run_eagerly"],
    )


def set_up_dirs(config):
    dataset = config["data_config"]["dataset_name"]
    model_dir = config["data_config"]["model_root_dir"]
    num_proj_layers = config["model_config"]["proj_config"]["num_proj_layers"]
    proj_out_dim = config["model_config"]["proj_config"]["proj_out_dim"]
    resnet_depth = config["model_config"]["resnet_config"]["resnet_depth"]
    sk_ratio = config["model_config"]["resnet_config"]["sk_ratio"]
    width_multiplier = config["model_config"]["resnet_config"]["width_multiplier"]

    # Set up checkpoint directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    model_substr = f"r{resnet_depth}_{width_multiplier}x_sk{int(sk_ratio > 0)}"
    checkpoint = os.path.join(
        home, f"Documents/D3M/simclr_tf2_models/pretrained/{model_substr}/saved_model"
    )

    # Set up experiment dir
    model_str = f"finetune_{dataset}_{model_substr}_proj{proj_out_dim}_{num_proj_layers}projLayers"
    log_name = f"{model_str}_{timestamp}"
    experiment_dir = os.path.join(model_dir, log_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up tensorboard dir
    tb_logdir = os.path.join(home, "Documents", "tensorboard", log_name)

    print(
        f"timestamp = {timestamp}\n"
        f"model_substr = {model_substr}\n"
        f"model_str = {model_str}\n"
        f"experiment_dir = {experiment_dir}\n"
        f"log_name = {log_name}\n"
        f"checkpoint = {checkpoint}"
    )

    config["data_config"].update(
        {
            "experiment_dir": experiment_dir,
            "log_name": log_name,
            "checkpoint_dir": checkpoint
        }
    )

    return checkpoint, experiment_dir, tb_logdir, log_name


def get_hparams_config_and_save(config):
    # Keep track of hyperparameters in Tensorboard
    experiment_dir = config["data_config"]["experiment_dir"]
    json.dump(
        config,
        open(os.path.join(model_dir, f"model_config.json"), "w"),
        indent=2,
    )

    tb_config = dict()
    exclude = ["msl_folder", "model_root_dir", "checkpoint_dir", "experiment_dir", "model_str",
               "log_name", "model_substr", ""]

    def add_to_tb_config(d):
        for k, v in d.items():
            if isinstance(v, dict):
                add_to_tb_config(v)
            else:
                if k not in exclude:
                    tb_config[k] = v

    add_to_tb_config(config)

    return tb_config


def set_up_callbacks(checkpoint_manager, config, tb_logdir, patience=7):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tb_logdir,
        write_graph=False,
        write_images=False,
        write_steps_per_second=False,
    )
    # remove unncessary/str hparams
    callbacks = [early_stop, tensorboard, hp.KerasCallback(tb_logdir, config)]
    callbacks.append(CheckpointCallback(checkpoint_manager))
    return callbacks


if __name__ == "__main__":
    # Config params
    config = {
        "data_config": {
            "dataset_name": "ai4mars_512px_masks_as_classes",
            "msl_folder": os.path.join(
                home,
                "Documents/D3M/ai4mars-dataset-merged-0.1/msl_preprocessed_512px_masks_as_classes/",
            ),
            "model_root_dir": os.path.join(home, "Documents/D3M/ai4mars_models/"),
            "image_size": 512,
            "num_classes": None,  # fill out later
            "checkpoint_dir": None,  # fill out later
            "batch_size": 1,
        },
        "model_config": {
            "train_mode": "finetune",
            "resnet_config": {
                "resnet_depth": 50,
                "width_multiplier": 2,
                "sk_ratio": 0.0625,
                "finetune_after_block": -1,
            },
            "proj_config": {
                "num_proj_layers": 3,
                "proj_out_dim": 128,
                "proj_head_mode": "nonlinear",
                "finetune_proj_selector": 0,
            },
            "compile_config": {
                "run_eagerly": False,
            },
            "linear_eval_while_pretraining": False,
        },
        "optimizer_config": {
            "optimizer_name": "lars",
            "learning_rate": None,
            "base_learning_rate": 0.002,
            "momentum": 0.9,
            "weight_decay": 0.0,
            "warmup_epochs": 2,
            "lr_scaling": "sqrt"  # sqrt or linear
        },
        "train_config": {
            "num_epochs": 50,
            "num_checkpoints_max": 30
        }
    }
    config["optimizer_config"]["learning_rate"] = config["optimizer_config"][
        "base_learning_rate"
    ] * np.sqrt(config["data_config"]["batch_size"])
