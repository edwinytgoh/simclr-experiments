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

# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.losses import Reduction, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Mean, MeanIoU, SparseCategoricalAccuracy, Sum

# add simclr folder to path;
# otherwise use setup.py and changes imports to simclr.objective, simclr.model, etc.
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(root, "simclr"))

import objective as obj_lib
from data_util import CROP_PROPORTION, center_crop, preprocess_image
from finetune import CheckpointCallback, try_restore_from_checkpoint
from model import Model, build_optimizer, generate_model
from segmentation_model import SegModel
from utils import get_files_and_labels, get_tf_dataset, read_class_label_map

logging.set_verbosity(logging.ERROR)

# Config params
imsize = 227
batch_size = 64
learning_rate = 0.002 * np.sqrt(batch_size)
resnet_depth = 101
width_multiplier = 2
sk_ratio = 0.0625
fine_tune_after_block = -1
ft_proj_selector = 0
num_proj_layers = 3
proj_out_dim = 128
weighted_loss = False
model_dir = "/home/goh/Documents/D3M/oxford_iiit_pet_models/"
dataset = "oxford_iiit_pet"
txt_file = "paper_2900_train.txt"
print(f"Batch size = {batch_size}")
print(f"Learning rate = {learning_rate:.3f}")

strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(["/gpu:1", "/gpu:2", "/gpu:3", "/gpu:4"])
# strategy = tf.distribute.MirroredStrategy(["/gpu:2", "/gpu:3"])
# strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:1")


# # Initialize TensorFlow Dataset
# batch_size = 16
dataset_name = "oxford_iiit_pet"

tfds_dataset, tfds_info = tfds.load(dataset_name, split="train", with_info=True)
num_images = tfds_info.splits["train"].num_examples
num_classes = tfds_info.features["label"].num_classes
num_classes = 3


def load_image_train(x):
    img = preprocess_image(
        x["image"], imsize, imsize, is_training=False, color_distort=False
    )
    mask_min = tf.cast(tf.reduce_min(x["segmentation_mask"]), tf.float32)
    mask_max = tf.cast(tf.reduce_max(x["segmentation_mask"]), tf.float32)
    mask = center_crop(x["segmentation_mask"], imsize, imsize, CROP_PROPORTION)
    mask = tf.reshape(mask, [imsize, imsize, 1])
    mask = tf.clip_by_value(mask, mask_min, mask_max)
    mask = tf.cast(tf.math.round(mask), tf.int32)
    mask = mask - 1
    return (img, mask)


def load_image_train_resize(x):
    img = tf.image.resize(x["image"], (imsize, imsize))
    img = tf.clip_by_value(img, 0.0, 1.0)
    mask_min = tf.cast(tf.reduce_min(x["segmentation_mask"]), tf.float32)
    mask_max = tf.cast(tf.reduce_max(x["segmentation_mask"]), tf.float32)
    mask = tf.image.resize(x["segmentation_mask"], (imsize, imsize))
    mask = tf.reshape(mask, [imsize, imsize, 1])
    mask = tf.clip_by_value(mask, mask_min, mask_max)
    mask = tf.cast(tf.math.round(mask), tf.int32)
    mask = mask - 1
    return (img, mask)


ds = tfds_dataset.map(load_image_train_resize, num_parallel_calls=tf.data.AUTOTUNE)

num_val = int(0.2 * num_images)
val_ds = ds.take(num_val)
train_ds = ds.skip(num_val)


# # Set up model params and paths
# Set up checkpoint directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
model_substr = f"r{resnet_depth}_{width_multiplier}x_sk{int(sk_ratio > 0)}"
model_str = (
    f"finetune_{dataset}_{model_substr}_proj{proj_out_dim}_{num_proj_layers}projLayers"
)
log_name = f"{model_str}_{timestamp}"
checkpoint = (
    f"/home/goh/Documents/D3M/simclr_tf2_models/pretrained/{model_substr}/saved_model"
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
        # Mean("weight_decay"),
        # Mean("total_loss_mean"),
        # Sum("total_loss_sum"),
        # Mean("supervised_loss_mean"),
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
        30,
        tf.Variable(0, dtype=tf.int64),
        optimizer,
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
config.update(
    {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": 0.9,
    }
)

json.dump(
    config,
    open(os.path.join(model_dir, f"model_config_{timestamp}.json"), "w"),
    indent=2,
)

temp_ds = train_ds.batch(batch_size)
val_ds_batched = val_ds.batch(batch_size)

monitor = "supervised_loss" if weighted_loss else "accuracy"
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7)

logdir = os.path.join(os.path.expanduser("~"), "Documents", "tensorboard", log_name)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    write_graph=False,
    write_images=False,
    write_steps_per_second=False,
)
# remove unncessary/str hparams
callbacks = [early_stop, tensorboard, hp.KerasCallback(logdir, config)]
callbacks.append(CheckpointCallback(checkpoint_manager))
model.fit(temp_ds, epochs=50, callbacks=callbacks, validation_data=val_ds_batched)

# model.save_weights(os.path.join(model_dir, "full_model_116_epochs.h5"), save_format="h5")
