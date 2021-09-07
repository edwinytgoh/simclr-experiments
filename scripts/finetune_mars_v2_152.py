#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys
from datetime import datetime
from functools import partial

import numpy as np
import tensorflow as tf
from absl import logging
from sklearn.preprocessing import MinMaxScaler

# https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
from tensorboard.plugins.hparams import api as hp

# add simclr folder to path;
# otherwise use setup.py and changes imports to simclr.objective, simclr.model, etc.
root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(root, "simclr"))

import objective as obj_lib
from finetune import CheckpointCallback, try_restore_from_checkpoint
from model import Model, build_optimizer, generate_model
from utils import get_files_and_labels, get_tf_dataset, read_class_label_map

logging.set_verbosity(logging.ERROR)

# Config params
imsize = 227
batch_size = 128
learning_rate = 0.002 * np.sqrt(batch_size)
resnet_depth = 50
width_multiplier = 2
sk_ratio = 0.0625
fine_tune_after_block = -1
ft_proj_selector = 0
num_proj_layers = 3
proj_out_dim = 128
weighted_loss = True
model_dir = "/home/goh/Documents/D3M/mars_simclr_models/"
dataset = "mars_v2_weighted_loss"
txt_file = "paper_2900_train.txt"
# find Google's simclr weights in simclr_weights_dir/r50_2x_sk0/saved_model; see L.76
simclr_weights_dir = f"/home/goh/Documents/D3M/simclr_tf2_models/pretrained"
print(f"Batch size = {batch_size}")
print(f"Learning rate = {learning_rate:.3f}")

strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.MirroredStrategy(['/gpu:1', '/gpu:2', '/gpu:3', '/gpu:4'])
# strategy = tf.distribute.OneDeviceStrategy(device=f"/gpu:1")


# # Initialize TensorFlow Dataset
data_dir = os.path.join(root, "data", "msl-labeled-data-set-v2.1")

img_folder = os.path.join(data_dir, "images")
file_list = os.path.join(data_dir, txt_file)
class_mapping = read_class_label_map(os.path.join(data_dir, "class_map.csv"))
X, num_files, num_classes, class_mapping = get_files_and_labels(
    img_folder, ext="jpg", metadata_file=file_list, mapping=class_mapping
)
classes = list(class_mapping.keys())
ds = get_tf_dataset(X, "jpg", preprocess=True, width=imsize, height=imsize)


# # Set up model params and paths
# Set up checkpoint directory
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
model_substr = f"r{resnet_depth}_{width_multiplier}x_sk{int(sk_ratio > 0)}"
model_str = (
    f"finetune_{dataset}_{model_substr}_proj{proj_out_dim}_{num_proj_layers}projLayers"
)
log_name = f"{model_str}_{timestamp}"
checkpoint = os.path.join(simclr_weights_dir, f"{model_substr}/saved_model")
model_dir = os.path.join(model_dir, log_name)
print(
    f"timestamp = {timestamp}\n"
    f"model_substr = {model_substr}\n"
    f"model_str = {model_str}\n"
    f"model_dir = {model_dir}\n"
    f"log_name = {log_name}\n"
    f"checkpoint = {checkpoint}"
)


# Get Classes and Labels
labels = [x[1] for x in X]
num_examples = [
    labels.count(i) for i in sorted(set(labels))
]  # gives num. training images in each class; len(num_examples) = num classes
class_count_reciprocal = [1 / n for n in num_examples]

class_count_reciprocal_scaled = (
    MinMaxScaler(feature_range=(1, 10))  # arbitrarily scale from 1 - 10
    .fit_transform(np.array(class_count_reciprocal)[:, np.newaxis])
    .astype(np.float32)
)

# Initialize Model
with strategy.scope():
    optimizer = build_optimizer(learning_rate, "lars", 0.9, weight_decay=0.0)

    # Build metrics
    metrics = [
        tf.keras.metrics.Mean("train/weight_decay"),
        tf.keras.metrics.Mean("train/total_loss"),
        tf.keras.metrics.Mean("train/supervised_loss"),
        tf.keras.metrics.Mean("train/supervised_acc"),
    ]

    model = Model(
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
        weighted_loss=weighted_loss,
    )

    model.build((None, imsize, imsize, 3))

    model.summary()
    checkpoint_manager = try_restore_from_checkpoint(
        model,
        model_dir,
        15,  # num. ckpts to keep
        tf.Variable(0, dtype=tf.int64),
        optimizer,
        checkpoint=checkpoint,
        zero_init_logits_layer=False,
    )

    # define loss function
    def loss_function(labels, predictions):
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, predictions)

    # https://www.tensorflow.org/guide/keras/train_and_evaluate#handling_losses_and_metrics_that_dont_fit_the_standard_signature
    if weighted_loss:
        loss = obj_lib.add_weighted_supervised_loss(batch_size)
    else:
        loss = obj_lib.add_supervised_loss

    model.compile(loss=loss, metrics=metrics, optimizer=optimizer, run_eagerly=False)


# Create log directory and log hyperparam config to the directory
os.makedirs(model_dir, exist_ok=True)
config = model.get_config()
config.update(
    {"batch_size": batch_size, "learning_rate": learning_rate, "momentum": 0.9}
)

json.dump(
    config,
    open(os.path.join(model_dir, f"model_config_{timestamp}.json"), "w"),
    indent=2,
)

# Set up batched dataset for training; optionally include class weights
if weighted_loss:
    weights = [class_count_reciprocal_scaled[int(l.numpy())] for _, l in iter(ds)]
    weights_ds = tf.data.Dataset.from_tensor_slices(weights)
    temp_ds = tf.data.Dataset.zip((ds, weights_ds))
    temp_ds = temp_ds.batch(batch_size)
else:
    temp_ds = ds.batch(batch_size)

# temp_ds = strategy.experimental_distribute_dataset(temp_ds)

# Define EarlyStop, Tensorboard and HParam callbacks
monitor = "train/supervised_loss" if weighted_loss else "train/supervised_acc"
early_stop = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=6)

logdir = os.path.join(os.path.expanduser("~"), "Documents", "tensorboard", log_name)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir=logdir, write_graph=False, write_images=False, write_steps_per_second=False,
)

callbacks = [early_stop, tensorboard, hp.KerasCallback(logdir, config)]
callbacks.append(CheckpointCallback(checkpoint_manager))


# Fit model
model.fit(temp_ds, epochs=300, callbacks=callbacks)
