import argparse
import os
from pprint import pformat, pprint

import numpy as np
import tensorflow as tf

import data as data_lib
import metrics
import model as model_lib
import objective as obj_lib
from utils import get_files_and_labels, read_class_label_map, get_tf_dataset
from metrics import update_finetune_metrics_train

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
train_args.add_argument(
    "--resnet_depth", type=int, default=50, help="Resnet depth"
)
train_args.add_argument(
    "--optimizer", type=str, default="lars", help="Optimizer to use. Choices: ['adam', 'lars', 'SGD']"
)
train_args.add_argument(
    "--learning_rate", type=float, default=0.3, help="Initial lr per batch size of 256"
)
train_args.add_argument(
    "--learning_rate_scaling", type=str, default='linear', help='How to scale the learning rate as a function of batch size. Can be `linear` or `sqrt`'
)
train_args.add_argument(
    "--warmup_epochs", type=int, default=5, help="Number of epochs of warmup"
)
train_args.add_argument(
    "--momentum", type=float, default=0.9, help="Momentum param for lars and SGD optimizers"
)
train_args.add_argument(
    "--weight_decay", type=float, default=1e-6, help='Amount of weight decay to use'
)

train_args.add_argument(
    "--run_eagerly", action='store_true', help="Set eager tracing to true for debugging"
)

# def try_restore_from_checkpoint(model, global_step, optimizer):
#   """Restores the latest ckpt if it exists, otherwise check FLAGS.checkpoint."""
#   checkpoint = tf.train.Checkpoint(
#       model=model, global_step=global_step, optimizer=optimizer)
#   checkpoint_manager = tf.train.CheckpointManager(
#       checkpoint,
#       directory=FLAGS.model_dir,
#       max_to_keep=FLAGS.keep_checkpoint_max)
#   latest_ckpt = checkpoint_manager.latest_checkpoint
#   if latest_ckpt:
#     # Restore model weights, global step, optimizer states
#     print('Restoring from latest checkpoint: %s', latest_ckpt)
#     checkpoint_manager.checkpoint.restore(latest_ckpt).expect_partial()
#   elif FLAGS.checkpoint:
#     # Restore model weights only, but not global step and optimizer states
#     print('Restoring from given checkpoint: %s', FLAGS.checkpoint)
#     checkpoint_manager2 = tf.train.CheckpointManager(
#         tf.train.Checkpoint(model=model),
#         directory=FLAGS.model_dir,
#         max_to_keep=FLAGS.keep_checkpoint_max)
#     checkpoint_manager2.checkpoint.restore(FLAGS.checkpoint).expect_partial()
#     if FLAGS.zero_init_logits_layer:
#       model = checkpoint_manager2.checkpoint.model
#       output_layer_parameters = model.supervised_head.trainable_weights
#       print('Initializing output layer parameters %s to zero',
#                    [x.op.name for x in output_layer_parameters])
#       for x in output_layer_parameters:
#         x.assign(tf.zeros_like(x))

#   return checkpoint_manager


if __name__ == "__main__":
    """
    Usage:
    ======
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/Mars_Classification/msl-labeled-data-set-v2.1/ --file_list train-set-v2.1.txt
    python simclr/finetune.py --data_dir /home/goh/Documents/D3M/UCMerced_LandUse_PNG/Images --ext png --gpu_ids [4,5,6,7]
    """

    args = parser.parse_args()
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

    ds = get_tf_dataset(X, args.ext, preprocess=True, width=image_size, height=image_size)

    if len(args.gpu_ids) > 1:
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/gpu:{int(d)}" for d in args.gpu_ids]
        )
        print(f"Running using MirroredStrategy on {strategy.num_replicas_in_sync} replicas")
    else:
        if len(args.gpu_ids) == 1:
            d = args.gpu_ids[0]
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

    with strategy.scope():

        # Build LR schedule and optimizer.
        learning_rate = model_lib.WarmUpAndCosineDecay(
            args.learning_rate,
            num_files,
            args.warmup_epochs,
            args.epochs,
            args.batch_size,
            args.learning_rate_scaling,
        )
        optimizer = model_lib.build_optimizer(learning_rate, args.optimizer, args.momentum)

        # Build metrics
        weight_decay_metric = tf.keras.metrics.Mean('train/weight_decay')
        total_loss_metric = tf.keras.metrics.Mean('train/total_loss')
        supervised_loss_metric = tf.keras.metrics.Mean("train/supervised_loss")
        supervised_acc_metric = tf.keras.metrics.Mean("train/supervised_acc")
        all_metrics = [
            weight_decay_metric, total_loss_metric, supervised_loss_metric, supervised_acc_metric
        ]
        metrics_dict = {
            "weight_decay": weight_decay_metric,
            "total_loss": total_loss_metric,
            "supervised_loss": supervised_loss_metric,
            "supervised_acc": supervised_acc_metric
        }
        # Restore checkpoint if available.
        # checkpoint_manager = try_restore_from_checkpoint(model, optimizer.iterations, optimizer)

        model = model_lib.Model(
            num_classes,
            image_size,
            train_mode="finetune",
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            resnet_depth=args.resnet_depth,
            sk_ratio=0.0,
            width_multiplier=1,
            proj_out_dim=128,
            num_proj_layers=3,
            ft_proj_selector=0,
            head_mode="linear",
            use_bias=False,  # whether to use bias in projection head
            use_bn=True,  # whether to use batch norm in projection head
            finetune_after_block=-1,
            linear_eval_while_pretraining=False,
        )

        model.compile(
            loss=obj_lib.add_supervised_loss,
            metrics=all_metrics,
            optimizer=optimizer,
            run_eagerly=args.run_eagerly
        )

        temp_ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        model.fit(temp_ds, epochs=args.epochs)

    print(model.summary())
    # tf.keras.utils.plot_model(model.model(images.shape[1:]), "model2.png", show_shapes=True, expand_nested=True)