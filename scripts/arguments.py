import argparse
parser = argparse.ArgumentParser()

data_args = parser.add_argument_group("Data args")
data_args.add_argument(
    "--data_dir",
    default="",
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
data_args.add_argument(
    "--train_pct",
    type=float,
    default=0.8,
    help="Percent of train dataset to retain for training. Rest will be used as validation."
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
    default=0.002,
    help="Initial learning rate per batch size of 256 (0.02 * sqrt(256))",
)
train_args.add_argument(
    "--learning_rate_scaling",
    type=str,
    default="sqrt",
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