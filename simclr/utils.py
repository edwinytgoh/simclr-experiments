from functools import partial
import os
import sys
import csv
import shutil
from glob import glob
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import splitfolders
import sklearn.model_selection
import tensorflow as tf

FLAGS_color_jitter_strength = 0.3
sys.path.insert(0, os.path.dirname(__file__))
from data_util import preprocess_image


def read_class_label_map(csv_file):
    assert os.path.isfile(
        csv_file
    ), f"Provided file {csv_file} is not a valid CSV file."
    reader = csv.reader(open(csv_file, "r"))
    return {c: int(i) for (i, c) in reader}


def infer_classes_from_filepaths(all_files: List):
    all_files = [os.path.abspath(f) for f in all_files]
    classes = [f.split(os.path.sep)[-2].strip() for f in all_files]
    idx_mapping = {c: i for i, c in enumerate(sorted(set(classes)))}
    # write idx_mapping to csv in dataset root dir
    folder = os.path.commonprefix(all_files)
    w = csv.writer(open(os.path.join(folder, "class_idx_map.csv"), "w"))
    w.writerows([[i, c] for c, i in idx_mapping.items()])
    mapping: dict = {f: idx_mapping[c] for (f, c) in zip(all_files, classes)}
    return mapping, idx_mapping


def get_files_and_labels(
    folder: str, ext: str = "png", mapping: dict = None, metadata_file=None
):

    if metadata_file is not None:
        lines = open(metadata_file, "r").readlines()
        all_files = []
        labels = []
        for l in lines:
            file_label = l.strip().split(f".{ext}")
            file = file_label[0].strip() + f".{ext}"
            if len(file.split(os.path.sep)) <= 1:
                file = os.path.join(folder, file)
            all_files.append(file)
            labels.append(int(file_label[1].strip()))
        idx_mapping = None
    else:
        path = os.path.join(os.path.abspath(folder), "**", f"**.{ext}")
        all_files = glob(path, recursive=True)
        all_files = [f for f in all_files if f".{ext}" in f]
        assert len(all_files) > 0, f"Couldn't find any files in {path}"

        if mapping is None:
            #file_mapping --> file: class_index
            # idx_mapping --> class: index
            file_mapping, idx_mapping = infer_classes_from_filepaths(all_files)
            mapping = idx_mapping

        labels = [file_mapping[f] for f in all_files]
    return list(zip(all_files, labels)), len(all_files), len(set(labels)), mapping


def get_pct_split(percent: int, X, y, total_sample_count: int, seed: int = 123):
    """Extract a stratified N% split from the provided dataset (X) based on labels (y).

    Parameters
    ----------
    percent : int
        Percentage of the FULL dataset to extract.
        (FULL in this case means train+test+val)
        N_extracted = percent/100 * total_sample_count
    X : Iterable
        Typically the full training set from which samples are extracted.
        We use the training set to generate splits in order to exclude test examples.
    y : Iterable
        Labels corresponding to X. Used to generate stratified samples
    total_sample_count : int
        Total size of the FULL dataset (train + test + val)
    seed : int, optional
        Random state to shuffle data before splitting, by default 123

    Returns
    -------
    X_split : Iterable
        Stratified subset of the provided X
        len(X_split) == int(percent/100 * total_sample_count)
    """
    if total_sample_count is None:
        total_sample_count = len(X)

    num_images_pct = int((percent / 100) * total_sample_count)
    ratio = num_images_pct / len(X)
    inv_ratio = (len(X) - num_images_pct) / len(X)
    X_split, _ = sklearn.model_selection.train_test_split(
        X, train_size=ratio, test_size=inv_ratio, stratify=y, random_state=seed,
    )
    print(
        f"Extracted {percent}% (N={len(X_split)}) "
        f"of full sample {total_sample_count} from "
        f"provided subset of size {len(X)} "
        f"({ratio*100:.2f}% of subset)"
    )
    return X_split


def load_jpg(filepath: str) -> Tuple[tf.Tensor]:
    """Read a JPG image into a tf.Tensor object
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def load_png(filepath: str) -> Tuple[tf.Tensor]:
    """Read a PNG image into a tf.Tensor object
    """
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)
    return img


def _preprocess(image, width=256, height=256):
    preprocessed = preprocess_image(
        image, width, height, is_training=False, color_distort=False
    )
    return preprocessed


def get_tf_dataset(X, ext, preprocess=False, width=256, height=256):
    files = [x[0] for x in X]
    labels = [x[1] for x in X]
    load_img = load_png if ext == "png" else load_jpg
    img_ds = tf.data.Dataset.from_tensor_slices(files).map(
        load_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    if preprocess:
        p = partial(_preprocess, width=width, height=height)
        img_ds = img_ds.map(p, num_parallel_calls=tf.data.AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    out_ds = tf.data.Dataset.zip((img_ds, label_ds))
    return out_ds#.shuffle(len(X), reshuffle_each_iteration=True)


def describe_folder(
    path: str,
    prefix: str = "",
    ext: str = "png",
    class_folders: bool = True,
    file_label_map: dict = None,
):
    if prefix == "":  # train, test or val
        prefix = path.split(os.path.sep)[-1]
    elif path.split(os.path.sep)[-1] != prefix:
        path = os.path.join(path, prefix)

    path = os.path.abspath(path)  # convert to absolute path

    class_count = defaultdict(int)
    if class_folders:  # images in separate class folders e.g., UCMerced
        files = glob(os.path.join(path, "**", f"**.{ext}"), recursive=True)
        for f in files:
            label = f.split(os.path.sep)[-2]
            class_count[label] += 1
    else:  # all images in one folder
        assert (
            file_label_map is not None
        ), "Please provide a dictionary that maps files to labels"
        files = glob(os.path.join(path, f"*.{ext}"))
        for f in files:
            filename = f.split(os.path.sep)[-1]
            label = file_label_map[filename]
            class_count[label] += 1
    num_files: int = len(files)
    num_classes: int = len(class_count.keys())
    avg_class_example_count: int = np.mean(list(class_count.values()))

    print(
        f"{prefix}:\t"
        f"Found {num_files} images across {num_classes} classes."
        f" Each class has {avg_class_example_count:.0f} images on average."
    )

    return num_files, num_classes, avg_class_example_count, dict(class_count)


def train_test_split(ratio: Tuple, input_folder: str, output_folder: str, seed: int):
    splitfolders.ratio(input_folder, output_folder, seed=seed, ratio=ratio)
