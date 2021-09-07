import csv
import os
import shutil
import sys
from collections import defaultdict
from functools import partial
from glob import glob
from typing import List, Tuple


import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import sklearn.model_selection

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


def split_uniformly_across_classes(X, seed=123):
    labels = [x[1] for x in X]
    unique_labels_sorted = sorted(set(labels))
    class_distribution = [labels.count(i) for i in unique_labels_sorted]
    min_samples = min(zip(unique_labels_sorted, class_distribution), key=lambda x: x[1])
    print(
        f"Class with minimum samples is {min_samples[0]} with {min_samples[1]} samples"
    )
    print(f"Will produce uniform dataset such that each class has {min_samples[1]}.")
    print(
        f"This dataset has {len(unique_labels_sorted)} classes, "
        f"which will result in {len(unique_labels_sorted)*min_samples[1]} total samples."
    )

    new_X = []
    for label in unique_labels_sorted:
        X_subset = [x for x in X if x[1] == label]
        random_indices = np.random.default_rng(seed).integers(
            low=0, high=len(X_subset), size=min_samples[1]
        )
        new_X += [X_subset[i] for i in random_indices]
    return new_X


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


def get_tf_dataset(X, ext, preprocess=False, width=256, height=256, shuffle=True):
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
    if shuffle:
        return out_ds.shuffle(len(X), reshuffle_each_iteration=True)
    else:
        return out_ds


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
