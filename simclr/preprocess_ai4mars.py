import argparse
import multiprocessing
import os
import shutil
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count
from pprint import pprint
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_DIR = "images/edr"
ROV_MASK_DIR = "images/mxy"
RNG_MASK_DIR = "images/rng-30m"

IMSIZE = 512
NUM_CPUS = 24


def preprocess_ai4mars_into_new_folder(msl_folder, output_dir=None, split="both"):
    (
        image_files,
        rover_mask_files,
        range_mask_files,
        label_files,
    ) = get_ai4mars_filepaths(msl_folder, select_files_with_labels=True, split=split)

    # # loop through files and preprocess
    get_output_dir = lambda files: os.path.commonpath(files).replace(
        msl_folder, output_dir
    )
    image_output_dir = get_output_dir(image_files)
    # # or: image_output_dir = os.path.commonpath(image_files).replace('edr', 'edr_preprocessed')
    # # or: image_output_dir = os.path.commonpath(image_files).replace('images', 'preprocessed_images')
    image_output_files = preprocess_image_files(
        image_files, image_output_dir, parallel=True
    )

    # mask_output_dir = get_output_dir(rover_mask_files)
    # mask_output_files = combine_preprocess_rover_range_masks(
    #     rover_mask_files, range_mask_files, mask_output_dir, parallel=True
    # )

    # TODO: combine masks and labels into new seg mask
    label_output_dir = get_output_dir(label_files)
    label_output_files = preprocess_labels_treat_masks_as_classes(
        label_files, rover_mask_files, range_mask_files, label_output_dir, parallel=True
    )

    # label_output_dir = get_output_dir(label_files)
    # label_output_files = preprocess_label_files(
    #     label_files, label_output_dir, parallel=False
    # )


def get_ai4mars_filepaths(msl_folder, select_files_with_labels=True, split="both"):
    join_msl = lambda path: os.path.join(msl_folder, path)
    images = sorted(glob(join_msl(f"{IMAGE_DIR}/*.JPG")))
    rover_masks = sorted(glob(join_msl(f"{ROV_MASK_DIR}/*.png")))
    range_masks = sorted(glob(join_msl(f"{RNG_MASK_DIR}/*.png")))

    if split == "both":
        labels = sorted(glob(join_msl("labels/**/*.png"), recursive=True))
    elif split == "train" or "test" in split:
        labels = sorted(glob(join_msl(f"labels/{split}/*.png")))
    else:
        raise NotImplementedError

    rov_mask_folder = os.path.commonpath(rover_masks) if len(rover_masks) > 0 else ""
    rng_mask_folder = os.path.commonpath(range_masks) if len(range_masks) > 0 else ""
    print(
        f"Found {len(images)} images in "
        f"{os.path.commonpath(images)}\n"
        f"Found {len(rover_masks)} rover mask files in "
        f"{rov_mask_folder}\n"
        f"Found {len(range_masks)} range mask files in "
        f"{rng_mask_folder}\n"
        f"Found {len(labels)} segmentation mask files in "
        f"{os.path.commonpath(labels)}\n"
    )

    if select_files_with_labels:
        images, rover_masks, range_masks = filter_files_with_labels_only(
            images, rover_masks, range_masks, labels
        )
    return images, rover_masks, range_masks, labels


def filter_files_with_labels_only(images, rover_masks, range_masks, labels):
    print(f"Filtering for files with corresponding segmentation masks only.")
    msl_folder = os.path.commonpath(images + rover_masks + range_masks + labels)
    join_msl = lambda path: os.path.join(msl_folder, path)

    image_subset = []
    rover_mask_subset, images_without_rover_mask = [], []
    range_mask_subset, images_without_range_mask = [], []
    for label in tqdm(labels, desc="Filtering files", unit="paths"):
        label_name = os.path.basename(label)
        label_name, label_ext = os.path.splitext(label_name)
        if "_merged" in label_name:
            #     # there are multiple versions of the same label, so don't repeat
            #     if (
            #         join_msl(f"{IMAGE_DIR}/{label_name.replace('_merged', '')}.JPG")
            #         in image_subset
            #     ):
            #         continue
            # else:
            label_name = label_name.replace("_merged", "")
        try:
            image_file = join_msl(f"{IMAGE_DIR}/{label_name}.JPG")
            image = images[images.index(image_file)]
            image_subset.append(image)
        except:
            raise FileNotFoundError(f"Missing corresponding image for {label}")
        try:
            rov_mask_file = join_msl(
                f"{ROV_MASK_DIR}/{label_name.replace('EDR', 'MXY')}.png"
            )
            rover_mask = rover_masks[rover_masks.index(rov_mask_file)]
            rover_mask_subset.append(rover_mask)
        except:
            # rover_mask_subset.append("")
            images_without_rover_mask.append(image)
        try:
            range_file = join_msl(
                f"{RNG_MASK_DIR}/{label_name.replace('EDR', 'RNG')}.png"
            )
            range_mask = range_masks[range_masks.index(range_file)]
            range_mask_subset.append(range_mask)
        except:
            # range_mask_subset.append("")
            images_without_range_mask.append(image)
    images = sorted(set(image_subset))
    rover_masks = sorted(set(rover_mask_subset))
    range_masks = sorted(set(range_mask_subset))
    print(
        f"{len(images_without_rover_mask)} image(s) without rover mask\n"
        f"{len(images_without_range_mask)} image(s) without range_mask"
    )

    if 0 < len(images_without_rover_mask) < 10:
        print("\nImages without rover mask:")
        pprint(images_without_rover_mask, compact=True)

    if 0 < len(images_without_range_mask) < 10:
        print("\nImages without range mask:")
        pprint(images_without_range_mask, compact=True)

    print(
        f"\nRemaining files:\n"
        f"{len(images)} images\n"
        f"{len(rover_masks)} rover masks\n"
        f"{len(range_masks)} range masks\n"
        f"{len(labels)} segmentation masks"
    )

    print(
        f"INFO: there are potentially multiple versions of the same label "
        f"(see info.txt); we will remove duplicate corresponding image files"
    )

    return images, rover_masks, range_masks


def preprocess_image_files(image_files, out_dir, parallel=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"Preprocessing {len(image_files)} images into {out_dir}")

    if parallel:
        pool = multiprocessing.Pool(processes=NUM_CPUS)
        out_files = list(
            tqdm(
                pool.imap_unordered(
                    partial(_preprocess_helper, out_dir=out_dir), image_files
                ),
                total=len(image_files),
                desc="Image preprocessing (parallel)",
                unit="images",
            )
        )
    else:
        out_files = [
            _preprocess_helper(file, out_dir)
            for file in tqdm(image_files, desc="Image preprocessing", unit="images")
        ]
    return out_files


def _preprocess_helper(file, out_dir):
    img = load_to_array(file, imsize=IMSIZE)
    img = preprocess(img)
    out_file = save_array_to_file(img, out_dir, file)
    return out_file


def preprocess(image_array):
    image_array = normalize(image_array)
    image_array = brighten(image_array, brightness=1.5)
    return image_array


def normalize(img):
    """MinMax Normalize from 0-255 to 0-1 with 0.5 mean?"""
    img = img.astype(np.float32)
    min_ = img.min()
    img = (img - min_) * (1.0 / (img.max() - min_))
    return img


def brighten(img, brightness=1.0, b=0.0, clip=True):
    # img = tf.keras.preprocessing.image.apply_brightness_shift(img, brightness)
    img = brightness * img + b
    if clip:
        return np.clip(img, 0.0, 1.0)
    else:
        return img


def preprocess_labels_treat_masks_as_classes(
    label_files, rover_files, range_files, out_dir, parallel=False
):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    rover_dir = os.path.commonpath(rover_files)
    range_dir = os.path.commonpath(range_files)

    if parallel:
        f = partial(
            _load_labels_combine_masks_and_save,
            rover_dir=rover_dir,
            range_dir=range_dir,
            out_dir=out_dir,
        )
        pool = multiprocessing.Pool(processes=NUM_CPUS)
        out_files = list(
            tqdm(
                pool.imap_unordered(f, label_files),
                total=len(label_files),
                desc="Preprocess labels and masks (parallel)",
                unit="images",
            )
        )
    else:
        out_files = []
        for file in tqdm(label_files, desc="Preprocess labels and masks", unit="files"):
            out_file = _load_labels_combine_masks_and_save(
                file, rover_dir, range_dir, out_dir
            )
            out_files.append(out_file)
    return out_files


def _load_labels_combine_masks_and_save(file, rover_dir, range_dir, out_dir):
    out_dir = handle_label_subfolders(file, out_dir)

    label_name = os.path.basename(file).replace("_merged", "")
    rover_file = os.path.join(rover_dir, label_name.replace("EDR", "MXY"))
    range_file = os.path.join(range_dir, label_name.replace("EDR", "RNG"))

    seg_mask = preprocess_labels(load_to_array(file, imsize=IMSIZE))
    rover_mask = load_mask_if_exists(rover_file)
    range_mask = load_mask_if_exists(range_file)

    seg_mask[np.where(rover_mask > 0)] = 5
    seg_mask[np.where(range_mask > 0)] = 6

    out_file = save_array_to_file(seg_mask, out_dir, file)
    return out_file


def combine_preprocess_rover_range_masks(
    rover_files, range_files, out_dir, parallel=False
):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    rover_dir = os.path.commonpath(rover_files)
    range_dir = os.path.commonpath(range_files)

    name_set = get_union_of_mask_filenames(rover_files, range_files)

    # for each unique name, check whether rover/range mask exists, load mask(s), combine, and save
    print(f"\nPreprocessing {len(name_set)} masks into {out_dir}")
    if parallel:
        pool = multiprocessing.Pool(processes=NUM_CPUS)
        out_files = list(
            tqdm(
                pool.imap_unordered(
                    partial(
                        _load_preprocess_save_masks,
                        out_dir=out_dir,
                        rover_dir=rover_dir,
                        range_dir=range_dir,
                    ),
                    name_set,
                ),
                total=len(name_set),
                desc="Combine masks (parallel)",
                unit="images",
            )
        )
    else:
        out_files = []
        for name in tqdm(name_set, desc="Combine masks", unit="files"):
            out_file = _load_preprocess_save_masks(name, out_dir, rover_dir, range_dir)
            out_files.append(out_file)
        return out_files


def get_union_of_mask_filenames(rover_mask_files, range_mask_files):
    """Combine rover and range mask names (take union) into a unique set"""
    rov_func = lambda file: os.path.basename(file).replace("MXY", "TEMP")
    rng_func = lambda file: os.path.basename(file).replace("RNG", "TEMP")
    rover_mask_name = list(map(rov_func, rover_mask_files))
    range_mask_name = list(map(rng_func, range_mask_files))
    name_set = sorted(set(rover_mask_name + range_mask_name))
    return name_set


def _load_preprocess_save_masks(file, out_dir, rover_dir, range_dir):
    rover_file = os.path.join(rover_dir, file.replace("TEMP", "MXY"))
    range_file = os.path.join(range_dir, file.replace("TEMP", "RNG"))
    rover_mask = load_mask_if_exists(rover_file)
    range_mask = load_mask_if_exists(range_file)

    # assert (
    #     range_file_exists or rover_file_exists
    # ), f"Neither range nor rover masks exist for {file}"

    image_mask = np.clip(rover_mask + range_mask, 0, 1).astype(np.uint8)
    mask_file = os.path.join(out_dir, file.replace("TEMP", "MXY"))
    out_file = save_array_to_file(image_mask, out_dir, mask_file)
    return out_file


def load_mask_if_exists(file):
    file_exists = os.path.isfile(file)
    if file_exists:
        return load_to_array(file, imsize=IMSIZE)
    else:
        return np.zeros((IMSIZE, IMSIZE, 3), dtype=np.uint8)


def preprocess_label_files(label_files, out_dir, parallel=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    print(f"\nPreprocessing {len(label_files)} labels into {out_dir}")

    if parallel:
        pool = multiprocessing.Pool(processes=NUM_CPUS)
        out_files = list(
            tqdm(
                pool.imap_unordered(
                    partial(_preprocess_label_helper, out_dir=out_dir), label_files
                ),
                total=len(label_files),
                desc="Preprocess labels (parallel)",
                unit="images",
            )
        )
    else:
        out_files = []
        for file in tqdm(label_files, desc="Preprocess labels", unit="files"):
            out_file = _preprocess_label_helper(file, out_dir)
            out_files.append(out_file)

    return out_files


def _preprocess_label_helper(file, out_dir):
    out_dir = handle_label_subfolders(file, out_dir)
    img_original = load_to_array(file, imsize=None)
    img = load_to_array(file, imsize=IMSIZE)
    try:
        assert len(np.unique(img_original)) == len(
            np.unique(img)
        ), "Some classes have been removed"
    except AssertionError:
        print(
            f"Original: {np.unique(img_original)} | "
            f"Resized: {np.unique(img)} | "
            f"Preprocessed: {np.unique(preprocess_labels(img))}"
        )
        pass
    img = preprocess_labels(img)
    out_file = save_array_to_file(img, out_dir, file)
    return out_file


def handle_label_subfolders(label_file, label_dir):
    # example label_dir: /home/goh/Documents/ai4mars-dataset-merged-0.1/msl_preprocessed/labels
    # add split/label_type to labels folder
    label_dir = os.path.join(label_dir, *label_file.split(os.path.sep)[-3:-1])
    # remove repeated folders:
    label_dir = remove_repeated_folders(label_dir)

    if not os.path.isdir(label_dir):
        os.makedirs(label_dir, exist_ok=True)
    return label_dir


def remove_repeated_folders(out_dir):
    components = out_dir.split(os.path.sep)
    out = [os.path.sep]  # start with "/" for absolute path
    for folder in components:
        if folder not in out and folder != "":
            out.append(folder)
    out_dir = os.path.join(*out)
    return os.path.abspath(out_dir)


def load_to_array(filepath, imsize=None):
    image = Image.open(filepath).convert("RGB")
    if imsize is not None:
        _, ext = os.path.splitext(os.path.basename(filepath))
        resample = Image.NEAREST if "png" in ext.lower() else Image.BILINEAR
        image = image.resize((imsize, imsize), resample=resample)
    return np.asarray(image)


def preprocess_labels(array):
    array[array == 255] = 4
    return array


def save_array_to_file(array, out_dir, file):
    out_file = os.path.join(out_dir, os.path.basename(file))
    if array.dtype == np.float32:
        array = (array * 255).astype(np.uint8)
    im = Image.fromarray(array)
    im.save(out_file)
    return out_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--msl_folder",
        default="/home/goh/Documents/D3M/ai4mars-dataset-merged-0.1/msl",
    )
    parser.add_argument("-o", "--output_folder", default="")
    args = parser.parse_args()

    msl_folder = args.msl_folder
    out_dir = args.output_folder

    assert os.path.isdir(
        msl_folder
    ), f"Provided MSL folder: {msl_folder} does not exist. Please check."

    if out_dir == "":
        msl_name = os.path.basename(msl_folder)
        out_dir = msl_folder.replace(msl_name, f"{msl_name}_preprocessed_512px_debug")

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=False)
        print(f"Writing preprocessed files to: {out_dir}")
    else:
        print(
            f"WARNING: Provided out_dir {out_dir} already exists. "
            f"Previously preprocessed files will be overwritten."
        )
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)

    preprocess_ai4mars_into_new_folder(msl_folder, output_dir=out_dir, split="both")
