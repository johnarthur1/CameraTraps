"""
Creates a classification dataset CSV with a corresponding JSON file determining
the train/val/test split.

This script takes as input a "json images file," whose keys are paths to images
and values are dictionaries containing information relevant for training
a classifier, including labels and (optionally) ground-truth bounding boxes.
The image paths are in the format `<dataset-name>/<blob-name>` where we assume
that the dataset name does not contain '/'.

{
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  # class from dataset
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],   # ground-truth bbox
        "label": ["monutain_lion"]  # labels to use in classifier
    },
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  # class from dataset
        "label": ["monutain_lion"]  # labels to use in classifier
    },
    ...
}

TODO: describe more

We assume that the tuple (dataset, location) identifies a unique location. In
other words, we assume that no two datasets have overlapping locations. This
probably isn't 100% true, but it's probably the best we can do in terms of
avoiding overlapping locations between the train/val/test splits.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from classification import detect_and_crop


def main(json_images_path: str,
         detector_version: str,
         detector_output_cache_base_dir: str,
         cropped_images_dir: str,
         confidence_threshold: float,
         csv_save_path: str,
         splits_json_save_path: str
         ) -> None:
    """
    Args:
        json_images_path: str, path to output of json_validator.py
        detector_version: str, detector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions,
            determines the subfolder of detector_output_cache_base_dir in
            which to find and save detector outputs
        detector_output_cache_base_dir: str, path to local directory
            where detector outputs are cached, 1 JSON file per dataset
        cropped_images_dir: str, path to local directory for saving crops of
            bounding boxes
        confidence_threshold: float, only crop bounding boxes above this value
        csv_save_path: str, path to save dataset csv
        splits_json_save_path: str, path to save splits JSON
    """
    result = create_crops_csv(json_images_path,
                              detector_version,
                              detector_output_cache_base_dir,
                              cropped_images_dir,
                              confidence_threshold,
                              csv_save_path)

    missing_detections, images_no_confident_detections, missing_crops = result
    print('Images missing detections:', len(missing_detections))
    print('Images without detections:', len(images_no_confident_detections))
    print('Missing crops:', len(missing_crops))

    crops_df = pd.read_csv(
        csv_save_path, index_col=False, float_precision='high')
    split_to_locs = create_splits(crops_df)
    with open(splits_json_save_path, 'w') as f:
        json.dump(split_to_locs, f)


def create_crops_csv(json_images_path: str,
                     detector_version: str,
                     detector_output_cache_base_dir: str,
                     cropped_images_dir: str,
                     confidence_threshold: float,
                     csv_save_path: str
                     ) -> Tuple[List[str], List[str], List[Tuple[str, int]]]:
    """Creates a classification dataset CSV.

    The classification dataset CSV contains the columns
    - path: str, <dataset>/<crop-filename>
    - dataset: str, name of camera trap dataset
    - location: str, location of image, provided by the camera trap dataset
    - dataset_class: image class, as provided by the camera trap dataset
    - confidence: float, confidence of bounding box, 1 if ground truth
    - label: str, comma-separated list of classification labels

    Args:
        json_images_path: str, path to output of json_validator.py
        detector_version: str, detector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions,
            determines the subfolder of detector_output_cache_base_dir in
            which to find and save detector outputs
        detector_output_cache_base_dir: str, path to local directory
            where detector outputs are cached, 1 JSON file per dataset
        cropped_images_dir: str, path to local directory for saving crops of
            bounding boxes
        confidence_threshold: float, only crop bounding boxes above this value
        csv_save_path: str, path to save dataset csv

    Returns:
        missing_detections: list of str, images without ground truth
            bboxes and not in detection cache
        images_no_confident_detections: list of str, images in detection cache
            with no bboxes above the confidence threshold
        images_missing_crop: list of tuple (img_path, i), where i is the i-th
            crop index
    """
    with open(json_images_path, 'r') as f:
        js = json.load(f)

    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')
    detection_cache = {}
    datasets = set(img_path[:img_path.find('/')] for img_path in js)
    print('loading detection cache...', end='')
    for ds in datasets:
        detection_cache[ds] = detect_and_crop.load_detection_cache(
            detector_output_cache_dir=detector_output_cache_dir, dataset=ds)
    print('done!')

    missing_detections = []  # no cached detections or ground truth bboxes
    images_no_confident_detections = []  # cached detections contain 0 bboxes
    images_missing_crop = []  # tuples: (img_path, crop_index)
    all_rows = []

    # True for ground truth, False for MegaDetector
    # always save as JPG for consistency
    crop_path_template = {
        True: '{img_path_root}_crop{n:>02d}.jpg',
        False: '{img_path_root}_mdv{v}_crop{n:>02d}.jpg'
    }

    for img_path, img_info in tqdm(js.items()):
        ds, img_file = img_path.split('/', maxsplit=1)

        # get bounding boxes
        if 'bbox' in img_info:  # ground-truth bounding boxes
            bbox_dicts = img_info['bbox']
            is_ground_truth = True
        else:  # get bounding boxes from detector cache
            if img_file in detection_cache[ds]:
                bbox_dicts = detection_cache[ds][img_file]['detections']
            else:
                missing_detections.append(img_path)
                continue
            is_ground_truth = False

        # check if crops are already downloaded, and ignore bboxes below the
        # confidence threshold
        rows = []
        for i, bbox_dict in enumerate(bbox_dicts):
            conf = 1.0 if is_ground_truth else bbox_dict['conf']
            if conf < confidence_threshold:
                continue
            img_path_root = os.path.splitext(img_path)[0]
            crop_path = crop_path_template[is_ground_truth].format(
                img_path_root=img_path_root, v=detector_version, n=i)
            full_crop_path = os.path.join(cropped_images_dir, crop_path)
            if not os.path.exists(full_crop_path):
                images_missing_crop.append((img_path, i))
            else:
                row = [crop_path, ds, img_info['location'], img_info['class'],
                       conf, ','.join(img_info['label'])]
                rows.append(row)
        if len(rows) == 0:
            images_no_confident_detections.append(img_path)
            continue
        all_rows.extend(rows)

    print('Saving classification dataset CSV...', end='')
    df = pd.DataFrame(
        data=all_rows,
        columns=['path', 'dataset', 'location', 'dataset_class', 'confidence',
                 'label'])
    df.to_csv(csv_save_path, index=False)
    print('done!')

    return (missing_detections,
            images_no_confident_detections,
            images_missing_crop)


def create_splits(df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
    """
    Args:
        df: pd.DataFrame, contains columns ['dataset', 'location', 'label']
            each row is a single image
            assumes each image is assigned exactly 1 label

    Returns: dict, keys are ['train', 'val', 'test'], values are lists of locs,
        where each loc is a tuple (dataset, location)
    """
    # merge dataset and location into a tuple (dataset, location)
    df['dataset_location'] = df[['dataset', 'location']].agg(tuple, axis=1)

    loc_to_label_sizes = df.groupby(['dataset_location', 'label']).size()

    seen_locs = set()
    split_to_locs = dict(train=[], val=[], test=[])
    label_sizes_by_split = {
        label: dict(train=0, val=0, test=0)
        for label in df['label'].unique()
    }

    def add_loc_to_split(loc: Tuple[str, str], split: str) -> None:
        split_to_locs[split].append(loc)
        for label, label_size in loc_to_label_sizes[loc].items():
            label_sizes_by_split[label][split] += label_size

    # sorted smallest to largest
    ordered_labels = df.groupby('label').size().sort_values()
    for label, label_size in tqdm(ordered_labels.items()):

        ordered_locs = (df[df['label'] == label]
                        .groupby('dataset_location')
                        .size()
                        .sort_values())
        for loc in ordered_locs.index:
            if loc in seen_locs:
                continue
            seen_locs.add(loc)

            # greedily add to test set until it has >= 15% of images
            if label_sizes_by_split[label]['test'] < 0.15 * label_size:
                add_loc_to_split(loc, 'test')
            elif label_sizes_by_split[label]['val'] < 0.15 * label_size:
                add_loc_to_split(loc, 'val')
            else:
                add_loc_to_split(loc, 'train')

    return split_to_locs


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Creates classification dataset.')
    parser.add_argument(
        'json_images',
        help='path to JSON file containing image paths and classification info')
    parser.add_argument(
        'cropped_images_dir',
        help='path to local directory for saving crops of bounding boxes')
    parser.add_argument(
        '-c', '--detector-output-cache-dir', required=True,
        help='path to directory where detector outputs are cached')
    parser.add_argument(
        '-v', '--detector-version', required=True,
        help='detector version string, e.g., "4.1"')
    parser.add_argument(
        '-t', '--confidence-threshold', type=float, default=0.8,
        help='confidence threshold above which to crop bounding boxes')
    parser.add_argument(
        '-d', '--csv-save-path', required=True,
        help='path to where dataset CSV should be saved')
    parser.add_argument(
        '-s', '--splits-json-save-path', required=True,
        help='path to where splits JSON file should be saved')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    assert 0 <= args.confidence_threshold <= 1
    main(json_images_path=args.json_images,
         detector_version=args.detector_version,
         detector_output_cache_base_dir=args.detector_output_cache_dir,
         cropped_images_dir=args.cropped_images_dir,
         confidence_threshold=args.confidence_threshold,
         csv_save_path=args.csv_save_path,
         splits_json_save_path=args.splits_json_save_path)
