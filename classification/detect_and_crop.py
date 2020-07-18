"""
Run MegaDetector on images, then save crops of the detected bounding boxes.

This script takes as input a "json images file," whose keys are paths to images
and values are dictionaries containing information relevant for training
a classifier, including labels and (optionally) ground-truth bounding boxes.

{
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  # class from dataset
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],
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

For an image with ground truth bounding boxes, the bounding boxes are cropped
and each saved to a file with the same name as the original image, except the
ending is changed from ".jpg" (for example) to "_cropXX.jpg", where "XX" ranges
from "00" to "99". We assume that no image contains over 100 bounding boxes and
that the ground truth bounding boxes are provided in a deterministic order. We
also assume that if an image has ground truth bounding boxes, then the ground
truth bounding boxes are exhaustive. In other words, there are no other bounding
boxes of interest for that image.

    Example cropped image path
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b_crop00.jpg"

For each image without any ground truth bounding boxes, we run MegaDetector
on the image. MegaDetector returns bounding boxes in deterministic order,  # TODO: verify the determinism
so we label the bounding boxes with confidence > 0.1 in order from 00 up to 99.
Based on the given confidence threshold, we may skip saving certain bounding box
crops, but we still increment the bounding box number for skipped boxes. We
change the file name ending from ".jpg" (for example) to "_mdvY.Y_cropXX.jpg"
where "Y.Y" indicates the MegaDetector version.

    Example cropped image path
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b_mdv4.1_crop00.jpg"

This script allows specifying a directory where MegaDetector outputs are cached
via the --megadetector-output-cache-dir argument. This directory must be
organized as
    <cache-dir>/<MegaDetector-version>/<dataset-name>.json

    Example: If the `cameratrapssc/classifier-training` Azure blob storage
    container is mounted to the local machine via blobfuse, it may be used as
    a MegaDetector output cache directory by passing
        "cameratrapssc/classifier-training/v4.1/caltech.json"
    as the value for --megadetector-output-cache-dir.

Note on image paths: the paths to images are in the format
    <dataset-name>/<blob-name>
where we assume that the dataset name does not contain '/'.
"""
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from azure.blob.storage import ContainerClient

from data_management.megadb.megadb_utils import MegadbUtils
from detection import run_tf_detector_batch
import sas_blob_utils  # from ai4eutils


def main(json_images_path: str,
         images_dir: str,
         megadetector_output_cache_dir: Optional[str] = None,
         megadetector_version: Optional[str] = None,
         local_megadetector: bool = False):
    with open(json_images_path, 'r') as f:
        js = json.load(f)
    images_to_detect = filter_detected_images(
        potential_images_to_detect=[k for k in js if 'bbox' not in js[k]],
        megadetector_output_cache_dir=megadetector_output_cache_dir,
        megadetector_version=megadetector_version)

    if local_megadetector:
        # download the necessary images locally
        # call run_tf_detector_batch()
        pass

    else:
        # convert local image paths to Azure blob URIs with SAS tokens
        image_sas_uris = get_image_sas_uris(images_to_detect)

        # call manage_api_submission.py to run batch processing API
        # wait for API to finish running
        # then
        

def filter_detected_images(potential_images_to_detect: Iterable[str],
                           megadetector_output_cache_dir: str,
                           megadetector_version: str) -> List[str]:
    """Checks image paths against cached MegaDetector outputs, and prepares
    the SAS URIs for each image not in the cache.

    Args:
        potential_images_to_detect: list of str, paths to images that do not
            have ground truth bounding boxes, each path has format
            <dataset-name>/<img-filename>, where <img-filename> is the blob name
        megadetector_output_cache_dir: str, path to folder where MegaDetector
            outputs are cached
        megadetector_version: str, MegaDetector version string, e.g., 'v4.1'

    Returns: list of str, image paths with SAS URLs ready to pass to the
        MegaDetector batch processing API
    """
    # cache of MegaDetector outputs: dataset name => set of detected img files
    cache: Dict[str, Set[str]] = {}

    images_to_detect = []
    for img_path in potential_images_to_detect:
        # img_path: <dataset-name>/<img-filename>
        dataset, img_file = img_path.split('/', maxsplit=1)

        if dataset not in cache:
            dataset_cache = os.path.join(
                megadetector_output_cache_dir, megadetector_version,
                f'{dataset}.json')
            if os.path.exists(dataset_cache):
                with open(dataset_cache, 'r') as f:
                    detected_images = json.load(f)['images']
                    cache[dataset] = {img['file'] for img in detected_images}
            else:
                cache[dataset] = set()

        if img_file not in cache[dataset]:
            images_to_detect.append(img_path)

    return images_to_detect


def get_image_sas_uris(local_images: Iterable[str]) -> List[str]:
    """Converts a list of local image paths to Azure Blob Storage blob URIs with
    SAS tokens.

    Args:
        local_images: list of str, <dataset-name>/<image-filename>

    Returns:
        image_sas_uris: list of str, blob URIs with SAS tokens
    """
    # we need the datasets table for getting SAS keys
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    image_sas_uris = []
    for img_path in local_images:
        dataset, img_file = img_path.split('/', maxsplit=1)

        image_sas_uri = sas_blob_utils.build_azure_storage_uri(
            account=datasets_table[dataset]['storage_account'],
            container=datasets_table[dataset]['container'],
            blob=img_file,
            sas_token=datasets_table[dataset]['container_sas_key'])
        image_sas_uris.append(image_sas_uri)
    return image_sas_uris


def download_and_crop(json_images: Mapping[str, Mapping[str, Any]]):
    # we need the datasets table for getting SAS keys
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    container_clients = {}  # Dict: (account, container_name) to ContainerClient
    for img_path in images_to_detect:
        account, container, blob = img_path.split('/', maxsplit=3)

        if (account, container) not in container_clients:
            pass
            # container_clients[(account, container)] = ContainerClient(
            #     account_url=f'https://{account}.blob.core.windows.net',
            #     container_name=container,
            #     credential=
            # )
        container_client = container_clients[(account, container)]


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Detects and crops images.')
    parser.add_argument(
        'json_images',
        help='path to JSON file containing image paths and classification info')
    parser.add_argument(
        'images_dir',
        help='path to local directory for saving crops of bounding boxes')
    parser.add_argument(
        '--megadetector-output-cache-dir', default=None,
        help='path to directory where MegaDetector outputs are cached')
    parser.add_argument(
        '--megadetector-version', default=None,
        help='MegaDetector version string including leading "v", e.g., "v4.1"')
    parser.add_argument(
        '--local-megadetector', action='store_true',
        help='run MegaDetector locally instead of calling the batch API')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(json_images_path=args.json_images,
         images_dir=args.images_dir,
         megadetector_output_cache_dir=args.megadetector_output_cache_dir,
         megadetector_version=args.megadetector_version,
         local_megadetector=args.local_megadetector)
