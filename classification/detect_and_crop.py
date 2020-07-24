"""
Run MegaDetector on images, then save crops of the detected bounding boxes.

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
truth bounding boxes are exhaustive--i.e., there are no other bounding boxes of
interest for that image.

    Example cropped image path
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b_crop00.jpg"

For each image without any ground truth bounding boxes, we run MegaDetector
on the image. MegaDetector returns bounding boxes in deterministic order,
so we label the bounding boxes in order from 00 up to 99.
Based on the given confidence threshold, we may skip saving certain bounding box
crops, but we still increment the bounding box number for skipped boxes. We
change the file name ending from ".jpg" (for example) to "_mdvY.Y_cropXX.jpg"
where "Y.Y" indicates the MegaDetector version.

    Example cropped image path
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b_mdv4.1_crop00.jpg"

MegaDetector can be run locally or via the Batch Detection API. If running
through the Batch Detection API, set the following environment variables for
the Azure Blob Storage container which will save the intermediate task lists.

    BATCH_DETECTION_API_URL                  # API URL
    CLASSIFICATION_BLOB_STORAGE_ACCOUNT      # storage account name
    CLASSIFICATION_BLOB_CONTAINER            # container name
    CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS  # SAS token, without leading '?'
    DETECTION_API_CALLER                     # allow-listed API caller

This script allows specifying a directory where MegaDetector outputs are cached
via the --detector-output-cache-dir argument. This directory must be
organized as
    <cache-dir>/<MegaDetector-version>/<dataset-name>.json

    Example: If the `cameratrapssc/classifier-training` Azure blob storage
    container is mounted to the local machine via blobfuse, it may be used as
    a MegaDetector output cache directory by passing
        "cameratrapssc/classifier-training/mdv4.1_cache/caltech.json"
    as the value for --detector-output-cache-dir.
"""
import argparse
from concurrent import futures
from datetime import datetime
import io
import json
import os
import time
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple)

from azure.storage.blob import download_blob_from_url
from PIL import Image
import requests
from tqdm import tqdm

from api.batch_processing.data_preparation.prepare_api_submission import (
    Task, TaskStatus, divide_list_into_tasks)
from api.batch_processing.postprocessing.combine_api_outputs import (
    combine_api_output_dictionaries)
from data_management.megadb.megadb_utils import MegadbUtils
# from detection import run_tf_detector_batch
import sas_blob_utils  # from ai4eutils


def main(json_images_path: str,
         detector_version: str,
         detector_output_cache_base_dir: str,
         cropped_images_dir: str,
         confidence_threshold: float = 0,
         local_detector: bool = False,
         images_dir: Optional[str] = None,
         threads: int = 1,
         resume_file_path: Optional[str] = None) -> None:
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
        local_detector: bool, whether to run detection locally or through
            the Batch Detection API
        images_dir: optional str, path to local directory where images are saved
        threads: int, number of threads to use for downloading images
        resume_file_path: optional str, path to save JSON file with list of info
            dicts on running tasks, or to resume from running tasks, only used
            if local_detector is False
    """
    with open(json_images_path, 'r') as f:
        js = json.load(f)
    detector_output_cache_dir = os.path.join(
        detector_output_cache_base_dir, f'v{detector_version}')
    if not os.path.exists(detector_output_cache_dir):
        os.makedirs(detector_output_cache_dir)
        print(f'Created directory at {detector_output_cache_dir}')
    images_to_detect = filter_detected_images(
        potential_images_to_detect=[k for k in js if 'bbox' not in js[k]],
        detector_output_cache_dir=detector_output_cache_dir)

    if len(images_to_detect) > 0:

        if local_detector:
            # download the necessary images locally
            # run_detection(images_to_detect, detector_version)
            # call run_tf_detector_batch()
            raise NotImplementedError

        else:
            assert resume_file_path is not None
            account = os.environ['CLASSIFICATION_BLOB_STORAGE_ACCOUNT']
            container = os.environ['CLASSIFICATION_BLOB_CONTAINER']
            sas_token = os.environ['CLASSIFICATION_BLOB_CONTAINER_WRITE_SAS']
            caller = os.environ['DETECTION_API_CALLER']
            batch_detection_api_url = os.environ['BATCH_DETECTION_API_URL']

            if os.path.exists(resume_file_path):
                tasks_by_dataset = resume_tasks(
                    resume_file_path,
                    batch_detection_api_url=batch_detection_api_url)
            else:
                tasks_by_dataset = submit_batch_detection_api(
                    images_to_detect=images_to_detect,
                    task_lists_dir=os.path.dirname(json_images_path),
                    detector_version=detector_version,
                    account=account, container=container, sas_token=sas_token,
                    caller=caller,
                    batch_detection_api_url=batch_detection_api_url,
                    resume_file_path=resume_file_path)
            wait_for_tasks(tasks_by_dataset, detector_output_cache_dir)

        # verify that there are no more images left to detect
        images_to_detect = filter_detected_images(
            potential_images_to_detect=[k for k in js if 'bbox' not in js[k]],
            detector_output_cache_dir=detector_output_cache_dir)
        assert len(images_to_detect) == 0

    images_missing_detections, images_failed_download = download_and_crop(
        json_images=js,
        detector_output_cache_dir=detector_output_cache_dir,
        detector_version=detector_version,
        cropped_images_dir=cropped_images_dir,
        confidence_threshold=confidence_threshold,
        images_dir=images_dir,
        threads=threads)
    print('Images with missing detections:', images_missing_detections)
    print('Images that failed to download:', images_failed_download)


def load_detection_cache(detector_output_cache_dir: str,
                         dataset: str) -> Dict[str, Dict[str, Any]]:
    """Loads detection cache for a given dataset. Returns an empty dictionary
    if the cache does not exist.

    Args:
        detector_output_cache_dir: str, path to folder where detector
            outputs are cached, stored as 1 JSON file per dataset
        cropp
            outputs are cached, stored as 1 JSON file per dataset
        dataset: str, name of dataset

    Returns: dict, maps str image path to dict of detection info
    """
    cache_path = os.path.join(detector_output_cache_dir, f'{dataset}.json')
    dataset_cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            detected_images = json.load(f)['images']
        dataset_cache = {img['file']: img for img in detected_images}
    return dataset_cache


def filter_detected_images(potential_images_to_detect: Iterable[str],
                           detector_output_cache_dir: str
                           ) -> List[str]:
    """Checks image paths against cached Detector outputs, and prepares
    the SAS URIs for each image not in the cache.

    Args:
        potential_images_to_detect: list of str, paths to images that do not
            have ground truth bounding boxes, each path has format
            <dataset-name>/<img-filename>, where <img-filename> is the blob name
        detector_output_cache_dir: str, path to folder where detector
            outputs are cached, stored as 1 JSON file per dataset
        cropp
            outputs are cached

    Returns: list of str, paths to images not in the detector output cache,
        with the format <dataset-name>/<img-filename>
    """
    # cache of Detector outputs: dataset name => {img_path => detection_dict}
    cache: Dict[str, Dict[str, Dict]] = {}

    images_to_detect = []
    for img_path in potential_images_to_detect:
        # img_path: <dataset-name>/<img-filename>
        ds, img_file = img_path.split('/', maxsplit=1)

        if ds not in cache:
            cache[ds] = load_detection_cache(detector_output_cache_dir, ds)

        if img_file not in cache[ds]:
            images_to_detect.append(img_path)

    return images_to_detect


def get_image_sas_uris(local_images: Iterable[str]) -> List[str]:
    """Converts a list of local image paths to Azure Blob Storage blob URIs with
    SAS tokens.

    Args:
        local_images: list of str, <dataset-name>/<image-filename>

    Returns:
        image_sas_uris: list of str, image blob URIs with SAS tokens, ready to
            pass to the batch detection API
    """
    # we need the datasets table for getting SAS keys
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    image_sas_uris = []
    for img_path in local_images:
        dataset, img_file = img_path.split('/', maxsplit=1)

        # strip leading '?' from SAS token
        sas_token = datasets_table[dataset]['container_sas_key']
        if sas_token[0] == '?':
            sas_token = sas_token[1:]

        image_sas_uri = sas_blob_utils.build_azure_storage_uri(
            account=datasets_table[dataset]['storage_account'],
            container=datasets_table[dataset]['container'],
            blob=img_file,
            sas_token=sas_token)
        image_sas_uris.append(image_sas_uri)
    return image_sas_uris


def split_images_list_by_dataset(images_to_detect: Iterable[str]
                                 ) -> Dict[str, List[str]]:
    """
    Args:
        images_to_detect: list of str, image paths with the format
            <dataset-name>/<image-filename>

    Returns: dict, maps dataset name to a list of image paths
    """
    images_by_dataset: Dict[str, List[str]] = {}
    for img_path in images_to_detect:
        dataset = img_path[:img_path.find('/')]
        if dataset not in images_by_dataset:
            images_by_dataset[dataset] = []
        images_by_dataset[dataset].append(img_path)
    return images_by_dataset


def submit_batch_detection_api(images_to_detect: Iterable[str],
                               task_lists_dir: str,
                               detector_version: str,
                               account: str,
                               container: str,
                               sas_token: str,
                               caller: str,
                               batch_detection_api_url: str,
                               resume_file_path: str
                               ) -> Dict[str, List[Task]]:
    """
    Args:
        images_to_detect: list of str, list of str, image paths with the format
            <dataset-name>/<image-filename>
        task_lists_dir: str, path to local directory for saving JSON files
            each containing a list of image URLs corresponding to an API task
        detector_version: str, MegaDetector version string, e.g., '4.1',
            see {batch_detection_api_url}/supported_model_versions
        account: str, Azure Storage account name
        container: str, Azure Blob Storage container name, where the task lists
            will be uploaded
        sas_token: str, SAS token with write permissions for the container
        caller: str, allow-listed caller
        batch_detection_api_url: str, URL to batch detection API
        resume_file_path: str, path to save resume file

    Returns: dict, maps str dataset name to list of Task objects
    """
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    images_by_dataset = split_images_list_by_dataset(images_to_detect)
    tasks_by_dataset = {}
    for dataset, image_paths in images_by_dataset.items():
        # get SAS URL for images container
        images_sas_token = datasets_table[dataset]['container_sas_key']
        if images_sas_token[0] == '?':
            images_sas_token = images_sas_token[1:]
        images_container_url = sas_blob_utils.build_azure_storage_uri(
            account=datasets_table[dataset]['storage_account'],
            container=datasets_table[dataset]['container'],
            sas_token=images_sas_token)

        # strip image paths of dataset name
        image_paths = [path.split('/', maxsplit=1)[1] for path in image_paths]

        tasks_by_dataset[dataset] = submit_batch_detection_api_by_dataset(
            dataset=dataset,
            image_paths=image_paths,
            images_container_url=images_container_url,
            task_lists_dir=task_lists_dir,
            detector_version=detector_version,
            account=account, container=container, sas_token=sas_token,
            caller=caller, batch_detection_api_url=batch_detection_api_url)

    # save list of dataset names and task IDs for resuming
    resume_json = [
        {
            'dataset': dataset,
            'task_name': task.name,
            'task_id': task.id,
            'local_images_list_path': task.local_images_list_path
        }
        for dataset in tasks_by_dataset
        for task in tasks_by_dataset[dataset]
    ]
    with open(resume_file_path, 'w') as f:
        json.dump(resume_json, f, indent=1)
    return tasks_by_dataset


def submit_batch_detection_api_by_dataset(
        dataset: str,
        image_paths: Sequence[str],
        images_container_url: str,
        task_lists_dir: str,
        detector_version: str,
        account: str,
        container: str,
        sas_token: str,
        caller: str,
        batch_detection_api_url: str
        ) -> List[Task]:
    """
    Args:
        dataset: str, name of dataset
        image_paths: list of str, image paths all from the same dataset
        images_container_url: str, URL to blob storage container where images
            are stored, including SAS token with read permissions if container
            is not public
        **see submit_batch_detection_api() for description of other args

    Returns: list of Task objects
    """
    os.makedirs(task_lists_dir, exist_ok=True)

    date = datetime.now().strftime('%Y%m%d_%H%M%S')  # e.g., '20200722_110816'
    task_list_base_filename = f'task_list_{dataset}_{date}.json'

    task_list_paths, _ = divide_list_into_tasks(
        file_list=image_paths,
        save_path=os.path.join(task_lists_dir, task_list_base_filename))

    # complete task name: 'detect_for_classifier_caltech_20200722_110816_task01'
    task_name_template = 'detect_for_classifier_{dataset}_{date}_task{n:>02d}'
    tasks: List[Task] = []
    for i, task_list_path in enumerate(task_list_paths):
        task = Task(
            name=task_name_template.format(dataset=dataset, date=date, n=i),
            images_list_path=task_list_path, api_url=batch_detection_api_url)
        task.upload_images_list(
            account=account, container=container, sas_token=sas_token)
        task.generate_api_request(
            caller=caller,
            input_container_url=images_container_url,
            model_version=detector_version)
        print(f'Submitting task for: {task_list_path}')
        task.submit()
        print(f'- task ID: {task.id}')
        tasks.append(task)
    return tasks


def resume_tasks(resume_file_path: str, batch_detection_api_url: str
                 ) -> Dict[str, List[Task]]:
    """
    Args:
        resume_file_path: str, path to resume file with list of info dicts on
            running tasks
        batch_detection_api_url: str, URL to batch detection API

    Returns: dict, maps str dataset name to list of Task objects
    """
    with open(resume_file_path, 'r') as f:
        resume_json = json.load(f)

    tasks_by_dataset: Dict[str, List[Task]] = {}
    for info_dict in resume_json:
        dataset = info_dict['dataset']
        if dataset not in tasks_by_dataset:
            tasks_by_dataset[dataset] = []
        task = Task(name=info_dict['task_name'],
                    task_id=info_dict['task_id'],
                    images_list_path=info_dict['local_images_list_path'],
                    validate=False,
                    api_url=batch_detection_api_url)
        tasks_by_dataset[dataset].append(task)
    return tasks_by_dataset


def wait_for_tasks(tasks_by_dataset: Mapping[str, Iterable[Task]],
                   detector_output_cache_dir: str,
                   poll_interval: int = 60) -> None:
    """Waits for the Batch Detection API tasks to finish running.

    For jobs that finish successfully, merges the output with cached detector
    outputs.

    Args:
        tasks_by_dataset: dict, maps str dataset name to list of Task objects
    """
    remaining_tasks: List[Tuple[str, Task]] = [
        (dataset, task) for dataset, tasks in tasks_by_dataset.items()
        for task in tasks]

    progbar = tqdm(total=len(remaining_tasks))
    while True:
        new_remaining_tasks = []
        for dataset, task in remaining_tasks:
            task.check_status()

            # task still running => continue
            if task.status == TaskStatus.RUNNING:
                new_remaining_tasks.append((dataset, task))
                continue

            progbar.update(1)
            progbar.write(f'Task {task.id} stopped with status {task.status}')

            if task.status in [TaskStatus.PROBLEM, TaskStatus.FAILED]:
                progbar.write('API response:')
                progbar.write(task.response)
                continue

            # task finished successfully
            assert task.status == TaskStatus.COMPLETED
            message = task.response['Status']['message']
            num_failed_shards = message['num_failed_shards']
            if num_failed_shards != 0:
                progbar.write(f'Task {task.id} completed with '
                              f'{num_failed_shards} failed shards.')

            detections_url = message['output_file_urls']['detections']
            assert task.id in detections_url
            detections = requests.get(detections_url).json()
            msg = cache_detections(
                detections=detections, dataset=dataset,
                detector_output_cache_dir=detector_output_cache_dir)
            progbar.write(msg)

        remaining_tasks = new_remaining_tasks
        if len(remaining_tasks) == 0:
            break
        progbar.write(f'Sleeping for {poll_interval} seconds...')
        time.sleep(poll_interval)

    progbar.close()


def cache_detections(detections: Mapping[str, Any],
                     dataset: str,
                     detector_output_cache_dir: str) -> str:
    """
    Args:
        detections: dict, represents JSON output of detector,
            see https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
        dataset: str, name of dataset
        detector_output_cache_dir: str, path to folder where detector
            outputs are cached, stored as 1 JSON file per dataset
        cropp
            cached, directory must already exist

    Returns: str, message
    """
    # combine detections with cache
    dataset_cache_path = os.path.join(
        detector_output_cache_dir, f'{dataset}.json')
    if os.path.exists(dataset_cache_path):
        with open(dataset_cache_path, 'r') as f:
            dataset_cache = json.load(f)
        merged_dataset_cache = combine_api_output_dictionaries(
            input_dicts=[dataset_cache, detections],
            require_uniqueness=False)
        msg = f'Merging detection output with {dataset_cache_path}'
    else:
        merged_dataset_cache = detections
        msg = ('No cached detection outputs found. Saving detection output to '
               f'{dataset_cache_path}')

    # write combined detections back out to cache
    with open(dataset_cache_path, 'w') as f:
        json.dump(merged_dataset_cache, f, indent=1)
    return msg


def download_and_crop(json_images: Mapping[str, Mapping[str, Any]],
                      detector_output_cache_dir: str,
                      detector_version: str,
                      cropped_images_dir: str,
                      confidence_threshold: float,
                      images_dir: Optional[str] = None,
                      threads: int = 1
                      ) -> Tuple[List[str], List[str]]:
    """

    Saves crops to a file with the same name as the original image, except the
    ending is changed from ".jpg" (for example) to:
    - if image has ground truth bboxes: "_cropXX.jpg", where "XX" indicates the
        bounding box index
    - if image has bboxes from MegaDetector: "_mdvY.Y_cropXX.jpg", where
        "Y.Y" indicates the MegaDetector version
    See module docstring for more info and examples.

    Args:
        json_images: dict, represents JSON output of json_validator.py
        detector_output_cache_dir: str, path to folder where detector
            outputs are cached, stored as 1 JSON file per dataset
        detector_version: str, detector version string, e.g., '4.1'
        cropped_images_dir: str, path to folder where cropped images are saved
        confidence_threshold: float, only crop bounding boxes above this value
        images_dir: optional str, path to folder where full images are saved
        threads: int, number of threads to use for downloading images

    Returns:
        images_missing_detections: list of str, image files without ground truth
            or cached detected bounding boxes
        images_failed_download: list of str, images with bounding boxes that
            failed to download
    """
    # we need the datasets table for getting SAS keys
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    # detection_cache_by_dataset: dataset => {img_path => detection_dict}
    detection_cache_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}

    images_missing_detections = []
    images_failed_download = []

    # True for ground truth, False for MegaDetector
    # always save as JPG for consistency
    crop_path_template = {
        True: os.path.join(cropped_images_dir,
                           '{img_file_root}_crop{n:>02d}.jpg'),
        False: os.path.join(cropped_images_dir,
                            '{img_file_root}_mdv{v}_crop{n:>02d}.jpg')
    }

    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_file = {}

    print('Getting bbox info for each image...')
    for img_file, info_dict in tqdm(json_images.items()):
        ds, blob_name = img_file.split('/', maxsplit=1)
        assert ds == info_dict['dataset']

        # get bounding boxes
        if 'bbox' in info_dict:  # ground-truth bounding boxes
            bbox_dicts = info_dict['bbox']
            is_bbox_ground_truth = True
        else:  # get bounding boxes from detector cache
            if ds not in detection_cache_by_dataset:
                detection_cache_by_dataset[ds] = load_detection_cache(
                    detector_output_cache_dir, ds)
            detection_cache = detection_cache_by_dataset[ds]
            if blob_name in detection_cache:
                bbox_dicts = detection_cache[blob_name]['detections']
            else:
                images_missing_detections.append(img_file)
                continue
            is_bbox_ground_truth = False

        # check if crops are already downloaded, and ignore bboxes below the
        # confidence threshold
        # crop_path => bbox_dict
        bboxes_tocrop: Dict[str, Dict[str, Any]] = {}
        for i, bbox_dict in enumerate(bbox_dicts):
            if 'conf' in bbox_dict and bbox_dict['conf'] < confidence_threshold:
                continue
            img_file_root = os.path.splitext(img_file)[0]
            crop_path = crop_path_template[is_bbox_ground_truth].format(
                img_file_root=img_file_root, v=detector_version, n=i)
            if not os.path.exists(crop_path):
                bboxes_tocrop[crop_path] = bbox_dict['bbox']
        if len(bboxes_tocrop) == 0:
            continue

        # get the image, either from disk or from Blob Storage
        future = pool.submit(load_and_crop, images_dir, img_file,
                             datasets_table[ds], bboxes_tocrop)
        future_to_img_file[future] = img_file

    print('Loading/downloading images and cropping...')
    n_futures = len(future_to_img_file)
    for future in tqdm(futures.as_completed(future_to_img_file), len=n_futures):
        img_file = future_to_img_file[future]
        try:
            future.result()
        except Exception as e:  # pylint: disable=broad-except
            tqdm.write(f'{img_file} - generated an exception: {e}')
            images_failed_download.append(img_file)
        else:
            tqdm.write(f'{img_file} - successfully loaded and cropped')

    pool.shutdown()
    return images_missing_detections, images_failed_download


def load_and_crop(images_dir: Optional[str], img_file: str,
                  dataset_info: Mapping[str, Any],
                  bboxes_tocrop: Mapping[str, Sequence[float]]):
    """Loads an image from disk or Azure Blob Storage, then crops it.

    Args:
        images_dir: optional str, path to local directory of images
        img_file: str, image file path with format `<dataset-name>/<blob-name>`
        dataset_info: dict, info about dataset from MegaDB
        bboxes_tocrop: dict, maps crop file name to [xmin, ymin, width, height]
            all in normalized coordinates
    """
    img = None
    if images_dir is not None:
        img_path = os.path.join(images_dir, img_file)
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img.load()
    if img is None:
        # download image from Blob Storage
        blob_url = sas_blob_utils.build_azure_storage_uri(
            account=dataset_info['storage_account'],
            container=dataset_info['container'],
            blob=img_file[img_file.find('/') + 1:])
        credential = dataset_info['container_sas_key']
        with io.BytesIO() as stream:
            download_blob_from_url(blob_url, stream, credential=credential)
            stream.seek(0)
            with Image.open(stream) as img:
                img.load()

    # crop the image
    for crop_path, bbox in bboxes_tocrop.items():
        save_crop(img, bbox_norm=bbox, save=crop_path)


def save_crop(img: Image.Image, bbox_norm: Sequence[float], save: str) -> None:
    """
    Args:
        img: PIL.Image.Image object, already loaded
        bbox_norm: list or tuple of float, [xmin, ymin, width, height] all in
            normalized coordinates
        save: str, path to save cropped image
    """
    os.makedirs(os.path.abspath(os.path.dirname(save)), exist_ok=True)
    img_width, img_height = img.size
    img.crop(box=[
        bbox_norm[0] * img_width,                    # left
        bbox_norm[1] * img_height,                   # upper
        (bbox_norm[0] + bbox_norm[2]) * img_width,   # right
        (bbox_norm[1] + bbox_norm[3]) * img_height,  # lower
    ]).save(save)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Detects and crops images.')
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
        '--local-detector', action='store_true',
        help='run detector locally instead of calling the batch API')
    parser.add_argument(
        '-i', '--images-dir', default=None,
        help='path to local directory where images are saved')
    parser.add_argument(
        '-n', '--threads', type=int, default=1,
        help='number of threads to use for downloading images (default=1)')
    parser.add_argument(
        '-r', '--resume-file', default=None,
        help='path to save JSON file with list of info dicts on running tasks, '
             'or to resume from running tasks. Only used if --local-detector '
             'flag is not set. Each dict has keys '
             '["dataset", "task_id", "task_name", "local_images_list_path", '
             '"remote_images_list_url"]')
    parser.add_argument(
        '-t', '--confidence-threshold', type=float,
        help='confidence threshold above which to crop bounding boxes')
    return parser.parse_args()


if __name__ == '__main__':
    # args = _parse_args()
    # assert 0 <= args.confidence_threshold <= 1
    # main(json_images_path=args.json_images,
    #      detector_version=args.detector_version,
    #      detector_output_cache_base_dir=args.detector_output_cache_dir,
    #      cropped_images_dir=args.cropped_images_dir,
    #      confidence_threshold=args.confidence_threshold,
    #      local_detector=args.local_detector,
    #      images_dir=args.images_dir,
    #      threads=args.threads,
    #      resume_file_path=args.resume_file)

    basedir = '/home/cyeh/CameraTraps/classification/'
    main(
        json_images_path=basedir + 'run_small/json_images.json',
        detector_version='4.1',
        detector_output_cache_base_dir=basedir + 'mdcache/',
        cropped_images_dir=basedir + 'run_small/cropped_images',
        confidence_threshold=0.8,
        local_detector=False,
        images_dir=None,
        threads=30,
        resume_file_path=basedir + 'run_small/resume_detections2.json')
