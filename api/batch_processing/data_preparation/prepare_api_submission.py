"""
This module is somewhere between "documentation" and "code".  It is intended to
capture the steps the precede running a job via the AI for Earth Camera Trap
Image Processing API, and it automates a couple of those steps.  We hope to
gradually automate all of these.

Here's the stuff we usually do before submitting a job:

1) Upload data to Azure... we do this with azcopy, not addressed in this script

2) List the files you want the API to process... see
    ai4eutils.ai4e_azure_utils.enumerate_blobs_to_file()

3) Divide that list into chunks that will become individual API submissions...
    this module supports that via divide_files_into_tasks.

3) Put each .json file in a blob container, and generate a read-only SAS
   URL for it.  Not automated right now.

4) Generate the API query(ies) you'll submit to the API... see
    generate_api_queries()

5) Submit the API query... I currently do this with Postman.

6) Monitor task status

7) Combine multiple API outputs

8) We're now into what we really call "postprocessing", rather than
    "data_preparation", but... possibly do some amount of partner-specific
    renaming, folder manipulation, etc. This is very partner-specific, but
    generally done via:

    find_repeat_detections.py
    subset_json_detector_output.py
    postprocess_batch_results.py
"""
#%% Imports and constants

import json
import string
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union)

import ai4e_azure_utils  # from ai4eutils
import path_utils  # from ai4eutils


DEFAULT_N_FILES_PER_API_TASK = 1_000_000

VALID_REQUEST_NAME_CHARS = f'-_{string.ascii_letters}{string.digits}'
REQUEST_NAME_CHAR_LIMIT = 100


#%% Dividing files into multiple tasks

def divide_chunks(l: Sequence[Any], n: int) -> List[Sequence[Any]]:
    """
    Divide list *l* into chunks of size *n*, with the last chunk containing
    <= n items.
    """
    # https://www.geeksforgeeks.org/break-list-chunks-size-n-python/
    chunks = [l[i * n:(i + 1) * n] for i in range((len(l) + n - 1) // n)]
    return chunks


def divide_files_into_tasks(
        file_list_json: str,
        n_files_per_task: int = DEFAULT_N_FILES_PER_API_TASK
        ) -> Tuple[List[str], List[Sequence[Any]]]:
    """
    Divides the file *file_list_json*, which contains a single json-encoded list
    of strings, into a set of json files, each containing *n_files_per_task*
    (the last file will contain <= *n_files_per_task* files).

    Output JSON files have extension `*.chunkXXX.json`. For example, if the
    input JSON file is `blah.json`, output files will be `blah.chunk000.json`,
    `blah.chunk001.json`, etc.

    Args:
        file_list_json: str, path to JSON file containing list of file names
        n_files_per_task: int, max number of files to include in each API task

    Returns:
        output_files: list of str, output JSON file names
        chunks: list of list of str, chunks[i] is the content of output_files[i]
    """
    with open(file_list_json) as f:
        file_list = json.load(f)

    chunks = divide_chunks(file_list, n_files_per_task)
    output_files = []

    for i_chunk, chunk in enumerate(chunks):
        chunk_id = 'chunk{0:0>3d}'.format(i_chunk)
        output_file = path_utils.insert_before_extension(
            file_list_json, chunk_id)
        output_files.append(output_file)
        with open(output_file, 'w') as f:
            json.dump(chunk, f, indent=1)

    return output_files, chunks


def clean_request_name(request_name: str,
                       whitelist: str = VALID_REQUEST_NAME_CHARS,
                       char_limit: int = REQUEST_NAME_CHAR_LIMIT) -> str:
    """Removes invalid characters from an API request name."""
    return path_utils.clean_filename(
        filename=request_name, whitelist=whitelist, char_limit=char_limit)


def generate_api_queries(
        input_container_sas_url: str,
        file_list_sas_urls: Iterable[str],
        request_name_base: str,
        caller: str,
        additional_args: Optional[Mapping] = None,
        image_path_prefixes: Optional[Union[str, Sequence[str]]] = None
        ) -> Tuple[List[str], List[Dict]]:
    """Generates JSON-formatted API input from input parameters.

    See
    github.com/microsoft/CameraTraps/tree/master/api/batch_processing#api-inputs

    Args:
        input_container_sas_url: str, SAS URL with list and read permissions to
            the Blob Storage container where the images are stored
        file_list_sas_urls: list of str, SAS URLs to individual file lists,
            all relative to the same container
        request_name_base: str, request name for the set
            if the base name is 'blah', individual requests will get request
            names 'blah_chunk000', 'blah_chunk001', etc.
        additional_args: dict, custom arguments to be added to each query
            (to specify different custom args per query, use multiple calls to
            generate_api_query())
        image_path_prefixes: str or list of str, one per request

    Returns:
        request_strings: list of str, request_strings[i] is a JSON string
            representation of request_dict[i]
        request_dicts: list of dict, each dict contains the parameters for a
            single request to be sent to the MegaDetector Batch Processing API
    """
    assert isinstance(file_list_sas_urls, list)

    request_name_orig = request_name_base
    request_name_base = clean_request_name(request_name_base)
    if request_name_base != request_name_orig:
        print(f'Warning: renamed {request_name_orig} to {request_name_base}')

    request_dicts = []
    request_strings = []
    for i_url, file_list_sas_url in enumerate(file_list_sas_urls):
        if len(file_list_sas_urls) > 1:
            request_name = f'{request_name_base}_chunk{i_url:0>3d}'
        else:
            request_name = request_name_base

        d = {
            'input_container_sas': input_container_sas_url,
            'images_requested_json_sas': file_list_sas_url,
            'request_name': request_name,
            'caller': caller,
        }
        if additional_args is not None:
            d.update(additional_args)

        if image_path_prefixes is not None:
            if not isinstance(image_path_prefixes, list):
                d['image_path_prefix'] = image_path_prefixes
            else:
                d['image_path_prefix'] = image_path_prefixes[i_url]
        request_dicts.append(d)
        request_strings.append(json.dumps(d, indent=1))

    return request_strings, request_dicts


def generate_api_query(input_container_sas_url: str,
                       file_list_sas_url: str,
                       request_name: str,
                       caller: str,
                       additional_args: Optional[Mapping] = None,
                       image_path_prefix: Optional[str] = None):
    """
    Convenience function to call generate_api_queries for a single batch.

    See generate_api_queries, and s/lists/single items.
    """
    file_list_sas_urls = [file_list_sas_url]
    image_path_prefixes = None
    if image_path_prefix is not None:
        image_path_prefixes = [image_path_prefix]
    request_strings, request_dicts = generate_api_queries(
        input_container_sas_url, file_list_sas_urls, request_name, caller,
        additional_args, image_path_prefixes)
    return request_strings[0], request_dicts[0]


#%% Tools for working with API output

# I suspect this whole section will move to a separate file at some point,
# so leaving these imports and constants here for now.
from posixpath import join as urljoin

import urllib
import tempfile
import os
import requests

ct_api_temp_dir = os.path.join(tempfile.gettempdir(),'camera_trap_api')
IMAGES_PER_SHARD = 2000

def fetch_task_status(endpoint_url,task_id):
    """
    Currently a very thin wrapper to fetch the .json content from the task URL

    Returns status dictionary,status code
    """
    response = requests.get(urljoin(endpoint_url,str(task_id)))
    return response.json(),response.status_code


def get_output_file_urls(response):
    """
    Given the dictionary returned by fetch_task_status, get the set of
    URLs returned at the end of the task, or None if they're not available.
    """
    try:
        output_file_urls = response['Status']['message']['output_file_urls']
    except:
        return None
    assert 'detections' in output_file_urls
    assert 'failed_images' in output_file_urls
    assert 'images' in output_file_urls
    return output_file_urls


def download_url(url, destination_filename, verbose=False):
    """
    Download a URL to a local file
    """
    if verbose:
        print('Downloading {} to {}'.format(url,destination_filename))
    urllib.request.urlretrieve(url, destination_filename)
    assert(os.path.isfile(destination_filename))
    return destination_filename


def get_temporary_filename():
    os.makedirs(ct_api_temp_dir,exist_ok=True)
    fn = os.path.join(ct_api_temp_dir,next(tempfile._get_candidate_names()))
    return fn


def download_to_temporary_file(url):
    return download_url(url,get_temporary_filename())


def get_missing_images(response,verbose=False):
    """
    Downloads and parses the list of submitted and processed images for a task,
    and compares them to find missing images.  Double-checks that 'failed_images'
    is a subset of the missing images.
    """
    output_file_urls = get_output_file_urls(response)
    if output_file_urls is None:
        return None

    # Download all three urls to temporary files
    #
    # detections, failed_images, images
    temporary_files = {}
    for s in output_file_urls.keys():
        temporary_files[s] = download_to_temporary_file(output_file_urls[s])

    # Load all three files
    results = {}
    for s in temporary_files.keys():
        with open(temporary_files[s]) as f:
            results[s] = json.load(f)

    # Diff submitted and processed images
    submitted_images = results['images']
    if verbose:
        print('Submitted {} images'.format(len(submitted_images)))

    detections = results['detections']
    processed_images = [detection['file'] for detection in detections['images']]
    if verbose:
        print('Received results for {} images'.format(len(processed_images)))

    failed_images = results['failed_images']
    if verbose:
        print('{} failed images'.format(len(failed_images)))

    n_failed_shards = int(response['Status']['message']['num_failed_shards'])
    estimated_failed_shard_images = n_failed_shards * IMAGES_PER_SHARD
    if verbose:
        print('{} failed shards (approimately {} images)'.format(n_failed_shards,estimated_failed_shard_images))

    missing_images = list(set(submitted_images) - set(processed_images))
    if verbose:
        print('{} images not in results'.format(len(missing_images)))

    # Confirm that the failed images are a subset of the missing images
    assert len(set(failed_images) - set(missing_images)) == 0, 'Failed images should be a subset of missing images'

    for fn in temporary_files.values():
        os.remove(fn)

    return missing_images


def download_detection_results(endpoint_url,task_id,output_file):
    """
    Download the detection results .json file for a task
    """
    response,_ = fetch_task_status(endpoint_url,task_id)
    output_file_urls = get_output_file_urls(response)
    if output_file_urls is None:
        return None
    detection_url = output_file_urls['detections']
    download_url(detection_url,output_file)
    return response


def generate_resubmission_list(endpoint_url,task_id,resubmission_file_list_name):
    """
    Finds all the image files that failed to process in a job and writes them to a file.
    """
    response,_ = fetch_task_status(endpoint_url,task_id)
    missing_files = get_missing_images(response)
    missing_images = path_utils.find_image_strings(missing_files)
    non_images = list(set(missing_files) - set(missing_images))
    ai4e_azure_utils.write_list_to_file(
        resubmission_file_list_name, missing_images)
    return missing_images,non_images


#%% Interactive driver

# if False:

#     #%%
#     account_name = ''
#     sas_token = 'st=...'
#     container_name = ''
#     rmatch = None # '^Y53'
#     output_file = r'output.json'

#     blobs = ai4e_azure_utils.enumerate_blobs_to_file(
#         output_file=output_file,
#         account_name=account_name,
#         sas_token=sas_token,
#         container_name=container_name,
#         rsearch=rsearch)

#     #%%

#     file_list_json = r"D:\temp\idfg_20190801-hddrop_image_list.json"
#     task_files = prepare_api_submission.divide_files_into_tasks(file_list_json)

#     #%%

#     file_list_sas_urls = [
#         '','',''
#     ]

#     input_container_sas_url = ''
#     request_name_base = ''
#     caller = 'blah@blah.com'

#     request_strings,request_dicts = generate_api_queries(
#         input_container_sas_url,
#         file_list_sas_urls,
#         request_name_base,
#         caller)

#     for s in request_strings:
#         print(s)
