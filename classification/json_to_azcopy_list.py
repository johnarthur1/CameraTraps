import json
import os

from data_management.megadb import megadb_utils
import sas_blob_utils

from tqdm import tqdm

images_dir = '/ssd/images/'
json_images_path = 'run_idfg/json_images.json'
output_dir = 'run_idfg/'

with open(json_images_path, 'r') as f:
    js = json.load(f)

datasets_table = megadb_utils.MegadbUtils().get_datasets_table()

output_files = {}

pbar = tqdm(js.items())
for img_path, img_info in pbar:
    save_path = os.path.join(images_dir, img_path)
    if os.path.exists(save_path):
        continue

    ds, img_file = img_path.split('/', maxsplit=1)
    if ds not in output_files:
        output_path = os.path.join(output_dir, f'{ds}_images.txt')
        output_files[ds] = open(output_path, 'w')

        dataset_info = datasets_table[ds]
        account = dataset_info['storage_account']
        container = dataset_info['container']

        if 'public' in datasets_table[ds]['access']:
            url = sas_blob_utils.build_azure_storage_uri(
                account, container)
        else:
            url = sas_blob_utils.build_azure_storage_uri(
                account, container,
                sas_token=dataset_info['container_sas_key'][1:])
        pbar.write(url)

    output_files[ds].write(img_file)

for f in output_files.values():
    f.close()
