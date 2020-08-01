"""
Converts CSV to JSON format for label specification.

There are 3 possible values for the 'type' column in the CSV:
- "row": this selects a specific rowfrom the master taxonomy CSV
    content syntax: <dataset_name>|<dataset_label>
- "datasettaxon": this selects all animals in a taxon from a particular dataset
    content syntax: <dataset_name>|<taxon_level>|<taxon_name>
- <taxon_level>: this selects all animals in a taxon across all datasets
    content syntax: <taxon_name>

Example CSV input:
    output_label,type,content

    cervid,row,idfg|deer
    cervid,row,idfg|elk
    cervid,row,idfg|prong
    cervid,row,idfg_swwlf_2019|elk
    cervid,row,idfg_swwlf_2019|muledeer
    cervid,row,idfg_swwlf_2019|whitetaileddeer

    cervid,family,cervidae
    cervid,datasettaxon,idfg|family|cervidae
    cervid,datasettaxon,idfg_swwlf_2019|family|cervidae

    bird,row,idfg_swwlf_2019|bird
    bird,class,aves

    !bird,row,idfg_swwlf_2019|turkey
    !bird,genus,meleagris

Example JSON output:

    {
        "cervid": {
            "dataset_labels": {
                "idfg": ["deer", "elk", "prong"],
                "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer"]
            },
            "taxa": [
                {
                    "level": "family",
                    "name": "cervidae"
                },
                {
                    "level": "family",
                    "name": "cervidae"
                    "datasets": ["idfg"]
                },
                {
                    "level": "family",
                    "name": "cervidae"
                    "datasets": ["idfg_swwlf_2019"]
                }
            ]
        },
        "bird": {
            "dataset_labels": {
                "idfg_swwlf_2019": ["bird"]
            },
            "taxa": [
                {
                    "level": "class",
                    "name": "aves"
                }
            ],
            "exclude": {
                "dataset_labels": {
                    "idfg_swwlf_2019": ["turkey"]
                },
                "taxa": [
                    {
                        "level": "genus",
                        "name": "meleagris"
                    }
                ]
            }
        }
    }
"""
import argparse
from collections import defaultdict
import json
from typing import Any, Dict

import pandas as pd


def main():
    args = _parse_args()
    js = csv_to_jsondict(args.input_csv_file)
    with open(args.output_json_path, 'w') as f:
        json.dump(js, f, indent=args.json_indent)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Converts CSV to JSON format for label specification.')
    parser.add_argument(
        'input_csv_file',
        help='path to CSV file containing label specification')
    parser.add_argument(
        'output_json_path',
        help='path to save converted JSON file')
    parser.add_argument(
        '--json-indent', type=int, default=None,
        help='number of spaces to use for JSON indent (default no indent)')
    return parser.parse_args()


def parse_csv_row(obj: Dict[str, Any], rowtype: str, content: str) -> None:
    """Parses a row in the CSV."""
    if rowtype == 'row':
        if 'dataset_labels' not in obj:
            obj['dataset_labels'] = defaultdict(list)

        assert '|' in content
        ds, ds_label = content.split('|')
        obj['dataset_labels'][ds].append(ds_label)

    elif rowtype == 'datasettaxon':
        if 'taxa' not in obj:
            obj['taxa'] = []

        assert '|' in content
        ds, taxon_level, taxon_name = content.split('|')
        obj['taxa'].append({
            'level': taxon_level,
            'name': taxon_name,
            'datasets': [ds]
        })

    else:
        if 'taxa' not in obj:
            obj['taxa'] = []

        taxon_level = rowtype
        taxon_name = content
        obj['taxa'].append({
            'level': taxon_level,
            'name': taxon_name
        })


def csv_to_jsondict(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """Converts CSV to json-style dictionary"""
    df = pd.read_csv(csv_path)
    assert (df.columns == ['output_label', 'type', 'content']).all()

    js: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for i in df.index:
        label, rowtype, content = df.loc[i]
        if label.startswith('!'):
            label = label[1:]
            if 'exclude' not in js[label]:
                js[label]['exclude'] = {}
            obj = js[label]['exclude']
        else:
            obj = js[label]
        parse_csv_row(obj, rowtype, content)

    return js


if __name__ == '__main__':
    main()