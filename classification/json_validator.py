"""Validates the JSON file."""
import argparse
from collections import defaultdict
import json
from typing import Any, Dict, Mapping, Set, Tuple

import pandas as pd


def main():
    """Main function."""
    args = _parse_args()
    with open(args.input_json, 'r') as f:
        js = json.load(f)
    taxonomy_df = pd.read_csv(args.input_taxonomy_csv)
    taxonomy_dict = build_taxonomy_dict(taxonomy_df)
    validate_json(js, taxonomy_dict)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Validates JSON.')
    parser.add_argument(
        'input_json',
        help='path to JSON file containing label specification')
    parser.add_argument(
        'input_taxonomy_csv',
        help='path to taxonomy CSV file')
    return parser.parse_args()


def build_taxonomy_dict(taxonomy_df: pd.DataFrame) -> Dict[str, Tuple[str, str]]:
    for i, row in taxonomy_df.iterrows():
        pass


def parse_taxa(taxa_dict: Mapping[str, str],
               taxonomy_dict: Mapping[str, Mapping[str, Set[str]]]
               ) -> Set[Tuple[str, str]]:
    results = set()
    for taxon in taxa_dict:
        taxon_name = taxon['name']
        taxon_level = taxon['level']
        taxon_id = f'{taxon_name}/{taxon_level}'
        datasets = taxon.get('datasets', None)
        ds_to_labels = taxonomy_dict[taxon_id]
        if datasets is None:
            for ds, ds_labels in ds_to_labels.items():
                results += {(ds, ds_label) for ds_label in ds_labels}
        else:
            for ds in datasets:
                results += {(ds, ds_label) for ds_label in ds_to_labels[ds]}
    return results


def parse_spec(spec_dict: Mapping[str, Any],
               taxonomy_dict: Mapping[str, Set[Tuple[str, str]]]
               ) -> Set[Tuple[str, str]]:
    results = set()
    if 'taxa' in spec_dict:
        results += parse_taxa(spec_dict['taxa'], taxonomy_dict)
    if 'dataset_labels' in spec_dict:
        for ds, ds_labels in spec_dict['dataset_labels'].items():
            for ds_label in ds_labels:
                results.add((ds, ds_label))
    return results


def validate_json(js: Dict[str, Dict[str, Any]],
                  taxonomy_dict: Mapping[str, Set[Tuple[str, str]]]):
    """Validates JSON"""
    # maps label name to set of (dataset, dataset_label) tuples
    label_to_inclusions = {}
    seen = set()
    for label, spec_dict in js.items():
        include_set = parse_spec(spec_dict, taxonomy_dict)
        if 'exclude' in spec_dict:
            include_set -= parse_spec(spec_dict['exclude'], taxonomy_dict)
        if seen.isdisjoint(include_set):
            label_to_inclusions[label] = include_set
        else:
            raise ValueError('Intersection between sets!')

