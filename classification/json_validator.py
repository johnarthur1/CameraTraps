"""Validates a classification label specification JSON file and optionally
outputs a JSON file listing the matching image files.

See README.md for an example of a classification label specification JSON file.

The output JSON file looks like:

{
    "caltech/cct_images/59f79901-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset
        "bbox": [{"category": "animal",
                  "bbox": [0, 0.347, 0.237, 0.257]}],
        "label": ["monutain_lion"]  // labels to use in classifier
    },
    "caltech/cct_images/59f5fe2b-23d2-11e8-a6a3-ec086b02610b.jpg": {
        "dataset": "caltech",
        "location": 13,
        "class": "mountain_lion",  // class from dataset
        "label": ["monutain_lion"]  // labels to use in classifier
    },
    ...
}

Example usage:

    python json_validator.py my_classes.json camera_trap_taxonomy_mapping.csv \
        --output-json my_images.json --json-indent 4
"""
# allow forward references in typing annotations
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Container, Dict, List, Mapping, Optional, Set, Tuple

import pandas as pd

from data_management.megadb.megadb_utils import MegadbUtils


class TaxonNode:
    r"""A node in a taxonomy tree, associated with a set of dataset labels.

    By default we support multiple parents for each TaxonNode because different
    taxonomies may have a different granularity of hierarchy. If the taxonomy
    was created from a mixture of different taxonomies, then we may see the
    following, for example:

        "eastern gray squirrel" (inat)     "squirrel" (gbif)
        ------------------------------     -----------------
    family:                        sciuridae
                                  /          \
    subfamily:          sciurinae             |  # skips subfamily
                                |             |
    tribe:               sciurini             |  # skips tribe
                                  \          /
    genus:                          sciurus
    """
    # class variables
    single_parent_only = False

    def __init__(self, level: str, name: str):
        """Initializes a TaxonNode."""
        self.level = level
        self.name = name
        self.ids: Set[Tuple[str, int]] = set()
        self.children: List[TaxonNode] = []
        self.parents: List[TaxonNode] = []
        self.dataset_labels: Set[Tuple[str, str]] = set()

    def __repr__(self):
        id_str = ', '.join(f'{source}={id}' for source, id in self.ids)
        return f'TaxonNode({id_str}, level={self.level}, name={self.name})'

    def add_id(self, source: str, taxon_id: int):
        assert source in ['gbif', 'inat', 'manual']
        self.ids.add((source, taxon_id))

    def add_parent(self, parent: TaxonNode):
        """Adds a TaxonNode to the list of parents of the current TaxonNode.

        Args:
            parent: TaxonNode, must be higher in the taxonomical hierarchy
        """
        if TaxonNode.single_parent_only and len(self.parents) > 0:
            assert len(self.parents) == 1
            assert self.parents[0] is parent, (
                f'self.parents: {self.parents}, new parent: {parent}')
            return
        if parent not in self.parents:
            self.parents.append(parent)
            parent.add_child(self)

    def add_child(self, child: TaxonNode):
        if child not in self.children:
            self.children.append(child)
            child.add_parent(self)

    def add_dataset_label(self, ds: str, ds_label: str) -> None:
        """
        Args:
            ds: str, name of dataset
            ds_label: str, name of label used by that dataset
        """
        self.dataset_labels.add((ds, ds_label))

    def get_dataset_labels(self,
                           include_datasets: Optional[Container[str]] = None
                           ) -> Set[Tuple[str, str]]:
        """Returns a set of all (ds, ds_label) tuples that belong to this taxon
        node or its descendants.

        Args:
            include_datasets: list of str, names of datasets to include
                if None, then all datasets are included

        Returns: set of (ds, ds_label) tuples
        """
        result = self.dataset_labels
        if include_datasets is not None:
            result = set(tup for tup in result if tup[0] in include_datasets)

        for child in self.children:
            result |= child.get_dataset_labels()
        return result


def main(input_json_path: str,
         taxonomy_csv_path: str,
         allow_multilabel: bool = False,
         single_parent_taxonomy: bool = False,
         output_json_path: str = None,
         json_indent: Optional[int] = None):
    """Main function."""
    print('Building taxonomy hierarchy')
    taxonomy_df = pd.read_csv(taxonomy_csv_path)
    if single_parent_taxonomy:
        TaxonNode.single_parent_only = True
    taxonomy_dict = build_taxonomy_dict(taxonomy_df)

    print('Validating input json')
    with open(input_json_path, 'r') as f:
        js = json.load(f)
    label_to_inclusions = validate_json(
        js, taxonomy_dict, allow_multilabel=allow_multilabel)

    # use MegaDB to generate list of images
    if output_json_path is not None:
        print('Generating output json')
        output_js = get_output_json(label_to_inclusions)
        with open(output_json_path, 'w') as f:
            json.dump(output_js, f, indent=json_indent)


def _parse_args() -> argparse.Namespace:
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Validates JSON.')
    parser.add_argument(
        'input_json',
        help='path to JSON file containing label specification')
    parser.add_argument(
        'taxonomy_csv',
        help='path to taxonomy CSV file')
    parser.add_argument(
        '--allow-multilabel', action='store_true',
        help='flag that allows assigning a (dataset, dataset_label) pair to '
             'multiple output labels')
    parser.add_argument(
        '--single-parent-taxonomy', action='store_true',
        help='flag that restricts the taxonomy to only allow a single parent '
             'for each taxon node')
    parser.add_argument(
        '--output-json',
        help='path to output JSON file with list of images')
    parser.add_argument(
        '--json-indent', type=int, default=None,
        help='number of spaces to use for JSON indent (default no indent)')
    return parser.parse_args()


def build_taxonomy_dict(taxonomy_df: pd.DataFrame
                        ) -> Dict[Tuple[str, str], TaxonNode]:
    """Creates a mapping from (taxon_level, taxon_name) to TaxonNodes, used for
    gathering all dataset labels associated with a given taxon.

    Args:
        taxonomy_df: pd.DataFrame, see taxonomy_mapping directory for more info

    Returns: dict, maps (taxon_level, taxon_name) to a TaxonNode
    """
    taxonomy_dict = {}
    for _, row in taxonomy_df.iterrows():
        ds = row['dataset_name']
        ds_label = row['query']
        taxa_ancestry = row['taxonomy_string']
        id_source = row['source']
        if pd.isna(taxa_ancestry):
            # taxonomy CSV rows without 'taxonomy_string' entries can only be
            # added to the JSON via the 'dataset_labels' key
            continue
        else:
            taxa_ancestry = eval(taxa_ancestry)  # pylint: disable=eval-used

        taxon_child = None
        for i, taxon in enumerate(taxa_ancestry):
            taxon_id, taxon_level, taxon_name, _ = taxon

            key = (taxon_level, taxon_name)
            if key not in taxonomy_dict:
                node = TaxonNode(level=taxon_level, name=taxon_name)
                if taxon_child is not None:
                    node.add_child(taxon_child)
                taxonomy_dict[key] = node

            node = taxonomy_dict[key]
            node.add_id(id_source, int(taxon_id))  # np.int64 -> int
            if i == 0:
                assert row['taxonomy_level'] == taxon_level, (
                    f'taxonomy CSV level: {row["taxonomy_level"]}, '
                    f'level from taxonomy_string: {taxon_level}')
                assert row['scientific_name'] == taxon_name
                node.add_dataset_label(ds, ds_label)

            taxon_child = node

    # returns a dict that maps (taxon_level, taxon_name) to a TaxonNode
    return taxonomy_dict


def parse_taxa(taxa_dicts: List[Dict[str, str]],
               taxonomy_dict: Dict[Tuple[str, str], TaxonNode]
               ) -> Set[Tuple[str, str]]:
    """Gathers the dataset labels requested by a "taxa" specification.

    Args:
        taxa_dicts: list of dict, corresponds to the "taxa" key in JSON, e.g.,
            [
                {'level': 'family', 'name': 'cervidae', 'datasets': ['idfg']},
                {'level': 'genus',  'name': 'meleagris'}
            ]
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode

    Returns: set of (ds, ds_label), dataset labels requested by the taxa spec
    """
    results = set()
    for taxon in taxa_dicts:
        key = (taxon['level'], taxon['name'])
        datasets = taxon.get('datasets', None)
        results |= taxonomy_dict[key].get_dataset_labels(datasets)
    return results


def parse_spec(spec_dict: Mapping[str, Any],
               taxonomy_dict: Dict[Tuple[str, str], TaxonNode]
               ) -> Set[Tuple[str, str]]:
    """
    Args:
        spec_dict: dict, contains keys ['taxa', 'dataset_labels']
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode

    Returns: set of (ds, ds_label), dataset labels requested by the spec
    """
    results = set()
    if 'taxa' in spec_dict:
        results |= parse_taxa(spec_dict['taxa'], taxonomy_dict)
    if 'dataset_labels' in spec_dict:
        for ds, ds_labels in spec_dict['dataset_labels'].items():
            for ds_label in ds_labels:
                results.add((ds, ds_label))
    return results


def validate_json(js: Dict[str, Dict[str, Any]],
                  taxonomy_dict: Dict[Tuple[str, str], TaxonNode],
                  allow_multilabel: bool) -> Dict[str, Set[Tuple[str, str]]]:
    """Validates JSON.

    Args:
        js: dict, Python dict representation of JSON file
            see classification/README.md
        taxonomy_dict: dict, maps (taxon_level, taxon_name) to a TaxonNode
        allow_multilabel: bool, whether to allow a dataset label to be assigned
            to multiple output labels

    Returns: dict, maps label name to set of (dataset, dataset_label) tuples
    """
    # maps output label name to set of (dataset, dataset_label) tuples
    label_to_inclusions: Dict[str, Set[Tuple[str, str]]] = {}
    for label, spec_dict in js.items():
        include_set = parse_spec(spec_dict, taxonomy_dict)
        if 'exclude' in spec_dict:
            include_set -= parse_spec(spec_dict['exclude'], taxonomy_dict)

        for label_b, set_b in label_to_inclusions.items():
            if not include_set.isdisjoint(set_b):
                print(f'Labels {label} and {label_b} will share images')
                if not allow_multilabel:
                    raise ValueError('Intersection between sets!')

        label_to_inclusions[label] = include_set
    return label_to_inclusions


def get_output_json(label_to_inclusions: Dict[str, Set[Tuple[str, str]]]
                    ) -> Dict[str, Dict[str, Any]]:
    """Queries MegaDB to get image paths matching dataset_labels.

    Args:
        label_to_inclusions: dict, maps label name to set of
            (dataset, dataset_label) tuples, output of validate_json()

    Returns: dict, maps image_path to a dict of properties
    - 'dataset': str, name of dataset that image is from
    - 'location': str or int, optional
    - 'class': str, class label from the dataset
    - 'label': str, assigned output label
    - 'bbox': list of dicts, optional
    """
    # because MegaDB is organized by dataset, we do the same
    # ds_to_labels = {
    #     'dataset_name': {
    #         'dataset_label': [output_label1, output_label2]
    #     }
    # }
    ds_to_labels: Dict[str, Dict[str, List]] = {}
    for output_label, ds_dslabels_set in label_to_inclusions.items():
        for (ds, ds_label) in ds_dslabels_set:
            if ds not in ds_to_labels:
                ds_to_labels[ds] = {}
            if ds_label not in ds_to_labels[ds]:
                ds_to_labels[ds][ds_label] = []
            ds_to_labels[ds][ds_label].append(output_label)


    # we need the datasets table for getting full image paths
    megadb = MegadbUtils()
    datasets_table = megadb.get_datasets_table()

    # The line
    #    [img.class[0], seq.class[0]][0] as class
    # selects the image-level class label if available. Otherwise it selects the
    # sequence-level class label. This line assumes the following conditions,
    # expressed in the WHERE clause:
    # - at least one of the image or sequence class label is given
    # - the image and sequence class labels are arrays with length at most 1
    # - the image class label takes priority over the sequence class label
    #
    # In Azure Cosmos DB, if a field is not defined, then it is simply excluded
    # from the result. For example, on the following JSON object,
    #     {
    #         "dataset": "camera_traps",
    #         "seq_id": "1234",
    #         "location": "A1",
    #         "images": [{"file": "abcd.jpeg"}],
    #         "class": ["deer"],
    #     }
    # the array [img.class[0], seq.class[0]] just gives ['deer'] because
    # img.class is undefined and therefore excluded.
    query = '''
    SELECT
        seq.dataset,
        seq.location,
        img.file,
        [img.class[0], seq.class[0]][0] as class,
        img.bbox
    FROM sequences seq JOIN img IN seq.images
    WHERE (ARRAY_LENGTH(img.class) = 1
            AND ARRAY_CONTAINS(@dataset_labels, img.class[0])
        )
        OR (ARRAY_LENGTH(seq.class) = 1
            AND ARRAY_CONTAINS(@dataset_labels, seq.class[0])
            AND (ARRAY_LENGTH(img.class) = 0
                OR NOT IS_DEFINED(img.class)
                OR (ARRAY_LENGTH(img.class) = 1 AND img.class[0] = seq.class[0])
            )
        )
    '''

    output_json = {}  # maps full image path to json object
    for ds in sorted(ds_to_labels.keys()):  # sort for determinism
        ds_labels = sorted(ds_to_labels[ds].keys())
        print(f'Querying dataset "{ds}" for dataset labels:', ds_labels)

        start = time.time()
        parameters = [dict(name='@dataset_labels', value=ds_labels)]
        results = megadb.query_sequences_table(
            query, partition_key=ds, parameters=parameters)
        elapsed = time.time() - start
        print(f'- query took {elapsed:.0f}s, found {len(results)} images')

        # if no path prefix, set it to the empty string '', because
        #     os.path.join('', x) = x
        img_path_prefix = os.path.join(
            ds, datasets_table[ds].get('path_prefix', ''))
        for result in results:
            # result keys
            # - already has: ['dataset', 'location', 'file', 'class', 'bbox']
            # - add ['label'], remove ['file']
            img_path = os.path.join(img_path_prefix, result['file'])
            del result['file']
            ds_label = result['class']
            result['label'] = ds_to_labels[ds][ds_label]
            output_json[img_path] = result

    return output_json


if __name__ == '__main__':
    args = _parse_args()
    main(input_json_path=args.input_json,
         taxonomy_csv_path=args.taxonomy_csv,
         allow_multilabel=args.allow_multilabel,
         single_parent_taxonomy=args.single_parent_taxonomy,
         output_json_path=args.output_json,
         json_indent=args.json_indent)

# main(
#     input_json_path='classification/idfg_classes.json',
#     taxonomy_csv_path='../camera-traps-private/camera_trap_taxonomy_mapping.csv',
#     output_json_path='classification/idfg_images.json',
#     json_indent=4)
