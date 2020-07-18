## Installation

Install miniconda3. Then create the conda environment using the following command. If you need to add/remove/modify packages, make the appropriate change in the `environment-classifier.yml` file and run the following command again.

```bash
conda env update -f environment-classifier.yml --prune
```

## 1. Select classification labels for training.

Create a classification labels specification JSON file. This file defines the labels that our classifier will be trained to distinguish, as well as the original dataset labels and/or biological taxons that will map to each classification label.

The classification labels specification JSON file must have the following format:

```javascript
{
    // name of classification label
    "cervid": {

        // select animals to include based on hierarchical taxonomy,
        // optionally restricting to a subset of datasets
        "taxa": [
            {
                "level": "family",
                "name": "cervidae"
                // include all datasets if no "datasets" key given
            },
            {
                "level": "family",
                "name": "cervidae",
                "datasets": ["idfg", "idfg_swwlf_2019"]
            }
        ],

        // select animals to include based on dataset labels
        "dataset_labels": {
            "idfg": ["deer", "elk", "prong", "moose", "wtd", "md"],
            "idfg_swwlf_2019": ["elk", "muledeer", "whitetaileddeer", "moose", "pronghorn"],
        },

        // exclude animals using the same format
        "exclude": {
            "taxa": [],  // same format as "taxa" above
            "dataset_labels": {}  // same format as "dataset_labels" above
        }
    },

    // name of another classification label
    "bird": {
        "taxa": [
            {
                "level": "class",
                "name": "aves",
            }
        ],
        "dataset_labels": {
            "idfg_swwlf_2019": ["bird"]
        }
        "exclude": {
            "taxa": [
                {
                    "level": "genus",
                    "name": "meleagris"
                }
            ],
            "dataset_labels": {
                "idfg_swwlf_2019": ["turkey"]
            }
        }
    }
}
```