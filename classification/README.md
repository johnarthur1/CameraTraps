## Installation

Install miniconda3. Then create the conda environment using the following command. If you need to add/remove/modify packages, make the appropriate change in the `environment-classifier.yml` file and run the following command again.

```bash
conda env update -f environment-classifier.yml --prune
```

## JSON format

JSON format for specifying classification classes

```javascript
{
    // name of label
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

    // name of another label
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