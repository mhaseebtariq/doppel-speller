# DoppelSpeller

Finds the best match (in a database of titles) for a misspelled title,
using a combination of Machine Learning and NLP techniques.<br/><br/>
![Project description](./description.jpg)

## Setup
**Pre-requisite**: Install [Docker](https://docs.docker.com/install/) (tested on the v3.7 engine).
* Setup an environment variable `$PROJECT_DATA_PATH` (open [settings.py](./doppelspeller/settings.py) to see how it is used!)
    - Defaults to [data/](./data/)
* Copy all the files from [example_dataset/*](./example_dataset/) to `$PROJECT_DATA_PATH`
* Check [cli.py](./doppelspeller/cli.py) and [Makefile](./Makefile) for cli definitions
* `make --always-make build`
* `make update-docker` - to only update the project on the Docker container
* `make generate-lsh-forest`
* `make prepare-data-for-features-generation`
* `make generate-train-and-evaluation-data-sets`
* `make train-model`
* `make prepare-predictions-data`
* `make generate-predictions`

## CLIs' Explanation
#### `make generate-lsh-forest`
Alias of `generate_lsh_forest` in [cli.py](./doppelspeller/cli.py)
* Given the "truth" database (see `GROUND_TRUTH_FILE` in [settings.py](./doppelspeller/settings.py)):
    - Generates a Locality-sensitive hashing (LSH) forest for fetching the nearest title matches
    - "Nearest", based on the Jaccard distance computed on ngrams (n=3) of the titles
* The computation can definitely be improved by using a different distance metric, computed over high dimensional matrices

#### `make prepare-data-for-features-generations`
Alias of `prepare_data_for_features_generations` in [cli.py](./doppelspeller/cli.py)
* Prepares training data for a `OneVsRestClassifier`
* Each "positive" get the nearest "n" matches that do not match

#### `make generate-train-and-evaluation-data-sets`
Alias of `generate_train_and_evaluation_data_sets` in [cli.py](./doppelspeller/cli.py)

#### `make train-model`
Alias of `train_model` in [cli.py](./doppelspeller/cli.py)
* `train-auc:1	evaluation-auc:0.999882	train-custom-error:7	evaluation-custom-error:213`
* See the definition of `custom_error` in [train.py](./doppelspeller/train.py)

#### `make prepare-predictions-data`
Alias of `prepare_predictions_data` in [cli.py](./doppelspeller/cli.py)

#### `make generate-predictions`
Alias of `generate_predictions` in [cli.py](./doppelspeller/cli.py)
```
true_positives      5872
true_negatives      3845
false_positives     128
false_negatives     155
```

#### `make extensive-search-single-title`
Alias of `extensive_search_single_title` in [cli.py](./doppelspeller/cli.py)

## TODO
* More details on README
* Document classes/methods in the code
* Write unit tests
