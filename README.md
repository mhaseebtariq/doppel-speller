# DoppelSpeller

Finds the best match (in a database of titles) for a misspelled title,
using a combination of Machine Learning and NLP techniques.<br/><br/>
![Project description](./description.jpg)<br/><br/>
#### Challenges:
* Matching search terms in a database of millions of "true" titles (for example, company names) could be computationally expensive
* Human beings can be really creative, even come up with new ways, to misspell words in a title

## Setup
* **Pre-requisites**:
    - Install [Docker](https://docs.docker.com/install/) (tested on engine v3.7)
    - Install `make`:
        - Windows: Install [Cygwin](https://www.cygwin.com/setup-x86_64.exe) [while on the screen that lets you select packages to install, find `make` and select it]
        - Debian: `apt-get install build-essential`
        - RHEL: `yum install make`
        - macOS: Xcode `xcode-select --install` | or using Homebrew `brew install make`
* Check [cli.py](./doppelspeller/cli.py) and [Makefile](./Makefile) for cli definitions
* `make --always-make build` - to build and prepare the Docker container for running the project
* `make update-docker` - to update the project setup on the Docker container
* `make stage-example-data-set` - to copy the "example" data set files to the Docker container
* `make inspect` - inspect the code for PEP-8 issues
* `make test` - run the unit tests

## Explanation

Run the following cli's in order:

#### `make train-model`
Alias of `train_model` in [cli.py](./doppelspeller/cli.py)
* Prepares training data for a `OneVsRest[*]Classifier` - "rest" being the nearest "n" (based on the Jaccard distance) matches
* Each "positive" match is trained along with the nearest n matches, that do not match with the title
* Generates `train` and `evaluation` data sets for the `train-model` cli
* Main features generation method: `construct_features` (in [feature_engineering.py](./doppelspeller/feature_engineering.py))
* XGBoost training output: `train-auc:0.999995	evaluation-auc:0.999747	train-custom-error:177	evaluation-custom-error:193`
* See the definition of `custom_error` in [train.py](./doppelspeller/train.py)
    - Also, the custom objective function `weighted_log_loss`

#### `make generate-predictions`
Alias of `generate_predictions` in [cli.py](./doppelspeller/cli.py)
* The algorithm first looks for exact matches
* Then the nearest "n" matches per (remaining) title are found using the the Jaccard (modified) distance
* Next, the nearest matches are "fuzzy" matched with each title
* Finally, the trained model is used to match the remaining titles
* Test set predictions accuracy (run `make get-predictions-accuracy` to calculate the following):
```
True Positives          5928
True Negatives          3920
False Positives         72
False Negatives         80
```

#### `make extensive-search-single-title title='PRO teome plc SCIs'`
Alias of `extensive_search_single_title` in [cli.py](./doppelspeller/cli.py)
* Predicts the best match using the `OneVsRestClassifier` for the entire (not just the nearest matches) "truth" database

## NOTES
* The "example" data set is auto-generated, therefore, it is actually not too hard to get a high accuracy
    - The solution produces similar accuracy on a data set with actual human errors as well
* All the computationally expensive tasks run in multi-processing mode
    - Those tasks, can therefore, be easily refactored to run on distributed computing clusters

## TODO
* Extend README to include more details/explanation of the solution
* Document all the classes/methods in the code
* Write more unit tests
* Refactor code to be more SOLID
