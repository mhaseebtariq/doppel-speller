# DoppelSpeller

Finds the best match for a misspelled title.<br/><br/>
![Project description](./description.jpg)

## Setup
Pre-requisite: Install [Docker](https://docs.docker.com/install/) (tested on the v3.7 engine).
* `make --always-make build`
* `make generate-lsh-forest`
* `make prepare-data-for-features-generation`
* `make generate-train-and-evaluation-data-sets`
* `make train-model`
* `make prepare-predictions-data`
* `make generate-predictions`
