clean:
	find . | grep -E '(\.pytest_cache|\.coverage|\.cache|__pycache__|\.pyc|\.pyo$\)' | xargs rm -rvf
	rm -r build/ dist/ .eggs/ .cache/ *.egg-info/ || echo '1'

build:
	docker-compose build
	make update-docker

update-docker:
	make clean-docker
	docker-compose up -d && docker exec -t doppelspeller /bin/sh /doppelspeller/docker-update.sh

clean-docker:
	docker-compose up -d && docker exec -t doppelspeller bash -c "cd /doppelspeller/ && make clean"

login:
	docker-compose up -d && docker attach doppelspeller

inspect:
	docker-compose up -d && docker exec -t doppelspeller bash -c "cd /doppelspeller/ && /usr/local/bin/flake8 doppelspeller/"

test:
	docker-compose up -d && docker exec -t doppelspeller bash -c "cd /doppelspeller/ && make clean && /usr/local/bin/py.test"

stage-example-data-set:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv stage-example-data-set-on-docker-container

train-model:
	make update-docker
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv train-model

generate-predictions:
	make update-docker
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv generate-predictions

get-predictions-accuracy:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv get-predictions-accuracy

closest-search-single-title:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv closest-search-single-title --title-to-search="$(title)"
