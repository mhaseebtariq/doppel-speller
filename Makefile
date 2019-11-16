clean:
	find . | grep -E '(\.cache|__pycache__|\.pyc|\.pyo$\)' | xargs rm -rvf
	rm -r build/ dist/ .eggs/ .cache/ *.egg-info/ || echo '1'

build:
	make clean
	docker-compose build
	make update-docker

login:
	docker-compose up -d && docker attach doppelspeller

update-docker:
	docker-compose up -d && docker exec -t doppelspeller /bin/sh /doppelspeller/docker-update.sh

generate-lsh-forest:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv generate-lsh-forest

prepare-data-for-features-generation:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv prepare-data-for-features-generation

generate-train-and-evaluation-data-sets:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv generate-train-and-evaluation-data-sets

train-model:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv train-model
