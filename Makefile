clean:
	find . | grep -E '(\.cache|__pycache__|\.pyc|\.pyo$\)' | xargs rm -rvf
	rm -r build/ dist/ .eggs/ .cache/ *.egg-info/ || echo '1'

build:
	make clean
	docker-compose build

login:
	docker-compose up -d && docker attach doppelspeller

update-docker:
	docker-compose up -d && docker exec -t doppelspeller /bin/sh /doppelspeller/update_docker.sh

generate-lsh-forest:
	docker-compose up -d && docker exec -t doppelspeller doppel-speller -vv generate-lsh-forest