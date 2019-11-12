init:
	pip install -r requirements.txt
clean:
	find . | grep -E '(\.cache|__pycache__|\.pyc|\.pyo$\)' | xargs rm -rvf
	rm -r build/ dist/ .eggs/ .cache/ *.egg-info/ || echo '1'

build:
	docker-compose build

up:
	docker-compose up -d && docker attach development
