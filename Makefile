test:
	pytest --cov=src --cov-report=term-missing tests/

test-html:
	pytest --cov=src --cov-report=html tests/

lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503

all: lint test
