PROJECT=dsvi

PYTHON_SOURCE_DIRS=$(PROJECT)

all: format check test

clean:
	@echo "Not implemented"

check:
	@flake8 $(PYTHON_SOURCE_DIRS)
	@mypy $(PYTHON_SOURCE_DIRS)
	@isort --quiet --recursive --atomic --diff $(PYTHON_SOURCE_DIRS)
	@black --diff --quiet $(PYTHON_SOURCE_DIRS)

docs:
	@echo "Not implemented"

format:
	@isort --quiet --recursive --atomic $(PYTHON_SOURCE_DIRS)
	@black --quiet $(PYTHON_SOURCE_DIRS)

help:
	@echo "all:\tFormat code and run checks"
	@echo "clean\tRemove temporary files"
	@echo "check:\tRun checks"
	@echo "docs:\tBuild html documentation"
	@echo "format:\tFormat code"
	@echo "help:\tShow this help"
	@echo "test:\tRun tests"

test:
	@pytest -q -s
