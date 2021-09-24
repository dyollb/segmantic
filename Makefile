.PHONY: test init init-dev

init: .venv

init-dev: init
	pip --quiet install -r requirements/dev.txt

test: init-dev
	pytest tests

.venv: # creates virtual environment (detectable by vscode)
	python3 -m venv $@
	$@/bin/pip3 --quiet install --upgrade \
		pip \
		wheel \
		setuptools
	@echo "To activate the venv, execute 'source .venv/bin/activate'"

.PHONY: docs
docs: .venv
	pip install -r requirements/docs.txt
	mkdocs build
	mkdocs serve