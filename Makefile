PY := .venv/bin/python
BB := .venv/bin/biobrain

.PHONY: setup data download preprocess validate analyze bench test

setup:            ## create the venv with the exact versions used for the recorded results
	python3 -m venv .venv
	$(PY) -m pip install -r requirements.lock
	$(PY) -m pip install -e . --no-deps

download:         ## ~1.1 GiB from Zenodo 10676866 + GitHub (checksums verified)
	$(BB) download

preprocess:
	$(BB) preprocess

validate:
	$(BB) validate

data: download preprocess validate

analyze:
	$(BB) analyze

bench:
	$(BB) bench-memory

test:
	$(PY) -m pytest -q
