# Root conftest: makes the repo root importable when tests are run with the
# plain `pytest` binary (which, unlike `python -m pytest`, does not add the
# current directory to sys.path). pyproject also sets pythonpath=["."].
