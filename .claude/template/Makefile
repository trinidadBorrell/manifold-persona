.PHONY: help install-dev hooks check
help:
	@printf 'install-dev: install optional tools into the existing .venv\nhooks: opt in to Git hooks\ncheck: run scoped pre-commit checks\n'
install-dev:
	@test -x .venv/bin/python || (echo 'Create .venv first; this target will not replace it.' && exit 1)
	@command -v uv >/dev/null || (echo 'Install uv or install requirements-dev.txt with your environment tool.' && exit 1)
	uv pip install --python .venv/bin/python -r requirements-dev.txt
hooks:
	@test -x .venv/bin/pre-commit || (echo 'Run make install-dev first.' && exit 1)
	.venv/bin/pre-commit install
check:
	@test -x .venv/bin/pre-commit || (echo 'Run make install-dev first.' && exit 1)
	.venv/bin/pre-commit run --all-files
