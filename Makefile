.PHONY: install prepare-data model-plan tiny-smoke test lint typecheck corpora coverage verify

install:
	python -m pip install -e '.[dev]'

prepare-data:
	PYTHONPATH=src python scripts/retail_bank/prepare_tool_sft_data.py \
		--output-dir data/banking-v3-tool-sft \
		--pilot-count 5000

model-plan:
	PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
		--manifest data/banking-v3-tool-sft/manifest.json

tiny-smoke:
	PYTHONPATH=src python scripts/retail_bank/cloud_train_tool_sft.py \
		--run-tiny-smoke \
		--family granite \
		--max-steps 1 \
		--output-dir artifacts/banking-v3-tool-sft-smoke

test:
	python -m pytest tests/test_banking_*.py poc/retail-bank-customer-service-poc/tests

lint:
	python -m ruff check src scripts tests poc/retail-bank-customer-service-poc

typecheck:
	python -m mypy src scripts tests

corpora:
	PYTHONPATH=src python scripts/retail_bank/check_corpora_reproduce.py

# Use-case coverage of each corpus against configs/corpus-coverage.toml. The
# report ranks declared cells below target -- that is the authoring order.
coverage:
	PYTHONPATH=src python scripts/retail_bank/measure_corpus_coverage.py --corpus router
	PYTHONPATH=src python scripts/retail_bank/measure_corpus_coverage.py --corpus alignment

# The whole gate, in the order that fails cheapest first. There is no hosted CI
# by decision, so this is the enforcement -- run it before pushing.
verify:
	uv lock --check
	uv run ruff check .
	PYTHONPATH=src uv run python -m pytest -q tests
	POC_SKIP_MODEL_LOAD=1 POC_SKIP_ROUTER_LOAD=1 \
		uv run python -m pytest -q poc/retail-bank-customer-service-poc/tests
	PYTHONPATH=src uv run python scripts/retail_bank/check_corpora_reproduce.py
	PYTHONPATH=src uv run python scripts/retail_bank/measure_corpus_coverage.py --corpus router --gate
	PYTHONPATH=src uv run python scripts/retail_bank/measure_corpus_coverage.py --corpus alignment --gate
