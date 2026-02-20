# MI-Entanglement TDC — Local targets (requires uv: https://docs.astral.sh/uv/)

BENCHMARK ?= CYP3A4_Substrate_CarbonMangels
MAX_QUBITS ?= 28
LOCAL_DEVICE ?= lightning.qubit
LOCAL_WORKERS ?= $(shell n=$$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4); w=$$((n - 2)); [ $$w -lt 1 ] && w=1; echo $$w)

.PHONY: sync sync-gpu test test-all features run run-baseline list-benchmarks

sync:
	uv sync --extra dev
	uv pip install --no-deps PyTDC

sync-gpu:
	uv sync --extra dev
	uv pip install --no-deps PyTDC
	uv run python -m pip install pennylane-lightning-gpu

test:
	uv run pytest tests/ -m 'not gpu' -v

test-all:
	uv run pytest tests/ -v

features:
	uv run python src/run_experiment.py \
		--generate-features \
		--benchmark $(BENCHMARK) \
		--cache-dir cache \
		--data-path data \
		--n-workers $(LOCAL_WORKERS)

run:
	uv run python src/run_experiment.py \
		--benchmark $(BENCHMARK) \
		--device $(LOCAL_DEVICE) \
		--cache-dir cache \
		--output-json results/results_$(BENCHMARK).json \
		--data-path data \
		--configs-dir configs \
		--max-qubits $(MAX_QUBITS) \
		--n-workers $(LOCAL_WORKERS)

run-baseline:
	uv run python src/run_experiment.py \
		--benchmark $(BENCHMARK) \
		--device $(LOCAL_DEVICE) \
		--cache-dir cache \
		--output-json results/baseline_$(BENCHMARK).json \
		--data-path data \
		--configs-dir configs \
		--baseline \
		--n-workers $(LOCAL_WORKERS)

# Benchmarks supported by the pipeline (roc-auc / pr-auc / mae metrics)
BENCHMARKS := \
	Caco2_Wang \
	HIA_Hou \
	Pgp_Broccatelli \
	Bioavailability_Ma \
	Lipophilicity_AstraZeneca \
	Solubility_AqSolDB \
	BBB_Martins \
	PPBR_AZ \
	CYP2C9_Veith \
	CYP2D6_Veith \
	CYP3A4_Veith \
	CYP2C9_Substrate_CarbonMangels \
	CYP2D6_Substrate_CarbonMangels \
	CYP3A4_Substrate_CarbonMangels \
	LD50_Zhu \
	hERG \
	AMES \
	DILI

list-benchmarks:
	@echo "TDC ADMET Benchmarks"
	@echo "===================="
	@for b in $(BENCHMARKS); do \
		lc=$$(echo "$$b" | tr '[:upper:]' '[:lower:]'); \
		if [ -f configs/$$lc.json ]; then \
			echo "  $$b  [config saved]"; \
		else \
			echo "  $$b"; \
		fi; \
	done
