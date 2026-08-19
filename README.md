# PH-HGNN

Research framework for testing whether hypergraph-native persistent homology
complements finite-hop hypergraph neural networks on hypergraph-level
classification.

The first milestone combines a native incidence HGNN readout with cached H0/H1
persistence diagrams through a learnable gated residual. It supports PH-only,
HGNN-only, concatenation, addition, and gated variants under one training
pipeline.

## Architecture

```text
hypergraph -> finite-hop incidence HGNN ---------\
                                                  gated residual -> classifier
hypergraph -> SCC/ResBS/RelBS -> H0/H1 -> DeepSet/
```

The implementation includes:

- safe adapters for six HIC/DHG-Bench hypergraph-level datasets plus Aktas's
  four datasets (Highschool/Primary/Makam/BBC);
- deterministic stratified 80/10/10 splits for HIC/DHG-Bench and 10-fold CV
  protocol for Aktas datasets;
- SCC, restricted barycentric, relative barycentric H0/H1, and incidence
  control constructions;
- atomic, content-addressed persistence cache;
- HGNN, HNHN, UniGCN, UniGIN, and AllDeepSets-style backbones;
- learned persistence DeepSets and Aktas-style five-statistic encoder;
- local-only, PH-only, concat, add, and gated residual experiments;
- a three-class RHG-style toy dataset for CPU smoke tests.
- a preregistered local/global witness dataset (one cycle versus two cycles)
  whose depth-L rooted incidence views match while H0 differs.

## Server installation

Python 3.11 and `uv` are recommended on the implementation server.

```bash
uv sync --extra dev
```

This decision-side checkout does not require a local Python environment.
Install the CUDA-compatible PyTorch build on the server, then change `device`
in a copied configuration rather than editing the CPU-safe test defaults.

## Verify

```bash
uv run ruff check .
uv run mypy src/ph_hgnn
uv run pytest
uv run ph-hgnn --quick
```

The smoke run writes a traceable `runs/rhg_toy/<timestamp>-seed0/metrics.json`
containing configuration, git SHA, validation history, test metrics, and wall
time. Model checkpoints, runs, caches, PDFs, and datasets are ignored by git.

## Run ablations

Copy `configs/base.yaml`, then vary:

- `fusion`: `hgnn`, `ph`, `concat`, `add`, `gated`
- `complex_kind`: `scc`, `resbs`, `relbs`, `incidence`
- `num_layers`: `1` through `4`
- `backbone`: `hgnn`, `hnhn`, `unigcn`, `unigin`, `alldeepsets`

Run:

```bash
uv run ph-hgnn --config configs/base.yaml --dataset rhg_toy
uv run ph-hgnn --config configs/base.yaml --dataset local_global_witnesses
uv run ph-hgnn --config configs/base.yaml --dataset rhg_3
uv run ph-hgnn --config configs/base.yaml --dataset highschool --fold 0
```

For real data, follow `data/README.md`. Do not commit upstream datasets or
reference PDFs.

Convert HIC `.txt` files after cloning upstream data locally:

```bash
git clone --depth 1 https://github.com/iMoonLab/HIC.git data/raw/HIC
uv run python scripts/convert_hic_datasets.py --datasets rhg_3 rhg_10
```

Batch runs:

```bash
uv run python scripts/run_sweep.py --dataset rhg_3 --seeds 0 1 2 3 4
uv run python scripts/run_sweep.py --dataset highschool --seeds 0 1 2 --folds 0 1 2 3 4 5 6 7 8 9
uv run python scripts/aggregate_results.py --runs runs --output artifacts/summary.json
```

## Research safeguards

The scientific positioning, evidence map, frozen protocol, and go/no-go gates
are in `docs/research`. In particular, this project does not claim that PH
recovers every invariant unavailable to finite-hop HGNNs. The initial RelBS
implementation targets H0/H1 and must be cross-validated against published
examples before final experiments.

## Upstream provenance

- DeepHypergraph: https://github.com/iMoonLab/DeepHypergraph
- HIC datasets: https://github.com/iMoonLab/HIC
- DHG-Bench: https://github.com/Coco-Hut/DHG-Bench

The code in this repository is MIT-licensed. Upstream code and data retain
their own licenses.
