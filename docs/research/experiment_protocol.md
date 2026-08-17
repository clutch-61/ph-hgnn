# Frozen experiment protocol

## Research questions

1. Can PH separate controlled hypergraph pairs that a depth-L incidence HGNN
   maps to the same representation?
2. Is the PH representation complementary rather than redundant with the
   local readout?
3. Does the residual improve real hypergraph-level classification under equal
   splits, search budgets, and approximately matched parameter counts?
4. How do gains and costs change with hop depth, filtration, homology
   dimension, noise, and hyperedge cardinality?

## Data

Primary modern benchmark: RHG-3, RHG-10, IMDB-Dir-Form, IMDB-Dir-Genre,
Steam-Player, and Twitter-Friend, using fixed stratified 80/10/10 splits.

Replication benchmark: Highschool, Primary, Makam, and BBC with the original
10-fold protocol when redistribution and preprocessing are verified.

Diagnostic benchmark: paired synthetic hypergraphs with controlled local
incidence neighborhoods and differing global Betti/overlap structure.

## Metrics and statistics

- Accuracy and macro-F1 for every dataset.
- Mean, standard deviation, and 95% confidence interval across at least five
  seeds; ten seeds for final reported results.
- Paired seed-level tests for the proposed model versus its exact backbone.
- PH preprocessing wall time, epoch time, inference time, peak GPU memory, and
  cache size.

## Model comparisons

- PH-only: five statistics, persistence image, and DeepSets encoder.
- Local-only: HGNN, HNHN, UniGCN/UniGIN, AllDeepSets/AllSetTransformer.
- Hybrid: concatenation, addition, and gated residual.
- External: HIC, HyperGCN, DPHGNN, TF-HNN, and the strongest reproducible
  DHG-Bench methods.

## Required ablations

- Depth L in 1, 2, 3, 4.
- H0, H1, and H0+H1.
- SCC, ResBS, RelBS, and incidence-bipartite control.
- Static versus learned filtration after the static milestone passes.
- Random diagrams, shuffled diagrams, and parameter-matched non-topological
  features as negative controls.
- Gate location and encoder capacity.

## Leakage prevention

Dataset splitting occurs before any learned normalization or model selection.
Filtration hyperparameters, diagram normalization, early stopping, and
architecture selection use training/validation data only. Test metrics are
computed after configuration freezing and are never used to select a method.
