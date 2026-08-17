# Scientific positioning

## Falsifiable claim

PH-HGNN tests whether hypergraph-native persistent homology supplies global,
multi-scale invariants that a fixed-depth incidence message-passing network
does not expose at graph readout. The claim is conditional: gains should occur
when labels depend on connectivity, cycles, or hyperedge-overlap patterns
outside the backbone's effective incidence-hop radius.

The first model is deliberately a static-PH residual:

`local readout + learned gate * encoded persistence diagrams`.

This isolates the information question before adding differentiable
filtrations. A learned diagram encoder is used so the comparison is not
restricted to five hand-designed barcode statistics.

## Novelty boundary

- Aktas et al. define SCC, ResBS, and RelBS filtrations for hypergraph
  classification, but use fixed barcode statistics and Random Forest rather
  than a native HGNN hybrid.
- TOGL and CliquePH inject persistent topology into ordinary graph neural
  networks, not hypergraph-level classifiers operating on native incidence.
- HIC and DHG-Bench establish modern hypergraph-level datasets and neural
  baselines but do not use persistent homology.
- THTN adds structural encodings and attention for node classification; these
  are not persistent, multi-scale hypergraph invariants.
- WidthWall/InvNet is the closest 2026 neighbor: it adds global pattern
  invariants beyond bounded-width local HGNNs. PH-HGNN must compare against
  this framing and must not claim that all global invariant methods are new.

## Claims allowed before stronger proofs

1. The complete classifier is invariant to vertex and hyperedge relabeling
   when its filtration is invariant/equivariant and readouts are symmetric.
2. Specific witness pairs can be indistinguishable to a selected depth-L
   backbone while having different persistence diagrams.
3. On specified datasets and perturbations, the PH residual improves measured
   accuracy, macro-F1, or stability under a controlled tuning budget.

## Claims not allowed

- PH always recovers all information lost by finite-hop HGNNs.
- The current RelBS implementation is equivalent to arbitrary-dimensional
  relative persistent homology; the initial implementation targets H0/H1.
- A performance gain alone proves an expressivity theorem.
- Results on synthetic RHG datasets establish general real-world superiority.
