# Evidence map

## Hypergraph persistent homology

- **Hypergraph Classification via Persistent Homology**: defines SCC, ResBS,
  and RelBS filtrations; evaluates H0/H1 statistics with Random Forest on
  Highschool, Primary, Makam, and BBC. It establishes PH-only feasibility but
  uses a weak clique-expansion GCN comparison.
- **Persistent hypergraph homology and its applications**: supplies a broader
  mathematical foundation and motivates comparing alternative hypergraph
  homology constructions.

## Persistent topology in graph learning

- **Topological Graph Neural Networks (TOGL)**: direct precedent for PH
  complementing finite-depth graph message passing.
- **On the Expressivity of Persistent Homology in Graph Learning**: motivates
  explicit separation witnesses instead of relying only on benchmark gains.
- **CliquePH**: demonstrates efficient low-dimensional PH on lifted
  higher-order structures and the importance of a learnable diagram encoder.
- **Localized Topological Features**, **Hourglass Persistence**, **TopInG**,
  **TopNets**, **Boosting Graph Pooling with PH**, and **Frequent
  Subgraph-Based PH** define the adjacent design space for locality,
  filtration, pooling, continuity, and interpretability.

## Hypergraph neural learning

- **HGNN/HGNN+**: canonical spectral incidence baselines.
- **HNHN, UniGNN, AllSetTransformer, DPHGNN, TF-HNN**: stronger native
  hypergraph baselines needed to avoid the weak-baseline issue in prior
  hypergraph PH work.
- **HIC**: source of RHG-3, RHG-10, IMDB-Dir-Form, IMDB-Dir-Genre,
  Steam-Player, and Twitter-Friend hypergraph-level datasets.
- **DHG-Bench**: standardized 80/10/10 split, accuracy and macro-F1 protocol,
  and broad baseline results for those six datasets.
- **WidthWall/InvNet (2026 preprint)**: closest global-invariant competitor and
  theoretical warning against an overbroad novelty claim.

## Decision

The defensible paper is not “PH works on hypergraphs.” It is a controlled
study and model showing when a hypergraph-native PH residual supplies
non-redundant information to bounded-hop native HGNNs, supported by witness
pairs, hop-depth interactions, strong baselines, and real-data results.
