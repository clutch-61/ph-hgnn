# Risk register and go/no-go gates

## Scientific risks

- **Weak baseline illusion**: PH may only beat clique-expanded GCN. Mitigation:
  compare native HGNN families and depth-matched variants.
- **Pure PH saturation**: Aktas-style features may already solve a dataset.
  Mitigation: prioritize modern real datasets and analyze representation
  complementarity, not only absolute accuracy.
- **Synthetic-only gain**: RHG classes may align too directly with topology.
  Gate: no general SOTA claim without consistent gains on real datasets.
- **Depth replacement**: deeper HGNN may erase the hybrid gain. This is a
  valid negative result and must be shown through the depth interaction.
- **Filtration selection bias**: choosing SCC/ResBS/RelBS on test data is
  prohibited. Selection is validation-only under a fixed budget.
- **Nearby global-invariant work**: WidthWall/InvNet can undermine novelty.
  Position PH as a stable, interpretable topological coordinate family and
  include it in the closest-related-work discussion.

## Engineering risks

- RelBS quotient details require independent mathematical validation before
  final experiments. The current H0/H1 construction is explicitly documented
  and tested, but must be cross-checked against published examples.
- Clique closure can grow exponentially. Enforce max dimension, hyperedge-size
  caps, timeouts, cache hashes, and per-sample complexity logging.
- Eight-gigabyte laptop GPU memory limits large sweeps. Use CPU PH caching,
  small per-sample graph batches, mixed precision only after correctness, and
  sequential seed scheduling.
- Third-party datasets and code have separate licenses. Keep raw data,
  reference PDFs, and upstream repositories out of git; distribute download
  and conversion instructions with provenance and checksums.

## Gates

1. **Construction gate**: known toy barcodes and permutation tests pass.
2. **Signal gate**: PH separates at least one preregistered local-indistinguishability family.
3. **Hybrid gate**: the gated model beats the exact local backbone across
   seeds without using test-set selection.
4. **Reality gate**: gains appear on at least two real datasets and do not
   depend on one pathological split.
5. **Scale gate**: preprocessing and memory costs are reported and remain
   practical for the target benchmark.

Failure at gates 1–2 triggers filtration redesign. Failure at gates 3–4
changes the paper to a diagnostic/negative-results contribution rather than a
universal performance claim.
