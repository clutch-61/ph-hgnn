from ph_hgnn.data.synthetic import make_rhg_toy


def test_toy_dataset_has_disjoint_complete_splits() -> None:
    dataset = make_rhg_toy(num_per_class=10, seed=7)
    dataset.validate()
    train = set(dataset.train_indices)
    val = set(dataset.val_indices)
    test = set(dataset.test_indices)
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(range(len(dataset.samples)))


def test_split_is_deterministic() -> None:
    first = make_rhg_toy(num_per_class=10, seed=9)
    second = make_rhg_toy(num_per_class=10, seed=9)
    assert first.train_indices == second.train_indices
    assert first.val_indices == second.val_indices
    assert first.test_indices == second.test_indices
