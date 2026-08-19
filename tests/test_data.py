from ph_hgnn.data.datasets import DATASET_SPECS, _split_kfold
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


def test_aktas_datasets_use_kfold10_protocol() -> None:
    for name in ("highschool", "primary", "makam", "bbc"):
        assert DATASET_SPECS[name].split_protocol == "kfold10"


def test_kfold_split_is_disjoint_and_deterministic() -> None:
    labels = [0] * 20 + [1] * 20
    first = _split_kfold(labels=labels, seed=0, fold_index=3, n_splits=10)
    second = _split_kfold(labels=labels, seed=0, fold_index=3, n_splits=10)
    assert first == second
    train, val, test = first
    train_set, val_set, test_set = set(train), set(val), set(test)
    assert not train_set & val_set
    assert not train_set & test_set
    assert not val_set & test_set
    assert train_set | val_set | test_set == set(range(len(labels)))
