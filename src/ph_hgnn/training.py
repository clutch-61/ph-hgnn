from __future__ import annotations

import copy
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from ph_hgnn.models.fusion import PHHGNNClassifier
from ph_hgnn.topology.complexes import ComplexKind
from ph_hgnn.topology.persistence import PersistenceCache
from ph_hgnn.types import DatasetBundle, HypergraphSample


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 30
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 32
    topo_dim: int = 32
    num_layers: int = 2
    backbone: str = "hgnn"
    fusion: str = "gated"
    topology_encoder: str = "deepset"
    complex_kind: str = "scc"
    device: str = "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _on_device(sample: HypergraphSample, device: torch.device) -> HypergraphSample:
    return HypergraphSample(
        sample.sample_id,
        sample.x.to(device),
        sample.incidence.to(device),
        sample.y,
        sample.num_hyperedges,
        sample.vertex_filtration.to(device)
        if sample.vertex_filtration is not None
        else None,
        sample.hyperedge_filtration.to(device)
        if sample.hyperedge_filtration is not None
        else None,
    )


def _evaluate(
    model: PHHGNNClassifier,
    dataset: DatasetBundle,
    indices: tuple[int, ...],
    diagrams: dict[int, dict[int, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for index in indices:
            sample = _on_device(dataset.samples[index], device)
            logits = model(sample, diagrams[index])
            predictions.append(int(logits.argmax()))
            labels.append(sample.y)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def _git_sha(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "uncommitted"


def run_experiment(
    dataset: DatasetBundle,
    config: TrainConfig,
    project_root: Path,
    run_dir: Path,
) -> dict[str, object]:
    seed_everything(config.seed)
    device = torch.device(config.device)
    cache = PersistenceCache(project_root / "artifacts" / "cache" / "persistence")
    preprocessing_started = time.perf_counter()
    diagrams = {
        index: cache.get_or_compute(sample, ComplexKind(config.complex_kind))
        for index, sample in enumerate(dataset.samples)
    }
    preprocessing_seconds = time.perf_counter() - preprocessing_started
    model = PHHGNNClassifier(
        input_dim=dataset.samples[0].x.shape[1],
        num_classes=dataset.num_classes,
        hidden_dim=config.hidden_dim,
        topo_dim=config.topo_dim,
        num_layers=config.num_layers,
        backbone=config.backbone,
        fusion=config.fusion,  # type: ignore[arg-type]
        topology_encoder=config.topology_encoder,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(config.epochs):
        model.train()
        order = list(dataset.train_indices)
        random.shuffle(order)
        total_loss = 0.0
        for index in order:
            sample = _on_device(dataset.samples[index], device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(sample, diagrams[index])
            loss = criterion(logits.unsqueeze(0), torch.tensor([sample.y], device=device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
        validation = _evaluate(model, dataset, dataset.val_indices, diagrams, device)
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": total_loss / len(order),
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
        }
        history.append(epoch_record)
        score = validation["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    inference_started = time.perf_counter()
    test_metrics = _evaluate(model, dataset, dataset.test_indices, diagrams, device)
    inference_seconds = time.perf_counter() - inference_started
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "config": asdict(config),
        "dataset": dataset.name,
        "git_sha": _git_sha(project_root),
        "ph_preprocessing_seconds": preprocessing_seconds,
        "elapsed_seconds": elapsed,
        "test_inference_seconds": inference_seconds,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "best_val_macro_f1": best_score,
        "test": test_metrics,
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    torch.save(best_state, run_dir / "model.pt")
    return result
