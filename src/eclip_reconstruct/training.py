"""Paired-replicate training and prediction."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import fit_cap, load_data, transform
from .model import CoverageAutoencoder, model_input


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loader(data: dict[str, np.ndarray], indices: np.ndarray,
            cap: float, batch_size: int, shuffle: bool) -> DataLoader:
    arrays = [torch.from_numpy(transform(data[name][indices], cap))
              for name in ("ip1", "ip2", "sminput")]
    arrays.append(torch.from_numpy(data["sequence"][indices].astype(np.float32)))
    return DataLoader(TensorDataset(*arrays), batch_size=batch_size, shuffle=shuffle)


def paired_loss(model: CoverageAutoencoder, batch: tuple[torch.Tensor, ...],
                device: torch.device, latent_weight: float) -> torch.Tensor:
    ip1, ip2, smi, sequence = (item.to(device) for item in batch)
    p2, z1 = model(model_input(sequence, ip1, smi))
    p1, z2 = model(model_input(sequence, ip2, smi))
    reconstruction = nn.functional.huber_loss(p1[:, 0], ip1)
    reconstruction += nn.functional.huber_loss(p2[:, 0], ip2)
    return reconstruction + latent_weight * nn.functional.mse_loss(z1, z2)


def train(data_path: Path, output: Path, seeds: list[int], *,
          learning_rate: float = 0.001, batch_size: int = 64,
          maximum_epochs: int = 8, patience: int = 2,
          latent_weight: float = 0.05, dropout: float = 0.10,
          device_name: str = "auto") -> list[Path]:
    data = load_data(data_path)
    splits = np.asarray(data["split"])
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "validation")
    if not len(train_idx) or not len(validation_idx):
        raise ValueError("training requires train and validation windows")
    train_chrom = set(data["chrom"][train_idx])
    validation_chrom = set(data["chrom"][validation_idx])
    if train_chrom & validation_chrom:
        raise ValueError("train and validation windows must be on disjoint chromosomes")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("provide at least one unique training seed")
    cap = fit_cap(data)
    device = select_device(device_name)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocessing.json").write_text(json.dumps({
        "transform": "min(log1p(max(coverage, 0)), cap)",
        "fit_partition": "train",
        "clip_percentile": 99.9,
        "clip_value": cap,
    }, indent=2) + "\n")
    histories = []
    checkpoints = []
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = CoverageAutoencoder(dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best = float("inf")
        best_epoch = 0
        failures = 0
        checkpoint_path = output / f"model_seed{seed}.pt"
        for epoch in range(1, maximum_epochs + 1):
            model.train()
            train_losses = []
            for batch in _loader(data, train_idx, cap, batch_size, True):
                optimizer.zero_grad()
                loss = paired_loss(model, batch, device, latent_weight)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))
            model.eval()
            validation_losses = []
            with torch.no_grad():
                for batch in _loader(data, validation_idx, cap, batch_size, False):
                    loss = paired_loss(model, batch, device, latent_weight)
                    validation_losses.append(float(loss.cpu()))
            observed = float(np.mean(validation_losses))
            histories.append({"seed": seed, "epoch": epoch,
                              "train_loss": float(np.mean(train_losses)),
                              "validation_loss": observed})
            if observed < best:
                best = observed
                best_epoch = epoch
                failures = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "clip_value": cap,
                    "dropout": dropout,
                    "best_epoch": best_epoch,
                    "validation_loss": best,
                    "input_channels": 6,
                }, checkpoint_path)
            else:
                failures += 1
                if failures >= patience:
                    break
        checkpoints.append(checkpoint_path)
    with (output / "training_history.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=histories[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(histories)
    return checkpoints


def predict(data_path: Path, checkpoints: list[Path], output: Path, *,
            batch_size: int = 64, device_name: str = "auto") -> None:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    data = load_data(data_path)
    device = select_device(device_name)
    predictions = []
    caps = []
    for path in checkpoints:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["input_channels"] != 6:
            raise ValueError("checkpoint does not use the selected six-channel architecture")
        cap = float(saved["clip_value"])
        caps.append(cap)
        model = CoverageAutoencoder(float(saved["dropout"]))
        model.load_state_dict(saved["model_state"])
        model = model.to(device).eval()
        source = [transform(data[name], cap) for name in ("ip1", "ip2", "sminput")]
        sequence = data["sequence"].astype(np.float32)
        p1_parts, p2_parts = [], []
        with torch.no_grad():
            for start in range(0, len(sequence), batch_size):
                end = start + batch_size
                ip1, ip2, smi = (torch.from_numpy(item[start:end]).to(device)
                                 for item in source)
                seq = torch.from_numpy(sequence[start:end]).to(device)
                p1, _ = model(model_input(seq, ip2, smi))
                p2, _ = model(model_input(seq, ip1, smi))
                p1_parts.append(p1[:, 0].cpu().numpy())
                p2_parts.append(p2[:, 0].cpu().numpy())
        predictions.append((np.concatenate(p1_parts), np.concatenate(p2_parts)))
    if not np.allclose(caps, caps[0], rtol=0, atol=1e-8):
        raise ValueError("checkpoints do not share the same preprocessing cap")
    p1 = np.mean([pair[0] for pair in predictions], axis=0).astype(np.float32)
    p2 = np.mean([pair[1] for pair in predictions], axis=0).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        p1=p1, p2=p2, consensus=(p1 + p2) / 2,
        sminput=transform(data["sminput"], caps[0]),
        **{name: data[name] for name in ("window_id", "chrom", "start", "end",
                                          "strand", "split", "window_class")},
    )

