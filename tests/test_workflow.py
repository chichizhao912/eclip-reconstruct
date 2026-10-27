from __future__ import annotations

import json

import numpy as np
import pyBigWig
import pysam
import torch

from eclip_reconstruct.caller import (_representatives, call, calibrate,
                                     genomic_coordinates, refine)
from eclip_reconstruct.data import load_data, one_hot, prepare
from eclip_reconstruct.model import CoverageAutoencoder
from eclip_reconstruct.training import predict, train


def synthetic_data(path):
    rng = np.random.default_rng(4)
    positions = np.arange(300)
    peak = np.exp(-0.5 * ((positions - 150) / 20) ** 2)
    base = np.stack([peak * (1 + i / 10) for i in range(6)])
    ip1 = np.maximum(base + rng.normal(0, 0.015, base.shape), 0).astype("float32")
    ip2 = np.maximum(base + rng.normal(0, 0.015, base.shape), 0).astype("float32")
    sminput = np.full_like(ip1, 0.03)
    sequence = np.stack([one_hot("ACGT" * 75) for _ in range(6)])
    np.savez_compressed(
        path, ip1=ip1, ip2=ip2, sminput=sminput, sequence=sequence,
        window_id=np.array([f"w{i}" for i in range(6)]),
        chrom=np.array(["chr1", "chr1", "chr2", "chr2", "chr3", "chr3"]),
        start=np.array([0, 300, 0, 300, 0, 300]),
        end=np.array([300, 600, 300, 600, 300, 600]),
        strand=np.array(["+", "-", "+", "-", "+", "-"]),
        split=np.array(["train", "train", "validation", "validation", "test", "test"]),
        window_class=np.array(["peak", "peak", "hard_background",
                               "hard_background", "peak", "peak"]),
    )


def test_model_and_train_predict_smoke(tmp_path):
    model = CoverageAutoencoder()
    assert sum(parameter.numel() for parameter in model.parameters()) == 45281
    with torch.no_grad():
        output, latent = model(torch.zeros(2, 6, 300))
    assert output.shape == (2, 1, 300)
    assert latent.shape == (2, 64, 75)
    assert torch.all(output >= 0)

    dataset = tmp_path / "windows.npz"
    synthetic_data(dataset)
    checkpoints = train(dataset, tmp_path / "models", [42],
                        maximum_epochs=1, batch_size=2, device_name="cpu")
    prediction_path = tmp_path / "predictions.npz"
    predict(dataset, checkpoints, prediction_path, batch_size=2, device_name="cpu")
    with np.load(prediction_path, allow_pickle=False) as result:
        assert result["consensus"].shape == (6, 300)
        assert np.allclose(result["consensus"], (result["p1"] + result["p2"]) / 2)
    threshold_path = tmp_path / "threshold.json"
    calibrate(prediction_path, threshold_path)
    assert json.loads(threshold_path.read_text())["n_hard_background_windows"] == 2
    call(prediction_path, threshold_path, tmp_path / "calls.bed", split="test")
    assert (tmp_path / "calls.bed").exists()


def test_refinement_and_negative_strand_coordinates():
    positions = np.arange(300)
    profile = np.exp(-0.5 * ((positions - 148) / 13) ** 2)
    start, end, fallback = refine(profile, 100)
    assert not fallback
    assert 100 <= start < 148 < end <= 200
    plus = genomic_coordinates(1000, 1300, "+", start, end)
    minus = genomic_coordinates(1000, 1300, "-", start, end)
    assert plus == (1000 + start, 1000 + end)
    assert minus == (1300 - end, 1300 - start)


def test_prepare_orients_coverage_and_sequence(tmp_path):
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">chr1\n" + "A" * 300 + "C" * 300 + "\n")
    pysam.faidx(str(fasta))
    windows = tmp_path / "windows.tsv"
    windows.write_text(
        "window_id\tchrom\tstart\tend\tstrand\tsplit\twindow_class\n"
        "plus\tchr1\t0\t300\t+\ttrain\tpeak\n"
        "minus\tchr1\t300\t600\t-\tvalidation\thard_background\n"
    )
    tracks = {}
    for name in ("ip1", "ip2", "sminput"):
        pair = []
        for strand in ("plus", "minus"):
            path = tmp_path / f"{name}.{strand}.bw"
            with pyBigWig.open(str(path), "w") as handle:
                handle.addHeader([("chr1", 600)])
                handle.addEntries(["chr1"] * 600, list(range(600)),
                                  ends=list(range(1, 601)),
                                  values=[float(i + 1) for i in range(600)])
            pair.append(path)
        tracks[name] = tuple(pair)
    output = tmp_path / "prepared.npz"
    prepare(windows, fasta, tracks, output)
    data = load_data(output)
    assert data["ip1"][0, 0] == 1
    assert data["ip1"][1, 0] == 600
    assert data["sequence"][0, 0].sum() == 300
    assert data["sequence"][1, 2].sum() == 300


def test_consolidation_keeps_strands_separate_and_parent_provenance():
    calls = [
        {"chrom": "chr3", "strand": "+", "start": 100, "end": 150,
         "window_id": "a", "detector_score": 2.0},
        {"chrom": "chr3", "strand": "+", "start": 140, "end": 170,
         "window_id": "b", "detector_score": 3.0},
        {"chrom": "chr3", "strand": "-", "start": 140, "end": 170,
         "window_id": "c", "detector_score": 1.0},
    ]
    chosen = _representatives(calls)
    assert len(chosen) == 2
    plus = next(item for item in chosen if item["strand"] == "+")
    assert plus["window_id"] == "b"
    assert plus["supporting_window_ids"] == "a,b"

