"""Background-calibrated detection and boundary refinement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

SCAN_LENGTH = 100
SCAN_STRIDE = 5
SMINPUT_WEIGHT = 0.25
BACKGROUND_QUANTILE = 0.95


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    required = {"consensus", "sminput", "window_id", "chrom", "start", "end",
                "strand", "split", "window_class"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"prediction file is missing {sorted(missing)}")
        data = {name: archive[name] for name in required}
    if data["consensus"].shape != data["sminput"].shape:
        raise ValueError("consensus and SMInput profiles must have equal shapes")
    if data["consensus"].ndim != 2 or data["consensus"].shape[1] != 300:
        raise ValueError("predictions must have shape [windows, 300]")
    return data


def local_scores(consensus: np.ndarray, sminput: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean consensus minus 0.25 times mean SMInput in each 100 nt scan."""
    consensus = np.asarray(consensus, dtype=np.float64)
    sminput = np.asarray(sminput, dtype=np.float64)
    if consensus.shape != sminput.shape or consensus.ndim != 2 or consensus.shape[1] != 300:
        raise ValueError("both profiles must have shape [windows, 300]")
    starts = np.arange(0, 300 - SCAN_LENGTH + 1, SCAN_STRIDE)
    net = consensus - SMINPUT_WEIGHT * sminput
    prefix = np.pad(np.cumsum(net, axis=1), ((0, 0), (1, 0)))
    scores = (prefix[:, starts + SCAN_LENGTH] - prefix[:, starts]) / SCAN_LENGTH
    return scores, starts


def calibrate(predictions: Path, output: Path,
              profile: str = "consensus") -> dict[str, float | int | str]:
    data = load_predictions(predictions)
    if profile not in {"consensus", "p1", "p2"}:
        raise ValueError("profile must be consensus, p1, or p2")
    with np.load(predictions, allow_pickle=False) as archive:
        selected_profile = archive[profile]
    background = ((data["split"] == "validation")
                  & (data["window_class"] == "hard_background"))
    if not background.any():
        raise ValueError("no validation hard-background windows for threshold calibration")
    scores, _ = local_scores(selected_profile, data["sminput"])
    parent_maxima = scores[background].max(axis=1)
    threshold = float(np.quantile(parent_maxima, BACKGROUND_QUANTILE, method="higher"))
    result = {
        "threshold": threshold,
        "n_hard_background_windows": int(background.sum()),
        "quantile": BACKGROUND_QUANTILE,
        "quantile_method": "higher",
        "scan_length_nt": SCAN_LENGTH,
        "scan_stride_nt": SCAN_STRIDE,
        "sminput_weight": SMINPUT_WEIGHT,
        "profile": profile,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def refine(net_profile: np.ndarray, scan_start: int) -> tuple[int, int, bool]:
    """Retain the central 80% of positive signal above a local baseline."""
    profile = np.asarray(net_profile, dtype=np.float64)
    if profile.shape != (300,):
        raise ValueError("net profile must contain 300 positions")
    scan_end = scan_start + SCAN_LENGTH
    if not 0 <= scan_start <= 300 - SCAN_LENGTH:
        raise ValueError("scan lies outside parent window")
    smoothed = gaussian_filter1d(profile, 2.0, mode="nearest")
    local = smoothed[scan_start:scan_end]
    baseline = float(np.quantile(local, 0.10))
    excess = np.maximum(local - baseline, 0.0)
    total = float(excess.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return scan_start, scan_end, True
    cumulative = np.cumsum(excess) / total
    start = scan_start + int(np.searchsorted(cumulative, 0.10, side="left"))
    end = scan_start + int(np.searchsorted(cumulative, 0.90, side="left")) + 1
    return start, max(start + 1, min(end, scan_end)), False


def genomic_coordinates(parent_start: int, parent_end: int, strand: str,
                        local_start: int, local_end: int) -> tuple[int, int]:
    if strand == "+":
        start, end = parent_start + local_start, parent_start + local_end
    elif strand == "-":
        start, end = parent_end - local_end, parent_end - local_start
    else:
        raise ValueError("strand must be + or -")
    return int(start), int(end)


def _representatives(calls: list[dict]) -> list[dict]:
    """Keep the highest-scoring call from each positive-overlap component."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for call in calls:
        groups.setdefault((call["chrom"], call["strand"]), []).append(call)
    chosen = []
    for group in groups.values():
        ordered = sorted(group, key=lambda call: (call["start"], call["end"]))
        component = []
        furthest_end = -1
        for call in ordered:
            if component and call["start"] >= furthest_end:
                representative = sorted(component, key=lambda item: (-item["detector_score"],
                                                                      item["start"], item["window_id"]))[0].copy()
                representative["supporting_window_ids"] = ",".join(sorted(
                    item["window_id"] for item in component
                ))
                chosen.append(representative)
                component = []
                furthest_end = -1
            component.append(call)
            furthest_end = max(furthest_end, call["end"])
        if component:
            representative = sorted(component, key=lambda item: (-item["detector_score"],
                                                                  item["start"], item["window_id"]))[0].copy()
            representative["supporting_window_ids"] = ",".join(sorted(
                item["window_id"] for item in component
            ))
            chosen.append(representative)
    return sorted(chosen, key=lambda item: (item["chrom"], item["start"],
                                             item["end"], item["strand"]))


def call(predictions: Path, threshold_path: Path, output: Path,
         split: str = "all", profile: str = "consensus") -> list[dict]:
    data = load_predictions(predictions)
    threshold_spec = json.loads(threshold_path.read_text())
    if profile not in {"consensus", "p1", "p2"}:
        raise ValueError("profile must be consensus, p1, or p2")
    if threshold_spec.get("profile", "consensus") != profile:
        raise ValueError("threshold was calibrated for a different profile")
    with np.load(predictions, allow_pickle=False) as archive:
        selected_profile = archive[profile]
    for name, expected in (("scan_length_nt", SCAN_LENGTH),
                           ("scan_stride_nt", SCAN_STRIDE),
                           ("sminput_weight", SMINPUT_WEIGHT)):
        if threshold_spec.get(name) != expected:
            raise ValueError(f"threshold file uses a different {name}")
    threshold = float(threshold_spec["threshold"])
    scores, starts = local_scores(selected_profile, data["sminput"])
    selected = np.ones(len(scores), dtype=bool) if split == "all" else data["split"] == split
    calls = []
    for index in np.flatnonzero(selected):
        best = int(np.argmax(scores[index]))
        detector_score = float(scores[index, best])
        if detector_score < threshold:
            continue
        net = selected_profile[index] - SMINPUT_WEIGHT * data["sminput"][index]
        local_start, local_end, fallback = refine(net, int(starts[best]))
        start, end = genomic_coordinates(int(data["start"][index]),
                                         int(data["end"][index]),
                                         str(data["strand"][index]),
                                         local_start, local_end)
        calls.append({"chrom": str(data["chrom"][index]), "start": start,
                      "end": end, "strand": str(data["strand"][index]),
                      "window_id": str(data["window_id"][index]),
                      "detector_score": detector_score,
                      "boundary_fallback": fallback})
    chosen = _representatives(calls)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for number, item in enumerate(chosen, 1):
            handle.write("\t".join(map(str, (
                item["chrom"], item["start"], item["end"], f"candidate_{number}",
                0, item["strand"], f"{item['detector_score']:.8g}",
                item["window_id"], int(item["boundary_fallback"]),
                item["supporting_window_ids"],
            ))) + "\n")
    return chosen

