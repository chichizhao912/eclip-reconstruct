"""Window extraction and train-fitted coverage transformation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

WINDOW_LENGTH = 300
BASES = "ACGT"
FIELDS = ("window_id", "chrom", "start", "end", "strand", "split", "window_class")
ARRAY_FIELDS = ("ip1", "ip2", "sminput", "sequence", *FIELDS)


def one_hot(sequence: str) -> np.ndarray:
    sequence = sequence.upper()
    return np.stack([(np.fromiter((base == name for base in sequence),
                                  dtype=np.float32)) for name in BASES])


def read_windows(path: Path) -> list[dict[str, str | int]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"window table must contain: {', '.join(FIELDS)}")
        windows = []
        seen = set()
        for row in reader:
            start, end = int(row["start"]), int(row["end"])
            if start < 0 or end - start != WINDOW_LENGTH:
                raise ValueError("each window must be a valid 300 nt half-open interval")
            if row["strand"] not in {"+", "-"}:
                raise ValueError("strand must be + or -")
            if row["window_id"] in seen:
                raise ValueError("window_id must be unique")
            seen.add(row["window_id"])
            windows.append({**row, "start": start, "end": end})
    if not windows:
        raise ValueError("window table is empty")
    return windows


def clean_coverage(values: np.ndarray, strand: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (WINDOW_LENGTH,):
        raise ValueError("coverage extraction did not return 300 values")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0)
    return values if strand == "+" else values[::-1].copy()


def prepare(windows_path: Path, fasta_path: Path,
            track_paths: dict[str, tuple[Path, Path]], output: Path) -> None:
    """Extract strand-oriented coverage and sequence into a portable NPZ."""
    import pyBigWig
    import pysam

    windows = read_windows(windows_path)
    if set(track_paths) != {"ip1", "ip2", "sminput"}:
        raise ValueError("tracks must include IP1, IP2, and SMInput")
    handles = {name: (pyBigWig.open(str(plus)), pyBigWig.open(str(minus)))
               for name, (plus, minus) in track_paths.items()}
    profiles = {name: [] for name in handles}
    sequences = []
    try:
        with pysam.FastaFile(str(fasta_path)) as genome:
            for row in windows:
                chrom, start, end, strand = (row["chrom"], row["start"],
                                              row["end"], row["strand"])
                sequence = genome.fetch(chrom, start, end).upper()
                if len(sequence) != WINDOW_LENGTH:
                    raise ValueError(f"incomplete reference sequence for {row['window_id']}")
                if strand == "-":
                    sequence = sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
                sequences.append(one_hot(sequence))
                for name, (plus, minus) in handles.items():
                    handle = plus if strand == "+" else minus
                    values = handle.values(chrom, start, end, numpy=True)
                    profiles[name].append(clean_coverage(values, strand))
    finally:
        for plus, minus in handles.values():
            plus.close()
            minus.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **{name: np.stack(values) for name, values in profiles.items()},
        sequence=np.stack(sequences),
        **{name: np.asarray([row[name] for row in windows]) for name in FIELDS},
    )


def load_data(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(ARRAY_FIELDS) - set(archive.files)
        if missing:
            raise ValueError(f"missing fields in {path}: {sorted(missing)}")
        data = {name: archive[name] for name in ARRAY_FIELDS}
    count = len(data["ip1"])
    if any(data[name].shape != (count, WINDOW_LENGTH)
           for name in ("ip1", "ip2", "sminput")):
        raise ValueError("coverage arrays must have shape [windows, 300]")
    if data["sequence"].shape != (count, 4, WINDOW_LENGTH):
        raise ValueError("sequence must have shape [windows, 4, 300]")
    if any(len(data[name]) != count for name in FIELDS):
        raise ValueError("window metadata and coverage arrays differ in length")
    return data


def fit_cap(data: dict[str, np.ndarray], percentile: float = 99.9) -> float:
    selected = np.asarray(data["split"]) == "train"
    if not selected.any():
        raise ValueError("at least one training window is required")
    logged = [np.log1p(np.maximum(data[name][selected], 0)).ravel()
              for name in ("ip1", "ip2", "sminput")]
    return float(np.percentile(np.concatenate(logged), percentile))


def transform(values: np.ndarray, cap: float) -> np.ndarray:
    return np.minimum(np.log1p(np.maximum(values, 0)), cap).astype(np.float32)

