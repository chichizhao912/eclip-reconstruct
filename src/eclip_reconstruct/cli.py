"""Command-line interface for preparation, training, and candidate calling."""

from __future__ import annotations

import argparse
from pathlib import Path

from .caller import calibrate, call
from .data import prepare
from .training import predict, train


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="eclip-reconstruct")
    sub = command.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare", help="extract 300 nt coverage and sequence windows")
    prep.add_argument("--windows", type=Path, required=True, help="tab-delimited window manifest")
    prep.add_argument("--fasta", type=Path, required=True, help="indexed reference FASTA")
    for track in ("ip1", "ip2", "sminput"):
        prep.add_argument(f"--{track}-plus", type=Path, required=True)
        prep.add_argument(f"--{track}-minus", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)

    training = sub.add_parser("train", help="train the selected paired-replicate model")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    training.add_argument("--device", default="auto")
    training.add_argument("--epochs", type=int, default=8)
    training.add_argument("--batch-size", type=int, default=64)

    prediction = sub.add_parser("predict", help="reconstruct both IP directions")
    prediction.add_argument("--data", type=Path, required=True)
    prediction.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    prediction.add_argument("--output", type=Path, required=True)
    prediction.add_argument("--device", default="auto")
    prediction.add_argument("--batch-size", type=int, default=64)

    calibration = sub.add_parser("calibrate", help="fit the hard-background score cutoff")
    calibration.add_argument("--predictions", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--profile", choices=("consensus", "p1", "p2"),
                             default="consensus")

    calling = sub.add_parser("call", help="write refined, consolidated candidate intervals")
    calling.add_argument("--predictions", type=Path, required=True)
    calling.add_argument("--threshold", type=Path, required=True)
    calling.add_argument("--output", type=Path, required=True)
    calling.add_argument("--split", choices=("all", "train", "validation", "test"),
                         default="all")
    calling.add_argument("--profile", choices=("consensus", "p1", "p2"),
                         default="consensus")
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        tracks = {name: (getattr(args, f"{name}_plus"), getattr(args, f"{name}_minus"))
                  for name in ("ip1", "ip2", "sminput")}
        prepare(args.windows, args.fasta, tracks, args.output)
        print(f"Prepared windows: {args.output}")
    elif args.command == "train":
        checkpoints = train(args.data, args.output, args.seeds,
                            device_name=args.device, maximum_epochs=args.epochs,
                            batch_size=args.batch_size)
        for checkpoint in checkpoints:
            print(f"Trained model: {checkpoint}")
    elif args.command == "predict":
        predict(args.data, args.checkpoint, args.output,
                batch_size=args.batch_size, device_name=args.device)
        print(f"Predictions: {args.output}")
    elif args.command == "calibrate":
        fitted = calibrate(args.predictions, args.output, args.profile)
        print(f"Threshold: {fitted['threshold']:.6g} ({args.output})")
    elif args.command == "call":
        calls = call(args.predictions, args.threshold, args.output, args.split,
                     args.profile)
        print(f"Candidates: {len(calls)} ({args.output})")


if __name__ == "__main__":
    main()

