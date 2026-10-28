# eCLIP reconstruction and candidate calling

Biological eCLIP replicates can differ in their local coverage patterns and reported peak calls, making candidate regions difficult to assess from either replicate alone. This Python tool reconstructs nucleotide-resolution coverage from two immunoprecipitation replicates (IP1 and IP2), size-matched input (SMInput), and local reference sequence. A background-calibrated caller then scores the reconstructed consensus and reports refined candidate intervals.

The design is inspired by [Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html): one replicate serves as the training target for the other, without treating either as a clean reference. Shared network weights predict IP1 from IP2 and IP2 from IP1, trained with paired Huber losses and a latent-agreement penalty. Candidate calling follows reconstruction, producing prioritised regions rather than validated binding sites or calibrated binding probabilities.

## Install

Python 3.10 or newer is required. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
eclip-reconstruct --help
```

The workflow was tested with Python 3.10.20, PyTorch 2.10.0, and NumPy 2.2.6 on an Apple-silicon MacBook Pro. Training runs on CPU, CUDA, or Apple MPS when available.

## Inputs

Prepare a tab-delimited window manifest with this header:

```text
window_id  chrom  start  end  strand  split  window_class
```

Use tabs between fields. Coordinates are zero-based and half-open, and every window must be 300 nt. `strand` is `+` or `-`. `split` is `train`, `validation`, or `test`. The training and validation chromosomes must be disjoint. Mark the validation windows used to fit the caller threshold as `hard_background` in `window_class`. Other class names are retained but do not affect training or prediction. The caller does not treat hard-background windows as confirmed negatives.

Provide six strand-specific bigWig tracks: positive and negative strands for IP1, IP2, and SMInput. Provide the reference genome as an indexed FASTA with matching chromosome names and a `.fai` index. The same genome assembly and coordinate convention must be used for every input. Published CLIPper peaks may be used to nominate windows, but they are not model targets.

## Quick start

Replace the example paths with your own coverage tracks, reference genome, and window manifest.

```bash
eclip-reconstruct prepare \
  --windows windows.tsv --fasta GRCh38.fa \
  --ip1-plus ip1.plus.bw --ip1-minus ip1.minus.bw \
  --ip2-plus ip2.plus.bw --ip2-minus ip2.minus.bw \
  --sminput-plus sminput.plus.bw --sminput-minus sminput.minus.bw \
  --output outputs/windows.npz

eclip-reconstruct train --data outputs/windows.npz \
  --output outputs/models --seeds 42 1337 2026

eclip-reconstruct predict --data outputs/windows.npz \
  --checkpoint outputs/models/model_seed42.pt \
  --output outputs/predictions_seed42.npz

eclip-reconstruct calibrate --predictions outputs/predictions_seed42.npz \
  --output outputs/threshold_seed42.json

eclip-reconstruct call --predictions outputs/predictions_seed42.npz \
  --threshold outputs/threshold_seed42.json --split test \
  --output outputs/candidates_seed42.bed
```

For an ensemble, pass several checkpoints to `predict`. The directional profiles are averaged across checkpoints before forming the consensus. Calibrate a new threshold on predictions from that same ensemble. For analyses that compare individual model seeds, keep the seed-specific predictions, thresholds, and calls separate.

The default caller uses the consensus profile. To compare directional calls, use `--profile p1` and `--profile p2` in separate `calibrate` and `call` commands. Each direction receives its own validation-calibrated threshold.

## Method and output

`prepare` extracts coverage and sequence in transcript-strand orientation. Non-finite and negative coverage values are set to zero. The sequence is encoded as four A/C/G/T channels, with ambiguous bases encoded as zero in all four. Training fits a common 99.9th-percentile cap to `log1p` coverage using training windows only, then applies it to all partitions.

The network has 45,281 trainable parameters. The default training settings are Adam with learning rate 0.001, batch size 64, at most eight epochs, and early stopping after two non-improving validation epochs. The loss combines the two directional Huber reconstruction terms with a 0.05-weighted mean-squared latent-agreement term. Training writes model checkpoints, a preprocessing record, and a tab-delimited loss history.

The caller scans 100 nt intervals at 5 nt steps in each 300 nt parent. Each local score is mean reconstructed-consensus coverage minus 0.25 times mean transformed SMInput coverage. The threshold is the empirical 95th percentile of the *highest* score from each validation hard-background parent, using the `higher` quantile convention. A passing parent's highest-scoring scan is refined by Gaussian smoothing (standard deviation 2 nt), subtracting its 10th-percentile local baseline, and retaining the central 80% of positive excess signal. The highest-scoring representative is kept from each cluster of overlapping same-strand calls.

The final output is BED6+4: chromosome, start, end, candidate ID, BED score (set to 0), strand, raw detector score, representative source window ID, a boundary-fallback flag (`1` if the winning 100 nt scan was retained because no positive excess remained), and comma-separated IDs of all overlapping parent windows in that call cluster. The detector score is in transformed-coverage units, not a probability. A threshold fitted from selected hard-background windows is not a transcriptome-wide false-discovery-rate guarantee.

## Related analyses

The companion [eclip-thesis-analysis](https://github.com/chichizhao912/eclip-thesis-analysis) repository provides workflows for evaluating replicate agreement, reconstructed profiles, candidate calls, read-depth robustness, and motif support. To run this tool, supply paired eCLIP coverage tracks, an indexed reference genome, and a window manifest. The `train` command generates model checkpoints from these inputs.

