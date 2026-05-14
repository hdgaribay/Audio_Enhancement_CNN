"""
scripts/plot_spectrogram_examples.py

Generate paper-ready noisy, clean, and enhanced spectrogram examples from a
trained CNNDenoiser checkpoint. By default this uses the epoch 15 checkpoint
and saves both PDF and PNG figures for easy Overleaf embedding.

Usage:
    python scripts/plot_spectrogram_examples.py

    python scripts/plot_spectrogram_examples.py \
        --config config.yaml \
        --checkpoint checkpoints/epoch_15.pt \
        --num-samples 3

Outputs:
    outputs/figures/spectrogram_examples/spectrogram_examples_epoch_15.pdf
    outputs/figures/spectrogram_examples/spectrogram_examples_epoch_15.png
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SpeechDataset
from src.models.cnn_denoiser import CNNDenoiser


def denormalize_lps(lps_tensor, mean, std):
    return lps_tensor.cpu().numpy() * std + mean


def lps_to_db(lps):
    """Convert natural-log power spectrogram to decibels."""
    return 10.0 * lps / np.log(10.0)


def checkpoint_epoch_label(checkpoint_path):
    match = re.search(r"epoch[_-]?(\d+)", Path(checkpoint_path).stem)
    if match:
        return f"epoch_{match.group(1)}"
    return Path(checkpoint_path).stem


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(device_arg)


def load_model(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint."
        )

    model = CNNDenoiser().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_indices(dataset_size, num_samples, seed, indices):
    if indices:
        bad = [idx for idx in indices if idx < 0 or idx >= dataset_size]
        if bad:
            raise ValueError(f"Sample indices out of range for dataset size {dataset_size}: {bad}")
        return indices

    count = min(num_samples, dataset_size)
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_size, size=count, replace=False).tolist()


def get_sample_id(dataset, idx):
    row = dataset.data.iloc[idx]
    if "id" in row:
        return str(row["id"])
    clean_path = Path(row["clean_path"])
    return clean_path.stem


def collect_examples(dataset, model, indices, device):
    examples = []

    with torch.no_grad():
        for idx in indices:
            noisy_lps, _noisy_phase, clean_lps, lps_mean, lps_std = dataset[idx]
            mean = float(lps_mean.item())
            std = float(lps_std.item())

            noisy_input = noisy_lps.unsqueeze(0).unsqueeze(0).to(device)
            enhanced_lps = model(noisy_input).squeeze(0).squeeze(0)

            noisy_db = lps_to_db(denormalize_lps(noisy_lps, mean, std))
            clean_db = lps_to_db(denormalize_lps(clean_lps, mean, std))
            enhanced_db = lps_to_db(denormalize_lps(enhanced_lps, mean, std))

            examples.append(
                {
                    "id": get_sample_id(dataset, idx),
                    "noisy": noisy_db,
                    "clean": clean_db,
                    "enhanced": enhanced_db,
                }
            )

    return examples


def plot_examples(examples, sample_rate, hop_length, output_stem, dpi):
    n_rows = len(examples)
    panel_names = ["Noisy", "Clean", "Enhanced"]
    keys = ["noisy", "clean", "enhanced"]

    all_values = np.concatenate(
        [example[key].ravel() for example in examples for key in keys]
    )
    vmin, vmax = np.percentile(all_values, [2, 98])

    fig_width = 7.1
    fig_height = max(2.2, 1.75 * n_rows)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    last_image = None
    for row_idx, example in enumerate(examples):
        for col_idx, (key, panel_name) in enumerate(zip(keys, panel_names)):
            ax = axes[row_idx][col_idx]
            spectrogram = example[key]
            freq_bins, frames = spectrogram.shape
            duration = frames * hop_length / sample_rate
            max_freq_khz = sample_rate / 2000.0

            last_image = ax.imshow(
                spectrogram,
                origin="lower",
                aspect="auto",
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                extent=[0, duration, 0, max_freq_khz],
            )

            if row_idx == 0:
                ax.set_title(panel_name, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{example['id']}\nFrequency (kHz)", fontsize=8)
            else:
                ax.set_yticklabels([])
            if row_idx == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)
            else:
                ax.set_xticklabels([])

            ax.tick_params(axis="both", labelsize=7, length=2)

    colorbar = fig.colorbar(last_image, ax=axes, shrink=0.9, pad=0.01)
    colorbar.set_label("Log power (dB)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)

    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return pdf_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot noisy, clean, and enhanced spectrogram examples."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/epoch_15.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--indices", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures/spectrogram_examples",
    )
    parser.add_argument("--output-stem", type=str, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    dataset = SpeechDataset(config["manifest"], split=args.split)
    indices = select_indices(len(dataset), args.num_samples, args.seed, args.indices)
    examples = collect_examples(dataset, model, indices, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_stem:
        output_stem = output_dir / args.output_stem
    else:
        output_stem = output_dir / f"spectrogram_examples_{checkpoint_epoch_label(args.checkpoint)}"

    pdf_path, png_path = plot_examples(
        examples=examples,
        sample_rate=int(config["sample_rate"]),
        hop_length=int(config["hop_length"]),
        output_stem=output_stem,
        dpi=args.dpi,
    )

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")
    print("Sample indices:", ", ".join(str(idx) for idx in indices))


if __name__ == "__main__":
    main()
