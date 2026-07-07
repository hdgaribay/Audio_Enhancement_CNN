# Audio Enhancement CNN

A deep-learning speech-enhancement project. A fully-convolutional 2D CNN learns to
remove noise from speech by mapping the **noisy log-power spectrogram (LPS)** of an
audio clip to its **clean LPS**. Trained and evaluated on the VoiceBank-DEMAND
dataset, with a classical spectral-subtraction baseline for comparison.

The model works entirely in the magnitude domain: it cleans the spectrogram's
magnitude and reuses the original (noisy) phase to reconstruct audio — a standard,
effective approach for speech enhancement.

---

## How it works

```
raw audio (.wav)
   → load & make mono (audio_io.py)
   → STFT → magnitude + phase (stft.py)
   → magnitude → log-power spectrogram, LPS (stft.py)
   → normalize + crop to fixed length → tensor (dataset.py)
   → CNN maps noisy LPS → clean LPS (cnn_denoiser.py)          [training: train.py]
   → denormalize → LPS→magnitude → ISTFT with noisy phase → audio
   → score with PESQ / STOI (eval_models.py)
```

The evaluation stage is the exact inverse of the data stage: it undoes the
normalization (`* std + mean`), converts LPS back to magnitude (`exp(lps / 2)`,
because `LPS = log(mag²)`), and reconstructs the waveform with the ISTFT using the
saved noisy phase.

---

## Project structure

```
Audio_Enhancement_CNN/
├── config.yaml                     # all settings (STFT params, training hyperparams, paths)
├── dataset_manifest.csv            # generated: table of clean/noisy file pairs
│
├── src/                            # importable building blocks
│   ├── audio_io.py                 # load/save .wav files as numpy arrays
│   ├── stft.py                     # STFT / ISTFT / LPS conversions
│   ├── dataset.py                  # PyTorch Dataset: file pairs → model-ready tensors
│   ├── models/
│   │   └── cnn_denoiser.py         # the fully-convolutional denoising CNN
│   ├── metrics/
│   │   ├── pesq.py                 # PESQ score wrapper
│   │   └── stoi.py                 # STOI score wrapper
│   └── baselines/
│       └── spectral_subtraction.py # classical noise-subtraction baseline
│
└── scripts/                        # entry points you run
    ├── extract_dataset.py          # one-time: parquet → .wav files
    ├── make_manifest.py            # one-time: build dataset_manifest.csv
    ├── train_cnn.py                # train the model
    ├── eval_models.py              # evaluate the trained CNN (PESQ / STOI)
    ├── eval_spectral_subtraction.py# evaluate the baseline
    └── plot_spectrogram_examples.py# generate noisy/clean/enhanced figures
```

**`src/` vs `scripts/`:** files in `src/` are reusable modules imported by other code;
files in `scripts/` are entry points you execute directly.

---

## The model

A fully-convolutional 2D CNN. There is **no spatial downsampling** — every layer keeps
the `(256, 308)` spectrogram shape; only the channel count changes:

```
1 → 16 → 32 → 64 → 32 → 16 → 1      (5×5 convs, BatchNorm, ReLU)
```

Each internal layer is `Conv → BatchNorm → ReLU`; the final layer is a bare conv
(no BatchNorm/ReLU, since the output LPS can be negative).

**Skip (residual) connections** add earlier feature maps back in:

```
layer4 output + layer2 output      (both 32 channels)
layer5 output + layer1 output      (both 16 channels)
```

The symmetric channel counts exist precisely so these additions line up in shape.
Skips let early, detailed features reach the later layers, so the network mostly
preserves structure and learns the *difference* (the noise to remove).

---

## Setup

Requires Python with PyTorch. A conda environment is recommended.

```bash
conda create -n Audio_Enhancement_CNN python=3.10
conda activate Audio_Enhancement_CNN
pip install torch numpy soundfile pyyaml pandas matplotlib tqdm scipy pesq pystoi
```

Run all commands from the **project root** (`Audio_Enhancement_CNN/`).

---

## Usage

### 1. Get the data (run once)

Download the VoiceBank-DEMAND dataset:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='JacobLinCool/VoiceBank-DEMAND-16k',
    repo_type='dataset',
    local_dir='data/voicebank_demand'
)
"
```

Extract the parquet files into `.wav` files, then build the manifest:

```bash
python scripts/extract_dataset.py
python scripts/make_manifest.py
```

This produces `data/voicebank_demand/{train,test}/{clean,noisy}/*.wav` and
`dataset_manifest.csv`.

### 2. Train

```bash
python scripts/train_cnn.py --config config.yaml
```

Trains for the number of epochs in `config.yaml`, printing train/val loss each epoch
and saving checkpoints to `checkpoints/` (per-epoch `epoch_XX.pt`, plus `best.pt`
for the lowest validation loss).

### 3. Evaluate

Evaluate the trained CNN on the test set:

```bash
python scripts/eval_models.py --config config.yaml --checkpoint checkpoints/best.pt
```

Evaluate the classical baseline for comparison:

```bash
python scripts/eval_spectral_subtraction.py --config config.yaml
```

Both print average **PESQ** (perceptual quality, −0.5 to 4.5, higher better) and
**STOI** (intelligibility, 0 to 1, higher better) for the noisy input vs. the
enhanced output.

### 4. Visualize (optional)

```bash
python scripts/plot_spectrogram_examples.py --checkpoint checkpoints/best.pt
```

Saves noisy/clean/enhanced spectrogram figures (PDF + PNG) to
`outputs/figures/spectrogram_examples/`.

---

## Configuration

All settings live in `config.yaml` — the single source of truth. Key values:

| Setting | Meaning |
|---|---|
| `n_fft`, `hop_length`, `win_length` | STFT parameters (define the `(256, 308)` spectrogram shape) |
| `frame_length` | fixed waveform length in samples (30700) |
| `batch_size`, `learning_rate`, `epochs` | training hyperparameters |
| `manifest` | path to `dataset_manifest.csv` |
| `checkpoint_dir`, `keep_last` | where checkpoints are saved and how many to retain |
| `seed` | random seed for reproducibility |

Change behavior by editing `config.yaml`, not the code.

---

## Notes

- **Sample rate is fixed at 16 kHz.** `audio_io.load_audio` raises an error on any
  other rate.
- **Train/val/test split:** the manifest's `train` rows are split 90/10 into train
  and val (val = first 10%); the `test` rows are the test set.
- **Data augmentation:** training uses a random fixed-length crop each epoch;
  validation and test use a deterministic crop from the start for repeatable results.
- **Phase handling:** the model only enhances magnitude. Reconstruction always uses
  the original noisy phase, since clean phase is unavailable at inference.