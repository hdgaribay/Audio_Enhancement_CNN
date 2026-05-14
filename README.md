# Audio_Enhancement_CNN
This is a read-me for the audio-enhancement-cnn project for Dr. Ozer Introduction to Neural Nets Class,
Spring 2026. 

This project is a convolutional nerual network that removes background noise from speech recordings. 
The model is trained on the VoiceBank-DEMAND and learns to map noisy speech back to clean speech by
working in the frequency domain. 

Requriements
Pythong 3.11
PyTorch 2.3 + torchaudio 2.3
CUDA 12.1
see environment.yml for full dependency list

How it works

Raw audio is convereted into a spectrogram - a 2D image of frequency content over time - using
short-time fourier transform. The CNN treats this like an image and learns to suppress noise patterns,
producing an enhanced spectrogram that is then convereted back into a .wav file via the inverse STFT

Setup:
1. Clone the Repository
2. Create Conda environement
3. Download the Dataset (hosted on huggingface)
4. Extract audio files
  The downloaded dataset is stored in a compressed format. This script converts it into an individual
  .wav file:
python scripts/extract_dataset.py
5. Build the dataset manifest
python scripts/make_manifest.py

Training:
use python scripts/train_cnn.py
or to specify a custom config:
python scripts/train_cnn.py --config config.yaml

Checkpoints are saved to checkpoints/ every epoch. 

Evaluation

Perceptual metrics (PESQ and STOI):
Evaulates the model on held-out test set and prints PESQ and STOI scores for the CNN output.

python scripts/eval_models.py --config config.yaml --checkpoint checkpoints/epoch_15.pt

Dataset sanity check
loads 10 random audio pairs, prints sample rate and duration, and saves their spectrogram to 
outputs/figures/. Useful for verifying the datset was extracted correctly before training

python scripts/sanity_check.py


Dataset
Voicebank-DEMAND is a standard benchmark dataset for speech enhancement research
Clean speech: recordings from VoiceBank
Noisy speech: clean speech mixed with real-world noise coinditions from DEMAND databse at various
SNR levels
Sample rate: 16,000 Hz
