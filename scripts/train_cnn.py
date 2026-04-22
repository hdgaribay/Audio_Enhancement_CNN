"""
scripts/train_cnn.py

Entry point for training CNNDenoiser.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --config config.yaml
"""
import argparse
import yaml
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)