"""How "pure" is a single-class model? Classify its samples and count the classes.

Uses the CIFAR-10 classifier from Experiment 1 only as a judge of finished samples; it plays
no part in sampling. That classifier is about 74% accurate on real images, so these counts are
a rough gauge, not exact.

    python separate_models/purity.py separate_models/results/preview_cat.pt separate_models/results/preview_dog.pt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
sys.path.insert(0, os.path.join(HERE, "..", "classifier_guided"))

import torch

from noisy_classifier import load_classifier
from score_sde_backend import get_device

CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def classify(x, clf, device):
    with torch.no_grad():
        s = torch.full((len(x),), 1e-5, device=device)
        return clf(x.to(device), s).argmax(1).cpu()


if __name__ == "__main__":
    device = get_device()
    clf = load_classifier(os.path.join(HERE, "..", "classifier_guided/checkpoints/cifar10_noisy_classifier.pt"), device)
    for path in sys.argv[1:]:
        x = torch.load(path, weights_only=False)
        pred = classify(x, clf, device)
        counts = torch.bincount(pred, minlength=10).tolist()
        top = sorted(((c, n) for c, n in zip(CIFAR, counts) if n), key=lambda t: -t[1])
        print(f"{path}  (n={len(x)}):  " + ", ".join(f"{c} {100 * n / len(x):.0f}%" for c, n in top))
