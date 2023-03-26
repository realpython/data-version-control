from joblib import load
import json
from pathlib import Path

from sklearn.metrics import accuracy_score

from train import load_data
import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def main(repo_path):
    data_path = repo_path / "data"
    data_dir_test = data_path / "hymenoptera_data/val"
    model = load(repo_path / "model/model.joblib")
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    accuracy = correct/total
    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)

