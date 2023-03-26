from joblib import load
import json
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.linear_model import SGDClassifier
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(repo_path):
    batch_size = 64
    data_path = repo_path / "data"
    data_dir_test = data_path / "hymenoptera_data/val"
    model = load(repo_path / "model/model.pkl")
       #loading the dataset
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    #loading the dataset
    dataset = torchvision.datasets.ImageFolder(root=data_dir_test, transform=transform)

    #loading the data into dataloader
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
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
    with open('metrics_acc.json', 'w') as f:
        json.dump(metrics, f, indent=5)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)

