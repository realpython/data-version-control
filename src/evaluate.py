from joblib import load
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

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
    eval_time = []
    eval_time_pure = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            val_start = time.time()
            images = images.to(device)
            labels = labels.to(device)
            val_start_pure = time.time()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            eval_time_pure.append(time.time()-val_start_pure)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            eval_time.append(time.time()-val_start)
            del images, labels, outputs
    accuracy = correct/total
    eval_time_avg = np.mean(eval_time)
    eval_time_pure_avg = np.mean(eval_time_pure)
    metrics = {"accuracy": accuracy, "eval_time_avg": eval_time_avg, "eval_time_pure_avg": eval_time_pure_avg}
    with open('metrics_eval.json', 'w') as f:
        json.dump(metrics, f, indent=3)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)

