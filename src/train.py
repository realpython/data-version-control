from joblib import dump
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(repo_path):
    data_path = repo_path / "data"
    data_dir_train = data_path / "hymenoptera_data/train"
    num_classes = 2
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.001

    
    #loading the dataset
    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    #loading the dataset
    train_dataset = torchvision.datasets.ImageFolder(root=data_dir_train, transform=transform)

    #loading the data into dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    total_step = len(train_dataset)
    epoch_time = []
    epoch_time_pure = []
    img_time_pure = []
    
    model_ft = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    
    model = model_ft.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)  
    for epoch in range(num_epochs):
        train_start = time.time()
        for images, labels in train_loader:
              
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            train_start_pure = time.time()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            img_time_pure.append(time.time() - train_start_pure)
        epoch_time.append(time.time() - train_start)
        epoch_time_pure.append(np.sum(img_time_pure))
        print ('Epoch [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, loss.item()))
    epoch_time_avg = np.mean(epoch_time)
    img_time_pure_avg = np.mean(img_time_pure)
    epoch_time_pure_avg = np.mean(epoch_time_pure)
    train_time = np.sum(epoch_time)
    train_time_pure = np.sum(epoch_time_pure)
    metrics = {
        
            "epoch_time_avg": epoch_time_avg,
            
            "epoch_time_pure_avg": epoch_time_pure_avg,
            "train_time": train_time,
            "train_time_pure": train_time_pure
        }
    

    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    trained_model = model
    dump(trained_model, repo_path / "model/model.pkl")
     
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
