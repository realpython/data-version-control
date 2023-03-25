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
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def load_images(data_frame, column_name):
    filelist = data_frame[column_name].to_list()
    image_list = imread_collection(filelist)
    return image_list


def load_labels(data_frame, column_name):
    label_list = data_frame[column_name].to_list()
    return label_list


def preprocess(image):
    resized = resize(image, (100, 100, 3))
    reshaped = resized.reshape((1, 30000))
    return reshaped


def load_data(data_path):
    df = pd.read_csv(data_path)
    labels = load_labels(data_frame=df, column_name="label")
    raw_images = load_images(data_frame=df, column_name="filename")
    processed_images = [preprocess(image) for image in raw_images]
    data = np.concatenate(processed_images, axis=0)
    return data, labels

def get_train_valid_loader(data_dir_train,
                           batch_size,
                           augment,
                           random_seed,
                           valid_size=0.5,
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    # define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227,227)),
            transforms.ToTensor(),
            normalize,
        ])


    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir_train, train=True,
        download=True, transform=train_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)


    return train_loader


def get_test_loader(data_dir_test,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.Resize((227,227)),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=data_dir_test, train=False,
        download=True, transform=transform,
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle
    )

    return data_loader


def main(repo_path):
    data_path = repo_path / "data"
    data_dir_train = data_path / "raw/train"
    data_dir_test = data_path / "raw/test"
    num_classes = 10
    num_epochs = 20
    batch_size = 64
    learning_rate = 0.005
    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  
    
    train_loader = get_train_valid_loader(data_dir_train = './data/raw/train',                                      batch_size = 64,
                        augment = False,                             		     random_seed = 1)

    test_loader = get_test_loader(data_dir_test = './data/raw/test',
                                batch_size = 64)
    # Train the model
    total_step = len(train_dataset)
    epoch_time = []
    img_time_pure = []
    for epoch in range(num_epochs):
        train_start = time.time()
        for i, (images, labels) in enumerate(train_loader):  
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
            img_time_pure.append(train_start_pure - time.time())
        epoch_time.append(train_start - time.time())
    epoch_time_avg = epoch_time.mean()
    img_time_pure_avg = img_time_pure.mean()
    epoch_time_pure_avg = img_time_pure_avg*batch_size
    train_time = epoch_time.sum()
    train_time_pure = img_time_pure.sum()
    metrics = {
        {
            "name": "epoch_time_avg",
            "type": "float",
            "value": epoch_time_avg
        },
        {
            "name": "epoch_time_pure_avg",
            "type": "float",
            "value": epoch_time_pure_avg
        },
        {
            "name": "train_time",
            "type": "float",
            "value": train_time
        },
        {
            "name": "train_time_pure",
            "type": "float",
            "value": train_time_pure
        }
    
        }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    trained_model = model
    dump(trained_model, repo_path / "model/model.joblib")

     
if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
