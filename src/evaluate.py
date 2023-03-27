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
import onnx
import onnxruntime as ort

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

def main(repo_path):
    batch_size = 8
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
    
    channels = 3
    height = 224
    width = 224
    sample_input = torch.cuda.FloatTensor(batch_size, channels, height, width)

    onnx_model_path = repo_path / "model"
    torch.onnx.export(model,sample_input,onnx_model_path,opset_version=12,input_names=['input'],output_names=['output'])
    # Load the ONNX model
    onnx_path = 'C:/Users/nasty/data-version-control/model/model.onnx'
    model = onnx.load(onnx_path)
    ort_session = ort.InferenceSession('model.onnx')
# Check that the IR is well formed
    onnx.checker.check_model(model)
    outputs = ort_session.run(
    None,
    {'input': np.random.randn(batch_size, channels, height, width).astype(np.float32)}
)



if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)

