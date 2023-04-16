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
import onnxruntime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

def main(repo_path):
    batch_size = 8
    data_path = repo_path / "data"
    data_dir_test = data_path / "hymenoptera_data/val"
    model = load(repo_path / "model/model.pkl")
    model.eval()
    # convert to onnx model
    dummy_input = torch.randn(batch_size, 3, 224, 224, device=device)
    input_names = ["input"]
    output_names = ["output"]
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    onnx_model_path = repo_path / "model/model.onnx"
    torch.onnx.export(model, dummy_input, onnx_model_path, input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes, opset_version=11)

    # create onnxruntime session
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model_path.as_posix(), sess_options=sess_options)

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
            # use onnxruntime to get predictions
            ort_inputs = {"input": images.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            outputs = torch.Tensor(ort_outs[0]).to(device)
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
    file = __file__
    repo_path = Path(file).parent.parent
    main(repo_path)