from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights

import time

import onnx
import onnxscript

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_image_path = Path('./carpet/train')
good_dataset = ImageFolder(root=train_image_path, transform=transform)

# Requires PyTorch 2.x: train_dataset, test_dataset = torch.utils.data.random_split(good_dataset, [0.8, 0.2])

# Set the batch size
BS = 16
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # disable GPU access
# print(torch.cuda.is_available())

# print torch version
print(torch.__version__)

test_loader = DataLoader(good_dataset, batch_size=BS, shuffle=True)

from AD_3_using_resnet_backbone_multilevel_model import FeatCAE, resnet_feature_extractor

model = FeatCAE(in_channels=1536, latent_dim=100)

backbone = resnet_feature_extractor()
#backbone = new_resnet_feature_extractor()

if torch.cuda.is_available():
    model = model.cuda()
    backbone.cuda()
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')

ckpoints = torch.load('autoencoder_with_resnet_deep_features.pth', map_location=map_location)
model.load_state_dict(ckpoints)

model.eval()
def decision_function(segm_map):  

    mean_top_10_values = []

    for map in segm_map:
        # Flatten the tensor
        flattened_tensor = map.reshape(-1)

        # Sort the flattened tensor along the feature dimension (descending order)
        sorted_tensor, _ = torch.sort(flattened_tensor,descending=True)

        # Take the top 10 values along the feature dimension
        mean_top_10_value = sorted_tensor[:10].mean()

        mean_top_10_values.append(mean_top_10_value)

    return torch.stack(mean_top_10_values)

# best threshold value was measured by AD_3_using_resnet_backbone_multilevel_best_threshold.py 
best_threshold = 0.02089686319231987

y_true=[]
y_pred=[]
y_score=[]

model.eval()
backbone.eval()

test_path = Path('carpet/test')

hw_target = "GPU" if torch.cuda.is_available() else "CPU"
print(f"Running inference on all carpet test images on {hw_target}")
total_inference_time = 0
inference_cnt = 0

for path in test_path.glob('*/*.png'):
    fault_type = path.parts[-2]
    test_image = transform(Image.open(path)).unsqueeze(0)

    if torch.cuda.is_available():
        test_image = test_image.cuda()
    
    with torch.no_grad():
        start = time.time()
        features = backbone(test_image)
        # Forward pass
        recon = model(features)
        inference_time = time.time()-start
        print(f"Inference time: {inference_time:.4f} s")
    
    inference_cnt += 1
    
    segm_map = ((features - recon)**2).mean(axis=(1))[:,3:-3,3:-3]
    y_score_image = decision_function(segm_map=segm_map)
    # y_score_image = segm_map.mean(axis=(1,2))
    
    y_pred_image = 1*(y_score_image >= best_threshold)
    
    y_true_image = 0 if fault_type == 'good' else 1
    
    y_true.append(y_true_image)
    y_pred.append(y_pred_image.cpu().numpy())
    y_score.append(y_score_image.cpu().numpy())
    
    total_inference_time += inference_time
    
print(f"Average inference time: {total_inference_time/inference_cnt:.4f} s")
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)
    
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

# Calculate AUC-ROC score
auc_roc_score = roc_auc_score(y_true, y_score)
print("AUC-ROC Score:", auc_roc_score)

# Plot ROC curve
# %% 
fpr, tpr, thresholds = roc_curve(y_true, y_score)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_roc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

    



