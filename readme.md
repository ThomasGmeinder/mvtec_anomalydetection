# Overview
Jupyter Notebooks are provided to provide executable documentation for 4 different approaches to anomaly detection.

There are also scripts that focusses on running inference with the pre-rained model created by [AD_3_using_resnet_backbone_multilevel_inference.ipynb](./AD_3_using_resnet_backbone_multilevel_inference.ipynb):
[autoencoder_with_resnet_deep_features.pth](./autoencoder_with_resnet_deep_features.pth)

# Running Inference
Here are the steps to run inference:
- Setup: `pip install -r requirements_minimal.txt`
- Download Dataset: `python download_dataset.py`
- FP32 Inference with original PyTorch model: `python AD_3_using_resnet_backbone_multilevel_inference.py`
- Quantize using Quark: `python quark_quantize.py`
- Run inference with the quantized model: `python AD_3_using_resnet_backbone_multilevel_inference_onnx.py`

**Note:** Different Execution Providers can be chosen for onnxruntime by changing the `EP` variable in [AD_3_using_resnet_backbone_multilevel_inference_onnx.py](./AD_3_using_resnet_backbone_multilevel_inference_onnx.py)

# Performance and accuracy

## PyTorch on GPU
Run this on Machine with GPUs enabled:
[AD_3_using_resnet_backbone_multilevel_inference.py](./AD_3_using_resnet_backbone_multilevel_inference.py):

```
Average inference time: 0.0074 s
AUC-ROC Score: 0.9867576243980738
```

## Autoencoder on ONNX, ResNet backbone on PyTorch. On GPU
Run this on Machine with GPUs enabled:
[AD_3_using_resnet_backbone_multilevel_inference_onnx.py](./AD_3_using_resnet_backbone_multilevel_inference_onnx.py) on GPU: 

# Further links
link to the Mvtech indivitual data= https://github.com/LuigiFederico/PatchCore-for-Industrial-Anomaly-Detection/blob/main/lib/data.py

link to entire mvtec data= https://www.mvtec.com/company/research/datasets/mvtec-ad

pytorch tutorial resources = https://pytorch.org/tutorials/beginner/basics/intro.html 

