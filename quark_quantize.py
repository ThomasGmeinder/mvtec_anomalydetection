from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

import quark
from AD_3_using_resnet_backbone_multilevel_model import FeatCAE, resnet_feature_extractor

# Prepare Dataset 
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_image_path = Path('./carpet/train')
good_dataset = ImageFolder(root=train_image_path, transform=transform)

# Load model
model = FeatCAE(in_channels=1536, latent_dim=100)

if torch.cuda.is_available():
    model = model.cuda()
    map_location=torch.device('cuda')
else:
    map_location=torch.device('cpu')

ckpoints = torch.load('autoencoder_with_resnet_deep_features.pth', map_location=map_location)
model.load_state_dict(ckpoints)
model.eval()

# Set quantization configuration
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
DEFAULT_INT8_PER_TENSOR_SYM_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                        qscheme=QSchemeType.per_tensor,
                                        observer_cls=PerTensorMinMaxObserver,
                                        symmetric=True,
                                        scale_type=ScaleType.float,
                                        round_method=RoundType.half_even,
                                        is_dynamic=False)

DEFAULT_W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=DEFAULT_INT8_PER_TENSOR_SYM_SPEC)
quant_config = Config(global_quant_config=DEFAULT_W_INT8_PER_TENSOR_CONFIG)

# Define calibration dataloader (still need this step for weight only and dynamic quantization)
# Set the batch size
BS = 1

# FeatureDataset converts images to features using backbone
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, backbone):
        self.dataset = dataset
        self.backbone = backbone

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            features = self.backbone(image.unsqueeze(0)).squeeze(0)
        return features

# ResNet18 backbone to generate features for the CAE
backbone = resnet_feature_extractor()
feature_dataset = FeatureDataset(good_dataset, backbone)
calib_dataloader = DataLoader(feature_dataset, batch_size=BS, shuffle=True)

# In-place replacement with quantized modules in model
from quark.torch import ModelQuantizer
quantizer = ModelQuantizer(quant_config)
quant_model = quantizer.quantize_model(model, calib_dataloader)


# Export to onnx
freezed_quantized_model = quantizer.freeze(quant_model)

from quark.torch import ModelExporter
# Get dummy input
for data in calib_dataloader:
    input_args = data
    break

if torch.cuda.is_available():
    quant_model = quant_model.to('cuda')
    input_args = input_args.to('cuda')

export_path = "./quantization_output"
from quark.torch import ModelExporter
from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized",
                                           pack_method="reorder")
export_config = ExporterConfig(json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
exporter = ModelExporter(config=export_config, export_dir=export_path)

exporter.export_onnx_model(quant_model, input_args)




