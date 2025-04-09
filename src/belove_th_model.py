"""
BeloveThaiAI: Transformer-based Model for Detecting Thailand's Unique Wildlife
==============================================================================
🐾 Objective:
Build a Transformer-based deep learning model to detect and classify 
endemic wildlife species in Thailand from image/video data.

🧠 Architecture:
- Vision Transformer (ViT) backbone (or Swin/BiT etc.)
- Optional CNN preprocessing layer
- Classification head / Detection head
- Extendable to multi-task (e.g. behavior, location tagging)

Author: BeloveThaiAI Team
"""

import torch
import torch.nn as nn
from torchvision import transforms
from timm import create_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BeloveThaiAIModel(nn.Module):
    """
    Vision Transformer-based classifier for Thai Wildlife Detection
    """

    def __init__(self, model_name="vit_base_patch16_224", num_classes=24, pretrained=True):
        """
        Args:
            model_name (str): Transformer model from `timm`
            num_classes (int): Number of species classes
            pretrained (bool): Whether to use ImageNet pretraining
        """
        super(BeloveThaiAIModel, self).__init__()
        self.backbone = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.backbone.num_features),
            nn.Linear(self.backbone.num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


# Example configuration
NUM_CLASSES = 24  # Example: 8 mammals + 2 birds + 31 reptiles + 13 amphibians (you can filter to endemic only)
MODEL_NAME = "vit_base_patch16_224"

# Instantiate model
model = BeloveThaiAIModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES).to(device)

# Print model summary (optional)
if __name__ == "__main__":
    print("✅ Model initialized: BeloveThaiAI using", MODEL_NAME)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"🔍 Output shape: {output.shape}")