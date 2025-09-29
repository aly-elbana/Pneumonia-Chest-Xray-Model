import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        return x

class CNNClassifier(nn.Module):
    def __init__(self, input_channels: int, output_dim: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        return x

def build_model(output_dim: int) -> nn.Module:
    base_densenet = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1
    )
    feature_extractor = DenseNetFeatureExtractor(base_densenet)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    
    classifier = CNNClassifier(input_channels=1024, output_dim=output_dim)
    classifier.train()

    model = nn.Sequential(
        feature_extractor,
        classifier
    )
    return model

model_cnn = build_model(output_dim=3)

if __name__ == "__main__":
    print(model_cnn)
