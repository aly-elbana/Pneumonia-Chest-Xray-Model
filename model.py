import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # (batch, 1024, H, W)
        x = self.pool(x)           # (batch, 1024, 1, 1)
        x = torch.flatten(x, 1)    # (batch, 1024)
        return x


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.drop5 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.drop3(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.drop4(x)
        x = self.fc5(x)
        x = self.relu5(x)
        x = self.drop5(x)
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
    
    classifier = FullyConnectedClassifier(input_dim=1024, output_dim=output_dim)
    classifier.train()
    
    model = nn.Sequential(
        feature_extractor,
        classifier
    )
    return model

model = build_model(output_dim=3)
if __name__ == "__main__":
    print(model)
