import torchvision
from torch import nn


class Resnet18EmbeddingModel(nn.Module):

    def __init__(self, download_pretrained: bool = False):
        super().__init__()
        self._model = torchvision.models.resnet18(weights="DEFAULT" if download_pretrained else None)

    def forward(self, x):
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)
        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        x = self._model.layer4(x)

        x = self._model.avgpool(x)

        x = x.view(x.size(0), -1)

        return x

    @property
    def embedding_size(self):
        return self._model.fc.in_features
