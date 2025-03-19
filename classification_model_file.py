import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class CustomResNet(nn.Module):
    def __init__(self, block, layers, feature_dim):
        super(CustomResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim
        self.fc = nn.Linear(512, feature_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class SignalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, feature_dim):
        super(SignalRNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.fc2 = nn.Linear(512, feature_dim)

    def forward(self, x):
        
        out, _ = self.rnn(x)
        # out = out[:, -1, :]
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class CLASSIFICATION_MODEL(nn.Module):
    def __init__(self, list_of_classes, signal_len, img_size=(425,550)):
        super(CLASSIFICATION_MODEL, self).__init__()
        self.list_of_classes = list_of_classes
        self.num_classes = len(self.list_of_classes)
        self.image_feature_dim = 256  # Dimension of the image feature vector
        self.signal_feature_dim = 256  # Dimension of the signal feature vector

        self.image_branch = CustomResNet(ResidualBlock, [2, 2, 2, 2], self.image_feature_dim)
        self.signal_branch = SignalRNN(input_dim=signal_len, hidden_dim=128, num_layers=4, feature_dim=self.signal_feature_dim)

        self.combined_feature_dim = self.image_feature_dim + self.signal_feature_dim

        # Convolutional layers for combined features
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8, 256)  # Adjust input dimension based on the output of conv layers
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, image, signal):
        image_features = self.image_branch(image)
        signal_features = self.signal_branch(signal)
        combined_features = torch.cat((image_features, signal_features), dim=1)

        # Reshape combined features for convolutional layers
        combined_features = combined_features.view(combined_features.size(0), 1, 32, 16)  # Example reshape, adjust as needed

        x = self.pool(F.relu(self.conv1(combined_features)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)  # Sigmoid activation for multilabel classification