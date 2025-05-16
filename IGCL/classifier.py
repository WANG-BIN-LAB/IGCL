from torch import nn


class FullyConnectedClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FullyConnectedClassifier, self).__init__()
        # First hidden layer, maps input dimension to 512
        self.fc1 = nn.Linear(input_dim, 512)
        # Batch normalization layer, helps accelerate training and stabilize the model
        self.bn1 = nn.BatchNorm1d(512)

        # ReLU activation function, introduces nonlinearity
        self.relu = nn.ReLU()

        # Second hidden layer, maps 512 to num_classes
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Forward propagation
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x