import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from attacks.foolboxattacks.L1_basic_iterative import L1BasicIterative
from attacks.artattacks.zeroth_order_optimization_bb_attack import ZeorthOrderOptimalization
import mlflow.pytorch
import torch
from attacks.foolboxattacks.brendel_bethge import L0BrendelBethge

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist


class Data:
    def __init__(self, inp, out):
        self.x = inp
        self.y = out
        self.input = inp
        self.output = out


if __name__ == '__main__':
    print("# Step 0: Define the neural network model, return logits instead of activation in forward method")

    # Step 0: Define the neural network model, return logits instead of activation in forward method
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
            self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
            self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
            self.fc_2 = nn.Linear(in_features=100, out_features=10)

        def forward(self, x):
            x = F.relu(self.conv_1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv_2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 10)
            x = F.relu(self.fc_1(x))
            x = self.fc_2(x)
            return x


    print("# Step 1: Load the MNIST dataset")
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

    print("# Step 1a: Swap axes to PyTorch's NCHW format")
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    print("# Step 2: Create the model")
    model = Net().eval()

    print("# Step 2a: Define the loss function and the optimizer")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    data = Data(x_train, y_train)

    # NotImplementedError
    print("# Step 3b: Initializing L1BasicIterative foolbox attack")
    foolboxAttack = L1BasicIterative({})
    print("# Step 4b: Conducting L1BasicIterative foolbox attack")
    print(foolboxAttack.conduct(model, data))
