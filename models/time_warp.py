import torch
import torch.nn as nn

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, n_hidden):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.ln1 = nn.LayerNorm(n_hidden)
        self.ln2 = nn.LayerNorm(n_hidden)

        self.act_func = torch.nn.Softplus()

    def forward(self, x):
        identity = x
        x = self.ln1(self.act_func(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        x += identity
        return self.act_func(x)


# Define the neural network model with residual blocks
class TimeWarpNet(nn.Module):
    def __init__(self, n_hidden=32):
        super(TimeWarpNet, self).__init__()
        # Simple MLP with a few layers
        self.fc1 = nn.Linear(1, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)

        self.res_block1 = ResidualBlock(n_hidden)  # First residual block
        self.res_block2 = ResidualBlock(n_hidden)  # Second residual block

        self.fc3 = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

        self.act_func = torch.nn.Softplus()

    def forward(self, x):
        # Pass through the network
        x = self.ln1(self.act_func(self.fc1(x)))
        x = self.res_block1(x)  # First residual block
        x = self.res_block2(x)  # Second residual block
        output = torch.sigmoid(self.fc3(x))  # Final layer

        return output
