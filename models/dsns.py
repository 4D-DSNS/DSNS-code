import torch
from torch import nn
from torch.nn import Softplus, Sequential


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=10, omega0=0.1):
        """
        Positional Embedding implementation (adapted from Pytorch3D).

        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**self.n_harmonic_functions * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**self.n_harmonic_functions * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluting the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            "frequencies",
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

class ResBlock(nn.Module):
    def __init__(self, hidden_size, activation):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = activation()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out + identity)

        return out

class DSNSModel(nn.Module):
    def __init__(self):
        super(DSNSModel, self).__init__()

        spatial_positional_encoding_size = 10  # Example positional encoding size
        temporal_positional_encoding_size = 10  # Example positional encoding size
        output_size = 3
        act_fun = Softplus

        self.spatial_positional_encoding = HarmonicEmbedding(spatial_positional_encoding_size)
        self.temporal_positional_encoding = HarmonicEmbedding(temporal_positional_encoding_size)

        modules = []

        input_size = 2*3*spatial_positional_encoding_size + 2*1*temporal_positional_encoding_size

        modules.append(nn.Linear(input_size, 1024))
        modules.append(act_fun())

        for layer in [1024]*6:
            block = ResBlock(layer, act_fun)
            modules.append(block)

        # output layer
        modules.append(nn.Linear(1024, output_size))
        self.mlp = Sequential(*modules)


    def forward(self, x):

        h_x = self.spatial_positional_encoding(x[:,:3] )
        h_t = self.temporal_positional_encoding(x[:,3].unsqueeze(1))
        h = torch.concat([h_x,h_t], dim=1)

        x = self.mlp(h)
        return x