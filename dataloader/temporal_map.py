import torch
from .spherical_parameterization import SphericalMixin


class TimeMap(SphericalMixin):

    def __init__(self, config):
        self.config     = config
        self.num_epochs = config.num_epochs
        self.time_samples = config.time_samples

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, index):


        times = torch.linspace(0,1,self.time_samples)


        data_dict = {
                    'time'              : times
                    }

        return data_dict