import torch
from .spherical_parameterization import SphericalMixin


class SurfaceMapDataset(SphericalMixin):

    def __init__(self, config):
        self.sample_path = config.sample_path
        self.num_points = config.num_points  # num points for each iteration
        self.num_epochs = config.num_epochs  # num epochs
        self.sample = self.read_torch_spherical_map(self.sample_path, config)

        temporal_size, spatial_size = self.sample['sphere_points'].size()[:2]

        # Create a list of all possible [Temporal Index, Point Index] pairs
        self.indices = [(t, p) for t in range(temporal_size) for p in range(spatial_size)]

        self.total_points = len(self.indices)

    def __len__(self):
        return self.num_epochs

    def __getitem__(self, index):
        sphere_points = self.sample['sphere_points'].float()
        sphere_faces = self.sample['sphere_faces'].long()
        surface_points = self.sample['surface_points'].float()
        surface_normals = self.sample['surface_normals'].float()

        # Randomly select batch_size indices
        selected_indices = torch.randint(0, self.total_points, (self.num_points,))

        # Create a block of indices containing [Temporal Index, Point Index] pairs
        block = [self.indices[i] for i in selected_indices]
        block = torch.tensor(block, dtype=torch.long)

        source_points = sphere_points[block[:, 0], block[:, 1]]
        points = surface_points[block[:, 0], block[:, 1]]
        normals = surface_normals[block[:, 0], block[:, 1]]
        faces = sphere_faces[block[:, 0], block[:, 1]]

        data_dict = {
            'source': source_points,
            'gt': points,
            'normals': normals,
            'faces': faces,
        }

        return data_dict