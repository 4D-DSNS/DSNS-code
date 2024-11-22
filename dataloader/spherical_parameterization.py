import os
import torch
import mat73

from utils.helpers import uv_sphere

class SphericalMixin:

    def create_sphere(self, sphere_size=32):
        vertices, face = uv_sphere((sphere_size, sphere_size))

        return torch.FloatTensor(vertices), self.adjust_face_orientation(torch.LongTensor(face))

    def split_to_blocks(self, size, num_blocks):
        '''
            Split the set of possible points into chuncks.
            Then permutes the indices to have random sampling.
        '''
        idxs = torch.randperm(size)
        block_size = int(float(idxs.size(0)) / float(num_blocks))
        blocks = []
        for i in range(num_blocks):
            blocks.append(idxs[block_size * i: block_size * (i + 1)])

        return blocks

    def load_data(self, path, dict_type, res_index):
        print(f"LOADING: \t {path}")
        if os.path.exists(path):
            if res_index is not None:
                return mat73.loadmat(path)["Surface"]
            else:
                return mat73.loadmat(path)[dict_type]
        return None

    def adjust_face_orientation(self, faces):
        return faces[:, [0, 2, 1]]

    def extract_surface_normals(self, vertices, faces):

        verts_normals = torch.zeros_like(vertices)
        vertices_faces = vertices[faces]

        faces_normals = torch.linalg.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        )

        # NOTE: this is already applying the area weighting as the magnitude
        # of the cross product is 2 x area of the triangle.
        verts_normals = verts_normals.index_add(
            0, faces[:, 0], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, faces[:, 1], faces_normals
        )
        verts_normals = verts_normals.index_add(
            0, faces[:, 2], faces_normals
        )

        return torch.nn.functional.normalize(
            verts_normals, p=2, eps=1e-6, dim=1
        )

    def read_torch_spherical_map(self, sample_path, config):

        sphere_vertices, sphere_faces = self.create_sphere(config.sphere_level)

        temporal_sphere_vertices, temporal_sphere_faces, temporal_surface_vertices, temporal_surface_normals = [], [], [], []
        template_uv = self.load_data(f"{sample_path}", config.dict_type, config.res_index)
        total_frames = len(template_uv)

        time_values = torch.linspace(0, 1, total_frames)
        selected_frames = torch.linspace(0, total_frames - 1, total_frames, dtype=torch.int32)

        for idx, frame_idx in enumerate(selected_frames):
            surface_vertices = torch.tensor(template_uv[frame_idx][config.dict_type].reshape(-1, 3))
            surface_normals = self.extract_surface_normals(surface_vertices, sphere_faces)
            time_domain = time_values[idx].clone().repeat(surface_vertices.size(0))
            temporal_sphere_vertices.append(torch.cat([sphere_vertices, time_domain.unsqueeze(1)], dim=1))
            temporal_sphere_faces.append(sphere_faces)

            temporal_surface_vertices.append(surface_vertices)
            temporal_surface_normals.append(surface_normals)

        self.spherical_map = {
            "sphere_points": torch.stack(temporal_sphere_vertices),
            "sphere_faces": torch.stack(temporal_sphere_faces),
            "surface_points": torch.stack(temporal_surface_vertices),
            "surface_normals": torch.stack(temporal_surface_normals)
        }

        return self.spherical_map