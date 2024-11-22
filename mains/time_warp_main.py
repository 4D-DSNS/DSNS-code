import torch
import numpy as np
import scipy.io as sio
import open3d as o3d
from pytorch_lightning.core import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataloader.temporal_map import TimeMap
from models.dsns import DSNSModel
from models.time_warp import TimeWarpNet
from loss.mse import MSDLoss
from utils.helpers import uv_sphere

from utils.spatiotemporal_helpers import curve_to_srvf,  project_surface_to_low_dim, apply_perturbations_on_sphere,  load_pca_basis, load_rotation_perturbation
from utils.visualization import visualize_spatiotemporal


class Spatiotemporal(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.time_samples = self.config.dataset.time_samples
        self.new_time_samples = self.config.dataset.time_samples
        self.increase_timestamps = self.config.dataset.time_samples

        self.net = self.load_model(TimeWarpNet())  # spatiotemporal alignment network between 4D surfaces
        self.f1 = self.load_model(DSNSModel(), path=config.dataset.sample_path_f1)  # surface map (fixed) 4D -> 3D
        self.f2 = self.load_model(DSNSModel(), path=config.dataset.sample_path_f2)  # surface map (fixed) 4D -> 3D

        self.alignment_loss = MSDLoss()

        self.sphere_level = self.config.dataset.sphere_size

        self.pca_mean, self.pca_eigen_vects = load_pca_basis(self.config.dataset.pca_basis_path)

        # Apply spatial registration to first neural representation
        self.f1_sphere_vertices, self.sphere_faces = self.load_spatially_registered_sphere(
            self.config.dataset.spatial_param_path_f1)

        # Apply spatial registration to second neural representation
        self.f2_sphere_vertices, self.sphere_faces = self.load_spatially_registered_sphere(
            self.config.dataset.spatial_param_path_f2)

        # Regularization
        self.lambda_monotonicity = self.config.dataset.lambda_monotonicity

    def load_model(self, network, path=None):
        if path is not None:
            state_dict = torch.load(path, weights_only=True)
            network.load_state_dict(state_dict)
            for param in network.parameters():
                param.requires_grad = False

        return network

    def train_dataloader(self):
        self.dataset = TimeMap(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers, persistent_workers=True)

        return dataloader

    def load_spatially_registered_sphere(self, path):
        rotation, perturbation = load_rotation_perturbation(path, self.sphere_level)
        sv, f = apply_perturbations_on_sphere(rotation, perturbation, self.sphere_level)
        return sv, f

    # Regularization loss to ensure strictly increasing order
    def monotonicity_regularization(self, output):
        diffs = output[1:] - output[:-1]
        return torch.mean(torch.relu(-diffs))  # Penalize if difference is non-positive

    def configure_optimizers(self):
        LR = 1.0e-4
        optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
        restart = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        return [optimizer], [scheduler]

    def log_data(self):
        pass

    def save_model(self):
        model_path = self.config.checkpointing.checkpoint_path + "/model.pth"
        torch.save(self.net.state_dict(), model_path)

    def update_time_stamp(self):
        self.new_time_samples = self.new_time_samples + self.increase_timestamps
        self.times = torch.linspace(0, 1, self.new_time_samples)
        if self.new_time_samples > self.config.dataset.max_temporal_batch_size:
            samples = self.config.dataset.max_temporal_batch_size - 2
            fixed_indices = torch.tensor([0, self.new_time_samples - 1])
            remaining_indices = torch.randperm(self.new_time_samples - 2)[:samples] + 1
            all_indices = torch.cat([fixed_indices, remaining_indices])
            sorted_indices = torch.sort(all_indices).values
            self.times = self.times[sorted_indices]

        if self.time_samples < self.config.dataset.max_temporal_batch_size:
            self.time_samples = self.time_samples + self.increase_timestamps

        self.times = self.times.cuda().unsqueeze(1)

    def training_step(self, batch, batch_idx):
        if self.global_step % 200 == 0:
            self.update_time_stamp()

        times = self.times
        pred_times = self.net(times)

        original_times_set = times.unsqueeze(1).expand(self.time_samples, self.f1_sphere_vertices.size(0), 1)
        predicted_times_set = pred_times.unsqueeze(1).expand(self.time_samples, self.f1_sphere_vertices.size(0), 1)
        f1_sv = self.f1_sphere_vertices.unsqueeze(0).expand(self.time_samples, -1, -1)
        f2_sv = self.f2_sphere_vertices.unsqueeze(0).expand(self.time_samples, -1, -1)

        S1 = torch.cat([f1_sv, original_times_set], dim=-1).reshape(-1, 4)
        S2 = torch.cat([f2_sv, predicted_times_set], dim=-1).reshape(-1, 4)

        S1 = self.f1(S1)
        S2 = self.f2(S2)

        S1 = S1.reshape(self.time_samples, self.sphere_level, self.sphere_level, 3)
        S2 = S2.reshape(self.time_samples, self.sphere_level, self.sphere_level, 3)

        M1 = project_surface_to_low_dim(S1, self.pca_mean, self.pca_eigen_vects)
        M2 = project_surface_to_low_dim(S2, self.pca_mean, self.pca_eigen_vects)

        q1, q2 = curve_to_srvf(M1, to_normalize=False), curve_to_srvf(M2, to_normalize=False)

        alignment_loss = torch.mean(self.alignment_loss(q1.permute(1, 0), q2.permute(1, 0)))
        monotonicity_regularization = self.monotonicity_regularization(pred_times)

        loss = alignment_loss + self.lambda_monotonicity * monotonicity_regularization

        if torch.isnan(alignment_loss):
            print("Nan in alignment loss")
            self.trainer.should_stop = True

        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # add here logging if needed
        self.log_data()

        return loss

    def save_discretized_frames(self):
        with torch.no_grad():
            self.net.cuda()
            self.f1.cuda()
            self.f2.cuda()

            self.sphere_vertices, _ = uv_sphere((self.config.dataset.visualization_resolution,self.config.dataset.visualization_resolution))
            self.sphere_vertices = torch.FloatTensor(self.sphere_vertices).cuda()
            self.times = torch.linspace(0,1, self.config.dataset.visualization_timestamps).unsqueeze(1).cuda()
            times = self.times
            pred_times = self.net(times)

            original_times_set = times.squeeze().repeat(self.sphere_vertices.size(0)).reshape(
                self.sphere_vertices.size(0), -1).cuda()
            predicted_times_set = pred_times.squeeze().repeat(self.sphere_vertices.size(0)).reshape(
                self.sphere_vertices.size(0), -1).cuda()

            source, target_reg, target_unreg = [], [], []
            for idx in range(times.shape[0]):
                F1_outputs = []
                F2_outputs = []
                F3_outputs = []
                points = self.config.dataset.num_points
                start = 0
                end = int(self.sphere_vertices.size(0) / points) + 1
                for i in range(start, end):
                    if (points * (i)) >= self.sphere_vertices.size(0):
                        continue

                    original_batch_sv = torch.cat([self.sphere_vertices[(points * i):(points * (i + 1)), :],
                                                   original_times_set[(points * i):(points * (i + 1)), idx].unsqueeze(
                                                       1)], dim=1)
                    F1 = self.f1(original_batch_sv)
                    F1_outputs.append(F1)

                    predicted_batch_sv = torch.cat([self.sphere_vertices[(points * i):(points * (i + 1)), :],
                                                    predicted_times_set[(points * i):(points * (i + 1)), idx].unsqueeze(
                                                        1)], dim=1)
                    F2 = self.f2(predicted_batch_sv)
                    F2_outputs.append(F2)

                    F3 = self.f2(original_batch_sv)
                    F3_outputs.append(F3)

                F1 = torch.cat(F1_outputs, dim=0)
                F2 = torch.cat(F2_outputs, dim=0)
                F3 = torch.cat(F3_outputs, dim=0)


                source.append(F1.reshape(self.sphere_level, self.sphere_level, 3))
                target_reg.append(F2.reshape(self.sphere_level, self.sphere_level, 3))
                target_unreg.append(F3.reshape(self.sphere_level, self.sphere_level, 3))

        source = torch.stack(source)
        target_reg = torch.stack(target_reg)
        target_unreg = torch.stack(target_unreg)

        sio.savemat(f'{self.config.checkpointing.checkpoint_path}/source.mat',
                    {"S": source.detach().cpu().numpy().astype(np.float64)})
        sio.savemat(f'{self.config.checkpointing.checkpoint_path}/target_unreg.mat',
                    {"S": target_unreg.detach().cpu().numpy().astype(np.float64)})
        sio.savemat(f'{self.config.checkpointing.checkpoint_path}/target_reg.mat',
                    {"S": target_reg.detach().cpu().numpy().astype(np.float64)})

    def visualize(self, save_discretization, show_registered):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        if save_discretization:
            self.save_discretized_frames()
        visualize_spatiotemporal(exp_path = f'{self.config.checkpointing.checkpoint_path}/', registered= show_registered, total_frames=self.config.dataset.visualization_timestamps, resolution= self.config.dataset.visualization_resolution)





