import os
import torch
import trimesh
import scipy.io as sio
import open3d as o3d

from pytorch_lightning.core import LightningModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from loss.mse import MSDLoss
from utils.differential import DifferentialMixin

from dataloader.spherical_map import SurfaceMapDataset
from models.dsns import DSNSModel
from utils.visualization import visualize_heatmap


class DSNS(DifferentialMixin, LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = DSNSModel()  # map
        self.loss_function = MSDLoss()  # loss

        self.sphere_level = self.config.dataset.sphere_level

    def train_dataloader(self):
        self.dataset = SurfaceMapDataset(self.config.dataset)
        dataloader = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers, persistent_workers=True)
        return dataloader

    def configure_optimizers(self):
        LR = 1.0e-4
        optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
        restart = int(self.config.dataset.num_epochs)
        scheduler = CosineAnnealingLR(optimizer, T_max=restart)
        return [optimizer], [scheduler]

    def save_model(self):
        model_path = self.config.checkpointing.checkpoint_path + "/model.pth"
        torch.save(self.net.state_dict(), model_path)

    def log_data(self):
        pass

    def training_step(self, batch, batch_idx):

        source = batch['source']  # Nx4
        gt = batch['gt']  # Nx3
        normals = batch['normals']  # Nx3

        # forward network
        out = self.net(source)

        loss_dist = self.loss_function(out, gt)

        # estimate normals through autodiff
        # pred_normals = self.compute_normals(out=out, wrt=source)
        # loss_normals = self.loss_function(pred_normals, normals)
        # loss = loss_dist + self.config.dataset.lambda_normal*loss_normals

        loss = loss_dist

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        # add here logging if needed
        self.log_data()

        return loss

    def evaluate(self, save_first_frame=True):
        self.net.eval().cuda()

        os.makedirs(f'{self.config.checkpointing.checkpoint_path}/dsns', exist_ok = True)
        os.makedirs(f'{self.config.checkpointing.checkpoint_path}/orig', exist_ok = True)

        per_frame_loss = []
        with torch.no_grad():
            total_frames = self.dataset.sample['sphere_points'].size(0)
            for idx in range(total_frames):
                sphere_points, surface_points, sphere_faces = self.dataset.sample['sphere_points'][idx].cuda(), \
                self.dataset.sample['surface_points'][idx].cuda(), self.dataset.sample['sphere_faces'][idx].cuda()

                outputs = []
                start, end = 0, int(sphere_points.size(0) / self.config.dataset.num_points) + 1

                for i in range(start, end):
                    outputs.append(self.net(sphere_points[(self.config.dataset.num_points * i):(self.config.dataset.num_points * (i + 1)),:]))

                out = torch.cat(outputs, dim=0)
                per_frame_loss.append(self.loss_function(out, surface_points))

                dsns_mesh = trimesh.Trimesh(vertices=out.detach().cpu().numpy(),
                                            faces=sphere_faces.detach().cpu().numpy())
                orig_mesh = trimesh.Trimesh(vertices=surface_points.detach().cpu().numpy(),
                                            faces=sphere_faces.detach().cpu().numpy())
                dsns_mesh.export(f'{self.config.checkpointing.checkpoint_path}/dsns/dsns_{idx}.obj')
                orig_mesh.export(f'{self.config.checkpointing.checkpoint_path}/orig/orig_{idx}.obj')

                if save_first_frame:
                    if idx==0:
                        sio.savemat(f'{self.config.checkpointing.checkpoint_path}/first_frame.mat', { 'S' : out.detach().cpu().numpy().reshape(self.config.dataset.sphere_level,self.config.dataset.sphere_level, 3)})

            return torch.tensor(per_frame_loss)

    def visualize(self):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
        visualize_heatmap(exp_path = f'{self.config.checkpointing.checkpoint_path}', total_frames= self.dataset.sample['sphere_points'].size(0))