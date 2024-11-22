import torch
import hydra
from omegaconf import DictConfig

import os

# Set the environment variable
torch.autograd.set_detect_anomaly(True)
os.environ['HYDRA_FULL_ERROR'] = '1'

from utils.config import compose_config_folders
from utils.config import copy_config_to_experiment_folder

from pytorch_lightning import Trainer

from mains.time_warp_main import Spatiotemporal


@hydra.main(config_path='configs', config_name='time_warp', version_base='1.1')
def main(cfg: DictConfig) -> None:

    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = Spatiotemporal(cfg)
    model.net.load_state_dict(torch.load('./_pretrained/timewarp.pth', weights_only=True))

    trainer = Trainer(max_epochs=1)
    trainer.fit(model)


    model.save_model()

    model.visualize(save_discretization = True, show_registered=True)

    # save surface map as sample for inter surface map
    # save_model(cfg.checkpointing.checkpoint_path, model.net)
    # potentially you can save the rotation in your
    # map (model.net) so you don't have to recompute it


if __name__ == '__main__':
    main()
