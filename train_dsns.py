import hydra
import torch
from pytorch_lightning import Trainer
from omegaconf import DictConfig

from mains.dsns_main import DSNS
from utils.config import compose_config_folders, copy_config_to_experiment_folder

@hydra.main(config_path='configs', config_name='dsns', version_base='1.1')
def main(cfg: DictConfig) -> None:

    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = DSNS(cfg)
    model.net.load_state_dict(torch.load("./_pretrained/model.pth", weights_only=True))

    trainer = Trainer(max_epochs=1)
    trainer.fit(model)

    # save dsns model
    print("Saving model weights.")
    model.save_model()

    print("Evaluating the performance. {est time. 10 minutes}")
    print(f"Performance: {torch.mean(model.evaluate(save_first_frame=True))}.")

    # visualize
    print("Running the heatmap visualization. {est time. 10 minutes}")
    model.visualize()

if __name__ == '__main__':
    main()