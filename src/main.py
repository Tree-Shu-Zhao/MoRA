import pyrootutils

pyrootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

import hydra
import nni
from loguru import logger
from omegaconf import OmegaConf

from src.data import build_dataloaders
from src.engine.trainer import Trainer
from src.models import build_model
from src.utils import init_env


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    # Setup logger, code backup, hyper-parameteres records, etc...
    status = init_env(cfg)

    # Use hyper-parameters provided by nni
    if "hparams_search" in status:
        cfg = status["cfg"]

    # Print configuration
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # Build dataloaders and model
    dataloaders = build_dataloaders(cfg.dataset)
    model = build_model(cfg.model)

    # Build trainer
    trainer = Trainer(cfg, dataloaders, model)

    if cfg.test.TEST_ONLY:
        logger.warning("Test only mode is enabled. Skip training.")
        trainer.load_checkpoint()
        score = trainer.evaluate(dataloaders["test"])
        logger.info(f"Test score: {score:.2f}")
    else:
        trainer.train()

        if "hparams_search" in status:
            nni.report_final_result(trainer.best_score)

if __name__ == "__main__":
    main()