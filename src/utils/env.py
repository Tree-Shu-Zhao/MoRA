import os
import random
import shutil

import nni
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from .hparams import reset_cfg


def init_env(cfg):
    # Log experiment note. If it is not be provided, we do not run the experiment.
    assert cfg.EXP_NOTE is not None, "You must provide a experiment note to run the experiment!"
    logger.info(f"Experiment Note: {cfg.EXP_NOTE}")

    cfg.OUTPUT_DIR = HydraConfig.get().run.dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Setup logger
    logger.add(os.path.join(cfg.OUTPUT_DIR, "main.log"))

    # Checkpoint Dir
    checkpoint_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint Dir: {checkpoint_dir}.")

    # Code Dir
    code_dir = os.path.join(cfg.OUTPUT_DIR, "codes")
    os.makedirs(code_dir, exist_ok=True)
    logger.info(f"Code Dir: {code_dir}")
    save_codes(code_dir)
    logger.info("Codes backup completed.")

    # Set seed
    random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    logger.info(f"Set seed to {cfg.SEED}.")

    # Set numpy print precision
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=2)

    # Set hyper-paremter search
    status = {}
    if "hparams_search" in cfg:
        logger.info("Enable nni to search hyper-parameters.")
        params = nni.get_next_parameter()
        cfg = reset_cfg(cfg, params)
        status["cfg"] = cfg
        status["hparams_search"] = True
    return status


def save_codes(code_dir):
    VALID_FILE_TYPES = (".py", ".ipynb", ".sh", ".yaml")
    VALID_DIRS = ("configs", "src", "scripts")

    logger.info(
        f"Prepare to backup codes to {code_dir}.\nValid file types: {VALID_FILE_TYPES}.\nValid dirs: {VALID_DIRS}.")

    for valid_dir in VALID_DIRS:
        for root, dirs, files in os.walk(valid_dir):
            for file in files:
                file_type = os.path.splitext(file)[1]
                if file_type in VALID_FILE_TYPES:
                    source_path = os.path.join(root, file)
                    backup_path = os.path.join(code_dir, source_path)
                    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                    shutil.copy2(source_path, backup_path)
