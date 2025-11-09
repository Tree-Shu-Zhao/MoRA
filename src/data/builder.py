from functools import partial

from loguru import logger
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

from .dataset import MissingDataset, collate_fn


def build_dataloaders(cfg):
    name = cfg.NAME.lower()
    assert name in ["food101", "mmimdb", "hatememes"], f"Unknown dataset name: {name}, choose from [food101, mmimdb, hatememes]"

    train_dataset = MissingDataset(
        cfg.DATA_DIR,
        cfg.missing_params,
        name,
        split="train",
    )
    val_dataset = MissingDataset(
        cfg.DATA_DIR,
        cfg.missing_params,
        name,
        split="val",
    )
    test_dataset = MissingDataset(
        cfg.DATA_DIR,
        cfg.missing_params,
        name,
        split="test",
    )

    logger.info(f"# train: {len(train_dataset)}")
    logger.info(f"# val: {len(val_dataset)}")
    logger.info(f"# test: {len(test_dataset)}")

    processor = CLIPProcessor.from_pretrained(cfg.BACKBONE_NAME)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.TEST_BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor),
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
