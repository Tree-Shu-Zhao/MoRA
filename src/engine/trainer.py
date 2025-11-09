import os
import time

import nni
import torch
import torch.nn as nn
from loguru import logger
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from utils import AverageMeter, ProgressMeter

from .evaluator import Evaluator


class Trainer:
    def __init__(self, cfg, dataloaders, model):
        super().__init__()

        self.cfg = cfg
        self.current_epoch = 0
        self.current_iter = 0
        self.epochs = cfg.train.EPOCHS
        self.dataloaders = dataloaders

        if cfg.train.GPU is None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(f"cuda:{cfg.train.GPU[0]}") if isinstance(cfg.train.GPU, list) else torch.device(f"cuda:{cfg.train.GPU}")

        self.model = model.to(self.device)

        # Build criterion
        if cfg.dataset.MULTI_LABEL:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer and LR scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=cfg.train.LR,
            weight_decay=cfg.train.WEIGHT_DECAY,
            eps=1e-8, 
            betas=(0.9, 0.98),
        )
        num_training_steps = len(dataloaders["train"]) * self.epochs
        num_warmup_steps = int(num_training_steps * cfg.train.WARMUP_RATIO)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Fp16 training
        self.scaler = GradScaler()

        # Build evaluator
        self.evaluator = Evaluator(cfg.test)
        
        # Initialize best score and best epoch
        self.best_score = 0.0
        self.best_epoch = -1
        self.num_no_improvement = 0

    def train(self):
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.train_loop()
            if (epoch + 1) % self.cfg.train.EVAL_FREQ == 0:
                score = self.evaluate(self.dataloaders["val"])
                logger.info(f"Validation score: {score:.2f}")

                if "hparams_search" in self.cfg:
                    nni.report_intermediate_result(score)

                if self.best_score < score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.num_no_improvement = 0
                    logger.info(f"New best score: {self.best_score:.2f} at epoch {self.best_epoch+1}")
                    if "hparams_search" not in self.cfg:
                        self.save_checkpoint()
                else:
                    self.num_no_improvement += 1
                    if self.num_no_improvement >= self.cfg.train.EARLY_STOPPING:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

        # Only evaluate at the end if no evaluation was done during training
        if self.best_score == 0.0:
            self.best_score = self.evaluate(self.dataloaders["val"])
            self.best_epoch = self.current_epoch
            logger.info(f"Validation score: {self.best_score:.2f}")
            if "nni" not in self.cfg:
                self.save_checkpoint()
        
        if "hparams_search" not in self.cfg:
            ckpt_path = os.path.join(self.cfg.OUTPUT_DIR, "checkpoints", "best_model.pt")
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_learned_weights(state_dict)
            test_score = self.evaluate(self.dataloaders["test"])
            logger.info(f"Test score: {test_score:.2f}")
        
    def train_loop(self):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.3f')
        progress = ProgressMeter(
            len(self.dataloaders["train"]),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}/{}]".format(self.current_epoch+1, self.epochs),
        )

        self.model.train()
        end = time.perf_counter()
        for it, batch in enumerate(self.dataloaders["train"]):
            self.current_iter += 1
            batch = move_to_device(batch, self.device)
            data_time.update(time.perf_counter() - end)

            self.optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = self.model(batch)
                loss = self.criterion(outputs["logits"], batch["labels"])

            losses.update(loss.item(), len(batch["labels"]))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            batch_time.update(time.perf_counter() - end)
            end = time.perf_counter()

            if (it+1) % self.cfg.train.PRINT_FREQ == 0:
                progress.display(it+1)

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        logits = []
        labels = []
        for it, batch in tqdm(enumerate(dataloader), desc=f"Evaluating", total=len(dataloader)):
            batch = move_to_device(batch, self.device)
            with autocast(device_type="cuda"):
                outputs = self.model(batch)
            logits.append(outputs["logits"].cpu())
            labels.append(batch["labels"].cpu())
        logits = torch.vstack(logits)
        if len(labels[0].shape) == 1:
            labels = torch.hstack(labels)
        else:
            # One-hot
            labels = torch.vstack(labels)
        score = self.evaluator(logits, labels).item() * 100.
        return score
    
    def save_checkpoint(self):
        best_model_path = os.path.join(self.cfg.OUTPUT_DIR, "checkpoints", "best_model.pt")
        torch.save(self.model.get_learned_weights(), best_model_path)
        
        # Save a metadata file with information about the best model
        metadata_path = os.path.join(self.cfg.OUTPUT_DIR, "checkpoints", "best_model_info.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Epoch: {self.best_epoch+1}\n")
            f.write(f"Score: {self.best_score:.4f}\n")
            f.write(f"Saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info(f"Best model saved at {best_model_path} (epoch {self.best_epoch+1}, score {self.best_score:.2f})")
    
    def load_checkpoint(self):
        logger.info(f"Model loaded from {self.cfg.test.CHECKPOINT_PATH}")
        state_dict = torch.load(self.cfg.test.CHECKPOINT_PATH, map_location=self.device)
        self.model.load_learned_weights(state_dict)


def move_to_device(batch, device):
    for k, v in batch.items():
        if hasattr(v, "to"):
            batch[k] = v.to(device)
    return batch