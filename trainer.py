import torch
import copy
from tqdm import tqdm
from diffusion.resample import UniformSampler
from utils.engine.nn import update_ema
from torch.optim import AdamW
import time
from diffusion.resample import UniformSampler, Batch_Same_Sampler
import os
from torch.utils.data import DataLoader
from utils.tools import mkdir_if_missing
import numpy as np
from utils.evaluation.scores import (
    discriminative_score,
    predictive_score,
    BMMD_score,
    BMMD_score_naive,
    VDS_score,
)
import pdb
from utils.evaluation.metric_utils import display_scores


class Trainer:
    def __init__(self, cfg, model, diffusion, dataset):
        self.cfg = cfg
        # Initialize the model, data, and optimizer
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            shuffle=cfg.DATALOADER.SHUFFLE,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=cfg.DATALOADER.DROP_LAST,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
        )
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.diffusion = diffusion
        self.model = model
        if cfg.DIFFUSION.SCHEDULE_SAMPLER == "batch":
            self.schedule_sampler = Batch_Same_Sampler(diffusion)
        elif cfg.DIFFUSION.SCHEDULE_SAMPLER == "uniform":
            self.schedule_sampler = UniformSampler(diffusion)
        else:
            raise NotImplementedError(f"Unkown sampler: {cfg.OPTIM.SCHEDULE_SAMPLER}")
        self.cuda = torch.cuda.is_available()
        self.device = (
            torch.device("cuda")
            if (self.cuda and cfg.USE_CUDA)
            else torch.device("cpu")
        )
        self.model.to(self.device)
        self.lr = cfg.OPTIM.LR
        self.weight_decay = cfg.OPTIM.WEIGHT_DECAY
        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Initialize the training loop parameters
        self.step = 0
        self.lr_anneal_steps = cfg.OPTIM.LR_ANNEAL_STEPS

        # Log and save intervals
        self.log_interval = cfg.TRAIN.LOG_INTERVAL
        self.save_interval = cfg.TRAIN.SAVE_INTERVAL

        self.save_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints/")
        os.makedirs(self.save_dir, exist_ok=True)

        self.alpha = cfg.TRAIN.MMD_ALPHA

    def train(self):
        """
        Train the model for a given number of steps.
        """
        self.step = 0
        cumulative_loss = 0.0
        loss_count = 0
        for self.step in range(self.lr_anneal_steps):
            self.opt.zero_grad()
            for batch_idx, batch in enumerate(self.dataloader):
                batch = batch.to(self.device)
                t, weights = self.schedule_sampler.sample(self.batch_size, self.device)
                losses = self.diffusion.training_losses(self.model, batch, t)

                mse_loss = (losses["mse"] * weights).mean()
                total_loss = mse_loss
                if "mmd" in losses:
                    mmd_loss = self.alpha * losses["mmd"]
                    total_loss += mmd_loss
                total_loss.backward()
                self._anneal_lr()
                self.opt.step()
                loss_val = total_loss.item()
                cumulative_loss += loss_val
                loss_count += 1
                avg_loss = cumulative_loss / loss_count
                if (batch_idx + 1) % self.log_interval == 0:
                    current_lr = self.opt.param_groups[0]["lr"]
                    print(
                        f"epoch [{self.step + 1}/{self.lr_anneal_steps}] batch [{batch_idx + 1}/{len(self.dataloader)}]"
                        f"loss {loss_val:.4f} ({avg_loss:.4f}) lr {current_lr:.4e}"
                    )

            if self.step % self.save_interval == 0:
                self.save()

    def eval(self):
        concatenated_tensor = self.sampling()
        save_dir = os.path.join(self.cfg.OUTPUT_DIR, "samples")
        mkdir_if_missing(save_dir)
        np.save(save_dir, concatenated_tensor.cpu())
        np_fake = np.array(concatenated_tensor.detach().cpu())
        if self.dataset.name == "sine":
            np_origin = np.load(
                os.path.join(
                    save_dir, f"sine_ground_truth_{self.dataset.window}_train.npy"
                )
            )
        else:
            np_origin = np.load(
                os.path.join(
                    save_dir,
                    f"{self.dataset.name}_norm_truth_{self.dataset.window}_train.npy",
                )
            )

        print("======Discriminative Score======")
        ds = discriminative_score(
            ori_data=np_origin,
            fake_data=np_fake,
            iterations=5,
            length=self.dataset.window,
        )
        display_scores(ds)
        print("======Predictive Score======")
        ps = predictive_score(
            ori_data=np_origin,
            fake_data=np_fake,
            iterations=5,
            length=self.dataset.window,
        )
        display_scores(ps)
        print("======VDS Score======")
        vs = VDS_score(
            ori_data=np_origin,
            fake_data=np_fake,
            length=self.dataset.window,
        )
        print(f"VDS Score: {vs}")
        print("======FDDS Score======")
        bs = BMMD_score(
            ori_data=np_origin,
            fake_data=np_fake,
            length=self.dataset.window,
        )
        print(f"BMMD Score: {bs}")

    def sampling(self, use_ddim=False):
        """Generate samples given the model."""
        model = self.model
        diffusion = self.diffusion
        model.eval()
        with torch.no_grad():
            sample_fn = (
                diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
            )
            res = []
            for i in tqdm(range((self.dataset.sample_num // self.batch_size) + 1)):
                res.append(
                    sample_fn(
                        model,
                        (self.batch_size, self.dataset.window, self.dataset.var_num),
                        clip_denoised=True,
                    )
                )
        concatenated_tensor = torch.cat(res, dim=0)
        return concatenated_tensor

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def next_batch(self):
        """
        Get the next batch of data.
        """
        while True:
            for batch in self.dataloader:
                yield batch

    def save(self):
        def save_checkpoint(rate, params):
            if not rate:
                state_dict = self.model.state_dict()
                filename = f"model_{(self.step):06d}.pt"
            else:
                state_dict = self._ema_params_to_state_dict(params)
                filename = f"ema_{rate}_{(self.step):06d}.pt"
            torch.save(
                {
                    "step": self.step,
                    "model_state_dict": state_dict,
                    "opt_state_dict": self.opt.state_dict(),
                },
                f"{self.save_dir}{filename}",
            )

        save_checkpoint(0, self.model)
