"""
Modifed based on code from https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/sine_dataset.py
"""

import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset
from utils.engine.builder import DATASET_REGISTRY
from utils.tools import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
import pdb
from utils.data.datasetbase import CustomDataset, noise_mask


@DATASET_REGISTRY.register()
class sine(Dataset):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.PERIOD in ["train", "test"], "period must be train or test."
        if cfg.PERIOD == "train":
            assert ~(
                cfg.DATASET.PREDICT_LENGTH is not None
                or cfg.DATASET.MISSING_RATIO is not None
            ), ""

        self.pred_len, self.missing_ratio = (
            cfg.DATASET.PREDICT_LENGTH,
            cfg.DATASET.MISSING_RATIO,
        )
        self.style, self.distribution, self.mean_mask_length = (
            cfg.DATASET.STYLE,
            cfg.DATASET.DISTRIBUTION,
            cfg.DATASET.MEAN_MASK_LENGTH,
        )

        self.dir = os.path.join(cfg.OUTPUT_DIR, "samples")
        os.makedirs(self.dir, exist_ok=True)

        self.rawdata = self.sine_data_generation(
            no=cfg.DATASET.NUM,
            seq_len=cfg.DATASET.WINDOW,
            dim=cfg.DATASET.DIM,
            save2npy=cfg.DATASET.SAVE2NPY,
            seed=cfg.SEED,
            dir=self.dir,
            period=cfg.PERIOD,
        )
        self.auto_norm = cfg.DATASET.NEG_ONE_TO_ONE
        self.samples = self.normalize(self.rawdata)
        self.var_num = cfg.DATASET.DIM
        self.sample_num = self.samples.shape[0]
        self.window = cfg.DATASET.WINDOW

        self.period, self.save2npy = cfg.PERIOD, cfg.DATASET.SAVE2NPY
        if cfg.PERIOD == "test":
            if cfg.DATASET.MISSING_RATIO is not None:
                self.masking = self.mask_data(cfg.SEED)
            elif cfg.DATASET.PREDICT_LENGTH is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -cfg.DATASET.PREDICT_LENGTH :, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data

    @staticmethod
    def sine_data_generation(
        no, seq_len, dim, save2npy=True, seed=123, dir="./", period="train"
    ):
        """Sine data generation.

        Args:
           - no: the number of samples
           - seq_len: sequence length of the time-series
           - dim: feature dimensions

        Returns:
           - data: generated data
        """
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        # Initialize the output
        data = list()
        # Generate sine data
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)

            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1) * 0.5
            # Stack the generated data
            data.append(temp)

        # Restore RNG.
        np.random.set_state(st0)
        data = np.array(data)
        if save2npy:
            np.save(
                os.path.join(dir, f"sine_ground_truth_{seq_len}_{period}.npy"), data
            )

        # print(f"Number of samples: {no}\nseq_len: {seq_len}\ndim: {dim}")
        # print(data.shape)
        return data

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"sine_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period == "test":
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num
