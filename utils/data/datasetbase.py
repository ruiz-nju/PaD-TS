"""
Modifed based on code from: https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/Data_utils/real_datasets.py
"""

import pdb
import os
import torch
import numpy as np
import pandas as pd
from scipy import io
from utils.tools import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class MM_TS_Dataset:
    """
     Multi-Modal Time-Series Dataset
    -------------------------------------------------
    | Date | Dim_0 | Dim_1 | ... | Dim_{n-1} | Text |
    -------------------------------------------------

    dataset = MM_TS_Dataset(cfg)
    train = dataset.train

    train["numerical"].shape  # (group, window, dim)
    type(train["numerical"])  # numpy.ndarray

    type(train["text"])  # list (group * window)
    """

    def __init__(self, cfg):
        super().__init__()
        self.proportion = cfg.DATASET.PROPORTION
        file = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.NAME + ".csv")
        if not os.path.exists(file):
            raise FileNotFoundError(f"File not found: {file}")
        df = pd.read_csv(file)
        self.group = len(df) - cfg.DATASET.WINDOW + 1
        if self.group <= 0:
            raise ValueError("The window size is too large.")
        self.window = cfg.DATASET.WINDOW
        self.dim = cfg.DATASET.DIM

        # 分割数据并转换为字典形式，train 和 test 中每个数据窗口的形状为 (group, window, dim)
        self.train, self.test = self.split_data(df, self.proportion)

    def split_data(self, df, proportion):
        split_index = int(len(df) * proportion)
        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df = df.iloc[split_index:].reset_index(drop=True)
        return self.dataframe_to_windows(
            train_df, self.window
        ), self.dataframe_to_windows(test_df, self.window)

    def dataframe_to_windows(self, df, window):
        # 数值数据转换为 numpy 数组，shape: (n, d)
        numerical = df.iloc[:, 1:-1].to_numpy()
        text = df.iloc[:, -1].tolist()
        numerical_windows = np.array(
            [numerical[i : i + window] for i in range(self.group)]
        )
        text_windows = [text[i : i + window] for i in range(self.group)]
        return {"numerical": numerical_windows, "text": text_windows}


class CustomDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.PERIOD in ["train", "test"], "period must be train or test."
        if cfg.PERIOD == "train":
            assert ~(
                cfg.DATASET.PREDICT_LENGTH is not None
                or cfg.DATASET.MISSING_RATIO is not None
            ), ""
        self.name, self.pred_len, self.missing_ratio = (
            cfg.DATASET.NAME,
            cfg.DATASET.PREDICT_LENGTH,  # None
            cfg.DATASET.MISSING_RATIO,  # None
        )
        self.style, self.distribution, self.mean_mask_length = (
            cfg.DATASET.STYLE,  # 'separate'
            cfg.DATASET.DISTRIBUTION,  # 'geometric'
            cfg.DATASET.MEAN_MASK_LENGTH,  # 3
        )
        # self.rawdata.shape = (3685, 6) 此处直接将原始数据读入
        self.rawdata, self.scaler = self.read_data(cfg.DATASET.ROOT, self.name)
        self.dir = os.path.join(cfg.OUTPUT_DIR, "samples")
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = cfg.DATASET.WINDOW, cfg.PERIOD  # 24, 'train'
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]  # 3685 6
        self.sample_num_total = max(self.len - self.window + 1, 0)  # 样本组的数量 3662
        self.save2npy = cfg.DATASET.SAVE2NPY  # True
        self.auto_norm = cfg.DATASET.NEG_ONE_TO_ONE  # True
        self.porportion = cfg.DATASET.PROPORTION  # 1.0
        self.data = self.__normalize(self.rawdata)  # 归一化后的数据
        self.seed = cfg.SEED
        train, inference = self.__getsamples(self.data, self.porportion, self.seed)

        self.samples = train if self.period == "train" else inference
        if self.period == "test":
            if self.missing_ratio is not None:
                self.masking = self.mask_data(self.seed)
            elif cfg.DATASET.PREDICT_LENGTH is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -cfg.DATASET.PREDICT_LENGTH :, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def __getsamples(self, data, proportion, seed):
        x = np.zeros(
            (self.sample_num_total, self.window, self.var_num)
        )  # （样本组数，窗口长度，变量数量)
        for i in range(self.sample_num_total):
            # 通过滑动窗口的方式读取
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        # 如果 save2npy 为 True，则将划分好的数据保存为 .npy 文件
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"
                    ),
                    self.unnormalize(test_data),
                )
            np.save(
                os.path.join(
                    self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"
                ),
                self.unnormalize(train_data),
            )
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"
                        ),
                        unnormalize_to_zero_to_one(test_data),
                    )
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"
                    ),
                    unnormalize_to_zero_to_one(train_data),
                )
            else:
                if 1 - proportion > 0:
                    np.save(
                        os.path.join(
                            self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"
                        ),
                        test_data,
                    )
                np.save(
                    os.path.join(
                        self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"
                    ),
                    train_data,
                )

        return train_data, test_data

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        # 先缩放到[0, 1]，在缩放到[-1, 1]
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # 保存当前的随机数生成器（RNG）的状态，以便后续恢复
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=""):
        """Reads a single .csv"""
        file = os.path.join(filepath, name + ".csv")
        df = pd.read_csv(file, header=0)  # header=0 表示第一行为列标题
        if name == "etth":
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = self.noise_mask(
                x,
                self.missing_ratio,
                self.mean_mask_length,
                self.style,
                self.distribution,
            )  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(
                os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks
            )

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


def noise_mask(
    X,
    masking_ratio,
    lm=3,
    mode="separate",
    distribution="geometric",
    exclude_feats=None,
):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
        mode: whether each variable should be masked separately ('separate'), or all variables at a certain positions
            should be masked concurrently ('concurrent')
        distribution: whether each mask sequence element is sampled independently at random, or whether
            sampling follows a markov chain (and thus is stateful), resulting in geometric distributions of
            masked squences of a desired mean length `lm`
        exclude_feats: iterable of indices corresponding to features to be excluded from masking (i.e. to remain all 1s)

    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    pass
