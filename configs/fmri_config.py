import argparse
import gaussian_diffusion_loss as gd


class Training_args(argparse.Namespace):
    data_dir = ""
    schedule_sampler = "batch"
    lr = 1e-4
    weight_decay = 0.0
    lr_anneal_steps = 70000
    microbatch = -1  # -1 disables microbatches
    log_interval = 10
    save_interval = 5e3
    mmd_alpha = 0.00002
    save_dir = "./OUTPUT/fmri_24/"


class Model_args(argparse.Namespace):
    hidden_size = 256
    num_heads = 4
    n_encoder = 1
    n_decoder = 3
    feature_last = True
    mlp_ratio = 4.0
    input_shape = (24, 50)


class Diffusion_args(argparse.Namespace):
    predict_xstart = True
    diffusion_steps = 500
    noise_schedule = "cosine"
    loss = "MSE_MMD"
    rescale_timesteps = False


class DataLoader_args(argparse.Namespace):
    batch_size = 32
    shuffle = True
    num_workers = 0
    drop_last = True
    pin_memory = True


class Data_args(argparse.Namespace):
    name = "fmri"
    data_root = "./dataset/fMRI"
    window = 24
    save2npy = True
    neg_one_to_one = True
    seed = 123
    period = "train"
    dim = 50
    proportion = 1.0
