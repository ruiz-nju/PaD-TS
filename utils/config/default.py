from yacs.config import CfgNode as CN

###########################
# Config definition
###########################
_C = CN()
# Directory to save the output files (like log.txt and model weights)
_C.OUTPUT_DIR = "./output"
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.USE_CUDA = True
# Print detailed information
_C.VERBOSE = True
_C.PERIOD = "train"

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ""
_C.DATASET.NAME = ""
_C.DATASET.PROPORTION = 1.0
_C.DATASET.SAVE2NPY = True
_C.DATASET.NEG_ONE_TO_ONE = True
_C.DATASET.DIM = 6
_C.DATASET.WINDOW = 24
_C.DATASET.PREDICT_LENGTH = None
_C.DATASET.MISSING_RATIO = None
_C.DATASET.STYLE = "separate"
_C.DATASET.DISTRIBUTION = "geometric"
_C.DATASET.MEAN_MASK_LENGTH = 3
_C.DATASET.NUM = -1

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 0
_C.DATALOADER.BATCH_SIZE = 64
_C.DATALOADER.PIN_MEMORY = True
_C.DATALOADER.DROP_LAST = True
_C.DATALOADER.SHUFFLE = True

###########################
# Model
###########################
_C.MODEL = CN()
_C.MODEL.NAME = ""
###########################
# Train
###########################
_C.TRAIN = CN()
_C.TRAIN.MICROBATCH = -1
_C.TRAIN.SAVE_INTERVAL = 5e3
_C.TRAIN.MMD_ALPHA = 0.0008
_C.TRAIN.LOG_INTERVAL = 10

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.LR = 1e-4
_C.OPTIM.WEIGHT_DECAY = 0.0
_C.OPTIM.LR_ANNEAL_STEPS = 70000
###########################
# Diffusion
###########################
_C.DIFFUSION = CN()
_C.DIFFUSION.PREDICT_XSTART = True
_C.DIFFUSION.DIFFUSION_STEPS = 250
_C.DIFFUSION.NOISE_SCHEDULE = "cosine"
_C.DIFFUSION.LOSS = "MSE_MMD"
_C.DIFFUSION.RESCALE_TIMESTEPS = False
_C.DIFFUSION.SCHEDULE_SAMPLER = "batch"

###########################
# Trainer
###########################
# _C.TRAINER = CN()


def get_cfg_default():
    return _C.clone()
