from utils.data.datasetbase import MM_TS_Dataset
from utils.engine.builder import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class demo(MM_TS_Dataset):
    def __init__(self, cfg):
        super().__init__(cfg)
