from utils.data.datasetbase import CustomDataset
from utils.engine.builder import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class energy(CustomDataset):
    def __init__(self, cfg):
        super().__init__(cfg)
