import os
from scipy import io
from sklearn.preprocessing import MinMaxScaler
from utils.data.datasetbase import CustomDataset
from utils.engine.builder import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class fMRI(CustomDataset):
    def __init__(self, cfg):
        super().__init__(cfg)

    @staticmethod
    def read_data(filepath, name=""):
        file = os.path.join(filepath, name, "sim4.mat")
        data = io.loadmat(file)["ts"]
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
