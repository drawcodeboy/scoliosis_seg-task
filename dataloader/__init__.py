from .scoliosis_dataloader import ScoliosisDataset
from .ICH_dataloader import ICHDataset

def load_dataset(dataset, mode):
    if dataset == "scoliosis":
        # Scoliosis Dataset
        return ScoliosisDataset('data/AIS.v1i.yolov8', mode, 'U-Net')
    elif dataset == "ICH_only":
        return ICHDataset("data/physionet.org/files/ct-ich/1.3.1/data_only", mode)
    elif dataset == "ICH_all":
        return ICHDataset("data/physionet.org/files/ct-ich/1.3.1/data_all", mode)