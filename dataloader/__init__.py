from .scoliosis_dataloader import ScoliosisDataset

def load_dataset(dataset, data_dir, mode):
    if dataset == "scoliosis":
        # Scoliosis Dataset
        return ScoliosisDataset(data_dir, mode, 'U-Net')