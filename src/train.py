from src.models.ct import CT
from src.data.cancer_sim.dataset import SyntheticDatasetCollection

def load_model_from_config(config):
    datamodule = SyntheticDatasetCollection(config)
    model = CT(args=config, dataset_collection=datamodule)
    return model, datamodule
