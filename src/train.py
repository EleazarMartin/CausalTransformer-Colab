from src.models.ct_model import CTModel
from src.utils import CTDataModule

def load_model_from_config(config):
    datamodule = CTDataModule(config)
    model = CTModel(config)
    return model, datamodule
