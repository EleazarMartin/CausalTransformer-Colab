import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.train import load_model_from_config
from src.models.utils import load_config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'], help='Mode to run')
    args = parser.parse_args()

    config = load_config(args.config)
    model, datamodule = load_model_from_config(config)

    if args.mode == 'train':
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            filename='best-checkpoint',
        )
        trainer = pl.Trainer(
            max_epochs=config['training']['epochs'],
            gpus=1 if config['training']['gpus'] else 0,
            callbacks=[checkpoint_callback],
            gradient_clip_val=config['training']['grad_clip']
        )
        trainer.fit(model, datamodule=datamodule)

    elif args.mode == 'evaluate':
        trainer = pl.Trainer(gpus=1 if config['training']['gpus'] else 0)
        trainer.test(model, datamodule=datamodule)
