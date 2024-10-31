import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from datetime import datetime
from pathlib import Path

from pet_seg_core.config import PetSegTrainConfig
from pet_seg_core.data import train_dataloader, val_dataloader
from pet_seg_core.model import UNet

def train():
    curr_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')
    results_folder = f"results/{curr_time}"
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    with open(f"{results_folder}/description.txt", "w") as f:
        f.write(PetSegTrainConfig.DESCRIPTION_TEXT)

    logger = CSVLogger(save_dir="", name=results_folder, version="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=results_folder,
        save_top_k=-1,
    )
    trainer = pl.Trainer(
        max_epochs=PetSegTrainConfig.EPOCHS, fast_dev_run=PetSegTrainConfig.FAST_DEV_RUN, logger=logger, callbacks=[checkpoint_callback], gradient_clip_val=1.0
    )
    model = UNet(3, 3, channels_list=PetSegTrainConfig.CHANNELS_LIST, depthwise_sep=PetSegTrainConfig.DEPTHWISE_SEP)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
