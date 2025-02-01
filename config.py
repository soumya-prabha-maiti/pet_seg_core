from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PetSegTrainConfig:
    EPOCHS = 5
    BATCH_SIZE = 8
    FAST_DEV_RUN = False
    TOTAL_SAMPLES = 100
    LEARNING_RATE = 1e-3
    TRAIN_VAL_TEST_DATA_PATH = "./data/train_val_test"
    DEPTHWISE_SEP = False
    CHANNELS_LIST = [16, 32, 64, 128, 256]
    DESCRIPTION_TEXT = None


@dataclass
class PetSegWebappConfig:
    MODEL_WEIGHTS_GDRIVE_FILE_ID = os.environ.get("MODEL_WEIGHTS_GDRIVE_FILE_ID")
    MODEL_WEIGHTS_LOCAL_PATH = os.environ.get(
        "MODEL_WEIGHTS_LOCAL_PATH", "pet-segmentation-pytorch_epoch=4-step=1840.ckpt"
    )
    DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE = (
        os.environ.get("DOWNLOAD_MODEL_WEIGTHS_FROM_GDRIVE", "True") == "True"
    )
