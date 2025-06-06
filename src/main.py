import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("DATASET_FILE_DIR") is None:
    print("DATASET_FILE_DIR environment variable not set.")
    sys.exit(1)

if os.getenv("MLFLOW_URI") is None:
    print("MLFLOW_URI environment variable not set.")
    sys.exit(1)

import mlflow
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import arguments
from dataset import ImageNetPatchDataset
import logs
from lightning.pytorch.loggers import MLFlowLogger
from model import AlexNet, alexnet_initialize_weights
from torch.utils.data import DataLoader
from pathlib import Path
import h5py

args = arguments.get_args()
logger = logs.get_logger("train")

current_directory = Path.cwd()

H5_DATASET_PATH = Path(os.getenv("DATASET_FILE_DIR"))
MLFLOW_URI = os.getenv("MLFLOW_URI")
MLFLOW_EXPERIMENT_NAME = "AlexNet"
SAVE_MODEL_DIR = "artifacts"

mlflow.set_tracking_uri(uri=MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🤖 Using device: {device}")

config = {
    "batch_size": 128,  # Original paper: 128
    "lr": 0.1,  # Original paper: 0.01  # "We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate."
    "epochs": 90,  # Original paper: 90
}

logger.info(f"☁️  Loading dataset...")
hdf5 = h5py.File(H5_DATASET_PATH, "r")

train_dataloader = DataLoader(
    ImageNetPatchDataset(hdf5, "train"),
    batch_size=config["batch_size"],
    num_workers=2,
    shuffle=True,
    pin_memory=True,
    # prefetch_factor=2,
    # in_order=True,
)

val_dataloader = DataLoader(
    ImageNetPatchDataset(hdf5, "validation"),
    batch_size=config["batch_size"],
    num_workers=2,
    shuffle=False,
    pin_memory=True,
    # prefetch_factor=2,
    # in_order=True,
)

# test_dataloader = DataLoader(
#     ImageNetPatchDataset(hdf5, "test"),
#     batch_size=config["batch_size"],
#     num_workers=os.cpu_count() // 3,
#     shuffle=False,
#     pin_memory=True,
#     # prefetch_factor=2,
#     # in_order=True,
# )

mlf_logger = MLFlowLogger(
    experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_URI
)
model = AlexNet(params=config)
alexnet_initialize_weights(model)

checkpoint_callback = ModelCheckpoint(
    dirpath=Path(SAVE_MODEL_DIR),
    every_n_epochs=1,
    mode="min",
    monitor="val_loss",
    save_last=True,
)

trainer = L.Trainer(
    default_root_dir=Path(SAVE_MODEL_DIR),
    callbacks=[checkpoint_callback],
    max_epochs=config["epochs"],
    logger=mlf_logger,
    enable_progress_bar=True,
    gradient_clip_val=10.0,
    gradient_clip_algorithm="norm",
    # profiler="simple",
    # profiler="advanced",
    # profiler="pytorch",
)

trainer.fit(model, train_dataloader, val_dataloader)

# trainer.test(model, test_dataloader)

logger.info("👋 Everything OK")
