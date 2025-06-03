import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_HOME") is None:
    print("HuggingFace HF_HOME environment variable not set.")
    sys.exit(1)
import mlflow
import lightning as L
import torch
import arguments
from dataset import ImageNetPatchDataset
import logs
from lightning.pytorch.loggers import MLFlowLogger
from model import AlexNet
from torch.utils.data import DataLoader
import optuna
from datasets import load_dataset
from pathlib import Path
from torchvision.transforms import v2
from PIL import Image
from torchinfo import summary
import h5py
import cProfile
import pstats
import tqdm

args = arguments.get_args()
logger = logs.get_logger("train")

current_directory = Path.cwd()

H5_DATASET_PATH = "dataset/imagenet_1k_256x256_float32-8.h5"
MLFLOW_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "AlexNet"
SAVE_MODEL_DIR = "artifacts"

mlflow.set_tracking_uri(uri=MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🤖 Using device: {device}")


# Helper functions
def get_dataloader(dataset, params):
    dataset = ImageNetPatchDataset(dataset)
    return DataLoader(
        dataset,
        batch_size=params["batch_size"],
        num_workers=os.cpu_count() // 2,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=4,
        in_order=False,
    )


config = {
    "batch_size": 128,  # Original paper: 128
    "lr": 0.01,  # "We used an equal learning rate for all layers, which we adjusted manually throughout training. The heuristic which we followed was to divide the learning rate by 10 when the validation error rate stopped improving with the current learning rate."
    "epochs": 1,  # Original paper: 90
}

logger.info(f"☁️  Loading dataset...")
hdf5 = h5py.File(H5_DATASET_PATH, "r")

train_dataloader = DataLoader(
    ImageNetPatchDataset(hdf5, "train"),
    batch_size=config["batch_size"],
    num_workers=os.cpu_count() // 2,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=4,
    in_order=False,
)

val_dataloader = DataLoader(
    ImageNetPatchDataset(hdf5, "validation"),
    batch_size=config["batch_size"],
    num_workers=os.cpu_count() // 2,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=4,
    in_order=False,
)

mlf_logger = MLFlowLogger(
    experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_URI
)
model = AlexNet(params=config)

trainer = L.Trainer(
    default_root_dir=Path(SAVE_MODEL_DIR),
    max_epochs=config["epochs"],
    logger=mlf_logger,
    enable_progress_bar=True,
    # profiler="simple",
    # profiler="advanced",
    # profiler="pytorch",
)

trainer.fit(model, train_dataloader, val_dataloader)

# trainer.test(model, test_dataloader)

logger.info("👋 Everything OK")
