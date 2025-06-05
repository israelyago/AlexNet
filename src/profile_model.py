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

# from torchvision.transforms.v2 import RandomHorizontalFlip

args = arguments.get_args()
logger = logs.get_logger("train")

current_directory = Path.cwd()

H5_DATASET_PATH = "dataset/imagenet_1k_256x256_float32.h5"
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
    "batch_size": 768,  # 32, 256, 512, 768
    "lr": 0.004398198831388467,
    "epochs": 1,
}

logger.info(f"☁️  Loading dataset...")
# dataset = load_dataset("imagenet-1k")
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
# logger.info(f"Running model on device: {model.device}")

# summary(model, input_size=(config["batch_size"], 3, 224, 224))

trainer = L.Trainer(
    # accelerator="gpu",
    default_root_dir=Path(SAVE_MODEL_DIR),
    max_epochs=config["epochs"],
    logger=mlf_logger,
    # enable_progress_bar=True,
    # profiler="simple",
    # profiler="advanced",
    # profiler="pytorch",
)
# ======================================================================
total_samples_computed = 256 * 1000
num_calls_to_profile = int(total_samples_computed / config["batch_size"])
logger.info(f"Profiling {num_calls_to_profile} calls to model.trainig_step()...")


# start_time = time.time()
# total_size = len(dataset)
model.cuda()
model.compile()
profiler = cProfile.Profile()
train_imgs_batch = torch.rand(config["batch_size"], 3, 224, 224).cuda()
train_labels_batch = (1000 * torch.rand(config["batch_size"])).long()
train_labels_batch = (
    torch.nn.functional.one_hot(train_labels_batch, 1000).float().cuda()
)
train_batch = (train_imgs_batch, train_labels_batch)
# print(train_labels_batch)
# exit(1)
profiler.enable()

for i in tqdm.trange(num_calls_to_profile):
    # _ = dataset[i % total_size]  # Cycle through indices
    _ = model.training_step(train_batch, batch_idx=0)
# end_time = time.time()

profiler.disable()

stats = pstats.Stats(profiler).sort_stats("cumtime")  # Sort by cumulative time
logger.info("\n--- cProfile Results (Top 10 by Cumulative Time) ---")
stats.print_stats(10)  # Print top 10 functions

logger.info("\n--- line_profiler Instructions ---")
logger.info(
    "To use line_profiler, uncomment the '@profile' decorator on model.trainig_step()."
)
logger.info("Then run in your terminal:")
logger.info("  kernprof -l src/main.py")
logger.info("After it runs, view the results with:")
logger.info("  python -m line_profiler my_dataset_profiler.py.lprof")

# ======================================================================

# trainer.fit(model, train_dataloader, val_dataloader)

# trainer.test(model, test_dataloader)

logger.info("👋 Everything OK")
