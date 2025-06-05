import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_HOME") is None:
    print("HuggingFace HF_HOME environment variable not set.")
    sys.exit(1)
from dataset import ImageNetPatchDataset, ImageNetHuggingFaceDataset
import logs
from datasets import load_dataset
import cProfile
import pstats
import h5py

H5_DATASET_PATH = "dataset/imagenet_1k_256x256_float32-6.h5"

logger = logs.get_logger("train")

logger.info(f"☁️  Loading dataset...")
# hdf5 = h5py.File(H5_DATASET_PATH, "r")
hf_dataset = load_dataset("imagenet-1k")
# dataset = ImageNetPatchDataset(hf_dataset, "train")
dataset = ImageNetHuggingFaceDataset(hf_dataset, "train")

# x, y = dataset[0]
# print(x.shape, "device=", x.device)
# print(y.shape, "device=", y.device)

num_calls_to_profile = 10000
logger.info(f"Profiling {num_calls_to_profile} calls to __getitem__...")


# start_time = time.time()
total_size = len(dataset)

profiler = cProfile.Profile()
profiler.enable()

for i in range(num_calls_to_profile):
    _ = dataset[i % total_size]  # Cycle through indices
# end_time = time.time()

profiler.disable()
# hdf5.close()

# print(f"It took {end_time - start_time}s to open 1000 indexes")
# stats = pstats.Stats(profiler).sort_stats("cumtime")  # Sort by cumulative time
logger.info("\n--- cProfile Results (Top 10 by Cumulative Time) ---")
# stats.print_stats(10)  # Print top 10 functions

# --- Option 2: Using line_profiler (Line-by-line profiling) ---
# 1. Install it: pip install line_profiler
# 2. Add @profile decorator to the __getitem__ method (as shown in the code above)
# 3. Run from terminal: kernprof -l my_dataset_profiler.py
# 4. View results: python -m line_profiler my_dataset_profiler.py.lprof
logger.info("\n--- line_profiler Instructions ---")
logger.info("To use line_profiler, uncomment the '@profile' decorator on __getitem__.")
logger.info("Then run in your terminal:")
logger.info("  kernprof -l src/profile_dataset.py")
logger.info("After it runs, view the results with:")
logger.info("  python -m line_profiler my_dataset_profiler.py.lprof")
