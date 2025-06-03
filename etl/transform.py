import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_HOME") is None:
    print("HuggingFace HF_HOME environment variable not set.")
    sys.exit(1)

import h5py
import torch
from datasets import load_dataset
from torchvision.transforms import v2
import os
import numpy as np
import tqdm
import time
import traceback

OUTPUT_H5_PATH = "dataset/imagenet_1k_256x256_float32.h5"
BATCH_SIZE = 8192

# Define preprocessing transforms
preprocess_transform = v2.Compose(
    [
        v2.Resize(256, antialias=True),
        v2.CenterCrop(256),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # ImageNet normalization
    ]
)


def process_single_index(index, dataset, transform):
    try:
        sample = dataset[index]
        pil_image, label = sample["image"], sample["label"]
        preprocessed_tensor = transform(pil_image)
        return preprocessed_tensor.numpy(), label
    except Exception as e:
        print(traceback.format_exc())
        print(f"There was en error Processing index {index}")
        sys.exit(1)


def process_batch(original_dataset, index, images_dset, labels_dset):

    num_images = len(original_dataset)
    start_index = index * BATCH_SIZE
    max_index = min(start_index + BATCH_SIZE, num_images)

    batch_tensors = []
    batch_labels = []
    indices_to_process = list(range(start_index, max_index))

    results = [
        process_single_index(i, original_dataset, preprocess_transform)
        for i in indices_to_process
    ]

    # Collect results
    for image, label in results:
        batch_tensors.append(image)
        batch_labels.append(label)

    images_dset[start_index:max_index] = batch_tensors
    labels_dset[start_index:max_index] = batch_labels


if __name__ == "__main__":

    # Create HDF5 file
    with h5py.File(OUTPUT_H5_PATH, "w") as f:
        for dataset_split in ["validation", "test", "train"]:

            original_dataset = load_dataset("imagenet-1k")[dataset_split]
            num_images = len(original_dataset)
            if dataset_split == "train":
                indices_to_keep = list(range(num_images))
                print("Removing corrupted entries")
                del indices_to_keep[883575]
                del indices_to_keep[1159337]  # Beware of the order of deletion!
                original_dataset = original_dataset.select(indices_to_keep)

            images_dset = f.create_dataset(
                dataset_split,
                shape=(num_images, 3, 256, 256),  # (N, C, H, W)
                dtype=np.float32,
            )

            # Create a dataset for labels
            labels_dset = f.create_dataset(
                f"{dataset_split}-labels",
                shape=(num_images,),
                dtype=np.int16,
            )

            # Populate the HDF5 datasets
            print(
                f"Preprocessing and saving {num_images} (split: {dataset_split}) images to {OUTPUT_H5_PATH}..."
            )
            print(f"Batch Size: {BATCH_SIZE}")

            start_time = time.time()
            for i in tqdm.tqdm(range(0, (num_images // BATCH_SIZE) + 1)):
                process_batch(original_dataset, i, images_dset, labels_dset)

            print(f"It took {time.time() - start_time} to run")

    print("Preprocessing complete and saved to HDF5.")
