from line_profiler import profile
from torch.utils.data import Dataset
import torch
from torch import nn
from torchvision.transforms import v2
from random import randrange

# --- Constants for Patching ---
IMAGE_PREPROCESSED_SIZE = 256
PATCH_SIZE = 224
PATCH_OFFSET_RANGE = IMAGE_PREPROCESSED_SIZE - PATCH_SIZE  # 32
BASE_PATCHES_GRID_SIZE = PATCH_OFFSET_RANGE * PATCH_OFFSET_RANGE  # 32 * 32 = 1024

# Total patches per original image (1024 non-flipped + 1024 flipped)
PATCHES_PER_ORIGINAL_IMAGE = BASE_PATCHES_GRID_SIZE * 2  # 2048
NUM_CLASSES = 1000

image_net_preprocess_transform = v2.Compose(
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


def get_single_patch(
    image_tensor: torch.Tensor, patch_idx_within_image: int
) -> torch.Tensor:
    """
    Extracts a single PATCH_SIZE x PATCH_SIZE patch from an IMAGE_PREPROCESSED_SIZE x IMAGE_PREPROCESSED_SIZE
    image tensor given its linear index, handling the first half non-flipped
    and second half horizontally flipped.

    Args:
        image_tensor (torch.Tensor): The preprocessed image tensor (C, H, W format, e.g., 3x256x256).
        patch_idx_within_image (int): The linear index of the patch (0 to PATCHES_PER_ORIGINAL_IMAGE - 1).

    Returns:
        torch.Tensor: The selected PATCH_SIZE x PATCH_SIZE patch (C, PATCH_SIZE, PATCH_SIZE),
                      potentially horizontally flipped.
    """
    # Determine if this patch should be flipped
    should_flip = patch_idx_within_image >= BASE_PATCHES_GRID_SIZE

    # Get the base index (0 to BASE_PATCHES_GRID_SIZE - 1) within the non-flipped grid
    base_linear_idx = patch_idx_within_image % BASE_PATCHES_GRID_SIZE

    # Calculate row (h_start) and column (w_start) for the base patch
    h_start = base_linear_idx // PATCH_OFFSET_RANGE
    w_start = base_linear_idx % PATCH_OFFSET_RANGE

    # Extract the base patch (C, H, W)
    patch = image_tensor[
        :, h_start : h_start + PATCH_SIZE, w_start : w_start + PATCH_SIZE
    ]

    # Apply horizontal flip if necessary.
    # For a (C, H, W) tensor, the W dimension is at index 2.
    if should_flip:
        patch = torch.flip(patch, dims=[2])

    return patch


class ImageNetPatchDataset(Dataset):
    def __init__(self, hdf5_dataset, split):

        VALID_SPLITS = ["train", "validation", "test"]
        if split not in VALID_SPLITS:
            raise BaseException(
                f"ImageNetPatchDataset split argument expected to be one of {VALID_SPLITS}, got {split}"
            )

        self.hdf5_dataset = hdf5_dataset
        self.split = split
        self.split_labels = f"{split}-labels"
        self.num_original_images = len(self.hdf5_dataset[split])

    def __len__(self):
        return self.num_original_images

    # @profile
    def __getitem__(self, idx):
        # Calculate which original image and which patch within that image to get
        original_image_idx = idx // PATCHES_PER_ORIGINAL_IMAGE
        patch_idx_within_image = randrange(
            PATCHES_PER_ORIGINAL_IMAGE
        )  # Randomly select one Patch from the image

        original_image = torch.from_numpy(
            self.hdf5_dataset[self.split][original_image_idx]
        )
        original_label = self.hdf5_dataset[self.split_labels][original_image_idx]

        selected_patch = get_single_patch(original_image, patch_idx_within_image)

        return selected_patch, torch.tensor(original_label, dtype=torch.long)


class ImageNetHuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, split):

        VALID_SPLITS = ["train", "validation", "test"]
        if split not in VALID_SPLITS:
            raise BaseException(
                f"ImageNetPatchDataset split argument expected to be one of {VALID_SPLITS}, got {split}"
            )

        self.hf_dataset = hf_dataset
        self.split = split
        self.split_labels = f"{split}-labels"
        self.num_original_images = len(self.hf_dataset[split])

    def __len__(self):
        return self.num_original_images * PATCHES_PER_ORIGINAL_IMAGE

    # @profile
    def __getitem__(self, idx):
        # Calculate which original image and which patch within that image to get
        original_image_idx = idx // PATCHES_PER_ORIGINAL_IMAGE
        patch_idx_within_image = idx % PATCHES_PER_ORIGINAL_IMAGE

        sample = self.hf_dataset[self.split][original_image_idx]
        original_image = sample["image"]
        original_label = sample["label"]

        # Apply the initial preprocessing (e.g., resize to 256x256)
        processed_256_image = image_net_preprocess_transform(original_image)

        selected_patch = get_single_patch(processed_256_image, patch_idx_within_image)

        label = nn.functional.one_hot(
            torch.tensor(original_label, dtype=torch.long), num_classes=NUM_CLASSES
        ).float()
        return selected_patch, label
