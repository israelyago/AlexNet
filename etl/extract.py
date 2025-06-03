import os
import sys
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_HOME") is None:
    print("HuggingFace HF_HOME environment variable not set.")
    sys.exit(1)

from datasets import load_dataset


ds = load_dataset("imagenet-1k")
print("Dataset downloaded. Len:", len(ds))
