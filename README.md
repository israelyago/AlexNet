# AlexNet Image classification

## Data

### Requirements

The data is acquired from HuggingFace [imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) (needs login).

This project needs some available space of at least:

|                  | Size (GB)  |
|------------------|-----------:|
| Raw data         | 333.3      |
| Transformed data | 1125.5     |
| Total            | 1458.8     |

### ETL

From the root of this project, run:

1. `python etl/extract.py`
1. `python etl/transform.py`

The local folder `dataset` should have the following structure:

```bash
dataset
└── imagenet_1k_256x256_float32.h5
```

## Running locally with pip
1. Run `python -m venv .venv`
1. Activate the virtual environment with `source .venv/bin/activate`
1. Install pip dependencies `pip install -r requirements.txt`
1. Start a local MLflow server inside a container:

    ```bash
    podman run -d -p 5000:5000 -v ./mlruns:/mlflow/mlruns ghcr.io/mlflow/mlflow mlflow server --backend-store-uri /mlflow/mlruns --default-artifact-root /mlflow/mlruns --host 0.0.0.0
    ```
1. Run `python src/main.py` to train the model

## Running locally with Podman/Docker
1. Build the image: `podman build -t alex-net:local .`
1. Example of run command (Adjust to your GPU config):
    ```bash
    podman run \
    -v "YOUR_DATASET_DIR":/app/dataset \
    -e DATASET_FILE_DIR="/app/dataset/imagenet_1k_256x256_float32.h5" \
    --device nvidia.com/gpu=all \
    --shm-size=512m \
    alex-net:local
    ```

## Development

When changing the dependencies of pip:

1. Run: `pip freeze > requirements.txt`
