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

## Running locally with conda/mamba

1. Create a `.env` file inside the root of this project, and define the ENV variable:
    - HF_HOME="~/.cache/huggingface" (Or your preferred location to store the HuggingFace cache)
1. Start a local MLflow server inside a container:

    ```bash
    podman run -d -p 5000:5000 -v ./mlruns:/mlflow/mlruns ghcr.io/mlflow/mlflow mlflow server --backend-store-uri /mlflow/mlruns --default-artifact-root /mlflow/mlruns --host 0.0.0.0
    ```

Then:

1. Create a new conda/mamba env: `mamba env create -n alex-net -f environment.yaml`
1. Activate the environment with `mamba activate alex-net`
1. Run `python src/main.py` to train the model

Monitor the training with MLflow

## Development

When changing the dependencies of mamba:

1. Run: `mamba env export --from-history > environment.yaml`
1. Edit `environment.yaml` (Remove the prefix field and change the name field to "base")
