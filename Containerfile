FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENV NVIDIA_VISIBLE_DEVICES=all
ENV MLFLOW_URI="http://host.docker.internal:5000"

COPY . /app

CMD ["python", "src/main.py"]