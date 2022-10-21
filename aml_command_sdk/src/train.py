"""Training and evaluation."""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple

import mlflow
import numpy as np
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema, TensorSpec
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork
from utils_train_nn import evaluate, fit

DATA_DIR = "aml_command_sdk/data"
MODEL_DIR = "aml_command_sdk/model/"


def load_train_val_data(
    data_dir: str, batch_size: int, training_fraction: float
) -> tuple[DataLoader[torch.Tensor], DataLoader[torch.Tensor]]:
    """
    Returns two DataLoader objects that wrap training and validation data.
    Training and validation data are extracted from the full original training
    data, split according to training_fraction.
    """
    full_train_data = datasets.FashionMNIST(data_dir,
                                            train=True,
                                            download=True,
                                            transform=ToTensor())
    full_train_len = len(full_train_data)
    train_len = int(full_train_len * training_fraction)
    val_len = full_train_len - train_len
    (train_data, val_data) = random_split(dataset=full_train_data,
                                          lengths=[train_len, val_len])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    return (train_loader, val_loader)


def save_model(model_dir: str, model: nn.Module) -> None:
    """
    Saves the trained model.
    """
    input_schema = Schema(
        [ColSpec(type="double", name=f"col_{i}") for i in range(784)])
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    code_paths = ["neural_network.py", "utils_train_nn.py"]
    full_code_paths = [
        Path(Path(__file__).parent, code_path) for code_path in code_paths
    ]
    shutil.rmtree(model_dir, ignore_errors=True)
    logging.info("Saving model to %s", model_dir)
    mlflow.pytorch.save_model(pytorch_model=model,
                              path=model_dir,
                              code_paths=full_code_paths,
                              signature=signature)


def train(data_dir: str, model_dir: str, device: str) -> None:
    """
    Trains the model for a number of epochs, and saves it.
    """
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataloader,
     val_dataloader) = load_train_val_data(data_dir, batch_size, 0.8)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        logging.info("Epoch %d", epoch + 1)
        (training_loss, training_accuracy) = fit(device, train_dataloader,
                                                 model, loss_fn, optimizer)
        (validation_loss,
         validation_accuracy) = evaluate(device, val_dataloader, model, loss_fn)

        metrics = {
            "training_loss": training_loss,
            "training_accuracy": training_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy
        }
        mlflow.log_metrics(metrics, step=epoch)

    save_model(model_dir, model)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    parser.add_argument("--model_dir", dest="model_dir", default=MODEL_DIR)
    args = parser.parse_args()
    logging.info("input parameters: %s", vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train(**vars(args), device=device)


if __name__ == "__main__":
    main()
