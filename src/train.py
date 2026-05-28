import os
import yaml
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.pytorch

import wandb

from data_loader import prepare_data
from model import SimpleCNN


# ---------------- LOGGING ---------------- #

logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# ---------------- BASE PARAMS ---------------- #

def load_params():

    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


# ---------------- EVALUATION ---------------- #

def evaluate_model(model, loader, criterion):

    model.eval()

    total_loss = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return total_loss, acc, precision, recall, f1


# ---------------- TRAIN FUNCTION ---------------- #

def train(config, run_name):

    train_loader, val_loader = prepare_data(config)

    model = SimpleCNN(
        config["model"]["num_classes"]
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    # ---------------- MLflow ---------------- #

    mlflow.set_experiment("CIFAR10_DVC_MLOPS")

    mlflow.start_run(run_name=run_name)

    mlflow.log_params(config["training"])
    mlflow.log_params(config["data"])

    # ---------------- W&B ---------------- #

    wandb.init(
        project="CIFAR10_MLOPS",
        name=run_name,
        config=config
    )

    best_val_loss = float("inf")

    logging.info(f"Training started: {run_name}")

    for epoch in range(config["training"]["epochs"]):

        model.train()

        train_loss = 0

        for images, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        # ---------------- VALIDATION ---------------- #

        val_loss, acc, prec, rec, f1 = evaluate_model(
            model,
            val_loader,
            criterion
        )

        avg_train_loss = train_loss / len(train_loader)

        # ---------------- MLflow LOGGING ---------------- #

        mlflow.log_metric(
            "train_loss",
            avg_train_loss,
            step=epoch
        )

        mlflow.log_metric(
            "val_loss",
            val_loss,
            step=epoch
        )

        mlflow.log_metric(
            "accuracy",
            acc,
            step=epoch
        )

        mlflow.log_metric(
            "f1_score",
            f1,
            step=epoch
        )

        # ---------------- W&B LOGGING ---------------- #

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        logging.info(
            f"Epoch {epoch+1} "
            f"Train Loss {avg_train_loss:.4f} "
            f"Val Loss {val_loss:.4f} "
            f"Accuracy {acc:.4f}"
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss {avg_train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Accuracy {acc:.4f}"
        )

        # ---------------- SAVE BEST MODEL ---------------- #

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            os.makedirs("artifacts", exist_ok=True)

            torch.save(
                model.state_dict(),
                f"artifacts/{run_name}_best_model.pth"
            )

    # ---------------- LOG ARTIFACTS ---------------- #

    mlflow.pytorch.log_model(
        model,
        "model"
    )

    mlflow.log_artifact(
        f"artifacts/{run_name}_best_model.pth"
    )

    wandb.save(
        f"artifacts/{run_name}_best_model.pth"
    )

    # ---------------- FINISH ---------------- #

    mlflow.end_run()

    wandb.finish()

    logging.info(f"Training finished: {run_name}")


# ======================================================
# CONFIG DICTIONARIES
# ====================================================== #

base_config = load_params()


config_small = {
    **base_config,

    "training": {
        "epochs": 5,
        "learning_rate": 0.001
    },

    "data": {
        "batch_size": 64,
        "num_chunks": 10,
        "train_chunks": [0, 1],
        "val_chunks": [2]
    }
}


config_medium = {
    **base_config,

    "training": {
        "epochs": 10,
        "learning_rate": 0.0005
    },

    "data": {
        "batch_size": 64,
        "num_chunks": 10,
        "train_chunks": [0, 1, 2, 3, 4],
        "val_chunks": [5]
    }
}


config_large = {
    **base_config,

    "training": {
        "epochs": 15,
        "learning_rate": 0.0001
    },

    "data": {
        "batch_size": 64,
        "num_chunks": 10,
        "train_chunks": [0, 1, 2, 3, 4, 5, 6],
        "val_chunks": [7, 8]
    }
}


# ======================================================
# MULTIPLE RUNS
# ====================================================== #

if __name__ == "__main__":

    train(
        config_small,
        "Run_1_Small_Config"
    )

    train(
        config_medium,
        "Run_2_Medium_Config"
    )

    train(
        config_large,
        "Run_3_Large_Config"
    )
```
