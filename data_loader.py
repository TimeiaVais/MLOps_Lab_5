import logging
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


def prepare_data(config):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transform = transforms.ToTensor()

    full_train = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data/raw",
        train=False,
        download=True,
        transform=test_transform
    )

    # ---------------- chunking ---------------- #

    num_chunks = config["data"]["num_chunks"]
    chunk_size = len(full_train) // num_chunks

    chunks = []

    for i in range(num_chunks):

        start = i * chunk_size
        end = len(full_train) if i == num_chunks - 1 else (i + 1) * chunk_size

        chunks.append(Subset(full_train, list(range(start, end))))

    train_chunks = [chunks[i] for i in config["data"]["train_chunks"]]
    val_chunks = [chunks[i] for i in config["data"]["val_chunks"]]

    train_dataset = ConcatDataset(train_chunks)
    val_dataset = ConcatDataset(val_chunks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False
    )

    logging.info("Data loaders created successfully")

    return train_loader, val_loader, test_loader