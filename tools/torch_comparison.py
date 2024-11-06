import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

FEAT_DIM = 784
NUM_EPOCHS = 10
NUM_LAYERS = 4
BATCH_SIZE = 64
LEARNING_RATE = 0.1


class MLP(nn.Module):
    def __init__(self, feat_dim: int, num_layers: int) -> None:
        super(MLP, self).__init__()
        layers = [
            (
                nn.Linear(feat_dim, feat_dim)
                if i < num_layers - 1
                else nn.Linear(feat_dim, 10)
            )
            for i in range(num_layers)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(-1, FEAT_DIM))


def get_dataloader(data_dir: Path) -> DataLoader:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_data = datasets.MNIST(
        data_dir, train=True, download=False, transform=transform
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def train_loop(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    log_filepath: Path,
) -> None:
    train_losses: List[float] = []
    with log_filepath.open("w") as f:
        for epoch in range(NUM_EPOCHS):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_losses.append(loss.item())
                f.write(f"{loss.item()}\n")
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item():.4f}")


def main(data_dir: Path, log_file: Path) -> None:
    train_loader = get_dataloader(data_dir)
    model = MLP(FEAT_DIM, NUM_LAYERS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train_loop(train_loader, model, criterion, optimizer, log_file)
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST data.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Path to the MNIST data directory")
    parser.add_argument("--log_file", type=Path, required=True, help="File path to save the training losses")
    
    args = parser.parse_args()
    main(args.data_dir, args.log_file)
