from typing import Dict, List

import torch
import matplotlib.pyplot as plt
from torch import Tensor, nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


_Optimizer = torch.optim.Optimizer


# 학습에 사용할 모델 정의
class MNISTNetwork(nn.Module):
    def __init__(self) -> None:
        super(MNISTNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# 학습 코드
def train(
    dataloader: DataLoader,
    device: str,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: _Optimizer,
) -> None:
    size = len(dataloader.dataset)
    model.train()

    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        train_loss += loss

    return train_loss / size


# 테스트 코드
def test(
    dataloader: DataLoader, device: str, model: nn.Module, loss_fn: nn.Module
) -> None:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# 추론 코드
def predict(test_data: Dataset, model: nn.Module) -> None:
    model.eval()
    x = test_data[0][0]
    y = test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = pred[0].argmax(0), y
        print(f"Predicted: {predicted}, Actual: {actual}")


# 학습과 테스트 실행하는 코드
def run_pytorch(batch_size: int, epochs: int) -> None:
    # 데이터 불러오기
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=16)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=8)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MNISTNetwork().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epoch_loss = []
    # 학습과 테스트 에폭
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss = train(train_dataloader, device, model, loss_fn, optimizer)
        epoch_loss.append(train_loss.detach().cpu().numpy())
        test(test_dataloader, device, model, loss_fn)
    print("Done!")

    model = MNISTNetwork()
    predict(test_data, model)

    plt.plot(range(1, epochs + 1), epoch_loss, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_pytorch(batch_size=128, epochs=5)
