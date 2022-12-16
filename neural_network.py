import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from data_processing import process_data


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class TorchDataset(Dataset):
    def __init__(self, x_values, y_values):
        X = x_values.iloc[0 : x_values.shape[0], 0 : x_values.shape[1]].values

        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = torch.tensor(y_values, dtype=torch.float32)

    def __len__(self):
        return len(self.X_tensor)

    def __getitem__(self, idx):
        return self.X_tensor[idx], self.y_tensor[idx]


class NeuralNetwork(nn.Module):
    def __init__(self, layers, num_nodes_per_layer):
        super().__init__()
        self.flatten = nn.Flatten()

        network_layers = []

        network_layers.append(nn.Linear(num_nodes_per_layer[0], num_nodes_per_layer[0]))
        network_layers.append(nn.Identity())

        for i in range(layers - 1):
            network_layers.append(
                nn.Linear(num_nodes_per_layer[i], num_nodes_per_layer[i + 1])
            )
            network_layers.append(nn.ReLU())

        network_layers[-1] = nn.Softmax()

        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
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


model = NeuralNetwork(5, [6, 4, 3]).to(device)
print(model)

batch_size = 25
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

(X_train, X_test, y_train, y_test, categories_mapping) = process_data(
    "star_classification.csv"
)

train_dataset = TorchDataset(X_train, y_train)
test_dataset = TorchDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
