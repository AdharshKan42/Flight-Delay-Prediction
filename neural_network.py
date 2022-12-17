import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from time import perf_counter

from data_processing import process_data


device = "cpu"
print(f"Using {device} device")


class TorchDataset(Dataset):
    def __init__(self, x_values, y_values):
        X = x_values.iloc[0 : x_values.shape[0], 0 : x_values.shape[1]].values

        self.X_tensor = torch.tensor(X, dtype=torch.float32)
        self.y_tensor = y_values

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

        network_layers.pop(-1)

        self.network = nn.Sequential(*network_layers)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


correct_per_class_train = [0, 0, 0]
totals_per_class_train = [0, 0, 0]

correct_per_class_test = [0, 0, 0]
totals_per_class_test = [0, 0, 0]


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    all_losses = []
    correct = 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X, y

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_losses.append(loss.item())
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        for i in range(len(pred)):
            if pred[i].argmax(0) == y[i]:
                correct_per_class_train[y[i]] += 1

            totals_per_class_train[y[i]] += 1

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss = sum(all_losses) / len(all_losses)
    correct /= len(dataloader.dataset)
    print(
        f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n"
    )

    return train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X, y
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            for i in range(len(pred)):
                if pred[i].argmax(0) == y[i]:
                    correct_per_class_test[y[i]] += 1

                totals_per_class_test[y[i]] += 1

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )

    return test_loss


model = NeuralNetwork(3, [6, 4, 3]).to(device)
print(model)

batch_size = 20
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

(X_train, X_test, y_train, y_test, categories_mapping) = process_data(
    "star_classification.csv"
)

train_dataset = TorchDataset(X_train, y_train)
test_dataset = TorchDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

train_avg_losses = []
test_avg_losses = []

time_taken = 0

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    start_time = perf_counter()
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    end_time = perf_counter()
    time_taken += end_time - start_time
    train_avg_losses.append(train_loss)

    test_loss = test(test_dataloader, model, loss_fn)
    test_avg_losses.append(test_loss)

print("Done!")
print(f"Training the neural network took {time_taken:.2f} seconds.")

plt.plot(
    [i + 1 for i in range(epochs)],
    train_avg_losses,
    marker="x",
    color="red",
    label="Train Avg Loss",
)
plt.plot(
    [i + 1 for i in range(epochs)],
    test_avg_losses,
    marker="o",
    color="blue",
    label="Validation Avg Loss",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

for i in range(3):
    category = list(categories_mapping.keys())[i]

    print(f"\nClass {category} Accuracy\n-------------------------------")

    train_acc = correct_per_class_train[i] / totals_per_class_train[i]
    test_acc = correct_per_class_test[i] / totals_per_class_test[i]

    print(f"Training Accuracy: {(100*train_acc):>0.1f}%")
    print(f"Test Accuracy: {(100*test_acc):>0.1f}%")

plt.show()
