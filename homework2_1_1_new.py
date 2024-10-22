import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

#  Setting up LaTEX rendering
plt.rcParams["text.usetex"] = True

# setting up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate training data
samples = 64  # 减少样本数量以增加过拟合的可能性
sample_min = -5
sample_max = 5

# new function creates training data
x = np.linspace(sample_min, sample_max, samples)
y = x ** 3 - 3 * x + 1

# Conversion to PyTorch tensor
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Creating datasets and data loaders
dataset = TensorDataset(x_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  # 减小batch_size
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)


# Model definition
class PolyNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(PolyNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, hidden_dim)  # 增加一层
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.activation_fn(self.layer2(x))
        x = self.activation_fn(self.layer3(x))
        x = self.activation_fn(self.layer4(x))
        x = self.layer5(x)
        return x


def get_model(activation_fn):
    return PolyNet(input_dim=1, hidden_dim=128, output_dim=1, activation_fn=activation_fn).to(device)


def train(train_loader, val_loader, learn_rate, epochs, activation_fn, lambda_l2=0.0):
    model = get_model(activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=lambda_l2)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")

    return model, train_losses, val_losses


def evaluate_and_plot(model):
    model.eval()
    with torch.no_grad():
        predictions = model(x_tensor.to(device)).cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='True function')
    plt.plot(x, predictions, label='Model predictions', linestyle='dashed')
    plt.legend()
    plt.show()


def plot_loss_curves(train_losses, val_losses, title="Loss Curves"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.show()


# Setting hyperarameters
learn_rate = 0.01
epochs = 2000  # 增加训练周期

# Select different activation functions for the experiment
activation_fns = {
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid()
}

# raining and evaluation models
for name, activation_fn in activation_fns.items():
    print(f"Using activation function: {name}")
    model, train_losses, val_losses = train(train_loader, val_loader, learn_rate, epochs, activation_fn, lambda_l2=0.0)

    # Plotting loss curves
    plot_loss_curves(train_losses, val_losses, title=f"Training and Validation Loss over Epochs with {name}")

    # Assessment and mapping
    evaluate_and_plot(model)

# Testing different hyperparameter settings
learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    print(f"Using learning rate: {lr}")
    model, train_losses, val_losses = train(train_loader, val_loader, lr, epochs, nn.Tanh(), lambda_l2=0.0)

    # Plotting loss curves
    plot_loss_curves(train_losses, val_losses,
                     title=f"Training and Validation Loss over Epochs with learning rate {lr}")

    # Assessment and mapping
    evaluate_and_plot(model)

# testing L_2 regularization
lambdas = [1e-5, 1e-2]
for lambda_l2 in lambdas:
    print(f"Using L2 regularization with lambda: {lambda_l2}")
    model, train_losses, val_losses = train(train_loader, val_loader, learn_rate, epochs, nn.Tanh(),
                                            lambda_l2=lambda_l2)

    # Plotting loss curves
    plot_loss_curves(train_losses, val_losses,
                     title=f"Training and Validation Loss over Epochs with L2 regularization (lambda={lambda_l2})")

    # Assessment and mapping
    evaluate_and_plot(model)
