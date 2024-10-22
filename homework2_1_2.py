import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable LaTeX text rendering in plots
plt.rcParams["text.usetex"] = True

# Generate training data
samples = 512
sample_min = -5
sample_max = 5

# Bivariate function: z = sin(x) + cos(y)
x = np.random.uniform(sample_min, sample_max, samples)
y = np.random.uniform(sample_min, sample_max, samples)
z = np.sin(x) + np.cos(y)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
z_tensor = torch.tensor(z, dtype=torch.float32).view(-1, 1)

# Concatenate input tensors
input_tensor = torch.cat((x_tensor, y_tensor), dim=1)

# Create dataset and dataloader
dataset = TensorDataset(input_tensor, z_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Model definition
class MultiVarNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
        super(MultiVarNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.activation_fn(self.layer2(x))
        x = self.layer3(x)
        return x


def get_model(activation_fn):
    return MultiVarNet(input_dim=2, hidden_dim=200, output_dim=1, activation_fn=activation_fn).to(device)


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
        predictions = model(input_tensor.to(device)).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, label='True function', color='blue', s=5)  # Adjust point size
    ax.scatter(x, y, predictions, label='Model predictions', color='red', s=5)  # Adjust point size
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
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


# Hyperparameter search space
learning_rates = [0.001, 0.01]
hidden_dim = 200
epochs = 500
activation_fns = {
    'Tanh': nn.Tanh(),
    'ReLU': nn.ReLU(),
    'Sigmoid': nn.Sigmoid()
}
lambdas = [0.0, 1e-5]

# Random search
best_val_loss = float('inf')
best_hyperparams = {}

for i in range(20):  # Number of random trials
    lr = random.choice(learning_rates)
    activation_fn_name, activation_fn = random.choice(list(activation_fns.items()))
    lambda_l2 = random.choice(lambdas)

    print(f"Trial {i + 1}: lr={lr}, hidden_dim={hidden_dim}, activation_fn={activation_fn_name}, lambda_l2={lambda_l2}")

    model, train_losses, val_losses = train(train_loader, val_loader, lr, epochs, activation_fn, lambda_l2=lambda_l2)

    avg_val_loss = val_losses[-1]

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_hyperparams = {
            'learning_rate': lr,
            'hidden_dim': hidden_dim,
            'activation_fn': activation_fn_name,
            'lambda_l2': lambda_l2
        }

    plot_loss_curves(train_losses, val_losses,
                     title=f"Loss Curves (lr={lr}, hidden_dim={hidden_dim}, activation_fn={activation_fn_name}, lambda_l2={lambda_l2})")
    evaluate_and_plot(model)

print(f"Best hyperparameters: {best_hyperparams}")
