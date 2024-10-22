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

# Generate noisy training data
samples = 512
sample_min = -5
sample_max = 5
noise_level = 3.0  # Increase noise to induce overfitting

x = np.random.uniform(sample_min, sample_max, samples)
y = np.random.uniform(sample_min, sample_max, samples)
z = np.sin(x) + np.cos(y) + np.random.normal(0, noise_level, samples)  # Adding noise

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
z_tensor = torch.tensor(z, dtype=torch.float32).view(-1, 1)
input_tensor = torch.cat((x_tensor, y_tensor), dim=1)

# Create dataset and dataloader
dataset = TensorDataset(input_tensor, z_tensor)
train_size = int(0.3 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)


# Model definition with Dropout
class MultiVarNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_fn, dropout_prob):
        super(MultiVarNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.layer3 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.layer4 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer5 = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.activation_fn(self.layer1(x))
        x = self.dropout(x)
        x = self.activation_fn(self.layer2(x))
        x = self.dropout(x)
        x = self.activation_fn(self.layer3(x))
        x = self.dropout(x)
        x = self.activation_fn(self.layer4(x))
        x = self.dropout(x)
        return self.layer5(x)


def train_and_evaluate(train_loader, val_loader, learn_rate, epochs, activation_fn, lambda_l2, dropout_prob):
    model = MultiVarNet(input_dim=2, hidden_dim=200, output_dim=1, activation_fn=activation_fn, dropout_prob=dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=lambda_l2)
    criterion = nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        train_losses.append(total_train_loss / len(train_loader))

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_output = model(x_val)
                val_loss = criterion(val_output, y_val)
                total_val_loss += val_loss.item()
        val_losses.append(total_val_loss / len(val_loader))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs}, Training Loss: {train_losses[-1]:.6f}, Validation Loss: {val_losses[-1]:.6f}")

    # Evaluate and plot results
    evaluate_and_plot(model, x, y, z)
    # Plotting loss curves
    plot_loss_curves(train_losses, val_losses, title="Loss Curves with lambda=" + str(lambda_l2) + " and dropout=" + str(dropout_prob))


def evaluate_and_plot(model, x, y, z):
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(np.column_stack((x, y)), dtype=torch.float32).to(device)).cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, label='True function', color='blue', s=5)
    ax.scatter(x, y, predictions.flatten(), label='Model predictions', color='red', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.show()


def plot_loss_curves(train_losses, val_losses, title="Loss Curves"):
    plt.figure(figsize=(10, 5))

    # Smooth the loss curves
    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points

    train_losses_smooth = smooth_curve(train_losses)
    val_losses_smooth = smooth_curve(val_losses)

    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss', alpha=0.3)
    plt.plot(train_losses_smooth, label='Smoothed Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Training Loss')

    plt.subplot(2, 1, 2)
    plt.plot(val_losses, label='Validation Loss', alpha=0.3)
    plt.plot(val_losses_smooth, label='Smoothed Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Validation Loss')

    plt.suptitle(title)
    plt.show()


# Set hyperparameters
learn_rate = 0.01
epochs = 3000
activation_fn = nn.ReLU()

# No regularization to induce overfitting
print("Training with high capacity model and no regularization to induce overfitting:")
train_and_evaluate(train_loader, val_loader, learn_rate, epochs, activation_fn, lambda_l2=0.0, dropout_prob=0.0)

# Adjust Dropout and L2 regularization parameters
lambda_l2 = 0.05
dropout_prob = 0.5
print("\nApplying higher L2 regularization and higher Dropout to reduce overfitting:")
train_and_evaluate(train_loader, val_loader, learn_rate, epochs, activation_fn, lambda_l2, dropout_prob)
