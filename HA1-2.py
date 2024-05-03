import torch
import matplotlib.pyplot as plt

#torch settings
torch.set_default_dtype(torch.float64) 
torch.set_num_threads(16)
#matplotlib settings 使用latex渲染，线长和字体大小
plt.rcParams["text.usetex"] = True
plt.rcParams["lines.markersize"] = 3 
plt.rcParams["font.size"] = 18

# Define the domain
Lx = 2
Ly = 1
Nx = 80 * Lx + 1
Ny = 80 * Ly + 1
shape1 = (Nx, Ny)
Nx = torch.tensor(Nx)
dx = (Lx / (Nx- 1))
dy = (Ly / (Ny- 1))
# Create a grid of points
X = torch.meshgrid(torch.linspace(0, Lx, Nx), torch.linspace(0, Ly, Ny), indexing='ij')
X = torch.cat((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), dim=1)

u = torch.stack((1*X[:, 0] + 2*X[:, 1], torch.pow(X[:, 1], 2)), dim=1)

# Plotting
fig,axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original points
axs[0].scatter(X[:, 0], X[:, 1], color='black', marker='o')
axs[0].set_title('Original Points')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')

# Plot the deformed points with colors based on Euclidean distance from the original position
euclidean_distances = torch.norm(u, dim=1)
axs[1].scatter(X[:, 0] + u[:, 0], X[:, 1] + u[:, 1], c=euclidean_distances, cmap='viridis', marker='o')
axs[1].set_title('Deformed Points Colored by Distance')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')
plt.colorbar(axs[1].collections[0], ax=axs[1], label='Euclidean Distance')

# Adjust layout to fit everything nicely
plt.tight_layout()
plt.show()