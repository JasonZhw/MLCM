import torch
import matplotlib.pyplot as plt

#torch settings
torch.set_default_dtype(torch.float64) 
torch.set_num_threads(16)
#matplotlib settings rendering with latex, line length and font size
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


u_A = torch.stack((2*X[:, 0] + X[:, 1], torch.pow(0.8*X[:, 0], 2)), dim=1)
u_Neu = torch.stack((1*X[:, 0] + 2*X[:, 1], torch.pow(X[:, 1], 2)), dim=1)

# Plotting
plt.figure(figsize=(12, 6))

# Plot the original points
plt.subplot(2, 2, 1)  
plt.scatter(X[:, 0], X[:, 1], color='black', marker='o')
plt.title('Original Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Plot the deformed points with colors based on Euclidean distance from the original position
euclidean_distances = torch.norm(u_A, dim=1)
euclidean_distances_Neu = torch.norm(u_Neu, dim=1)

plt.subplot(2, 2, 3) 
scatter = plt.scatter(X[:, 0] + u_A[:, 0], X[:, 1] + u_A[:, 1], c=euclidean_distances, cmap='viridis', marker='o')
plt.title('Deformed Points Colored by Distance')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(scatter, label='Euclidean Distance')


# Plot the original points
plt.subplot(2, 2, 2)  
plt.scatter(X[:, 0], X[:, 1], color='black', marker='o')
plt.title('Original Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Plot the deformed points with colors based on Euclidean distance from the original position
euclidean_distances = torch.norm(u_Neu, dim=1)
plt.subplot(2, 2, 4) 
scatter = plt.scatter(X[:, 0] + u_Neu[:, 0], X[:, 1] + u_Neu[:, 1], c=euclidean_distances_Neu, cmap='viridis', marker='o')
plt.title('Deformed Points Colored by Distance')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(scatter, label='Euclidean Distance')
# Adjust layout to fit everything nicely

plt.tight_layout()
plt.show()
