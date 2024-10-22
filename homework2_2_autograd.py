import torch
import matplotlib.pyplot as plt

#torch settings
torch.set_default_dtype(torch.float64) 
torch.set_num_threads(16)

#matplotlib settings rendering with latex, line length and font size
plt.rcParams["text.usetex"] = True
plt.rcParams["lines.markersize"] = 3 
plt.rcParams["font.size"] = 18

#######################################################################################################################################

# Define the domain
Lx = 2  # Define the length in the x-direction
Ly = 1  # Define the length in the y-direction
Nx = 80 * Lx + 1   # the number of grid points in the x-direction
Ny = 80 * Ly + 1   # the number of grid points in the y-direction
shape1 = (Nx, Ny)   # Shape of the grid
Nx = torch.tensor(Nx)   # Convert number of x-direction grid points into a tensor
dx = (Lx / (Nx- 1))   # Interval between each grid point in the x-direction
dy = (Ly / (Ny- 1))   # Interval between each grid point in the y-direction

# Using PyTorch to generate a grid of points
X = torch.meshgrid(torch.linspace(0, Lx, Nx,requires_grad=True), torch.linspace(0, Ly, Ny,requires_grad=True), indexing='ij')   #Create a mesh grid
X = torch.cat((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), dim=1)   #Reshape and combine the grid tensors

#Using PyTorch to define a two-dimensional displacement field 'u' and enables gradient computation to make u usable for subsequent automatic differentiation operations
u = torch.stack((1*X[:, 0] + 2*X[:, 1], torch.pow(X[:, 1], 2)), dim=1)   #Defining the displacement field 'u'
u.requires_grad_(True)   #Enabling gradient tracking


#Use PyTorch autograd feature to calculate the displacement gradient
#To compute the gradients of each component of the displacement field 'u' with respect to the input 'X'
du1_dx = torch.autograd.grad(outputs=u[:, 0].sum(), inputs=X, create_graph=True)[0]   #Computing the gradient in X_Direction
du2_dx = torch.autograd.grad(outputs=u[:, 1].sum(), inputs=X, create_graph=True)[0]   #Computing the gradient in Y_Direction

#To stack the gradient tensors computed from different components of the displacement field into a single three-dimensional tensor
grad_u = torch.stack([du1_dx, du2_dx], dim=1)  # Shape will be [13041, 2, 2]

#Calculate the deformation gradient 'F'
F = torch.eye(2).unsqueeze(0).expand_as(grad_u) + grad_u   # Adds the expanded identity matrix to the gradient tensor 'grad_u'

#Calculate the right Cauchy-Green tensor 'C'
C = torch.matmul(F.transpose(-1, -2), F)

#Calculate the left Cauchy-Green tensor 'B'
B = torch.matmul(F, F.transpose(-1, -2))

#Calculate the Green-Lagrange strain tensor 'E'
E = 0.5 * (C - torch.eye(2, device=X.device).unsqueeze(0).expand_as(C))   #The strain tenor E is half the difference between the right Cauchy-Green tensor C and the identity matrix

########################################################################################################################################

#Plotting
#Create a grid with 10 subplots arranged in a 5x2 layout
fig, axs = plt.subplots(5, 2, figsize=(12, 10))   #Specifie the overall size of the figure to be 12x10 inches
fig.subplots_adjust(hspace=0.4, wspace=0.4)   #Adjust the spacing between subplots within the figure

#Plot the distribution of the displacement gradients in the x-direction
#Creates a scatter plot where the x/y-axis represents displacement in the x/y-direction and the color of the points corresponds to the magnitude of the displacement gradient in the x-direction
scatter = axs[0, 0].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(grad_u[:, 0, 0].detach().numpy() + grad_u[:, 1, 0].detach().numpy()), cmap='viridis', marker='o')
axs[0, 0].set_title('grad_u_x')  #Set the title of the subplot to 'grad_u_x'
axs[0, 0].set_xlabel('X-axis')   #Set the labels for the x-axis
axs[0, 0].set_ylabel('Y-axis')   #Set the labels for the y-axis
fig.colorbar(scatter, ax=axs[0, 0], label='grad_u_x')   #Add a color bar to indicate the correspondence between colors and gradient values

#Plot the distribution of the displacement gradients in the y-direction
scatter = axs[0, 1].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(grad_u[:, 0, 1].detach().numpy() + grad_u[:, 1, 1].detach().numpy()), cmap='viridis', marker='o')
axs[0, 1].set_title('grad_u_y')
axs[0, 1].set_xlabel('X-axis')
axs[0, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[0, 1], label='grad_u_y')

#Plot the distribution of the deformation gradient in the x-direction
scatter = axs[1, 0].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(F[:, 0, 0].detach().numpy() + F[:, 1, 0].detach().numpy()), cmap='viridis', marker='o')
axs[1, 0].set_title('F_x')
axs[1, 0].set_xlabel('X-axis')
axs[1, 0].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[1, 0], label='F_x')

#Plot the distribution of the deformation gradient in the y-direction
scatter = axs[1, 1].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(F[:, 0, 1].detach().numpy() + F[:, 1, 1].detach().numpy()), cmap='viridis', marker='o')
axs[1, 1].set_title('F_y')
axs[1, 1].set_xlabel('X-axis')
axs[1, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[1, 1], label='F_y')

#Plot the right Cauchy-Green tensor in the x-direction
scatter = axs[2, 0].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(C[:, 0, 0].detach().numpy() + F[:, 1, 0].detach().numpy()), cmap='viridis', marker='o')
axs[2, 0].set_title('C_x')
axs[2, 0].set_xlabel('X-axis')
axs[2, 0].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[2, 0], label='C_x')

#Plot the right Cauchy-Green tensor in the y-direction
scatter = axs[2, 1].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(C[:, 0, 1].detach().numpy() + C[:, 1, 1].detach().numpy()), cmap='viridis', marker='o')
axs[2, 1].set_title('C_y')
axs[2, 1].set_xlabel('X-axis')
axs[2, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[2, 1], label='C_y')

#Plot the left Cauchy-Green tensor in the x-direction
scatter = axs[3, 0].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(B[:, 0, 0].detach().numpy() + B[:, 1, 0].detach().numpy()), cmap='viridis', marker='o')
axs[3, 0].set_title('B_x')
axs[3, 0].set_xlabel('X-axis')
axs[3, 0].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[3, 0], label='B_x')

#Plot the left Cauchy-Green tensor in the y-direction
scatter = axs[3, 1].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(B[:, 0, 1].detach().numpy() + B[:, 1, 1].detach().numpy()), cmap='viridis', marker='o')
axs[3, 1].set_title('B_y')
axs[3, 1].set_xlabel('X-axis')
axs[3, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[3, 1], label='B_y')

#Plot the Green-Lagrange strain tensor in the x-direction
scatter = axs[4, 0].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(E[:, 0, 0].detach().numpy() + E[:, 1, 0].detach().numpy()), cmap='viridis', marker='o')
axs[4, 0].set_title('E_x')
axs[4, 0].set_xlabel('X-axis')
axs[4, 0].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[4, 0], label='E_x')

#Plot the Green-Lagrange strain tensor in the y-direction
scatter = axs[4, 1].scatter(u[:, 0].detach().numpy(), u[:, 1].detach().numpy(), c=(E[:, 0, 1].detach().numpy() + E[:, 1, 1].detach().numpy()), cmap='viridis', marker='o')
axs[4, 1].set_title('E_y')
axs[4, 1].set_xlabel('X-axis')
axs[4, 1].set_ylabel('Y-axis')
fig.colorbar(scatter, ax=axs[4, 1], label='E_y')

plt.tight_layout()   #Automatically adjust the spacing and size between subplots to better fill the entire plotting area
plt.show()           #display the drawn graph