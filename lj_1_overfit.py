import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

##################### 1) Dataset #############################
class LJ_Dataset(Dataset):
    def __init__(self):
        self.n_samples = 10
        self.x_range = np.array([0.9, 2], dtype=np.float32)
        x_np = np.linspace(self.x_range[0], self.x_range[1], self.n_samples, dtype=np.float32)
        target = torch.from_numpy(self.leonard_jones_toten(x_np))

        # Reshape into 2D tensors for PyTorch [batch_size x features]
        self.x = torch.reshape(torch.from_numpy(x_np), (len(x_np), 1))                
        self.target = torch.reshape(target, (len(target), 1))

    def leonard_jones_toten(self, x_np):
        """ x_np is a 1D NumPy array of positions; returns a 1D array of energies """
        sigma = 1.0
        epsilon = 1.0
        return 4 * epsilon * ((sigma / x_np)**12 - (sigma / x_np)**6)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.target[idx]

# Create dataset & data loader
my_dataset = LJ_Dataset()
batch_size = 10
train_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"Num training samples: {len(my_dataset)}")
for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    break

######################### 2) Model #################################
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_basis_functions = 30
        nodes_per_layer = 5
        x_start = 0.9
        x_end = 3

        # Option A) Put mu directly on the chosen device:
        self.mu = torch.linspace(x_start, x_end, num_basis_functions, device=device)
        # If you prefer to keep a float for sigma, that's okay:
        self.sigma = (x_end - x_start) / num_basis_functions

        # If you want self.sigma as a tensor, also move it to the device:
        # self.sigma = torch.tensor((x_end - x_start)/num_basis_functions, device=device)

        # Alternatively (Option B), you can register it as a buffer:
        # self.register_buffer("mu", torch.linspace(x_start, x_end, num_basis_functions))

        self.multilayer_perceptron = nn.Sequential(
            nn.Linear(num_basis_functions, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, 1),
        )

    def gaussian(self, x):
        """
        x: [batch_size x 1]
        returns: [batch_size x num_basis_functions]
        """
        # Make sure x and self.mu are on the same device (handled by setting self.mu = ... device=device)
        return 1/(self.sigma * np.sqrt(2*np.pi)) * torch.exp(
            -0.5 * (x - self.mu)**2 / (self.sigma**2)
        )

    def forward(self, x):
        descriptors = self.gaussian(x)
        pred = self.multilayer_perceptron(descriptors)
        return pred

model = NeuralNetwork().to(device)
print(model)

###################### 3) Loss function ################################
loss_fn = nn.MSELoss()

###################### 4) Minimize loss #################################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(train_dataloader, model, loss_fn, optimizer):
    """ does a single epoch, i.e. updates the parameters based on each batch """
    epoch_loss = 0.0
    num_batches = len(train_dataloader.dataset)  # = 10 in your case
    model.train()

    for batch_idx, (X, y) in enumerate(train_dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Return the average loss over the entire epoch (not just the first batch!)
    return epoch_loss / num_batches

# Training Loop
epochs = 100
for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3} \t train loss: {train_loss:.6f}")

#################### 5) make a prediction ############################
model.eval()  # set the model into evaluation mode

# Collect training points for plotting
train_x = []
train_y = []
for X, y in train_dataloader:
    train_x.append(X.cpu().numpy())
    train_y.append(y.cpu().numpy())
train_x = np.concatenate(train_x, axis=0)
train_y = np.concatenate(train_y, axis=0)

# Calculate the real LJ curve
x_lin = np.linspace(0.9, 3, 100, dtype=np.float32)
y_ref = my_dataset.leonard_jones_toten(x_lin)

# Model predictions
with torch.no_grad():
    # shape: [100, 1]
    x_torch = torch.from_numpy(x_lin).reshape(-1, 1).to(device)
    y_pred = model(x_torch).cpu().numpy()  # bring back to CPU for plotting

# Plot
fig, ax = plt.subplots()
ax.plot(x_lin, y_ref, label="Target (LJ Curve)")
ax.scatter(train_x, train_y, color="red", label="Training Data")
ax.plot(x_lin, y_pred, label="NN Prediction")
ax.set_xlabel("Interatomic distance (units)")
ax.set_ylabel("Energy (units)")
ax.legend()
plt.savefig("lj_1_results.png")
plt.show()
plt.close()
