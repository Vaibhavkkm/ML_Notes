import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Get cpu, gpu or mps device for training
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
        self.n_samples = 40
        self.x_range = np.array([0.9, 2], dtype=np.float32)
        x_np = np.linspace(self.x_range[0], self.x_range[1], self.n_samples, dtype=np.float32)
        target = torch.from_numpy(self.leonard_jones_toten(x_np))

        # Reshape into 2D tensors
        self.x = torch.reshape(torch.from_numpy(x_np), (len(x_np), 1))
        self.target = torch.reshape(target, (len(target), 1))

    def leonard_jones_toten(self, x_np):
        """x_np is a 1D array of positions; returns a 1D array of LJ energies."""
        sigma = 1.0
        epsilon = 1.0
        return 4 * epsilon * ((sigma / x_np)**12 - (sigma / x_np)**6)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.target[idx]

# Instantiate dataset
my_dataset = LJ_Dataset()

# Split dataset (60% train, 20% val, 20% test)
generator1 = torch.Generator().manual_seed(42)
train_data, val_data, test_data = torch.utils.data.random_split(
    my_dataset, [0.6, 0.2, 0.2], generator=generator1
)

# Create data loaders
train_dataloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=len(val_data), drop_last=True)

print(f"Num training samples: {len(train_data)}")
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

        # Ensure mu is on the same device as the inputs
        self.mu = torch.linspace(x_start, x_end, num_basis_functions, device=device)
        self.sigma = (x_end - x_start) / num_basis_functions

        # MLP
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
        """ Converts position tensor x -> [batch_size, 1] into
            Gaussian basis descriptors -> [batch_size, num_basis_functions]. """
        return (1.0 / (self.sigma * np.sqrt(2 * np.pi))) * torch.exp(
            -0.5 * (x - self.mu)**2 / (self.sigma**2)
        )

    def forward(self, x):
        descriptors = self.gaussian(x)
        pred = self.multilayer_perceptron(descriptors)
        return pred

# Move model to device
model = NeuralNetwork().to(device)
print(model)

###################### 3) Loss function ################################
loss_fn = nn.MSELoss()

###################### 4) Minimize loss #################################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(train_dataloader, model, loss_fn, optimizer):
    """Does a single epoch of training over the entire dataset."""
    model.train()
    epoch_loss = 0.0
    num_samples = len(train_dataloader.dataset)

    for batch_idx, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Return average loss over all samples
    return epoch_loss / num_samples

def val(dataloader, model, loss_fn):
    """Computes validation loss on the entire validation set."""
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            num_batches += 1

    # Return average loss per batch
    return val_loss / num_batches if num_batches > 0 else 0.0

# Training loop
epochs = 300
skip_report = 5
train_loss_history = []
val_loss_history = []

for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)

    if epoch % skip_report == 0:
        val_loss = val(val_dataloader, model, loss_fn)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"Epoch {epoch:3} \t Train loss: {train_loss:.6f} \t Val loss: {val_loss:.6f}")

# Plot training curves
fig, ax = plt.subplots()
x = np.arange(len(train_loss_history)) * skip_report
ax.semilogy(x, train_loss_history, label="train loss")
ax.semilogy(x, val_loss_history, label="val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
plt.savefig("lj_2_results.png")
plt.show()
plt.close()
