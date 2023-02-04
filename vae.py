import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataset as dataset

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim * 2)
        self.zdim = z_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        mean, log_var = x[:, :self.z_dim], x[:, self.z_dim:]
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(z))
        x = self.fc3(z)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, z_dim)
        self.decoder = Decoder(z_dim, hidden_dim, input_dim)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        x_hat = self.decoder(z)
        return x_hat, mean, log_var

# def loss_function(x, x_hat, mean, log_var, cat_index):
#     bce_loss = F.binary_cross_entropy(torch.Sigmoid(x_hat[cat_index:]), x[cat_index:], reduction='none').mean(0).sum()
#     mse_loss = F.mse_loss(x_hat[:cat_index], x[:cat_index], reduction='none').mean(0).sum()
#     reconstruction_loss = mse_loss + bce_loss
#     kl_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
#     return reconstruction_loss + kl_divergence

# def train(model, optimizer, train_loader):
#     model.train()
#     train_loss = 0
#     for x in train_loader:
#         optimizer.zero_grad()
#         x_hat, mean, log_var = model(x)
#         loss = loss_function(x, x_hat, mean, log_var, 4)
#         loss.backward()
#         train_loss += loss.item()
#         optimizer.step()
#     return train_loss / len(train_loader.dataset)

# class CustomDataset(dataset):
#     def __init__(self, data):
#         self.data = data

#     def __len__(self):
#         return self.data.shape[0]

#     def __getitem__(self,idx):
#         return self.data[idx]
