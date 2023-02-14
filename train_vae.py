import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vae import VAE
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", help="location of data", type=str, default="../Data/data1.csv"
)
parser.add_argument(
    "-w", "--weights", help="location of weights/checkpoint", type=str, required=False
)
parser.add_argument("-c", "--load_checkpoint", default=False, type=bool)
parser.add_argument("-x", "--device", help="device", type=int, default=-1)
# parser.add_argument("-i", "--index_sensitive_attribute",
#                     default=-1, help='index of sensitive attribute', type=int)
parser.add_argument("-e", "--epochs", default=10, type=int)
# parser.add_argument('-p', '--patience_param_percentage', default=5,
#                     type=int, help='patience parameter for early stopping (in per cent)')
parser.add_argument(
    "-o",
    "--stopping_epochs",
    default=5,
    type=int,
    help="early stop after these many epochs",
)
parser.add_argument("-l", "--learning_rate", default=0.001, type=float)
# parser.add_argument('-s', "--early_stopping", choices=['hard', 'soft'], default='hard', type=str, help='Hard early stopping checks whether the loss is continually decreasing or not.\nSoft early stop checks whether the loss is within some epsilon-region of the best loss.'
#                     )
args = parser.parse_args()

if args.device >= 0:
    device = torch.device(
        f"cuda:{torch.device}" if torch.cuda.is_available() else "cpu"
    )
else:
    device = torch.device("cpu")


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]


def loss_function(x, x_hat, mean, log_var, cat_index):
    mse_loss = F.mse_loss(x_hat, x, reduction="mean")
    reconstruction_loss = mse_loss
    kl_divergence = -0.5 * torch.sum(1 + log_var - mean**2 - log_var.exp())
    return reconstruction_loss + kl_divergence


def train(model, optimizer, train_loader):
    model.train()
    train_loss = 0
    for x in train_loader:
        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var, 4)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(train_loader.dataset)


def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for x in test_loader:
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var, 4)
            test_loss += loss.item()
    return test_loss / len(test_loader.dataset)


def prepare_data():
    df = pd.read_csv(args.data).drop(columns="Unnamed: 0")
    X_data = df.drop(columns="income")
    X_train, X_test = train_test_split(X_data, test_size=0.3, random_state=7)
    X1_train = torch.from_numpy(X_train[X_train["gender_Female"] == 1.0].values).to(
        torch.float32
    )
    X1_test = torch.from_numpy(X_test[X_test["gender_Female"] == 1.0].values).to(
        torch.float32
    )
    X1_train.to(device)
    X1_test.to(device)
    X0_train = torch.from_numpy(X_train[X_train["gender_Female"] == 0.0].values).to(
        torch.float32
    )
    X0_test = torch.from_numpy(X_test[X_test["gender_Female"] == 0.0].values).to(
        torch.float32
    )
    X0_train.to(device)
    X0_test.to(device)
    X1_trainset = CustomDataset(X1_train)
    X1_testset = CustomDataset(X1_test)
    X1_trainloader = DataLoader(X1_trainset, batch_size=128, shuffle=True)
    X1_testloader = DataLoader(X1_testset, batch_size=128)
    X0_trainset = CustomDataset(X0_train)
    X0_testset = CustomDataset(X0_test)
    X0_trainloader = DataLoader(X0_trainset, batch_size=128, shuffle=True)
    X0_testloader = DataLoader(X0_testset, batch_size=128)
    return X0_trainloader, X0_testloader, X1_trainloader, X1_testloader


def save_checkpoint(epoch, model, optimizer, train_losses, test_losses):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
        },
        args.weights,
    )


def hard_early_stop(model, optimizer, train_loader, test_loader):
    best_loss = 10e5
    start_epoch = 0
    es_count = 0
    train_losses = []
    test_losses = []
    if args.load_checkpoint:
        checkpoint = torch.load(args.weights)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_losses, test_losses = (
            checkpoint["train_losses"],
            checkpoint["test_losses"],
        )
        print("Checkpoint loaded: ")
        printer(start_epoch, train_loss=train_losses[-1], test_loss=test_losses[-1])
        print("")
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(model, optimizer, train_loader)
        test_loss = test(model, test_loader)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        printer(epoch, train_loss, test_loss)
        if test_loss < best_loss:
            best_loss = test_loss
            es_count = 0
        else:
            es_count += 1
        if es_count == args.stopping_epochs:
            save_checkpoint(epoch, model, optimizer, train_losses, test_losses)
            print("Early stopping...")
            return model, train_losses, test_losses
        if epoch % 5 == 0:
            save_checkpoint(epoch, model, optimizer, train_losses, test_losses)
    save_checkpoint(
        (start_epoch + args.epochs), model, optimizer, train_losses, test_losses
    )
    return model, train_losses, test_losses


def printer(epoch, train_loss, test_loss):
    print(f"epoch: {epoch}, train loss: {train_loss}, test loss: {test_loss}")


if __name__ == "__main__":
    X0_trainloader, X0_testloader, X1_trainloader, X1_testloader = prepare_data()

    vae = VAE(102, 64, 16).to(device)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=args.learning_rate)
    model, train_losses, test_losses = hard_early_stop(
        vae, optimizer, X1_trainloader, X1_testloader
    )
