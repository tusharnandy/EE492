import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).contiguous().to(device)
        self.y = torch.Tensor(y).contiguous().to(device) 
    
    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_loaders(file_loc, y_col, test_frac=0.2):
    df = pd.read_csv(file_loc).drop(columns=["Unnamed: 0"])
    X, y = df.drop(columns=y_col).values, df[y_col].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_frac, random_state=108)
    train_data = CustomDataset(X_train, y_train)
    val_data = CustomDataset(X_val, y_val)
    return DataLoader(train_data, batch_size=64, shuffle=True), DataLoader(val_data, batch_size=128, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(102, 50)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(50,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x

def regularizer(x,y_hat,a=101):
    cat1 = x[:,a] == 1
    cat2 = x[:,a] == 0
    prob1 = y_hat[cat1]
    prob2 = y_hat[cat2]
    
    if list(prob1.shape)[0] == 0:
        demo_parity = torch.square(prob2.mean())
    elif list(prob2.shape)[0] == 0:
        demo_parity = torch.square(prob1.mean())
    else:
        demo_parity = torch.square(prob1.mean() - prob2.mean())
    
    return demo_parity

def training_loop(l=1):
    net.train()
    loss_train_epoch = []
    demo_parity_epoch = []
    i = 0
    for x,y in train_loader:
        optimizer.zero_grad()
        y_hat = net(x).reshape(x.shape[0])
        
        loss = loss_fn(y_hat, y)
        reg = regularizer(x.clone().detach(), y_hat)
        q = loss + l*reg
        
        loss_train_epoch.append(loss.item())
        demo_parity_epoch.append(reg.item())
        
        q.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
    return np.mean(loss_train_epoch), np.mean(demo_parity_epoch)

def val_loop():
    net.eval()
    loss_val_epoch = []
    demo_parity_epoch = []
    with torch.no_grad():
        for x,y in val_loader:
            y_hat = net(x).reshape(x.shape[0])
            loss = loss_fn(y_hat, y)
            reg = regularizer(x.clone().detach(), y_hat)
            loss_val_epoch.append(loss.item())
            demo_parity_epoch.append(reg.item())
    return np.mean(loss_val_epoch), np.mean(demo_parity_epoch)


if __name__ == "__main__":
    train_loader, val_loader = get_loaders("../Data/data1.csv", "income", test_frac=0.2)
    
    net = Net()
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    loss_train = []
    loss_val = []

    for epoch in range(10):
        loss_t, reg_t = training_loop(10)
        # break
        loss_v, reg_v = val_loop()

        loss_train.append(loss_t)
        loss_val.append(loss_v)
        
        print(f"Epoch: {epoch+1}, Training Loss: {np.round(loss_t, 3)}, Demo Parity Train: {np.round(np.sqrt(reg_t), 3)}, Val Loss: {np.round(loss_v, 3)}, Demo Parity Val: {np.round(np.sqrt(reg_v), 3)}")


    torch.save(net.state_dict(), "checkpoints/blackbox_preaudit.pth")