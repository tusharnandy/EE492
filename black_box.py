from torch import nn
import pandas as pd
import numpy as np
import torch

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

class BlackBox():
    def __init__(self, model_filename, device_id=0, save_queries=False):
        # setting device
        self.device = torch.device(f'cuda:{num_device}' if torch.cuda.is_available() else 'cpu')
        
        # loading weights of the classifier
        self.clf = Net().to(self.device)
        self.clf.load_state_dict(torch.load(model_filename))
        self.clf.eval()

        self.save_queries = save_queries
        self.query_points = None
    
    def query(self, X):
        with torch.no_grad():
            y  = self.clf(X)

        if self.save_queries:
            if self.query_points is None:
                self.query_points = torch.cat((X, y >= 0.5), dim=1)
            else:
                self.query_points = torch.cat((self.query_points, torch.cat((X, y >= 0.5), dim=1)), dim=0)
        return X, torch.FloatTensor([y >= 0.5])

    def save_query_points(self, filename):
        if self.query_points is not None:
            torch.save(self.query_points, filename)

