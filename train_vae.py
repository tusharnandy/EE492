import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from vae import VAE
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    prog= 'train_vae.py'
    description='train vae for sensitive attribute = 0 and 1\nWeights will be stored at weights/vae[a]_weights.pt'
)
parser.add_argument("-d", "--data", help='location of data', type=str)
parser.add_argument("-w", "--weights")
parser.add_argument("-x", "--device", help='device', type=int, default=0)
parser.add_argument("-i", "--index_sensitive_attribute", default=-1, help='index of sensitive attribute', type=int)