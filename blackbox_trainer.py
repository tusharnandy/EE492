import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from blackbox_models import *
import utils
from main_logger import get_logger
import argparse
import random
import datetime
import os
from tqdm import tqdm
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="Name of the dataset", default="adult_income", choices=["adult_income"])
parser.add_argument('--run_number', type=str, help = 'Number for the run. If not given - assigned to datetime of the run')
parser.add_argument('--epochs', type=int, help="Number of epochs", default=30)
parser.add_argument('--learning_rate', type=float, help="Learning rate", default=1e-3)
parser.add_argument('--train_batch_size', type=int, help="Training batch size", default=64)
parser.add_argument('--test_batch_size', type=int, help="Testing batch size", default=64)
parser.add_argument('--patience', type=int, help="Patience for early stopping", default=7)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--net', type=str, default='Logistic', choices=['Logistic', 'Linear', 'SVM'])
args = parser.parse_args()


def train(model, dataloader, len_train, criterion, optimizer, device, ):
    model.train()
    running_loss = 0.0
    correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len_train/dataloader.batch_size)):
        x, labels = data
        x, labels = x.to(device), labels.to(device)
        x = x.view(x.size(0), -1)
        optimizer.zero_grad()
        if args.net == 'Logistic':
           output = model(x)
           loss = criterion(output.squeeze(), labels.float())
        running_loss += loss.item()
        predicted = torch.round(output.squeeze())
        correct += (predicted == labels).sum()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    train_acc = 100 * (correct / len_train)
    return train_loss, train_acc

def validate(model, dataloader, len_test, criterion, device, ):
    model.eval()
    running_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len_test/dataloader.batch_size)):
            x, labels = data
            x, labels = x.to(device), labels.to(device)
            x = x.view(x.size(0), -1)
            if args.net == 'Logistic':
                output = model(x)
                loss = criterion(output.squeeze(), labels.float())
                running_loss += loss.item()
                predicted = torch.round(output.squeeze())
                correct += (predicted == labels).sum()


    val_loss = running_loss/len(dataloader.dataset)
    val_acc = 100 * (correct / len_test)
    return val_loss, val_acc


def trainer(model, train_loader, test_loader, len_train, len_test, device, save_path=f'checkpoints/{args.dataset}/blackbox/{args.net}', ):
    early_stopping = utils.EarlyStopping(args.patience, args.verbose, save_path=save_path, trace_func=file_logger.info)
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_model_path = f'{save_path}/best.pt'
    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_acc = train(model, train_loader, len_train, criterion, optimizer, device)
        val_epoch_loss, val_epoch_acc = validate(model, test_loader, len_test, criterion, device)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        early_stopping(val_epoch_loss, model)
        file_logger.info(f"Epoch: {epoch+1}/{args.epochs}\tTrain Loss: {train_epoch_loss:.4f}\tVal Loss: {val_epoch_loss:.4f}\tTrain Acc: {train_epoch_acc:.4f}\tVal Acc: {val_epoch_acc:.4f}")
        if early_stopping.early_stop:
            file_logger.critical('Triggering Early Stop')
            break

    
        
    model.load_state_dict(torch.load(best_model_path))

    return model, train_loss, val_loss, train_acc, val_acc, epoch




if __name__ == '__main__':
    if args.run_number is None:
        args.run_number = datetime.datetime.now().isoformat()
    
    log_dir = f'logs/{args.dataset}/{args.net}_logs'
    log_file = f'{log_dir}/{args.run_number}_training.log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok = True)
    file_logger = get_logger(f'Blackbox: {args.net} Training', log_file)
    file_logger.info(f'Run Number: {args.run_number}')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    train_loader, test_loader, len_train, len_test, feature_dim, num_cols = utils.get_dataset(args.dataset, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
    model = BlackBox(args.net, feature_dim, 1).to(device)
    model, train_loss, val_loss, train_acc, val_acc, last_epoch = trainer(model, train_loader, test_loader, len_train, len_test, device)
    
