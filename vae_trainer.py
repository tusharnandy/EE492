import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from vae_models import *
import utils
from main_logger import get_logger
import argparse
import random
import datetime
import os
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help="Name of the dataset", default="adult_income", choices=["adult_income"])
parser.add_argument('--run_number', type=str, help = 'Number for the run. If not given - assigned to datetime of the run')
parser.add_argument('--epochs', type=int, help="Number of epochs", default=30)
parser.add_argument('--learning_rate', type=float, help="Learning rate", default=1e-3)
parser.add_argument('--train_batch_size', type=int, help="Training batch size", default=64)
parser.add_argument('--test_batch_size', type=int, help="Testing batch size", default=64)
parser.add_argument('--vae_hidden_dim', type=int, help="VAE hidden dimension", default=64)
parser.add_argument('--vae_latent_dim', type=int, help="VAE latent dimension", default=16)
parser.add_argument('--patience', type=int, help="Patience for early stopping", default=7)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--split', action='store_true', help='Condition dataset on some x_A')
parser.add_argument('--xA', type=str, help="Value of xA: will be considered only if --split is provided")
parser.add_argument('--vae', type=str, help="Type of VAE to be used", default="vae", choices=["vae", "rbvae", "mgvae"])
args = parser.parse_args()


def train(model, dataloader, len_train, criterion, optimizer, device, **kwargs):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len_train/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        if args.vae == 'vae':
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = model.vae_loss(bce_loss, mu, logvar)
        
        elif args.vae == 'rbvae':
            reconstruction, mu = model(data, **kwargs)
            bce_loss = criterion(reconstruction, data)
            loss = model.vae_loss(bce_loss, mu)

        elif args.vae == 'mgvae':
            #print('$$$$', data.shape)            
            reconstruction, mu, logvar, logits_ = model(data, device, **kwargs)
            #print('####', reconstruction.shape)
            bce_loss = criterion(reconstruction, data)
            loss = model.vae_loss(bce_loss, mu, logvar, logits_)

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss

def validate(model, dataloader, len_test, criterion, device, **kwargs):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len_test/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            if args.vae == 'vae':
                reconstruction, mu, logvar = model(data)
                bce_loss = criterion(reconstruction, data)
                loss = model.vae_loss(bce_loss, mu, logvar)
            elif args.vae == 'rbvae':
                reconstruction, mu = model(data, **kwargs)
                bce_loss = criterion(reconstruction, data)
                loss = model.vae_loss(bce_loss, mu)
            elif args.vae == 'mgvae':
                reconstruction, mu, logvar, logits_ = model(data, device, **kwargs)
                bce_loss = criterion(reconstruction, data)
                loss = model.vae_loss(bce_loss, mu, logvar, logits_)

            running_loss += loss.item()

    val_loss = running_loss/len(dataloader.dataset)
    return val_loss


def trainer(model, train_loader, test_loader, len_train, len_test, device, save_path=f'checkpoints/{args.dataset}/{args.vae}', **kwargs):
    early_stopping = utils.EarlyStopping(args.patience, args.verbose, save_path=save_path, trace_func=file_logger.info)
    train_loss = []
    val_loss = []
    criterion = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    best_model_path = f'{save_path}/best.pt'
    for epoch in range(args.epochs):
        train_epoch_loss = train(model, train_loader, len_train, criterion, optimizer, device, **kwargs)
        val_epoch_loss = validate(model, test_loader, len_test, criterion, device, **kwargs)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        early_stopping(val_epoch_loss, model)
        file_logger.info(f"Epoch: {epoch+1}/{args.epochs}\tTrain Loss: {train_epoch_loss:.4f}\tVal Loss: {val_epoch_loss:.4f}")
        if early_stopping.early_stop:
            file_logger.critical('Triggering Early Stop')
            break

        if args.vae == 'rbvae':
            kwargs['steps'] += 1
            if kwargs['steps'] % kwargs['update_time'] == 0:
                kwargs['tau'] = max(kwargs['tau'] * np.exp(-kwargs['anneal_rate']*kwargs['steps']), kwargs['tau_min'])
        
    model.load_state_dict(torch.load(best_model_path))

    return model, train_loss, val_loss, epoch




if __name__ == '__main__':
    if args.run_number is None:
        args.run_number = datetime.datetime.now().isoformat()
    
    log_dir = f'logs/{args.dataset}/{args.vae}_logs'
    log_file = f'{log_dir}/{args.run_number}_training.log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok = True)
    file_logger = get_logger('VAE Training', log_file)
    file_logger.info(f'Run Number: {args.run_number}')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    VAE, kwargs = utils.vae_map(args.vae)

    if not args.split:
        train_loader, test_loader, len_train, len_test, feature_dim, num_cols = utils.get_dataset(args.dataset, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
        model = VAE(feature_dim, args.vae_hidden_dim, args.vae_latent_dim).to(device)
        model, train_loss, val_loss, last_epoch = trainer(model, train_loader, test_loader, len_train, len_test, device, **kwargs)
        utils.plot_loss(range(0, last_epoch+1), train_loss, val_loss, args.dataset, args.run_number, vae=args.vae)
    
    else:
        # xA = 1
        
        train_loader_1, test_loader_1, len_train_1, len_test_1, feature_dim_1, num_cols_1, train_loader_0, test_loader_0, len_train_0, len_test_0, feature_dim_0, num_cols_0 = utils.get_split_dataset(args.dataset, args.xA, train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size)
        file_logger.info(f'Training for {args.xA} = 1 with {len_train_1} datapoints')
        model_1 = VAE(feature_dim_1, args.vae_hidden_dim, args.vae_latent_dim).to(device)
        model_1, train_loss_1, val_loss_1, last_epoch_1 = trainer(model_1, train_loader_1, test_loader_1, len_train_1, len_test_1, device, save_path=f'checkpoints/{args.dataset}/{args.vae}_xA=1', **kwargs)
        utils.plot_loss(range(0, last_epoch_1+1), train_loss_1, val_loss_1, args.dataset, args.run_number, specific_name=f'{args.xA}=1', vae=args.vae)

        # xA = 0
        file_logger.info(f'Training for {args.xA} = 0 with {len_train_0} datapoints')
        model_0 = VAE(feature_dim_0, args.vae_hidden_dim, args.vae_latent_dim).to(device)
        model_0, train_loss_0, val_loss_0, last_epoch_0 = trainer(model_0, train_loader_0, test_loader_0, len_train_0, len_test_0, device, save_path=f'checkpoints/{args.dataset}/{args.vae}_xA=0', **kwargs)
        utils.plot_loss(range(0, last_epoch_0+1), train_loss_0, val_loss_0, args.dataset, args.run_number, specific_name=f'{args.xA}=0', vae=args.vae)